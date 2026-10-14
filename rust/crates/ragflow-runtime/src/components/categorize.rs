//! Categorize: one LLM call picks a category; the run goes to that
//! category's route.
//!
//! Prompt text, parsing and matching follow upstream's `categorize.go`
//! (main) byte for byte where it matters: categories sorted in the
//! prompt, examples quoted Go-style (`%q`), the answer matched exactly,
//! then case-insensitively, else the default category (no substring
//! matching). Routes come from `items[].to[0]` (else `uuid`) and are
//! overridden by the legacy `category_description[name].to[0]`.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::{json, Map, Value};

use crate::graph::Node;
use crate::models::{ChatMessage, ChatOptions, ModelRef, Models};
use crate::run::{Component, NodeCtx};
use crate::state::Outputs;

pub struct Categorize {
    id: String,
    models: Arc<dyn Models>,
    /// `None` fails the run when the node runs, as upstream does.
    model: Option<ModelRef>,
    /// DSL order; the default category is the first.
    categories: Vec<String>,
    descriptions: BTreeMap<String, String>,
    examples: BTreeMap<String, Vec<String>>,
    routes: BTreeMap<String, String>,
    sys_prompt: String,
    default_category: String,
    query_ref: String,
}

pub fn factory(node: &Node, models: Arc<dyn Models>) -> Result<Box<dyn Component>, String> {
    let p = node.params.as_object().cloned().unwrap_or_default();
    let model_id = text(&p, "model_id").or_else(|| text(&p, "llm_id"));
    let mut categories = Vec::new();
    let mut descriptions = BTreeMap::new();
    let mut examples: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut routes = BTreeMap::new();

    for item in p
        .get("items")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some(item) = item.as_object() else {
            continue;
        };
        let Some(name) = text(item, "name") else {
            continue;
        };
        categories.push(name.clone());
        if let Some(desc) = text(item, "description") {
            descriptions.insert(name.clone(), desc);
        }
        if let Some(route) = first_route(item.get("to")).or_else(|| text(item, "uuid")) {
            routes.insert(name.clone(), route);
        }
        for example in item
            .get("examples")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let value = match example {
                Value::String(s) => Some(s.clone()),
                Value::Object(m) => text(m, "value"),
                _ => None,
            };
            if let Some(value) = value.filter(|v| !v.is_empty()) {
                examples.entry(name.clone()).or_default().push(value);
            }
        }
    }
    if let Some(legacy) = p.get("category_description").and_then(Value::as_object) {
        let mut names: Vec<String> = legacy.keys().cloned().collect();
        names.sort();
        if categories.is_empty() {
            categories = names;
        }
        for (name, child) in legacy {
            match child {
                Value::String(desc) => {
                    descriptions.insert(name.clone(), desc.clone());
                }
                Value::Object(nested) => {
                    if let Some(desc) = nested.get("description").and_then(Value::as_str) {
                        descriptions.insert(name.clone(), desc.to_string());
                    }
                    let list: Vec<String> = nested
                        .get("examples")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                        .filter_map(|e| match e {
                            Value::String(s) if !s.is_empty() => Some(s.clone()),
                            Value::Object(m) => text(m, "value"),
                            _ => None,
                        })
                        .collect();
                    examples.insert(name.clone(), list);
                    if let Some(route) =
                        first_route(nested.get("to")).or_else(|| text(nested, "uuid"))
                    {
                        routes.insert(name.clone(), route);
                    }
                }
                _ => {}
            }
        }
    }
    if categories.is_empty() {
        return Err("categories: at least one category is required".into());
    }
    let default_category = text(&p, "default_category").unwrap_or_else(|| categories[0].clone());
    Ok(Box::new(Categorize {
        id: node.id.clone(),
        models,
        model: model_id.as_deref().map(ModelRef::parse).transpose()?,
        categories,
        descriptions,
        examples,
        routes,
        sys_prompt: text(&p, "sys_prompt")
            .or_else(|| text(&p, "system_prompt"))
            .unwrap_or_default(),
        default_category,
        query_ref: text(&p, "query").unwrap_or_else(|| "sys.query".into()),
    }))
}

fn text(map: &Map<String, Value>, key: &str) -> Option<String> {
    map.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

fn first_route(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(s) if !s.is_empty() => Some(s.clone()),
        Value::Array(items) => items
            .first()?
            .as_str()
            .filter(|s| !s.is_empty())
            .map(str::to_string),
        _ => None,
    }
}

impl Categorize {
    pub fn system_prompt(&self) -> String {
        if !self.sys_prompt.is_empty() {
            return self.sys_prompt.clone();
        }
        let mut cats = self.categories.clone();
        cats.sort();
        let mut b = String::from(
            "You are an advanced classification system that categorizes user questions into specific types. Analyze the input question and classify it into ONE of the following categories:\n",
        );
        for c in &cats {
            b.push_str(&format!("- {c}\n"));
        }
        if !self.descriptions.is_empty() {
            b.push_str("\nHere's description of each category:\n");
            for c in &cats {
                let desc = self
                    .descriptions
                    .get(c)
                    .map(|d| d.trim())
                    .unwrap_or_default();
                if !desc.is_empty() {
                    b.push_str(&format!("\n------\nCategory: {c}\nDescription: {desc}\n"));
                }
            }
        }
        b.push_str("\n---- Instructions ----\n");
        b.push_str("- Consider both explicit mentions and implied context\n");
        b.push_str("- Prioritize the most specific applicable category\n");
        b.push_str("- Return only the category name without explanations\n");
        if cats.iter().any(|c| c == "Other") {
            b.push_str("- Use \"Other\" only when no other category fits\n");
        }
        let mut lines = Vec::new();
        for c in &cats {
            for example in self.examples.get(c).into_iter().flatten() {
                let example = example.replace('\n', "    ");
                let example = example.trim();
                if !example.is_empty() {
                    lines.push(format!("USER: {} -> {c}", go_quote(example)));
                }
            }
        }
        if !lines.is_empty() {
            b.push_str("\n---- Examples ----\n");
            for line in lines {
                b.push_str(&line);
                b.push('\n');
            }
        }
        b
    }

    /// With no session history the window holds only the current query.
    pub fn user_prompt(query: &str) -> String {
        let content = query.replace("\r\n", "").replace(['\n', '\r'], "");
        format!("\n---- Real Data ----\nUSER: {} ->\n", go_quote(&content))
    }

    fn pick(&self, response: &str) -> String {
        let r = response.trim().trim_matches(|c| "\"'`\n\r\t ".contains(c));
        let r = r.strip_prefix("category:").unwrap_or(r);
        let r = r.strip_prefix("Category:").unwrap_or(r).trim();
        if let Some(c) = self.categories.iter().find(|c| *c == r) {
            return c.clone();
        }
        let lower = r.to_lowercase();
        self.categories
            .iter()
            .find(|c| c.to_lowercase() == lower)
            .cloned()
            .unwrap_or_else(|| self.default_category.clone())
    }
}

impl Component for Categorize {
    fn invoke(&self, ctx: &mut NodeCtx<'_>) -> Result<Outputs, String> {
        let Some(model) = &self.model else {
            return Err(r#"component: invalid param "model_id": required"#.into());
        };
        let query = match ctx.inputs.get("query") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Null) | None => ctx
                .state
                .lookup(&self.query_ref)
                .map(|v| crate::template::to_text(&v))
                .unwrap_or_default(),
            Some(other) => other.to_string(),
        };
        let messages = [
            ChatMessage {
                role: "system",
                content: self.system_prompt(),
            },
            ChatMessage {
                role: "user",
                content: Self::user_prompt(&query),
            },
        ];
        let answer = self
            .models
            .chat_for(&self.id, model, &messages, &ChatOptions::default())
            .map_err(|e| format!("component: Categorize.Invoke: {e}"))?;
        let chosen = self.pick(&answer);
        let scores: Map<String, Value> = self
            .categories
            .iter()
            .map(|c| (c.clone(), json!(if *c == chosen { 1.0 } else { 0.0 })))
            .collect();
        let next: Vec<String> = self.routes.get(&chosen).cloned().into_iter().collect();
        let mut out = Outputs::new();
        out.insert("category".into(), json!(chosen));
        out.insert("category_name".into(), json!(chosen));
        out.insert("scores".into(), Value::Object(scores));
        out.insert("_next".into(), json!(next));
        Ok(out)
    }
}

/// Go's `%q` (`strconv.Quote`): printable characters, CJK included, stay
/// as they are; quotes, backslashes and control characters are escaped.
pub fn go_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{07}' => out.push_str("\\a"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\u{0b}' => out.push_str("\\v"),
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32))
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

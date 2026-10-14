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
use crate::template::go_v;

pub struct Categorize {
    id: String,
    models: Arc<dyn Models>,
    /// Raw model id; empty fails the run when the node runs, as upstream
    /// does. The data upstream's `model_id` / `llm_id` replaces it.
    model_id: String,
    /// DSL order; the default category is the first.
    categories: Vec<String>,
    descriptions: BTreeMap<String, String>,
    examples: BTreeMap<String, Vec<String>>,
    routes: BTreeMap<String, String>,
    sys_prompt: String,
    default_category: String,
    query_ref: String,
    /// `message_history_window_size`: how many history entries (the current
    /// turn included) the "Real Data" prompt lists.
    history_window: i64,
}

pub fn factory(node: &Node, models: Arc<dyn Models>) -> Result<Box<dyn Component>, String> {
    let p = node.params.as_object().cloned().unwrap_or_default();
    // upstream's stringFrom: a present string counts, even empty.
    let model_id = string_from(&p, "model_id")
        .or_else(|| string_from(&p, "llm_id"))
        .unwrap_or_default()
        .to_string();
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
    let default_category = string_from(&p, "default_category")
        .unwrap_or_default()
        .to_string();
    Ok(Box::new(Categorize {
        id: node.id.clone(),
        models,
        model_id,
        categories,
        descriptions,
        examples,
        routes,
        sys_prompt: string_from(&p, "sys_prompt")
            .or_else(|| string_from(&p, "system_prompt"))
            .unwrap_or_default()
            .to_string(),
        default_category,
        query_ref: string_from(&p, "query").unwrap_or_default().to_string(),
        // categorize.go: defaults to 1; intFrom truncates a JSON number.
        history_window: p
            .get("message_history_window_size")
            .and_then(Value::as_f64)
            .map_or(1, |v| v as i64),
    }))
}

fn text(map: &Map<String, Value>, key: &str) -> Option<String> {
    map.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

fn string_from<'m>(map: &'m Map<String, Value>, key: &str) -> Option<&'m str> {
    map.get(key).and_then(Value::as_str)
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
    /// The generated prompt, or `sys_prompt` verbatim when set.
    pub fn system_prompt(&self) -> String {
        self.system_prompt_with(&self.sys_prompt)
    }

    fn system_prompt_with(&self, sys_prompt: &str) -> String {
        if !sys_prompt.is_empty() {
            return sys_prompt.to_string();
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

    /// `buildCategorizePrompt(categorizeHistory(...))`: the last `window`
    /// history entries (the current turn is the last one), each as
    /// `ROLE: "content"`, joined by ` | `; the last entry's content is the
    /// resolved query. With no usable history the window is the query alone.
    pub fn user_prompt(history: &[Map<String, Value>], window: i64, query: &str) -> String {
        let mut turns: Vec<(String, String)> = Vec::new();
        if window > 0 {
            let start = history.len().saturating_sub(window as usize);
            for entry in &history[start..] {
                let role = string_from(entry, "role").unwrap_or_default();
                let content = string_from(entry, "content").unwrap_or_default();
                if !role.is_empty() && !content.is_empty() {
                    turns.push((role.to_string(), content.to_string()));
                }
            }
        }
        if turns.is_empty() {
            turns.push(("user".into(), String::new()));
        }
        if let Some(last) = turns.last_mut() {
            last.1 = query.to_string();
        }
        let parts: Vec<String> = turns
            .iter()
            .map(|(role, content)| {
                let role = role.trim().to_uppercase();
                let role = if role.is_empty() { "USER".into() } else { role };
                let content = content.replace("\r\n", "").replace(['\n', '\r'], "");
                format!("{role}: {}", go_quote(&content))
            })
            .collect();
        format!("\n---- Real Data ----\n{} ->\n", parts.join(" | "))
    }

    fn pick(&self, response: &str, default_category: &str) -> String {
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
            .unwrap_or_else(|| default_category.to_string())
    }
}

impl Component for Categorize {
    fn invoke(&self, ctx: &mut NodeCtx<'_>) -> Result<Outputs, String> {
        // mergeCategorizeParam: the data upstream's outputs override the
        // string params this runtime acts on.
        let inputs = ctx.inputs;
        let model_id = string_from(inputs, "model_id")
            .or_else(|| string_from(inputs, "llm_id"))
            .unwrap_or(&self.model_id);
        let query_ref = string_from(inputs, "query").unwrap_or(&self.query_ref);
        let sys_prompt = string_from(inputs, "sys_prompt")
            .or_else(|| string_from(inputs, "system_prompt"))
            .unwrap_or(&self.sys_prompt);
        let default_category =
            match string_from(inputs, "default_category").unwrap_or(&self.default_category) {
                "" => self.categories[0].as_str(),
                d => d,
            };
        if model_id.is_empty() {
            return Err(r#"component: invalid param "model_id": required"#.into());
        }
        let model = ModelRef::parse(model_id)?;
        // resolveCategorizeQuery: the upstream's `query`, else the upstream
        // value under the query reference, else the reference itself.
        let query_ref = match query_ref.trim() {
            "" => "sys.query",
            r => r,
        };
        let any_text = |v: &Value| match v {
            Value::String(s) => s.clone(),
            other => go_v(other),
        };
        let query = match inputs.get("query").filter(|v| !v.is_null()) {
            Some(v) => any_text(v),
            None => match inputs
                .get(query_ref)
                .filter(|v| !v.is_null())
                .map(any_text)
                .filter(|v| !v.trim().is_empty())
            {
                Some(v) => v,
                None => ctx
                    .state
                    .lookup(query_ref)
                    .map(|v| any_text(&v))
                    .unwrap_or_default(),
            },
        };
        // mergeCategorizeParam: the upstream's window overrides the param.
        let window = inputs
            .get("message_history_window_size")
            .and_then(Value::as_f64)
            .map_or(self.history_window, |v| v as i64);
        let messages = [
            ChatMessage {
                role: "system",
                content: self.system_prompt_with(sys_prompt),
            },
            ChatMessage {
                role: "user",
                content: Self::user_prompt(&ctx.state.history, window, &query),
            },
        ];
        let answer = self
            .models
            .chat_for(&self.id, &model, &messages, &ChatOptions::default())
            .map_err(|e| format!("component: Categorize.Invoke: {e}"))?;
        let chosen = self.pick(&answer, default_category);
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

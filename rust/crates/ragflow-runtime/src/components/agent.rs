//! Agent with no tools: one chat call with the node's system and user
//! prompts, both strictly rendered, following upstream's `agent.go`
//! (`invokeNow`, `mergeAgentParam`) on main.
//!
//! Tools, MCP servers and citations (`cite: true`) are refused when the
//! canvas is built, so a Bot never silently runs without them. A missing
//! model id is not: upstream reports it when the Agent runs
//! (`component: invalid param "model_id": required`), and so does this.

use std::sync::Arc;

use serde_json::{json, Map, Value};

use crate::graph::Node;
use crate::models::{ChatMessage, ChatOptions, ModelRef, Models};
use crate::run::{Component, NodeCtx};
use crate::state::Outputs;
use crate::template::render_strict;

/// The placeholder upstream's editor puts in an empty user prompt.
const USER_PROMPT_PLACEHOLDER: &str = "This is the order you need to send to the agent.";

pub struct Agent {
    id: String,
    models: Arc<dyn Models>,
    model: Option<ModelRef>,
    system_prompt: String,
    user_prompt: String,
    options: ChatOptions,
}

pub fn factory(node: &Node, models: Arc<dyn Models>) -> Result<Box<dyn Component>, String> {
    let p = node.params.as_object().cloned().unwrap_or_default();
    let model_id = text(&p, "model_id").or_else(|| text(&p, "llm_id"));
    for key in ["tools", "mcp"] {
        if p.get(key)
            .and_then(Value::as_array)
            .is_some_and(|a| !a.is_empty())
        {
            return Err(format!("{key}: not supported by this runtime yet"));
        }
    }
    if p.get("cite").and_then(Value::as_bool) == Some(true) {
        return Err("cite: not supported by this runtime yet".into());
    }

    let mut system_prompt = text(&p, "system_prompt")
        .or_else(|| text(&p, "sys_prompt"))
        .unwrap_or_default();
    let mut user_prompt = String::new();
    if let Some((system, user)) = prompt_messages(p.get("prompts")) {
        system_prompt = append_prompt(&system_prompt, &system);
        if !user.trim().is_empty() {
            user_prompt = user;
        }
    }
    if let Some(v) = text(&p, "user_prompt").filter(|v| !falls_back(v)) {
        user_prompt = v;
    }
    // A sampler is sent only when set and its editor switch is not off
    // (`gateSampler`, llm.go; applied in mergeAgentParam, agent.go).
    let enabled = |flag: &str| p.get(flag).and_then(Value::as_bool) != Some(false);
    let options = ChatOptions {
        temperature: p
            .get("temperature")
            .and_then(Value::as_f64)
            .filter(|_| enabled("temperatureEnabled")),
        top_p: p
            .get("top_p")
            .and_then(Value::as_f64)
            .filter(|_| enabled("topPEnabled")),
        max_tokens: p
            .get("max_tokens")
            .and_then(Value::as_u64)
            .filter(|_| enabled("maxTokensEnabled")),
    };
    Ok(Box::new(Agent {
        id: node.id.clone(),
        models,
        model: model_id.as_deref().map(ModelRef::parse).transpose()?,
        system_prompt,
        user_prompt,
        options,
    }))
}

fn text(map: &Map<String, Value>, key: &str) -> Option<String> {
    map.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

fn falls_back(prompt: &str) -> bool {
    let p = prompt.trim();
    p.is_empty() || p == USER_PROMPT_PLACEHOLDER
}

fn append_prompt(base: &str, extra: &str) -> String {
    if extra.trim().is_empty() {
        base.to_string()
    } else if base.trim().is_empty() {
        extra.to_string()
    } else {
        format!("{base}\n{extra}")
    }
}

/// `prompts` as a string (user) or `[{role, content}]`, joined per role.
fn prompt_messages(value: Option<&Value>) -> Option<(String, String)> {
    match value? {
        Value::String(s) => Some((String::new(), s.clone())),
        Value::Array(items) => {
            let (mut systems, mut users) = (Vec::new(), Vec::new());
            for item in items.iter().filter_map(Value::as_object) {
                let Some(content) = item.get("content").and_then(Value::as_str) else {
                    continue;
                };
                match item
                    .get("role")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .trim()
                    .to_lowercase()
                    .as_str()
                {
                    "system" => systems.push(content.to_string()),
                    "user" | "" => users.push(content.to_string()),
                    _ => {}
                }
            }
            (!systems.is_empty() || !users.is_empty())
                .then(|| (systems.join("\n"), users.join("\n")))
        }
        _ => None,
    }
}

impl Component for Agent {
    fn invoke(&self, ctx: &mut NodeCtx<'_>) -> Result<Outputs, String> {
        if ctx.inputs.get("_ERROR").and_then(Value::as_str) == Some("No dataset is selected.") {
            let mut out = Outputs::new();
            out.insert("content".into(), json!("No dataset is selected."));
            return Ok(out);
        }
        let system = render_strict(&self.system_prompt, ctx.state).map_err(|e| e.to_string())?;
        let mut user = render_strict(&self.user_prompt, ctx.state).map_err(|e| e.to_string())?;
        if falls_back(&user) && system.trim().is_empty() {
            user = ctx
                .state
                .lookup("sys.query")
                .map(|v| crate::template::to_text(&v))
                .unwrap_or_default();
        }
        let Some(model) = &self.model else {
            return Err(r#"component: invalid param "model_id": required"#.into());
        };
        if user.is_empty() && system.is_empty() {
            return Err(r#"component: invalid param "user_prompt": at least one of user_prompt or system_prompt must be set"#.into());
        }
        if user.is_empty() {
            user = system.clone();
        }
        let mut messages = Vec::new();
        if !system.trim().is_empty() {
            messages.push(ChatMessage {
                role: "system",
                content: system,
            });
        }
        messages.push(ChatMessage {
            role: "user",
            content: user,
        });
        let content = self
            .models
            .chat_for(&self.id, model, &messages, &self.options)?;

        let mut out = Outputs::new();
        out.insert("content".into(), json!(content));
        out.insert("thinking".into(), json!(""));
        out.insert("tool_calls".into(), json!([]));
        out.insert("artifacts".into(), json!([]));
        Ok(out)
    }
}

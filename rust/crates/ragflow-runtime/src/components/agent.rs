//! Agent with no tools: one chat call with the node's system and user
//! prompts, both strictly rendered, following upstream's `agent.go`
//! (`invokeNow`, `mergeAgentParam`) on main.
//!
//! Params are layered as upstream layers them: the DSL params over the
//! defaults when the canvas is built, then the data upstream's outputs over
//! those on every run (`mergeAgentParam(c.param, inputs)`), so an upstream
//! output named like a param (`user_prompt`, `sys_prompt`, `llm_id`, ...)
//! replaces it. A string param counts when present, even empty.
//!
//! Tools, MCP servers and citations (`cite: true`) are refused when the
//! canvas is built (and when an upstream output switches them on), so a Bot
//! never silently runs without them. A missing model id is not refused at
//! build: upstream reports it when the Agent runs
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
    base: Params,
}

/// The subset of upstream's `AgentParam` this runtime acts on.
#[derive(Clone, Default)]
struct Params {
    model_id: String,
    system_prompt: String,
    user_prompt: String,
    temperature: Option<f64>,
    temperature_off: bool,
    top_p: Option<f64>,
    top_p_off: bool,
    max_tokens: Option<u64>,
    max_tokens_off: bool,
    tools: bool,
    mcp: bool,
    cite: bool,
}

impl Params {
    fn refusal(&self) -> Option<String> {
        if self.tools {
            Some("tools: not supported by this runtime yet".into())
        } else if self.mcp {
            Some("mcp: not supported by this runtime yet".into())
        } else if self.cite {
            Some("cite: not supported by this runtime yet".into())
        } else {
            None
        }
    }

    fn options(&self) -> ChatOptions {
        ChatOptions {
            temperature: self.temperature,
            top_p: self.top_p,
            max_tokens: self.max_tokens,
        }
    }
}

/// upstream's `stringFrom`: a string value, empty included.
fn string_from<'m>(map: &'m Map<String, Value>, key: &str) -> Option<&'m str> {
    map.get(key).and_then(Value::as_str)
}

/// upstream's `floatFrom` / `intFrom` on JSON numbers (`intFrom`
/// truncates).
fn float_from(map: &Map<String, Value>, key: &str) -> Option<f64> {
    map.get(key).and_then(Value::as_f64)
}

/// upstream's `mergeAgentParam`, for the params this runtime acts on.
fn merge(base: &Params, m: &Map<String, Value>) -> Params {
    let mut p = base.clone();
    if let Some(v) = string_from(m, "model_id").or_else(|| string_from(m, "llm_id")) {
        p.model_id = v.to_string();
    }
    if let Some(v) = string_from(m, "system_prompt").or_else(|| string_from(m, "sys_prompt")) {
        p.system_prompt = v.to_string();
    }
    if let Some((system, user)) = prompt_messages(m.get("prompts")) {
        p.system_prompt = append_prompt(&p.system_prompt, &system);
        if !user.trim().is_empty() {
            p.user_prompt = user;
        }
    }
    if let Some(v) = string_from(m, "user_prompt").filter(|v| !falls_back(v)) {
        p.user_prompt = v.to_string();
    }
    // gateSampler (llm.go): a switch set to false keeps the value off for
    // every later merge.
    let gate = |flag: &str, off: &mut bool| {
        if let Some(enabled) = m.get(flag).and_then(Value::as_bool) {
            *off = !enabled;
        }
        *off
    };
    if let Some(v) = float_from(m, "top_p") {
        p.top_p = Some(v);
    }
    if gate("topPEnabled", &mut p.top_p_off) {
        p.top_p = None;
    }
    if let Some(v) = float_from(m, "max_tokens") {
        p.max_tokens = Some(v as u64);
    }
    if gate("maxTokensEnabled", &mut p.max_tokens_off) {
        p.max_tokens = None;
    }
    if let Some(v) = float_from(m, "temperature") {
        p.temperature = Some(v);
    }
    if gate("temperatureEnabled", &mut p.temperature_off) {
        p.temperature = None;
    }
    if let Some(v) = m.get("mcp") {
        p.mcp = match v {
            Value::Null => false,
            Value::Array(a) => !a.is_empty(),
            Value::Object(o) => !o.is_empty(),
            _ => true,
        };
    }
    if let Some(Value::Array(tools)) = m.get("tools") {
        p.tools = tools.iter().any(|t| match t {
            Value::String(s) => !s.trim().is_empty(),
            _ => true,
        });
    }
    if let Some(v) = m.get("cite").and_then(Value::as_bool) {
        p.cite = v;
    }
    p
}

pub fn factory(node: &Node, models: Arc<dyn Models>) -> Result<Box<dyn Component>, String> {
    let params = node.params.as_object().cloned().unwrap_or_default();
    let base = merge(&Params::default(), &params);
    if let Some(refusal) = base.refusal() {
        return Err(refusal);
    }
    Ok(Box::new(Agent {
        id: node.id.clone(),
        models,
        base,
    }))
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

/// upstream's `formatAgentRuntimePrompt`: an upstream `user_prompt` is
/// wrapped with the upstream's `reasoning` / `context` when present.
fn runtime_prompt(inputs: &Map<String, Value>, user: &str) -> String {
    let mut b = String::new();
    if let Some(r) = string_from(inputs, "reasoning").filter(|r| !r.is_empty()) {
        b.push_str(&format!("\nREASONING:\n{r}\n"));
    }
    if let Some(c) = string_from(inputs, "context").filter(|c| !c.is_empty()) {
        b.push_str(&format!("\nCONTEXT:\n{c}\n"));
    }
    if b.is_empty() {
        return user.to_string();
    }
    b.push_str(&format!("\nQUERY:\n{user}\n"));
    b
}

impl Component for Agent {
    fn invoke(&self, ctx: &mut NodeCtx<'_>) -> Result<Outputs, String> {
        let p = merge(&self.base, ctx.inputs);
        if let Some(refusal) = p.refusal() {
            return Err(refusal);
        }
        let runtime_user = string_from(ctx.inputs, "user_prompt").is_some_and(|v| !falls_back(v));
        if ctx.inputs.get("_ERROR").and_then(Value::as_str) == Some("No dataset is selected.") {
            let mut out = Outputs::new();
            out.insert("content".into(), json!("No dataset is selected."));
            return Ok(out);
        }
        let system = render_strict(&p.system_prompt, ctx.state).map_err(|e| e.to_string())?;
        let mut user = render_strict(&p.user_prompt, ctx.state).map_err(|e| e.to_string())?;
        if runtime_user {
            user = runtime_prompt(ctx.inputs, &user);
        } else if falls_back(&user) && system.trim().is_empty() {
            if let Some(Value::String(q)) = ctx.state.sys.get("query") {
                if !q.trim().is_empty() {
                    user = q.clone();
                }
            }
        }
        if p.model_id.is_empty() {
            return Err(r#"component: invalid param "model_id": required"#.into());
        }
        if user.is_empty() && system.is_empty() {
            return Err(r#"component: invalid param "user_prompt": at least one of user_prompt or system_prompt must be set"#.into());
        }
        if user.is_empty() {
            user = system.clone();
        }
        let model = ModelRef::parse(&p.model_id)?;
        let mut messages = Vec::new();
        if !system.is_empty() {
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
            .chat_for(&self.id, &model, &messages, &p.options())?;

        let mut out = Outputs::new();
        out.insert("content".into(), json!(content));
        out.insert("thinking".into(), json!(""));
        out.insert("tool_calls".into(), json!([]));
        out.insert("artifacts".into(), json!([]));
        Ok(out)
    }
}

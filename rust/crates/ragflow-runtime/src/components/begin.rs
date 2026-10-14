//! Begin: puts the visitor's query into `sys.query` and fills the canvas's
//! declared input fields.

use serde_json::Value;

use crate::graph::Node;
use crate::run::{Component, NodeCtx};
use crate::state::Outputs;

pub struct Begin {
    fields: Vec<String>,
}

pub fn factory(node: &Node) -> Result<Box<dyn Component>, String> {
    let mut fields: Vec<String> = node
        .params
        .get("inputs")
        .and_then(Value::as_object)
        .map(|m| m.keys().cloned().collect())
        .unwrap_or_default();
    fields.sort();
    Ok(Box::new(Begin { fields }))
}

impl Component for Begin {
    fn invoke(&self, ctx: &mut NodeCtx<'_>) -> Result<Outputs, String> {
        let query = ctx.inputs.get("query").cloned().unwrap_or(Value::Null);
        let text = match &query {
            Value::String(s) => s.clone(),
            Value::Null => String::new(),
            other => other.to_string(),
        };
        ctx.state.sys.insert("query".into(), Value::String(text));
        for key in ["user_id", "webhook_payload"] {
            if let Some(value) = ctx.inputs.get(key) {
                ctx.state.sys.insert(key.into(), value.clone());
            }
        }

        // Upstream passes its inputs through (begin.go `mapsCopy`) and adds
        // only the declared fields; there is no `inputs` output.
        let mut out = ctx.inputs.clone();
        for field in &self.fields {
            let cache = format!("begin@{field}");
            let value = ctx
                .inputs
                .get(field)
                .cloned()
                .or_else(|| query.get(field).cloned())
                .or_else(|| ctx.state.globals.get(&cache).cloned())
                .or_else(|| query.is_string().then(|| query.clone()));
            if let Some(value) = value {
                ctx.state
                    .globals
                    .entry(cache)
                    .or_insert_with(|| value.clone());
                out.insert(field.clone(), value);
            }
        }
        Ok(out)
    }
}

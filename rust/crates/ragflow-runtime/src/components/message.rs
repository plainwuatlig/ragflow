//! Message: renders the reply the visitor reads.
//!
//! Which text it renders follows upstream: a `text` or `content` string
//! arriving from the data upstream wins over the node's own template,
//! except when that upstream is an Agent deferring its answer to this
//! Message, where the template wins. With no template at all it falls back
//! to an upstream `formalized_content`.

use serde_json::{json, Value};

use crate::event::Event;
use crate::graph::Node;
use crate::run::{Component, NodeCtx};
use crate::state::Outputs;
use crate::template::{render_display, render_strict};

pub struct Message {
    template: String,
}

pub fn factory(node: &Node) -> Result<Box<dyn Component>, String> {
    let params = &node.params;
    let template = first_text(params.get("text"))
        .or_else(|| first_text(params.get("content")))
        .unwrap_or_default();
    Ok(Box::new(Message { template }))
}

/// A string, or the first element of a list (upstream takes `[0]`).
fn first_text(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(s) => Some(s.clone()),
        Value::Array(items) => items.first()?.as_str().map(str::to_string),
        _ => None,
    }
}

impl Component for Message {
    fn invoke(&self, ctx: &mut NodeCtx<'_>) -> Result<Outputs, String> {
        let from_upstream = if ctx.upstream_defers {
            None
        } else {
            first_text(ctx.inputs.get("text")).or_else(|| first_text(ctx.inputs.get("content")))
        };
        let template = match from_upstream {
            Some(text) => text,
            None if !self.template.is_empty() => self.template.clone(),
            None => first_text(ctx.inputs.get("formalized_content")).unwrap_or_default(),
        };
        // Upstream checks every reference strictly, then renders for display
        // (non-string values as JSON).
        render_strict(&template, ctx.state).map_err(|e| e.to_string())?;
        let content = render_display(&template, ctx.state);
        ctx.events.emit(Event::Message {
            content: content.clone(),
        });
        let mut out = Outputs::new();
        out.insert("content".into(), Value::String(content));
        out.insert("downloads".into(), json!([]));
        Ok(out)
    }
}

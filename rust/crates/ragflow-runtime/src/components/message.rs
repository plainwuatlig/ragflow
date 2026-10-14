//! Message: renders the reply the visitor reads, following upstream's
//! `message.go` on main (`Invoke`, `resolveDeferredTemplate`).
//!
//! Which text it renders: an upstream `text` input overrides; otherwise the
//! node's own template is authoritative; with no template, an upstream
//! `text`/`content` string; then an upstream `formalized_content`.
//!
//! References are checked strictly, then rendered for display (non-string
//! values as JSON). When the template references a deferred Agent's
//! `content`, the Message runs that Agent in place: literal text, the
//! Agent's answer and other values are emitted as separate `message` events
//! in template order, and the Agent finishes as soon as its answer is in.

use serde_json::{json, Value};

use crate::event::Event;
use crate::graph::Node;
use crate::run::{Component, NodeCtx};
use crate::state::Outputs;
use crate::template::{reference_spans, render_display, TemplateError};

pub struct Message {
    template: String,
}

pub fn factory(node: &Node) -> Result<Box<dyn Component>, String> {
    Ok(Box::new(Message {
        template: extract_text(&node.params),
    }))
}

/// upstream's `extractMessageText`: a string `text`, else a string
/// `content` or the first element of a `content` list.
fn extract_text(map: &Value) -> String {
    if let Some(Value::String(s)) = map.get("text") {
        return s.clone();
    }
    match map.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(items)) => items
            .first()
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        _ => String::new(),
    }
}

/// The deferred Agent a reference names, if it is `<agent>@content` and
/// that Agent is still waiting to run.
fn deferred_agent<'r>(ctx: &NodeCtx<'_>, reference: &'r str) -> Option<&'r str> {
    match reference.split_once('@') {
        Some((cpn, "content")) if ctx.deferred.is_pending(cpn) => Some(cpn),
        _ => None,
    }
}

impl Component for Message {
    fn invoke(&self, ctx: &mut NodeCtx<'_>) -> Result<Outputs, String> {
        let inputs = Value::Object(ctx.inputs.clone());
        let runtime_text = ctx.inputs.contains_key("text");
        let mut text = if runtime_text {
            extract_text(&inputs)
        } else {
            String::new()
        };
        if !runtime_text && !self.template.is_empty() {
            text = self.template.clone();
        }
        if text.is_empty() && !runtime_text {
            text = extract_text(&inputs);
        }
        if text.is_empty() {
            if let Some(f) = ctx
                .inputs
                .get("formalized_content")
                .and_then(Value::as_str)
                .filter(|f| !f.trim().is_empty())
            {
                text = f.to_string();
            }
        }

        let spans = reference_spans(&text);
        for (_, _, reference) in &spans {
            if deferred_agent(ctx, reference).is_none() && ctx.state.lookup(reference).is_none() {
                return Err(TemplateError::NotFound(reference.clone()).to_string());
            }
        }
        let streams = spans
            .iter()
            .any(|(_, _, r)| deferred_agent(ctx, r).is_some());

        let content = if !streams {
            let content = render_display(&text, ctx.state);
            if !content.is_empty() {
                ctx.events.emit(Event::Message {
                    content: content.clone(),
                });
            }
            content
        } else {
            let mut out = String::new();
            let mut last = 0;
            for (start, end, reference) in &spans {
                let literal = &text[last..*start];
                if !literal.is_empty() {
                    ctx.events.emit(Event::Message {
                        content: literal.to_string(),
                    });
                    out.push_str(literal);
                }
                if let Some(agent) = deferred_agent(ctx, reference) {
                    let outputs = ctx
                        .deferred
                        .open(agent, ctx.state, ctx.events)
                        .map_err(|e| format!("Message: consume deferred Agent stream: {e}"))?;
                    let answer = outputs
                        .get("content")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    if !answer.is_empty() {
                        ctx.events.emit(Event::Message {
                            content: answer.clone(),
                        });
                    }
                    out.push_str(&answer);
                } else {
                    let resolved = render_display(&text[*start..*end], ctx.state);
                    ctx.events.emit(Event::Message {
                        content: resolved.clone(),
                    });
                    out.push_str(&resolved);
                }
                last = *end;
            }
            let tail = &text[last..];
            if !tail.is_empty() {
                ctx.events.emit(Event::Message {
                    content: tail.to_string(),
                });
                out.push_str(tail);
            }
            out
        };

        let mut out = Outputs::new();
        out.insert("content".into(), Value::String(content));
        out.insert("downloads".into(), json!([]));
        Ok(out)
    }
}

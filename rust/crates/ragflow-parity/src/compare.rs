//! Compares two event streams (Go oracle vs Rust runtime) for one case.
//!
//! Normalisation: ids, `created_at`, `elapsed_time` and message/session
//! ids are not compared at all; the `_created_time` / `_elapsed_time` keys
//! Go's node wrapper stamps into every output bucket are dropped; numbers
//! compare as f64 (`1` == `1.0`).
//!
//! Hard checks (any mismatch fails the case): the order of `node_finished`
//! component ids, each component's outputs, every LLM call (model,
//! messages, options) keyed by `<component>#<n>`, the concatenated
//! `message` content, the final answer (`workflow_finished.outputs`), and
//! whether the run ended in an `error`. Soft checks are reported but do not
//! fail: the error text, and how `message` content is split into frames.
//!
//! Known differences are ruled divergences: they are reported as KNOWN and
//! neither pass nor fail a case. There is exactly one (Plain, 2026-10-14):
//! a deferred Agent's `node_finished` outputs. Go emits that event from the
//! workflow engine's copy of the run state, which still holds the unopened
//! stream (`{"content":{"Open":null}}`, canvas/scheduler.go RegisterDeferredNode
//! callback); Rust reports the Agent's real outputs.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use serde_json::Value;

/// Output keys that only carry timing.
const TIMING_KEYS: &[&str] = &["_created_time", "_elapsed_time"];

#[derive(Default, Debug)]
pub struct Stream {
    pub finished: Vec<String>,
    pub outputs: BTreeMap<String, Value>,
    pub calls: BTreeMap<String, Value>,
    pub messages: Vec<String>,
    pub answer: Option<Value>,
    pub error: Option<String>,
}

pub fn parse(lines: &[Value]) -> Stream {
    let mut s = Stream::default();
    for line in lines {
        let data = line.get("data").cloned().unwrap_or(Value::Null);
        match line
            .get("event")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "node_finished" => {
                let id = text(&data, "component_id");
                let label = match data.get("error") {
                    Some(Value::String(e)) => format!("{id} (error: {e})"),
                    _ => id.clone(),
                };
                s.finished.push(label);
                let mut outputs = data.get("outputs").cloned().unwrap_or(Value::Null);
                if let Value::Object(map) = &mut outputs {
                    map.retain(|k, _| !TIMING_KEYS.contains(&k.as_str()));
                }
                s.outputs.insert(id, outputs);
            }
            "llm_call" => {
                let key = text(&data, "key");
                let mut call = data.clone();
                if let Value::Object(map) = &mut call {
                    map.retain(|k, _| matches!(k.as_str(), "model" | "messages" | "options"));
                }
                s.calls.insert(key, call);
            }
            "message" => s.messages.push(text(&data, "content")),
            "workflow_finished" => s.answer = data.get("outputs").cloned(),
            "error" => s.error = Some(text(&data, "message")),
            _ => {}
        }
    }
    s
}

fn text(v: &Value, key: &str) -> String {
    match v.get(key) {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

pub struct Report {
    pub hard: Vec<(&'static str, String)>,
    pub soft: Vec<(&'static str, String)>,
    pub known: Vec<(&'static str, String)>,
}

/// The one known difference: Go's outputs for a deferred Agent.
fn is_unopened_deferred_stream(go_outputs: &Value) -> bool {
    same(
        go_outputs,
        &serde_json::json!({ "content": { "Open": null } }),
    )
}

impl Report {
    pub fn passed(&self) -> bool {
        self.hard.is_empty()
    }
}

pub fn compare(go: &Stream, rust: &Stream) -> Report {
    let mut hard = Vec::new();
    let mut soft = Vec::new();
    let mut known = Vec::new();

    if go.finished != rust.finished {
        hard.push((
            "node_sequence",
            format!("go:   {:?}\nrust: {:?}", go.finished, rust.finished),
        ));
    }

    let mut out = String::new();
    let ids: BTreeSet<&String> = go.outputs.keys().chain(rust.outputs.keys()).collect();
    for id in ids {
        match (go.outputs.get(id), rust.outputs.get(id)) {
            (Some(g), Some(r)) if is_unopened_deferred_stream(g) => {
                if !same(g, r) {
                    known.push(("deferred_agent_outputs", format!("{id}: go {g}, rust {r}")));
                }
            }
            (Some(g), Some(r)) => {
                if !same(g, r) {
                    let _ = writeln!(out, "{id}:");
                    describe_value_diff(&mut out, "  ", g, r);
                }
            }
            (Some(_), None) => {
                let _ = writeln!(out, "{id}: finished in go only");
            }
            (None, Some(_)) => {
                let _ = writeln!(out, "{id}: finished in rust only");
            }
            (None, None) => {}
        }
    }
    if !out.is_empty() {
        hard.push(("outputs", out));
    }

    let mut out = String::new();
    let keys: BTreeSet<&String> = go.calls.keys().chain(rust.calls.keys()).collect();
    for key in keys {
        match (go.calls.get(key), rust.calls.get(key)) {
            (Some(g), Some(r)) => {
                if !same(g, r) {
                    let _ = writeln!(out, "{key}:");
                    describe_call_diff(&mut out, g, r);
                }
            }
            (Some(_), None) => {
                let _ = writeln!(out, "{key}: called in go only");
            }
            (None, Some(_)) => {
                let _ = writeln!(out, "{key}: called in rust only");
            }
            (None, None) => {}
        }
    }
    if !out.is_empty() {
        hard.push(("llm_prompts", out));
    }

    let (gm, rm) = (go.messages.concat(), rust.messages.concat());
    if gm != rm {
        hard.push(("message_content", text_diff(&gm, &rm)));
    } else if go.messages != rust.messages {
        soft.push((
            "message_frames",
            format!("go:   {:?}\nrust: {:?}", go.messages, rust.messages),
        ));
    }

    let answer_same = match (&go.answer, &rust.answer) {
        (Some(g), Some(r)) => same(g, r),
        (None, None) => true,
        _ => false,
    };
    if !answer_same {
        hard.push((
            "answer",
            format!(
                "go:   {}\nrust: {}",
                show(go.answer.as_ref()),
                show(rust.answer.as_ref())
            ),
        ));
    }

    match (&go.error, &rust.error) {
        (Some(g), Some(r)) if g != r => {
            soft.push(("error_text", format!("go:   {g}\nrust: {r}")));
        }
        (Some(g), None) => hard.push(("error", format!("go failed, rust did not\ngo:   {g}"))),
        (None, Some(r)) => hard.push(("error", format!("rust failed, go did not\nrust: {r}"))),
        _ => {}
    }

    Report { hard, soft, known }
}

fn show(v: Option<&Value>) -> String {
    v.map(Value::to_string).unwrap_or_else(|| "(none)".into())
}

/// JSON equality with numbers compared as f64.
pub fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (
                x.as_f64().unwrap_or(f64::NAN),
                y.as_f64().unwrap_or(f64::NAN),
            );
            x == y || (x - y).abs() < 1e-9
        }
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(a, b)| same(a, b))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| same(v, w)))
        }
        _ => a == b,
    }
}

fn describe_value_diff(out: &mut String, indent: &str, g: &Value, r: &Value) {
    match (g, r) {
        (Value::Object(gm), Value::Object(rm)) => {
            let keys: BTreeSet<&String> = gm.keys().chain(rm.keys()).collect();
            for k in keys {
                match (gm.get(k), rm.get(k)) {
                    (Some(a), Some(b)) if same(a, b) => {}
                    (Some(a), Some(b)) => {
                        if let (Value::String(sa), Value::String(sb)) = (a, b) {
                            let _ = writeln!(out, "{indent}{k}:");
                            for line in text_diff(sa, sb).lines() {
                                let _ = writeln!(out, "{indent}  {line}");
                            }
                        } else {
                            let _ = writeln!(out, "{indent}{k}: go={a} rust={b}");
                        }
                    }
                    (Some(a), None) => {
                        let _ = writeln!(out, "{indent}{k}: go only ({a})");
                    }
                    (None, Some(b)) => {
                        let _ = writeln!(out, "{indent}{k}: rust only ({b})");
                    }
                    (None, None) => {}
                }
            }
        }
        _ => {
            let _ = writeln!(out, "{indent}go={g}\n{indent}rust={r}");
        }
    }
}

fn describe_call_diff(out: &mut String, g: &Value, r: &Value) {
    for field in ["model", "options"] {
        if !same(&g[field], &r[field]) {
            let _ = writeln!(out, "  {field}: go={} rust={}", g[field], r[field]);
        }
    }
    let empty = Vec::new();
    let gm = g["messages"].as_array().unwrap_or(&empty);
    let rm = r["messages"].as_array().unwrap_or(&empty);
    if gm.len() != rm.len() {
        let roles = |m: &[Value]| -> Vec<String> { m.iter().map(|x| text(x, "role")).collect() };
        let _ = writeln!(
            out,
            "  messages: go has {} {:?}, rust has {} {:?}",
            gm.len(),
            roles(gm),
            rm.len(),
            roles(rm)
        );
    }
    for (i, (a, b)) in gm.iter().zip(rm).enumerate() {
        if !same(a, b) {
            let _ = writeln!(
                out,
                "  messages[{i}] ({} / {}):",
                text(a, "role"),
                text(b, "role")
            );
            for line in text_diff(&text(a, "content"), &text(b, "content")).lines() {
                let _ = writeln!(out, "    {line}");
            }
        }
    }
}

/// A line diff (`-` go, `+` rust, ` ` both), via LCS; texts here are small.
pub fn text_diff(go: &str, rust: &str) -> String {
    let a: Vec<&str> = go.split('\n').collect();
    let b: Vec<&str> = rust.split('\n').collect();
    let (n, m) = (a.len(), b.len());
    let mut lcs = vec![vec![0usize; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            lcs[i][j] = if a[i] == b[j] {
                lcs[i + 1][j + 1] + 1
            } else {
                lcs[i + 1][j].max(lcs[i][j + 1])
            };
        }
    }
    let mut out = String::new();
    let (mut i, mut j) = (0, 0);
    while i < n || j < m {
        if i < n && j < m && a[i] == b[j] {
            let _ = writeln!(out, "  {:?}", a[i]);
            i += 1;
            j += 1;
        } else if j < m && (i == n || lcs[i][j + 1] >= lcs[i + 1][j]) {
            let _ = writeln!(out, "+ {:?}", b[j]);
            j += 1;
        } else {
            let _ = writeln!(out, "- {:?}", a[i]);
            i += 1;
        }
    }
    out
}

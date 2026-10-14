//! What one run knows: `sys.*`, `env.*`, plain globals, and every
//! finished component's outputs.

use std::collections::BTreeMap;

use chrono::{DateTime, FixedOffset};
use serde_json::{Map, Value};

pub type Outputs = Map<String, Value>;

/// Output keys upstream never copies into a component's bucket.
pub const RESERVED_OUTPUT_KEYS: &[&str] = &["__cpn_id__", "state", "__legacy_noop__"];

#[derive(Debug, Default, Clone)]
pub struct RunState {
    pub sys: Map<String, Value>,
    pub env: Map<String, Value>,
    pub globals: Map<String, Value>,
    pub outputs: BTreeMap<String, Outputs>,
    /// Citation references recorded by Retrieval (`CanvasState.Retrieval`):
    /// `chunks` (deduped by `id`, first wins) and `doc_aggs` (by doc name).
    pub retrieval: Map<String, Value>,
}

impl RunState {
    /// Seeds a run from the canvas globals: `sys.x` keys go to `sys`,
    /// `env.x` to `env`, the rest stay globals. `sys.date` is written only
    /// when the globals lack it or hold a blank value, as upstream's
    /// `EnsureSysDate` does.
    pub fn seed(
        globals: &BTreeMap<String, Value>,
        query: &str,
        now: DateTime<FixedOffset>,
    ) -> Self {
        let mut state = RunState::default();
        for (key, value) in globals {
            if let Some(name) = key.strip_prefix("sys.") {
                state.sys.insert(name.to_string(), value.clone());
            } else if let Some(name) = key.strip_prefix("env.") {
                state.env.insert(name.to_string(), value.clone());
            } else {
                state.globals.insert(key.clone(), value.clone());
            }
        }
        state
            .sys
            .insert("query".into(), Value::String(query.to_string()));
        let has_date = state
            .sys
            .get("date")
            .and_then(Value::as_str)
            .is_some_and(|d| !d.trim().is_empty());
        if !has_date {
            state.sys.insert(
                "date".into(),
                Value::String(now.format("%Y-%m-%d %H:%M:%S").to_string()),
            );
        }
        let turns = state
            .sys
            .get("conversation_turns")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        state
            .sys
            .insert("conversation_turns".into(), Value::from(turns as i64 + 1));
        state
    }

    /// Resolves a reference name (no braces). `None` covers both an unknown
    /// name and a null value, as upstream treats both as "can't find".
    pub fn lookup(&self, reference: &str) -> Option<Value> {
        let found = match reference {
            "item" => self.globals.get("__item__").cloned(),
            "index" => self.globals.get("__index__").cloned(),
            _ => {
                if let Some(rest) = reference.strip_prefix("sys.") {
                    first_then_path(&self.sys, rest)
                } else if let Some(rest) = reference.strip_prefix("env.") {
                    first_then_path(&self.env, rest)
                } else {
                    let (cpn, param) = reference.split_once('@')?;
                    first_then_path(self.outputs.get(cpn)?, param)
                }
            }
        };
        found.filter(|v| !v.is_null())
    }
}

fn first_then_path(map: &Map<String, Value>, dotted: &str) -> Option<Value> {
    let mut segments = dotted.split('.');
    let mut value = map.get(segments.next()?)?.clone();
    for segment in segments {
        value = step(value, segment)?;
    }
    Some(value)
}

fn step(value: Value, segment: &str) -> Option<Value> {
    match value {
        Value::Object(map) => map.get(segment).cloned(),
        Value::Array(items) => items.get(segment.parse::<usize>().ok()?).cloned(),
        Value::String(text) => match serde_json::from_str::<Value>(&text).ok()? {
            nested @ (Value::Object(_) | Value::Array(_)) => step(nested, segment),
            _ => None,
        },
        _ => None,
    }
}

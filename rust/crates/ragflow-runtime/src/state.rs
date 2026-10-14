//! What one run knows: `sys.*`, `env.*`, plain globals, and every
//! finished component's outputs.

use std::collections::BTreeMap;

use chrono::{DateTime, FixedOffset};
use serde_json::{json, Map, Value};

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
    /// The conversation so far, current turn included once
    /// [`RunState::append_current_user`] ran: `{role, content, payload}`
    /// (upstream's `CanvasState.History`). `content` is the text components
    /// read; `payload` is the value the session DSL stores.
    pub history: Vec<Map<String, Value>>,
    /// Index in `history` of the in-flight user turn, so prompts can leave
    /// it out (`SnapshotPriorHistory`).
    pub active_history_index: Option<usize>,
    /// Tool-call memory, carried through untouched: `{user, assistant,
    /// summary}`.
    pub memory: Vec<Map<String, Value>>,
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

/// upstream's `historyContent`: the text of a history payload.
pub fn history_content(payload: &Value) -> String {
    match payload {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Object(m) => m
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        other => crate::template::go_v(other),
    }
}

/// `canvas.decodeHistory`: a session DSL's `history`, either the persisted
/// `[[role, payload], ...]` or the runtime `[{role, content|payload}, ...]`.
pub fn decode_history(raw: &Value) -> Vec<Map<String, Value>> {
    let Some(items) = raw.as_array() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for item in items {
        let (role, payload) = match item {
            Value::Array(pair) if pair.len() >= 2 => {
                (pair[0].as_str().unwrap_or_default(), pair[1].clone())
            }
            Value::Object(m) => {
                let payload = match m.get("payload") {
                    Some(p) => p.clone(),
                    None => m.get("content").cloned().unwrap_or(Value::Null),
                };
                (
                    m.get("role").and_then(Value::as_str).unwrap_or_default(),
                    payload,
                )
            }
            _ => continue,
        };
        if role.is_empty() {
            continue;
        }
        out.push(history_entry(role, payload));
    }
    out
}

fn history_entry(role: &str, payload: Value) -> Map<String, Value> {
    let mut entry = Map::new();
    entry.insert("role".into(), json!(role));
    entry.insert("content".into(), json!(history_content(&payload)));
    entry.insert("payload".into(), payload);
    entry
}

/// `canvas.decodeMemory`: `[[user, assistant, summary], ...]` or maps.
pub fn decode_memory(raw: &Value) -> Vec<Map<String, Value>> {
    let Some(items) = raw.as_array() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for item in items {
        let mut m = Map::new();
        match item {
            Value::Array(v) if v.len() >= 3 => {
                m.insert("user".into(), v[0].clone());
                m.insert("assistant".into(), v[1].clone());
                m.insert("summary".into(), v[2].clone());
            }
            Value::Object(o) => {
                for key in ["user", "assistant", "summary"] {
                    m.insert(key.into(), o.get(key).cloned().unwrap_or(Value::Null));
                }
            }
            _ => continue,
        }
        out.push(m);
    }
    out
}

/// `canvas.EncodeHistory`: `[[role, payload], ...]`.
pub fn encode_history(history: &[Map<String, Value>]) -> Value {
    Value::Array(
        history
            .iter()
            .filter_map(|entry| {
                let role = entry.get("role")?.as_str().filter(|r| !r.is_empty())?;
                let payload = entry
                    .get("payload")
                    .or_else(|| entry.get("content"))
                    .cloned()
                    .unwrap_or(Value::Null);
                Some(json!([role, payload]))
            })
            .collect(),
    )
}

/// `canvas.EncodeMemory`: `[[user, assistant, summary], ...]`.
pub fn encode_memory(memory: &[Map<String, Value>]) -> Value {
    Value::Array(
        memory
            .iter()
            .map(|m| {
                json!([
                    m.get("user").cloned().unwrap_or(Value::Null),
                    m.get("assistant").cloned().unwrap_or(Value::Null),
                    m.get("summary").cloned().unwrap_or(Value::Null)
                ])
            })
            .collect(),
    )
}

/// `service.pythonHistoryRepr`: a value the way Python's `repr` prints it,
/// which is how `sys.history` stores an assistant turn. Map keys follow the
/// order Python's output dictionaries have: `content`, `downloads`, any
/// other key, then `_created_time` and `_elapsed_time`.
pub fn python_history_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::String(s) => {
            let escaped = s
                .replace('\\', "\\\\")
                .replace('\'', "\\'")
                .replace('\n', "\\n")
                .replace('\r', "\\r")
                .replace('\t', "\\t");
            format!("'{escaped}'")
        }
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Object(map) => {
            let priority = |key: &str| match key {
                "content" => 0,
                "downloads" => 1,
                "_created_time" => 3,
                "_elapsed_time" => 4,
                _ => 2,
            };
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_by(|a, b| priority(a).cmp(&priority(b)).then_with(|| a.cmp(b)));
            let parts: Vec<String> = keys
                .into_iter()
                .map(|k| {
                    format!(
                        "{}: {}",
                        python_history_repr(&Value::String(k.clone())),
                        python_history_repr(&map[k])
                    )
                })
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
        Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(python_history_repr).collect();
            format!("[{}]", parts.join(", "))
        }
        other => crate::template::go_v(other),
    }
}

/// `service.renderUserHistoryValue`: a user turn in `sys.history`.
pub fn render_user_history_value(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Object(_) => value.to_string(),
        other => python_history_repr(other),
    }
}

impl RunState {
    /// `CanvasState.SetHistory` / `SetMemory` from a session DSL.
    pub fn set_conversation(&mut self, history: &Value, memory: &Value) {
        self.history = decode_history(history);
        self.active_history_index = None;
        self.memory = decode_memory(memory);
    }

    /// `AppendCurrentUser`: the in-flight user turn.
    pub fn append_current_user(&mut self, payload: Value) {
        self.history.push(history_entry("user", payload));
        self.active_history_index = Some(self.history.len() - 1);
    }

    /// `AppendHistory`.
    pub fn append_history(&mut self, role: &str, payload: Value) {
        if role.is_empty() {
            return;
        }
        self.history.push(history_entry(role, payload));
        self.active_history_index = None;
    }

    /// `SnapshotPriorHistory`: completed turns before the in-flight one.
    pub fn prior_history(&self) -> &[Map<String, Value>] {
        match self.active_history_index {
            Some(i) if i + 1 == self.history.len() => &self.history[..i],
            _ => &self.history,
        }
    }

    /// `AppendSysHistory`.
    pub fn append_sys_history(&mut self, entry: String) {
        let mut items = match self.sys.remove("history") {
            Some(Value::Array(items)) => items,
            _ => Vec::new(),
        };
        items.push(Value::String(entry));
        self.sys.insert("history".into(), Value::Array(items));
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

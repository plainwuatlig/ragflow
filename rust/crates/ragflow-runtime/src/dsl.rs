//! The canvas as RAGFlow stores it (agent DSL v2).
//!
//! Only `components` and `globals` reach the runtime; `graph` is the
//! editor's layout, and for the supported components it has no runtime
//! effect.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
pub struct Dsl {
    pub components: BTreeMap<String, Component>,
    #[serde(default)]
    pub globals: BTreeMap<String, Value>,
    /// The conversation a session DSL carries: `[[role, payload], ...]`.
    #[serde(default)]
    pub history: Value,
    /// Tool-call memory: `[[user, assistant, summary], ...]`.
    #[serde(default)]
    pub memory: Value,
}

/// A component in either stored shape: the canonical
/// `{obj: {component_name, params}, downstream, upstream}` or the flat
/// `{name, params, downstream, upstream}` that `NormalizeForCanvas` emits
/// (`decodeComponentFields`).
#[derive(Debug, Deserialize)]
#[serde(from = "RawComponent")]
pub struct Component {
    pub obj: ComponentObj,
    pub downstream: Vec<String>,
    pub upstream: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct ComponentObj {
    pub component_name: String,
    #[serde(default)]
    pub params: Value,
}

#[derive(Deserialize)]
struct RawComponent {
    obj: Option<RawObj>,
    name: Option<String>,
    params: Option<Value>,
    downstream: Option<Vec<Value>>,
    upstream: Option<Vec<Value>>,
}

#[derive(Deserialize)]
struct RawObj {
    component_name: Option<String>,
    params: Option<Value>,
    downstream: Option<Vec<Value>>,
}

fn strings(v: Option<Vec<Value>>) -> Vec<String> {
    v.into_iter()
        .flatten()
        .filter_map(|x| match x {
            Value::String(s) => Some(s),
            _ => None,
        })
        .collect()
}

impl From<RawComponent> for Component {
    fn from(raw: RawComponent) -> Self {
        let (mut name, mut params, mut downstream) = (String::new(), None, Vec::new());
        if let Some(obj) = raw.obj {
            name = obj.component_name.unwrap_or_default();
            params = obj.params.filter(Value::is_object);
            downstream = strings(obj.downstream);
        }
        if name.is_empty() {
            name = raw.name.unwrap_or_default();
        }
        if params.is_none() {
            params = raw.params.filter(Value::is_object);
        }
        if downstream.is_empty() {
            downstream = strings(raw.downstream);
        }
        Component {
            obj: ComponentObj {
                component_name: name,
                params: params.unwrap_or(Value::Null),
            },
            downstream,
            upstream: strings(raw.upstream),
        }
    }
}

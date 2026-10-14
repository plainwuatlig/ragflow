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

#[derive(Debug, Deserialize)]
pub struct Component {
    pub obj: ComponentObj,
    #[serde(default)]
    pub downstream: Vec<String>,
    #[serde(default)]
    pub upstream: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct ComponentObj {
    pub component_name: String,
    #[serde(default)]
    pub params: Value,
}

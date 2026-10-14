//! A corpus case: which canvas, what the visitor asks, and the recorded
//! model answers both runtimes replay.
//!
//! ```json
//! {
//!   "canvas": "canvases/switch_ops.json",
//!   "query": "I want a refund",
//!   "patch": { "Agent:A": { "llm_id": "parity-chat@OpenAI" } },
//!   "globals": { "env.tier": "" },
//!   "recordings": { "Agent:A#0": "fixed answer" }
//! }
//! ```
//!
//! `canvas` is relative to the case file and may be an upstream template
//! (`{"dsl": {...}}`) or a bare DSL. `patch` merges into a component's
//! params (templates ship `llm_id: ""`); `globals` merges into the DSL's
//! globals. `sys.date` is pinned so both runtimes render the same prompt.
//! Recording keys are `<component id>#<call index>`.

use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

/// The `sys.date` every case runs with unless it sets its own.
pub const PINNED_DATE: &str = "2026-01-01 00:00:00";

pub struct Case {
    pub name: String,
    pub canvas_path: PathBuf,
    pub dsl: Value,
    pub query: String,
    pub recordings: Map<String, Value>,
}

pub fn load(path: &Path) -> Result<Case, String> {
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_string();
    let spec: Value = read_json(path)?;
    let dir = path.parent().unwrap_or(Path::new("."));
    let canvas_rel = spec
        .get("canvas")
        .and_then(Value::as_str)
        .ok_or("case: \"canvas\" (a path) is required")?;
    let canvas_path = dir.join(canvas_rel);
    let loaded = read_json(&canvas_path)?;
    let mut dsl = match loaded.get("dsl") {
        Some(inner @ Value::Object(_)) => inner.clone(),
        _ => loaded,
    };

    if let Some(patch) = spec.get("patch").and_then(Value::as_object) {
        for (cpn, params) in patch {
            let target = dsl
                .pointer_mut(&format!("/components/{}/obj/params", escape(cpn)))
                .and_then(Value::as_object_mut)
                .ok_or_else(|| format!("case: patch names {cpn}, which has no params"))?;
            for (k, v) in params.as_object().into_iter().flatten() {
                target.insert(k.clone(), v.clone());
            }
        }
    }

    let globals = dsl
        .as_object_mut()
        .ok_or("canvas: not a JSON object")?
        .entry("globals")
        .or_insert_with(|| Value::Object(Map::new()))
        .as_object_mut()
        .ok_or("canvas: globals is not an object")?;
    for (k, v) in spec
        .get("globals")
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
    {
        globals.insert(k.clone(), v.clone());
    }
    let has_date = globals
        .get("sys.date")
        .and_then(Value::as_str)
        .is_some_and(|d| !d.trim().is_empty());
    if !has_date {
        globals.insert("sys.date".into(), Value::String(PINNED_DATE.into()));
    }

    Ok(Case {
        name,
        canvas_path,
        dsl,
        query: spec
            .get("query")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        recordings: spec
            .get("recordings")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default(),
    })
}

fn escape(segment: &str) -> String {
    segment.replace('~', "~0").replace('/', "~1")
}

pub fn read_json(path: &Path) -> Result<Value, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

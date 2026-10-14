//! The DSL transforms around a run, as upstream's Go server applies them
//! (`internal/agent/dsl`, `internal/service/agent.go`): preparing a stored
//! canvas for the runtime, starting a fresh session from it, and building the
//! snapshot a session stores after each turn.

use std::sync::OnceLock;

use ragflow_runtime::{encode_history, encode_memory, RunState};
use regex::Regex;
use serde_json::{json, Map, Value};

/// `dsl.NormalizeForRun` for the components this server runs: a deep copy
/// whose React Flow handle ids are repaired, a graph derived for a canvas
/// that has none (which also flattens its components), and the legacy
/// `IterationItem:<id>@item|index` aliases rewritten to `item|index`.
///
/// Upstream also folds legacy Loop and Iteration pairs and renames Iteration
/// to Parallel; none of that can touch a canvas of the supported components,
/// and a canvas with other components never gets as far as being persisted.
pub fn normalize_for_run(dsl: &Value) -> Value {
    let mut out = dsl.clone();
    enforce_handle_ids(&mut out);
    if !graph_has_nodes(&out) {
        derive_graph(&mut out);
    }
    repair_parallel_leaks(&mut out);
    rewrite_aliases(&mut out);
    out
}

fn enforce_handle_ids(dsl: &mut Value) {
    let Some(edges) = dsl
        .pointer_mut("/graph/edges")
        .and_then(Value::as_array_mut)
    else {
        return;
    };
    for edge in edges.iter_mut().filter_map(Value::as_object_mut) {
        if matches!(
            edge.get("sourceHandle").and_then(Value::as_str),
            Some("end" | "start")
        ) {
            edge.insert("sourceHandle".into(), Value::String("start".into()));
        }
        if matches!(
            edge.get("targetHandle").and_then(Value::as_str),
            Some("start" | "end")
        ) {
            edge.insert("targetHandle".into(), Value::String("end".into()));
        }
    }
}

fn graph_has_nodes(dsl: &Value) -> bool {
    dsl.pointer("/graph/nodes")
        .and_then(Value::as_array)
        .is_some_and(|n| !n.is_empty())
}

/// `componentNameToNodeType`.
fn node_type(name: &str) -> &'static str {
    match name {
        "Begin" => "beginNode",
        "Retrieval" => "ragNode",
        "Categorize" => "categorizeNode",
        "Message" | "Answer" => "messageNode",
        "RewriteQuestion" => "rewriteNode",
        "ExeSQL" | "Tool" | "Code" => "toolNode",
        "Switch" => "switchNode",
        "File" => "fileNode",
        "Parser" => "parserNode",
        "Tokenizer" => "tokenizerNode",
        "GeneralChunker" | "TokenChunker" | "TitleChunker" | "OneChunker" | "QAChunker"
        | "TableChunker" | "PageChunker" => "chunkerNode",
        "Extractor" => "contextNode",
        "Loop" => "loopNode",
        "LoopStart" => "loopStartNode",
        "ExitLoop" => "exitLoopNode",
        "Iteration" => "iterationNode",
        "IterationStart" => "iterationStartNode",
        "Parallel" => "parallelNode",
        "DataOperations" => "dataOperationsNode",
        "ListOperations" => "listOperationsNode",
        "VariableAssigner" => "variableAssignerNode",
        "VariableAggregator" => "variableAggregatorNode",
        "Keyword" => "keywordNode",
        "Note" => "noteNode",
        "Placeholder" => "placeholderNode",
        _ => "agentNode",
    }
}

/// `buildGraphFromComponents`: a canvas stored without a graph gets a
/// single-row layout (x = 50 + 350 i, y = 200, ids sorted) and its
/// components in the flat `{id, name, downstream, upstream, params}` shape.
fn derive_graph(dsl: &mut Value) {
    let Some(components) = dsl.get("components").and_then(Value::as_object) else {
        return;
    };
    if components.is_empty() {
        return;
    }
    let mut keys: Vec<&String> = components.keys().collect();
    keys.sort();
    let (mut nodes, mut edges, mut flat) = (Vec::new(), Vec::new(), Map::new());
    let strings = |v: Option<&Value>| -> Value {
        let list: Vec<Value> = v
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|x| x.as_str().is_some_and(|s| !s.is_empty()))
            .cloned()
            .collect();
        if list.is_empty() {
            Value::Null
        } else {
            Value::Array(list)
        }
    };
    let mut i = 0usize;
    for key in keys {
        let Some(comp) = components[key].as_object() else {
            continue;
        };
        let obj = comp.get("obj").and_then(Value::as_object);
        let mut name = obj
            .and_then(|o| o.get("component_name"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let mut params = obj
            .and_then(|o| o.get("params"))
            .filter(|p| p.is_object())
            .cloned();
        let mut downstream: Vec<Value> = match strings(obj.and_then(|o| o.get("downstream"))) {
            Value::Array(a) => a,
            _ => Vec::new(),
        };
        if name.is_empty() {
            name = comp
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
        }
        if params.is_none() {
            params = comp.get("params").filter(|p| p.is_object()).cloned();
        }
        if let Value::Array(a) = strings(comp.get("downstream")) {
            downstream.extend(a);
        }
        if name.is_empty() {
            name = key.clone();
        }
        let params = params.unwrap_or(Value::Null);
        nodes.push(json!({
            "id": key,
            "type": node_type(&name),
            "position": {"x": 50 + i * 350, "y": 200},
            "data": {"label": name, "name": name, "form": params},
            "sourcePosition": "right",
            "targetPosition": "left",
        }));
        for dst in &downstream {
            if let Some(dst) = dst.as_str() {
                edges.push(json!({
                    "id": format!("xy-edge__{key}-{dst}"),
                    "source": key,
                    "target": dst,
                    "sourceHandle": "start",
                    "targetHandle": "end",
                }));
            }
        }
        flat.insert(
            key.clone(),
            json!({
                "id": key,
                "name": name,
                "downstream": strings(comp.get("downstream")),
                "upstream": strings(comp.get("upstream")),
                "params": params,
            }),
        );
        i += 1;
    }
    if nodes.is_empty() {
        return;
    }
    if let Some(map) = dsl.as_object_mut() {
        map.insert("graph".into(), json!({"nodes": nodes, "edges": edges}));
        map.insert("components".into(), Value::Object(flat));
    }
}

/// `repairParallelLeaksForCanvas`: a leaked runtime-only Parallel goes back to
/// the editor's Iteration.
fn repair_parallel_leaks(dsl: &mut Value) {
    if let Some(components) = dsl.get_mut("components").and_then(Value::as_object_mut) {
        for comp in components.values_mut().filter_map(Value::as_object_mut) {
            if let Some(obj) = comp.get_mut("obj").and_then(Value::as_object_mut) {
                if obj.get("component_name").and_then(Value::as_str) == Some("Parallel") {
                    obj.insert("component_name".into(), Value::String("Iteration".into()));
                }
            }
            if comp.get("name").and_then(Value::as_str) == Some("Parallel") {
                comp.insert("name".into(), Value::String("Iteration".into()));
            }
        }
    }
    let Some(nodes) = dsl
        .pointer_mut("/graph/nodes")
        .and_then(Value::as_array_mut)
    else {
        return;
    };
    for node in nodes.iter_mut().filter_map(Value::as_object_mut) {
        if node.get("type").and_then(Value::as_str) == Some("parallelNode") {
            node.insert("type".into(), Value::String("iterationNode".into()));
        }
        if let Some(data) = node.get_mut("data").and_then(Value::as_object_mut) {
            for key in ["label", "name"] {
                if data.get(key).and_then(Value::as_str) == Some("Parallel") {
                    data.insert(key.into(), Value::String("Iteration".into()));
                }
            }
        }
    }
}

fn alias_pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"IterationItem:[A-Za-z0-9_:-]+@(item|index)\b").expect("alias pattern")
    })
}

fn rewrite_aliases(value: &mut Value) {
    match value {
        Value::String(s) => {
            if s.contains("IterationItem:") {
                *s = alias_pattern().replace_all(s, "$1").into_owned();
            }
        }
        Value::Array(items) => items.iter_mut().for_each(rewrite_aliases),
        Value::Object(map) => map.values_mut().for_each(rewrite_aliases),
        _ => {}
    }
}

/// `dsl.ResetForNewSession`: a copy with the conversation-owned state
/// cleared (`history`, `path`, `globals["sys.history"]`).
pub fn reset_for_new_session(dsl: &Value) -> Value {
    let mut out = match dsl {
        Value::Object(_) => dsl.clone(),
        _ => Value::Object(Map::new()),
    };
    let map = out.as_object_mut().expect("an object");
    map.insert("history".into(), Value::Array(Vec::new()));
    map.insert("path".into(), Value::Array(Vec::new()));
    let globals = match map.get_mut("globals") {
        Some(Value::Object(g)) => g,
        _ => {
            map.insert("globals".into(), Value::Object(Map::new()));
            map.get_mut("globals")
                .and_then(Value::as_object_mut)
                .expect("just inserted")
        }
    };
    globals.insert("sys.history".into(), Value::Array(Vec::new()));
    out
}

/// `buildPersistedAgentDSL`: the run DSL with the state the run left behind.
/// `globals` keeps its keys, each refreshed from the run (`sys.*`, `env.*`,
/// plain) when the run has it, gains every `begin@*` cache and the `sys` keys
/// the next turn reads; `history` and `memory` are re-encoded.
pub fn persisted_dsl(run_dsl: &Value, state: &RunState) -> Value {
    let mut dsl = match run_dsl {
        Value::Object(m) => m.clone(),
        _ => Map::new(),
    };
    let mut globals = match dsl.get("globals") {
        Some(Value::Object(g)) => g.clone(),
        _ => Map::new(),
    };
    for key in globals.keys().cloned().collect::<Vec<_>>() {
        let fresh = if let Some(name) = key.strip_prefix("sys.") {
            state.sys.get(name)
        } else if let Some(name) = key.strip_prefix("env.") {
            state.env.get(name)
        } else {
            state.globals.get(&key)
        };
        if let Some(value) = fresh {
            globals.insert(key, value.clone());
        }
    }
    for (key, value) in &state.globals {
        if key.starts_with("begin@") {
            globals.insert(key.clone(), value.clone());
        }
    }
    for key in [
        "query",
        "user_id",
        "conversation_turns",
        "files",
        "history",
        "date",
    ] {
        if let Some(value) = state.sys.get(key) {
            globals.insert(format!("sys.{key}"), value.clone());
        }
    }
    dsl.insert("globals".into(), Value::Object(globals));
    dsl.insert("history".into(), encode_history(&state.history));
    dsl.insert("memory".into(), encode_memory(&state.memory));
    Value::Object(dsl)
}

/// What building the components does to the run DSL in place, and so to the
/// snapshot upstream saves: the Retrieval factory copies `dataset_ids` to
/// `kb_ids` in the node's params (`parseRetrievalParams`).
pub fn apply_build_effects(dsl: &mut Value) {
    let Some(components) = dsl.get_mut("components").and_then(Value::as_object_mut) else {
        return;
    };
    for comp in components.values_mut() {
        let name = {
            let obj = comp.get("obj").unwrap_or(comp);
            obj.get("component_name")
                .and_then(Value::as_str)
                .or_else(|| comp.get("name").and_then(Value::as_str))
                .unwrap_or_default()
                .to_string()
        };
        if !name.eq_ignore_ascii_case("Retrieval") {
            continue;
        }
        let params = if comp.pointer("/obj/params").is_some_and(Value::is_object) {
            comp.pointer_mut("/obj/params")
        } else {
            comp.get_mut("params")
        };
        if let Some(params) = params.and_then(Value::as_object_mut) {
            if let Some(ids) = params.get("dataset_ids").cloned() {
                params.insert("kb_ids".into(), ids);
            }
        }
    }
}

/// `deriveAgentSessionName`: the first 250 characters of the question (of a
/// form input's `query` or `question`).
pub fn session_name(user_input: &Value) -> String {
    let text = match user_input {
        Value::Object(m) => {
            let named = ["query", "question"]
                .iter()
                .find_map(|k| m.get(*k).and_then(Value::as_str).filter(|s| !s.is_empty()));
            match named {
                Some(s) => s.to_string(),
                None if !m.is_empty() => stringify_input(user_input),
                None => String::new(),
            }
        }
        other => stringify_input(other),
    };
    text.chars().take(250).collect()
}

/// `stringifyAgentUserInput`: a string as it is, anything else as JSON.
pub fn stringify_input(user_input: &Value) -> String {
    match user_input {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        other => crate::gojson::marshal(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn handles_and_aliases_are_repaired_on_a_copy() {
        let dsl = json!({
            "graph": {"edges": [
                {"sourceHandle": "end", "targetHandle": "start"},
                {"sourceHandle": "agentBottom", "targetHandle": "end"}
            ], "nodes": [{"id": "Message:a"}]},
            "components": {"Message:a": {"obj": {"component_name": "Message", "params":
                {"content": ["x {IterationItem:Foo@item} {IterationItem:Foo@items}"]}}}}
        });
        let out = normalize_for_run(&dsl);
        assert_eq!(out["graph"]["edges"][0]["sourceHandle"], "start");
        assert_eq!(out["graph"]["edges"][0]["targetHandle"], "end");
        assert_eq!(out["graph"]["edges"][1]["sourceHandle"], "agentBottom");
        assert_eq!(
            out["components"]["Message:a"]["obj"]["params"]["content"][0],
            "x {item} {IterationItem:Foo@items}"
        );
        assert_eq!(
            dsl["graph"]["edges"][0]["sourceHandle"], "end",
            "input untouched"
        );
    }

    #[test]
    fn a_new_session_starts_without_history() {
        let out = reset_for_new_session(&json!({"history": [["user", "x"]], "path": ["a"],
            "globals": {"sys.history": ["user: x"], "sys.query": "x"}}));
        assert_eq!(out["history"], json!([]));
        assert_eq!(out["path"], json!([]));
        assert_eq!(out["globals"]["sys.history"], json!([]));
        assert_eq!(out["globals"]["sys.query"], "x");
        let bare = reset_for_new_session(&json!({}));
        assert_eq!(bare["globals"], json!({"sys.history": []}));
    }

    #[test]
    fn the_persisted_snapshot_carries_the_turn() {
        let run = json!({"components": {}, "history": [], "globals": {
            "sys.history": [], "sys.query": "", "sys.user_id": "", "sys.files": [],
            "sys.conversation_turns": 0, "env.k": "v", "plain": 1, "sys.unused": "keep"}});
        let mut state = RunState::default();
        state.sys.insert("query".into(), json!("q"));
        state.sys.insert("user_id".into(), json!("u"));
        state.sys.insert("conversation_turns".into(), json!(1));
        state
            .sys
            .insert("date".into(), json!("2026-01-01 00:00:00"));
        state.sys.insert("history".into(), json!(["user: q"]));
        state.sys.insert("files".into(), json!([]));
        state.env.insert("k".into(), json!("v2"));
        state.globals.insert("begin@name".into(), json!("n"));
        state.globals.insert("plain".into(), json!(2));
        state.globals.insert("stray".into(), json!(3));
        state.set_conversation(&json!([]), &json!([]));
        state.append_current_user(json!("q"));
        state.append_history("assistant", json!({"content": "a"}));
        let out = persisted_dsl(&run, &state);
        let g = &out["globals"];
        assert_eq!(g["sys.query"], "q");
        assert_eq!(g["sys.history"], json!(["user: q"]));
        assert_eq!(g["sys.user_id"], "u");
        assert_eq!(g["sys.conversation_turns"], 1);
        assert_eq!(
            g["sys.date"], "2026-01-01 00:00:00",
            "new sys keys are added"
        );
        assert_eq!(g["sys.unused"], "keep");
        assert_eq!(g["env.k"], "v2");
        assert_eq!(g["plain"], 2);
        assert_eq!(g["begin@name"], "n");
        assert!(
            g.get("stray").is_none(),
            "only existing globals are refreshed"
        );
        assert_eq!(
            out["history"],
            json!([["user", "q"], ["assistant", {"content": "a"}]])
        );
        assert_eq!(out["memory"], json!([]));
    }

    #[test]
    fn building_a_retrieval_node_copies_its_dataset_ids() {
        let mut dsl = json!({"components": {
            "Retrieval:a": {"obj": {"component_name": "Retrieval", "params": {"dataset_ids": ["k"]}}},
            "Retrieval:b": {"obj": {"component_name": "Retrieval", "params": {"query": "q"}}},
            "Agent:c": {"obj": {"component_name": "Agent", "params": {"dataset_ids": ["k"]}}},
        }});
        apply_build_effects(&mut dsl);
        assert_eq!(
            dsl["components"]["Retrieval:a"]["obj"]["params"]["kb_ids"],
            json!(["k"])
        );
        assert!(dsl["components"]["Retrieval:b"]["obj"]["params"]
            .get("kb_ids")
            .is_none());
        assert!(dsl["components"]["Agent:c"]["obj"]["params"]
            .get("kb_ids")
            .is_none());
    }

    #[test]
    fn names_are_cut_at_250_characters() {
        assert_eq!(session_name(&json!("hello")), "hello");
        let long = "令".repeat(300);
        assert_eq!(session_name(&json!(long)).chars().count(), 250);
        assert_eq!(session_name(&json!({"query": "q", "x": 1})), "q");
        assert_eq!(session_name(&json!({"question": "w"})), "w");
        assert_eq!(session_name(&json!({"a": 1})), r#"{"a":1}"#);
        assert_eq!(session_name(&json!({})), "");
        assert_eq!(session_name(&Value::Null), "");
    }

    /// Upstream's own results (`fixtures/gen`) for the same inputs.
    fn go_session() -> Value {
        serde_json::from_str(include_str!("../fixtures/go_session.json")).unwrap()
    }

    #[test]
    fn python_reprs_and_user_turns_match_go() {
        for case in go_session()["repr"].as_array().unwrap() {
            let input = &case["input"];
            assert_eq!(
                ragflow_runtime::python_history_repr(input),
                case["repr"].as_str().unwrap(),
                "repr of {input}"
            );
            assert_eq!(
                ragflow_runtime::render_user_history_value(input),
                case["user_history"].as_str().unwrap(),
                "user turn of {input}"
            );
        }
    }

    #[test]
    fn session_names_match_go() {
        for case in go_session()["names"].as_array().unwrap() {
            assert_eq!(
                session_name(&case["input"]),
                case["name"].as_str().unwrap(),
                "name of {}",
                case["input"]
                    .to_string()
                    .chars()
                    .take(60)
                    .collect::<String>()
            );
        }
    }

    #[test]
    fn normalize_and_reset_match_go() {
        let fixtures = go_session();
        for case in fixtures["normalize"].as_array().unwrap() {
            assert_eq!(normalize_for_run(&case["input"]), case["normalized"]);
        }
        for case in fixtures["reset"].as_array().unwrap() {
            assert_eq!(reset_for_new_session(&case["input"]), case["reset"]);
        }
    }

    #[test]
    fn the_persisted_snapshot_matches_go() {
        for case in go_session()["persisted"].as_array().unwrap() {
            let mut state = RunState::default();
            let take = |key: &str| case[key].as_object().cloned().unwrap_or_default();
            state.sys = take("sys");
            state.env = take("env");
            state.globals = take("globals");
            state.set_conversation(&case["history"], &case["memory"]);
            assert_eq!(
                persisted_dsl(&case["run_dsl"], &state),
                case["persisted"],
                "scenario {}",
                case["name"]
            );
        }
    }
}

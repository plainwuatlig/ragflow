use ragflow_runtime::{compile, CompileError};

const GREETING: &str = include_str!("fixtures/greeting.json");

#[test]
fn compiles_a_supported_canvas() {
    let plan = compile(GREETING).unwrap();
    assert_eq!(plan.begin, "begin");
    assert_eq!(plan.dsl.components.len(), 3);
}

#[test]
fn refuses_an_unsupported_component_by_name() {
    let dsl = GREETING.replace(
        "\"component_name\": \"Agent\"",
        "\"component_name\": \"Iteration\"",
    );
    assert_eq!(
        compile(&dsl).unwrap_err(),
        CompileError::Unsupported {
            id: "Agent:Reply".into(),
            name: "Iteration".into()
        }
    );
}

#[test]
fn refuses_an_edge_to_a_missing_component() {
    let dsl = GREETING.replace(
        "\"downstream\": [\"Message:Out\"]",
        "\"downstream\": [\"Message:Gone\"]",
    );
    assert_eq!(
        compile(&dsl).unwrap_err(),
        CompileError::DanglingEdge {
            from: "Agent:Reply".into(),
            to: "Message:Gone".into()
        }
    );
}

fn with(components: &str) -> String {
    format!("{{\"components\": {{{components}}}}}")
}

#[test]
fn matches_component_names_case_insensitively() {
    let dsl = with(r#""b": {"obj": {"component_name": "begin"}}"#);
    assert_eq!(compile(&dsl).unwrap().graph.nodes["b"].name, "Begin");
}

#[test]
fn refuses_unknown_upstream_self_edges_and_cycles() {
    let unknown = with(
        r#""b": {"obj": {"component_name": "Begin"}}, "m": {"obj": {"component_name": "Message"}, "upstream": ["x"]}"#,
    );
    assert_eq!(
        compile(&unknown).unwrap_err(),
        CompileError::UnknownUpstream {
            id: "m".into(),
            upstream: "x".into()
        }
    );
    let selfie = with(
        r#""b": {"obj": {"component_name": "Begin"}}, "m": {"obj": {"component_name": "Message"}, "upstream": ["m"]}"#,
    );
    assert_eq!(
        compile(&selfie).unwrap_err(),
        CompileError::SelfEdge("m".into())
    );
    let cycle = with(
        r#""b": {"obj": {"component_name": "Begin"}}, "x": {"obj": {"component_name": "Message"}, "upstream": ["y"]}, "y": {"obj": {"component_name": "Message"}, "upstream": ["x"]}"#,
    );
    assert_eq!(
        compile(&cycle).unwrap_err(),
        CompileError::Cycle(vec!["x".into(), "y".into()])
    );
}

#[test]
fn orders_nodes_topologically_with_ids_breaking_ties() {
    let dsl = with(
        r#""b": {"obj": {"component_name": "Begin"}}, "z": {"obj": {"component_name": "Message"}, "upstream": ["b"]}, "a": {"obj": {"component_name": "Message"}, "upstream": ["b"]}"#,
    );
    assert_eq!(compile(&dsl).unwrap().graph.order, ["b", "a", "z"]);
}

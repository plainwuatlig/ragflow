//! Switch semantics against upstream's switch.go (main).

use chrono::{FixedOffset, TimeZone};
use ragflow_runtime::{
    compile, Bot, Component, Event, Node, NodeCtx, Outputs, Registry, RunError, RunInput,
};
use serde_json::{json, Value};

/// Emits fixed outputs, standing in for a Retrieval.
struct Fixed(Value);
impl Component for Fixed {
    fn invoke(&self, _: &mut NodeCtx<'_>) -> Result<Outputs, String> {
        Ok(self.0.as_object().unwrap().clone())
    }
}

fn run(source: Value, switch: Value, query: &str) -> Result<Vec<String>, RunError> {
    let dsl = json!({ "components": {
        "begin": { "obj": { "component_name": "Begin" }, "upstream": [], "downstream": ["Retrieval:r"] },
        "Retrieval:r": { "obj": { "component_name": "Retrieval", "params": { "fixed": source } }, "upstream": ["begin"], "downstream": ["Switch:s"] },
        "Switch:s": { "obj": { "component_name": "Switch", "params": switch }, "upstream": ["Retrieval:r"], "downstream": ["Message:yes", "Message:no"] },
        "Message:yes": { "obj": { "component_name": "Message", "params": { "content": ["yes"] } }, "upstream": ["Switch:s"], "downstream": [] },
        "Message:no": { "obj": { "component_name": "Message", "params": { "content": ["no"] } }, "upstream": ["Switch:s"], "downstream": [] },
    }});
    let mut registry = Registry::with_core();
    registry.register("Retrieval", |n: &Node| {
        Ok(Box::new(Fixed(n.params["fixed"].clone())) as Box<dyn Component>)
    });
    let bot = Bot::new(compile(&dsl.to_string()).unwrap(), &registry).unwrap();
    let mut events = Vec::new();
    let input = RunInput {
        query: query.into(),
        now: FixedOffset::east_opt(0)
            .unwrap()
            .with_ymd_and_hms(2026, 1, 1, 0, 0, 0)
            .unwrap(),
        deadline: None,
    };
    bot.run(input, &mut events)?;
    Ok(events
        .into_iter()
        .filter_map(|e| match e {
            Event::Message { content } => Some(content),
            _ => None,
        })
        .collect())
}

fn legacy(cpn_id: &str, operator: &str, value: Value) -> Value {
    json!({
        "conditions": [{ "logical_operator": "and", "items": [{ "cpn_id": cpn_id, "operator": operator, "value": value }], "to": ["Message:yes"] }],
        "end_cpn_ids": ["Message:no"],
    })
}

#[test]
fn legacy_empty_routes_on_an_empty_retrieval() {
    // A common shape: empty formalized_content routes to a no-info reply.
    let sw = legacy("Retrieval:r@formalized_content", "empty", json!(""));
    assert_eq!(
        run(json!({ "formalized_content": "" }), sw.clone(), "q").unwrap(),
        ["yes"]
    );
    assert_eq!(
        run(json!({ "formalized_content": "[ID:1] x" }), sw, "q").unwrap(),
        ["no"]
    );
}

#[test]
fn equality_ignores_case_and_compares_numbers_numerically() {
    assert_eq!(
        run(
            json!({ "v": "Parking" }),
            legacy("Retrieval:r@v", "=", json!("parking")),
            "q"
        )
        .unwrap(),
        ["yes"]
    );
    assert_eq!(
        run(
            json!({ "v": 3 }),
            legacy("Retrieval:r@v", "==", json!("3.0")),
            "q"
        )
        .unwrap(),
        ["yes"]
    );
    assert_eq!(
        run(
            json!({ "v": "a" }),
            legacy("Retrieval:r@v", "≠", json!("A")),
            "q"
        )
        .unwrap(),
        ["no"]
    );
}

#[test]
fn string_operators_lowercase_both_sides() {
    let src = json!({ "v": "Night Rate 50%" });
    assert_eq!(
        run(
            src.clone(),
            legacy("Retrieval:r@v", "contains", json!("RATE")),
            "q"
        )
        .unwrap(),
        ["yes"]
    );
    assert_eq!(
        run(
            src.clone(),
            legacy("Retrieval:r@v", "start with", json!("night")),
            "q"
        )
        .unwrap(),
        ["yes"]
    );
    assert_eq!(
        run(
            src.clone(),
            legacy("Retrieval:r@v", "end with", json!("50%")),
            "q"
        )
        .unwrap(),
        ["yes"]
    );
    assert_eq!(
        run(
            src,
            legacy("Retrieval:r@v", "not contains", json!("day")),
            "q"
        )
        .unwrap(),
        ["yes"]
    );
}

#[test]
fn numeric_comparison_on_text_fails_the_run() {
    let err = run(
        json!({ "v": "many" }),
        legacy("Retrieval:r@v", ">", json!(2)),
        "q",
    )
    .unwrap_err();
    assert!(
        matches!(err, RunError::Component { ref id, ref message } if id == "Switch:s" && message.contains("requires numeric operands")),
        "{err:?}"
    );
    assert_eq!(
        run(
            json!({ "v": "5" }),
            legacy("Retrieval:r@v", "≥", json!(5)),
            "q"
        )
        .unwrap(),
        ["yes"]
    );
}

#[test]
fn sys_query_can_be_matched_as_a_phrase_shortcut() {
    let sw = legacy("sys.query", "==", json!("營業時間"));
    assert_eq!(run(json!({}), sw.clone(), "營業時間").unwrap(), ["yes"]);
    assert_eq!(run(json!({}), sw, "營業時間？").unwrap(), ["no"]);
}

#[test]
fn new_clause_form_with_or() {
    let sw = json!({
        "conditions": [{ "op": "or", "clauses": [
            { "left": "{Retrieval:r@v}", "op": "==", "right": "x" },
            { "left": "{Retrieval:r@v}", "op": "==", "right": "y" },
        ], "to": "Message:yes" }],
        "default": "Message:no",
    });
    assert_eq!(run(json!({ "v": "y" }), sw.clone(), "q").unwrap(), ["yes"]);
    assert_eq!(run(json!({ "v": "z" }), sw, "q").unwrap(), ["no"]);
}

#[test]
fn an_empty_group_never_matches_and_a_group_without_to_routes_nowhere() {
    let empty_group = json!({ "conditions": [{ "items": [], "to": ["Message:yes"] }], "end_cpn_ids": ["Message:no"] });
    assert_eq!(run(json!({}), empty_group, "q").unwrap(), ["no"]);
    // matched_0 is not in downstream, so nothing after the Switch runs and
    // the run fails, as upstream's does (no terminal reached).
    let no_to = json!({ "conditions": [{ "items": [{ "cpn_id": "sys.query", "operator": "not empty" }] }], "end_cpn_ids": ["Message:no"] });
    assert!(matches!(
        run(json!({}), no_to, "q"),
        Err(RunError::NoTerminal { last, .. }) if last == "Switch:s"
    ));
}

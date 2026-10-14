//! Scheduler behaviour, with fake Agent and Switch components standing in
//! for the ones that call models.

use std::time::{Duration, Instant};

use chrono::{FixedOffset, TimeZone};
use ragflow_runtime::{
    compile, Bot, Component, Event, Node, NodeCtx, Outputs, Registry, RunError, RunInput,
};
use serde_json::{json, Value};

/// Answers with its `say` param, rendered against the run state.
struct FakeAgent(String);
impl Component for FakeAgent {
    fn invoke(&self, ctx: &mut NodeCtx<'_>) -> Result<Outputs, String> {
        let text = ragflow_runtime::template::render_strict(&self.0, ctx.state)
            .map_err(|e| e.to_string())?;
        Ok(json!({ "content": text }).as_object().unwrap().clone())
    }
}

/// Routes to its `route` param.
struct FakeRouter(Value);
impl Component for FakeRouter {
    fn invoke(&self, _: &mut NodeCtx<'_>) -> Result<Outputs, String> {
        Ok(json!({ "_next": self.0 }).as_object().unwrap().clone())
    }
}

fn registry() -> Registry {
    let mut r = Registry::with_core();
    r.register("Agent", |n: &Node| {
        Ok(Box::new(FakeAgent(
            n.params["say"].as_str().unwrap_or_default().to_string(),
        )) as Box<dyn Component>)
    })
    .register("Switch", |n: &Node| {
        Ok(Box::new(FakeRouter(n.params["route"].clone())) as Box<dyn Component>)
    });
    r
}

fn input(query: &str) -> RunInput {
    RunInput {
        query: query.into(),
        now: FixedOffset::east_opt(8 * 3600)
            .unwrap()
            .with_ymd_and_hms(2026, 10, 13, 16, 21, 0)
            .unwrap(),
        deadline: None,
    }
}

fn bot(dsl: Value) -> Bot {
    Bot::new(compile(&dsl.to_string()).unwrap(), &registry()).unwrap()
}

fn node(name: &str, params: Value, up: &[&str], down: &[&str]) -> Value {
    json!({ "obj": { "component_name": name, "params": params }, "upstream": up, "downstream": down })
}

fn messages(events: &[Event]) -> Vec<String> {
    events
        .iter()
        .filter_map(|e| match e {
            Event::Message { content } => Some(content.clone()),
            _ => None,
        })
        .collect()
}

fn ran(events: &[Event]) -> Vec<String> {
    events
        .iter()
        .filter_map(|e| match e {
            Event::NodeFinished { component_id, .. } => Some(component_id.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn a_linear_canvas_renders_sys_query_and_sys_date() {
    let b = bot(json!({ "components": {
        "begin": node("Begin", json!({}), &[], &["Message:a"]),
        "Message:a": node("Message", json!({ "content": ["You asked {sys.query} at {{ sys.date }}"] }), &["begin"], &[]),
    }}));
    let mut events = Vec::new();
    let out = b.run(input("fees?"), &mut events).unwrap();
    assert_eq!(
        messages(&events),
        ["You asked fees? at 2026-10-13 16:21:00"]
    );
    assert_eq!(
        out.answer["content"],
        "You asked fees? at 2026-10-13 16:21:00"
    );
    let names: Vec<_> = events.iter().map(Event::name).collect();
    assert_eq!(names.first(), Some(&"workflow_started"));
    assert_eq!(
        &names[names.len() - 2..],
        ["message_end", "workflow_finished"]
    );
}

#[test]
fn a_router_runs_only_the_chosen_child_and_a_join_still_runs() {
    // begin -> Switch -> {A, B}; A and B both feed Message:end.
    let b = bot(json!({ "components": {
        "begin": node("Begin", json!({}), &[], &["Switch:r"]),
        "Switch:r": node("Switch", json!({ "route": ["Agent:A"] }), &["begin"], &["Agent:A", "Agent:B"]),
        "Agent:A": node("Agent", json!({ "say": "from A" }), &["Switch:r"], &["Message:end"]),
        "Agent:B": node("Agent", json!({ "say": "from B" }), &["Switch:r"], &["Message:end"]),
        "Message:end": node("Message", json!({ "content": ["{Agent:A@content}"] }), &["Agent:A", "Agent:B"], &[]),
    }}));
    let mut events = Vec::new();
    b.run(input("q"), &mut events).unwrap();
    assert_eq!(
        ran(&events),
        ["begin", "Switch:r", "Agent:A", "Message:end"]
    );
    assert_eq!(messages(&events), ["from A"]);
}

#[test]
fn a_route_missing_from_downstream_is_dropped() {
    let b = bot(json!({ "components": {
        "begin": node("Begin", json!({}), &[], &["Switch:r"]),
        "Switch:r": node("Switch", json!({ "route": "Agent:A" }), &["begin"], &[]),
        "Agent:A": node("Agent", json!({ "say": "never" }), &["Switch:r"], &[]),
    }}));
    let mut events = Vec::new();
    b.run(input("q"), &mut events).unwrap();
    assert_eq!(ran(&events), ["begin", "Switch:r"]);
}

#[test]
fn an_agent_deferring_to_a_message_lets_the_template_win() {
    // Every child of the Agent is a Message, so the Message renders its own
    // template rather than the Agent's raw content.
    let b = bot(json!({ "components": {
        "begin": node("Begin", json!({}), &[], &["Agent:a"]),
        "Agent:a": node("Agent", json!({ "say": "raw" }), &["begin"], &["Message:m"]),
        "Message:m": node("Message", json!({ "content": ["Answer: {Agent:a@content}"] }), &["Agent:a"], &[]),
    }}));
    let mut events = Vec::new();
    b.run(input("q"), &mut events).unwrap();
    assert_eq!(messages(&events), ["Answer: raw"]);
}

#[test]
fn an_agent_not_deferring_overrides_the_message_template() {
    // The Agent also feeds a second Agent, so it does not defer: upstream's
    // Message then prints the incoming content, not its template.
    let b = bot(json!({ "components": {
        "begin": node("Begin", json!({}), &[], &["Agent:a"]),
        "Agent:a": node("Agent", json!({ "say": "raw" }), &["begin"], &["Message:m", "Agent:b"]),
        "Agent:b": node("Agent", json!({ "say": "b" }), &["Agent:a"], &[]),
        "Message:m": node("Message", json!({ "content": ["Answer: {Agent:a@content}"] }), &["Agent:a"], &[]),
    }}));
    let mut events = Vec::new();
    b.run(input("q"), &mut events).unwrap();
    assert_eq!(messages(&events), ["raw"]);
}

#[test]
fn an_unresolved_reference_fails_the_run() {
    let b = bot(json!({ "components": {
        "begin": node("Begin", json!({}), &[], &["Message:m"]),
        "Message:m": node("Message", json!({ "content": ["{Agent:gone@content}"] }), &["begin"], &[]),
    }}));
    let mut events = Vec::new();
    let err = b.run(input("q"), &mut events).unwrap_err();
    assert_eq!(
        err,
        RunError::Component {
            id: "Message:m".into(),
            message: "Can't find variable: 'Agent:gone@content'".into()
        }
    );
    assert_eq!(events.last().map(Event::name), Some("error"));
}

#[test]
fn a_spent_budget_stops_before_the_next_component() {
    let b = bot(json!({ "components": {
        "begin": node("Begin", json!({}), &[], &["Message:m"]),
        "Message:m": node("Message", json!({ "content": ["hi"] }), &["begin"], &[]),
    }}));
    let mut run = input("q");
    run.deadline = Some(Instant::now() - Duration::from_millis(1));
    let err = b.run(run, &mut Vec::new()).unwrap_err();
    assert_eq!(
        err,
        RunError::Deadline {
            next: "begin".into()
        }
    );
}

#[test]
fn begin_fills_declared_fields_from_the_query() {
    let b = bot(json!({ "components": {
        "begin": node("Begin", json!({ "inputs": { "topic": {} } }), &[], &["Message:m"]),
        "Message:m": node("Message", json!({ "content": ["{begin@topic}"] }), &["begin"], &[]),
    }}));
    let mut events = Vec::new();
    b.run(input("parking"), &mut events).unwrap();
    assert_eq!(messages(&events), ["parking"]);
}

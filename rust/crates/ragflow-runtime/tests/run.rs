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
        if let Some(error) = self.0.strip_prefix("!") {
            return Err(error.to_string());
        }
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
        ids: Default::default(),
        user_input: None,
        cancel: None,
        finish: None,
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
fn a_child_outside_the_router_downstream_is_not_gated() {
    // Agent:A lists the Switch as upstream but is not in its downstream, so
    // the Switch's branch does not cover it (multibranch.go): it runs
    // whatever the Switch picks.
    let b = bot(json!({ "components": {
        "begin": node("Begin", json!({}), &[], &["Switch:r"]),
        "Switch:r": node("Switch", json!({ "route": "Message:m" }), &["begin"], &["Message:m"]),
        "Message:m": node("Message", json!({ "content": ["m"] }), &["Switch:r"], &[]),
        "Agent:A": node("Agent", json!({ "say": "ungated" }), &["Switch:r"], &[]),
    }}));
    let mut events = Vec::new();
    b.run(input("q"), &mut events).unwrap();
    assert_eq!(ran(&events), ["begin", "Switch:r", "Agent:A", "Message:m"]);
}

#[test]
fn a_run_where_no_terminal_ran_fails_like_upstream() {
    let canvas = |router: &str| {
        json!({ "components": {
            "begin": node("Begin", json!({}), &[], &[router]),
            router: node("Switch", json!({ "route": [] }), &["begin"], &["Message:m"]),
            "Message:m": node("Message", json!({ "content": ["m"] }), &[router], &[]),
        }})
    };
    let mut events = Vec::new();
    let err = bot(canvas("Switch:r"))
        .run(input("q"), &mut events)
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "agent invoke: Switch routing stopped because no connected branch matched the condition; check the Switch branches"
    );
    assert_eq!(events.last().map(Event::name), Some("error"));
    let mut events = Vec::new();
    let err = bot(canvas("router"))
        .run(input("q"), &mut events)
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "agent invoke: [GraphRunError] no tasks to execute, last completed nodes: [router]"
    );
}

#[test]
fn a_deferred_agent_runs_inside_the_message_in_template_order() {
    // Both Agents feed only a terminal Message, so both defer: they start
    // when scheduled and finish when the Message reaches their reference.
    let b = bot(json!({ "components": {
        "begin": node("Begin", json!({}), &[], &["Agent:a", "Agent:b"]),
        "Agent:a": node("Agent", json!({ "say": "A" }), &["begin"], &["Message:m"]),
        "Agent:b": node("Agent", json!({ "say": "B" }), &["begin"], &["Message:m"]),
        "Message:m": node("Message", json!({ "content": ["1 {Agent:b@content} 2 {Agent:a@content}"] }), &["Agent:a", "Agent:b"], &[]),
    }}));
    let mut events = Vec::new();
    let out = b.run(input("q"), &mut events).unwrap();
    assert_eq!(ran(&events), ["begin", "Agent:b", "Agent:a", "Message:m"]);
    assert_eq!(messages(&events), ["1 ", "B", " 2 ", "A"]);
    assert_eq!(out.answer["content"], "1 B 2 A");
}

#[test]
fn an_unreferenced_deferred_agent_finishes_at_the_end_without_running() {
    let b = bot(json!({ "components": {
        "begin": node("Begin", json!({}), &[], &["Agent:a"]),
        "Agent:a": node("Agent", json!({ "say": "!must not run" }), &["begin"], &["Message:m"]),
        "Message:m": node("Message", json!({ "content": ["fixed"] }), &["Agent:a"], &[]),
    }}));
    let mut events = Vec::new();
    b.run(input("q"), &mut events).unwrap();
    assert_eq!(ran(&events), ["begin", "Message:m", "Agent:a"]);
    assert_eq!(messages(&events), ["fixed"]);
}

#[test]
fn a_failing_deferred_agent_becomes_the_reply_without_workflow_finished() {
    let b = bot(json!({ "components": {
        "begin": node("Begin", json!({}), &[], &["Agent:a"]),
        "Agent:a": node("Agent", json!({ "say": "!component: invalid param \"model_id\": required" }), &["begin"], &["Message:m"]),
        "Message:m": node("Message", json!({ "content": ["{Agent:a@content}"] }), &["Agent:a"], &[]),
    }}));
    let mut events = Vec::new();
    let out = b.run(input("q"), &mut events).unwrap();
    assert_eq!(
        out.deferred_failure.as_deref(),
        Some(r#"component: invalid param "model_id": required"#)
    );
    let names: Vec<_> = events.iter().map(Event::name).collect();
    assert_eq!(
        names,
        [
            "workflow_started",
            "node_started",
            "node_finished",
            "node_started",
            "node_started",
            "node_finished",
            "message",
            "message_end"
        ]
    );
    assert_eq!(
        messages(&events),
        [r#"component: invalid param "model_id": required"#]
    );
}

#[test]
fn an_eager_agent_does_not_override_the_message_template() {
    // The Agent also feeds a second Agent, so it runs eagerly; the Message
    // template still wins over the Agent's incoming `content` (main).
    let b = bot(json!({ "components": {
        "begin": node("Begin", json!({}), &[], &["Agent:a"]),
        "Agent:a": node("Agent", json!({ "say": "raw" }), &["begin"], &["Message:m", "Agent:b"]),
        "Agent:b": node("Agent", json!({ "say": "b" }), &["Agent:a"], &[]),
        "Message:m": node("Message", json!({ "content": ["Answer: {Agent:a@content}"] }), &["Agent:a"], &[]),
    }}));
    let mut events = Vec::new();
    b.run(input("q"), &mut events).unwrap();
    assert_eq!(messages(&events), ["Answer: raw"]);
}

#[test]
fn an_upstream_text_input_overrides_the_template_and_message_edges_carry_no_data() {
    // Message:m1 gets Agent:a's {text} and prints it over its template.
    // Message:m2's only upstream is a Message, so it reads the workflow
    // input ({query}), which has no text to print.
    let mut r = registry();
    r.register("Agent", |_: &Node| {
        struct Text;
        impl Component for Text {
            fn invoke(&self, _: &mut NodeCtx<'_>) -> Result<Outputs, String> {
                Ok(json!({ "text": "override" }).as_object().unwrap().clone())
            }
        }
        Ok(Box::new(Text) as Box<dyn Component>)
    });
    let dsl = json!({ "components": {
        "begin": node("Begin", json!({}), &[], &["Agent:a"]),
        "Agent:a": node("Agent", json!({}), &["begin"], &["Message:m1", "Message:m2"]),
        "Message:m1": node("Message", json!({ "content": ["template"] }), &["Agent:a"], &["Message:m2"]),
        "Message:m2": node("Message", json!({ "content": [""] }), &["Message:m1"], &[]),
    }});
    let b = Bot::new(compile(&dsl.to_string()).unwrap(), &r).unwrap();
    let mut events = Vec::new();
    b.run(input("the query"), &mut events).unwrap();
    assert_eq!(messages(&events), ["override"]);
    let m2_inputs = events.iter().find_map(|e| match e {
        Event::NodeStarted {
            component_id,
            inputs,
            ..
        } if component_id == "Message:m2" => Some(inputs.clone()),
        _ => None,
    });
    assert_eq!(
        m2_inputs.unwrap(),
        *json!({ "query": "the query" }).as_object().unwrap()
    );
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

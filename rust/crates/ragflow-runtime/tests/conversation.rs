//! Multi-turn runs: the conversation a session DSL carries reaches the
//! prompts, the service identities reach `sys.*`, and the finish hook runs
//! before the closing events.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use chrono::{FixedOffset, TimeZone};
use ragflow_runtime::models::{ChatMessage, ChatOptions, ModelRef, Models};
use ragflow_runtime::{
    compile, python_history_repr, Bot, Event, EventSink, Registry, RunError, RunInput,
};
use serde_json::{json, Value};

/// One chat request: the component and its `(role, content)` messages.
type Call = (String, Vec<(String, String)>);

/// Records every chat request and answers with a fixed reply per component.
struct Recorder {
    calls: Mutex<Vec<Call>>,
    category: String,
}

impl Models for Recorder {
    fn chat(&self, m: &ModelRef, msgs: &[ChatMessage], o: &ChatOptions) -> Result<String, String> {
        self.chat_for("", m, msgs, o)
    }
    fn chat_for(
        &self,
        cpn: &str,
        _: &ModelRef,
        msgs: &[ChatMessage],
        _: &ChatOptions,
    ) -> Result<String, String> {
        self.calls.lock().unwrap().push((
            cpn.to_string(),
            msgs.iter()
                .map(|m| (m.role.to_string(), m.content.clone()))
                .collect(),
        ));
        Ok(if cpn.starts_with("Categorize") {
            self.category.clone()
        } else {
            "the answer".into()
        })
    }
    fn embed(&self, _: &ModelRef, _: &[String]) -> Result<Vec<Vec<f32>>, String> {
        Err("no embeddings".into())
    }
}

fn node(name: &str, params: Value, up: &[&str], down: &[&str]) -> Value {
    json!({ "obj": { "component_name": name, "params": params }, "upstream": up, "downstream": down })
}

fn canvas(history: Value, agent_window: Option<i64>) -> Value {
    let mut agent = json!({
        "llm_id": "m@Prov", "sys_prompt": "be brief", "prompts": [{"role": "user", "content": "{sys.query}"}]
    });
    if let Some(w) = agent_window {
        agent["message_history_window_size"] = json!(w);
    }
    json!({
        "globals": { "sys.history": [], "sys.conversation_turns": 1 },
        "history": history,
        "components": {
            "begin": node("Begin", json!({}), &[], &["Categorize:c"]),
            "Categorize:c": node("Categorize", json!({
                "llm_id": "m@Prov", "message_history_window_size": 2,
                "items": [{"name": "ask", "description": "", "examples": [], "to": ["Agent:a"]}]
            }), &["begin"], &["Agent:a"]),
            "Agent:a": node("Agent", agent, &["Categorize:c"], &["Message:m"]),
            "Message:m": node("Message", json!({"content": ["{Agent:a@content}"]}), &["Agent:a"], &[]),
        }
    })
}

fn run_with(
    dsl: Value,
    query: &str,
    finish: Option<ragflow_runtime::FinishHook>,
) -> (Vec<Call>, Vec<Event>, Bot) {
    let models = Arc::new(Recorder {
        calls: Mutex::new(Vec::new()),
        category: "ask".into(),
    });
    let bot = Bot::new(
        compile(&dsl.to_string()).unwrap(),
        &Registry::with_models(models.clone()),
    )
    .unwrap();
    let mut input = RunInput::new(
        query,
        FixedOffset::east_opt(0)
            .unwrap()
            .with_ymd_and_hms(2026, 1, 1, 0, 0, 0)
            .unwrap(),
    );
    input.ids.user_id = "u1".into();
    input.ids.canvas_id = "c1".into();
    input.ids.session_id = "s1".into();
    input.ids.tenant_id = "t1".into();
    input.finish = finish;
    let mut events = Vec::new();
    bot.run(input, &mut events).unwrap();
    let calls = models.calls.lock().unwrap().clone();
    (calls, events, bot)
}

fn prior() -> Value {
    json!([
        ["user", "first question"],
        ["assistant", {"content": "first answer"}],
        ["user", "second question"],
        ["assistant", {"content": "second answer", "downloads": []}],
    ])
}

#[test]
fn the_agent_prompt_carries_the_last_2n_minus_1_prior_turns() {
    let (calls, _, _) = run_with(canvas(prior(), Some(2)), "third", None);
    let agent = calls.iter().find(|(c, _)| c == "Agent:a").unwrap();
    let roles: Vec<_> = agent
        .1
        .iter()
        .map(|(r, c)| (r.as_str(), c.as_str()))
        .collect();
    assert_eq!(
        roles,
        [
            ("system", "be brief"),
            ("assistant", "first answer"),
            ("user", "second question"),
            ("assistant", "second answer"),
            ("user", "third"),
        ],
        "window 2 keeps the last 3 completed turns, then the current prompt"
    );
}

#[test]
fn a_zero_window_sends_no_history() {
    let (calls, _, _) = run_with(canvas(prior(), Some(0)), "third", None);
    let agent = calls.iter().find(|(c, _)| c == "Agent:a").unwrap();
    assert_eq!(agent.1.len(), 2);
}

#[test]
fn categorize_lists_its_window_with_the_current_turn_last() {
    let (calls, _, _) = run_with(canvas(prior(), None), "third", None);
    let cat = calls.iter().find(|(c, _)| c == "Categorize:c").unwrap();
    assert_eq!(
        cat.1[1].1,
        "\n---- Real Data ----\nASSISTANT: \"second answer\" | USER: \"third\" ->\n"
    );
}

#[test]
fn a_fresh_session_lists_only_the_query() {
    let (calls, _, _) = run_with(canvas(json!([]), None), "hello", None);
    let cat = calls.iter().find(|(c, _)| c == "Categorize:c").unwrap();
    assert_eq!(cat.1[1].1, "\n---- Real Data ----\nUSER: \"hello\" ->\n");
    let agent = calls.iter().find(|(c, _)| c == "Agent:a").unwrap();
    assert_eq!(agent.1.len(), 2, "system + current prompt only");
}

/// Streams "ab" then "cd" as deltas.
struct Streamer;
impl Models for Streamer {
    fn chat(&self, _: &ModelRef, _: &[ChatMessage], _: &ChatOptions) -> Result<String, String> {
        Ok("abcd".into())
    }
    fn chat_stream_for(
        &self,
        _: &str,
        _: &ModelRef,
        _: &[ChatMessage],
        _: &ChatOptions,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String, String> {
        on_delta("ab");
        on_delta("cd");
        Ok("abcd".into())
    }
    fn embed(&self, _: &ModelRef, _: &[String]) -> Result<Vec<Vec<f32>>, String> {
        Err("no".into())
    }
}

#[test]
fn a_deferred_agent_streams_before_it_finishes_and_the_message_does_not_repeat_it() {
    let dsl = json!({"components": {
        "begin": node("Begin", json!({}), &[], &["Agent:a"]),
        "Agent:a": node("Agent", json!({"llm_id": "m@Prov", "prompts": [{"role": "user", "content": "{sys.query}"}]}),
            &["begin"], &["Message:m"]),
        "Message:m": node("Message", json!({"content": ["<{Agent:a@content}>"]}), &["Agent:a"], &[]),
    }});
    let bot = Bot::new(
        compile(&dsl.to_string()).unwrap(),
        &Registry::with_models(Arc::new(Streamer)),
    )
    .unwrap();
    let input = RunInput::new(
        "q",
        FixedOffset::east_opt(0)
            .unwrap()
            .with_ymd_and_hms(2026, 1, 1, 0, 0, 0)
            .unwrap(),
    );
    let mut events = Vec::new();
    bot.run(input, &mut events).unwrap();
    let shape: Vec<String> = events
        .iter()
        .map(|e| match e {
            Event::Message { content } => format!("message:{content}"),
            Event::NodeFinished { component_id, .. } => format!("finished:{component_id}"),
            other => other.name().to_string(),
        })
        .collect();
    let at = |what: &str| shape.iter().position(|s| s == what).unwrap();
    assert!(at("message:<") < at("message:ab") && at("message:ab") < at("message:cd"));
    assert!(at("message:cd") < at("finished:Agent:a"), "{shape:?}");
    assert!(at("finished:Agent:a") < at("message:>"), "{shape:?}");
    let text: String = shape
        .iter()
        .filter_map(|s| s.strip_prefix("message:"))
        .collect();
    assert_eq!(text, "<abcd>", "the answer is said once");
}

struct Names(Arc<Mutex<Vec<String>>>);
impl EventSink for Names {
    fn emit(&mut self, e: Event) {
        self.0.lock().unwrap().push(e.name().to_string());
    }
}

#[test]
fn the_finish_hook_sees_the_state_before_the_closing_events() {
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let log = seen.clone();
    let hook: ragflow_runtime::FinishHook = Box::new(move |f| {
        log.lock().unwrap().push("hook".into());
        assert_eq!(f.visible_answer, "the answer");
        assert!(f.deferred_failure.is_none());
        assert_eq!(f.state.sys["session_id"], "s1");
        assert_eq!(f.state.sys["user_id"], "u1");
        assert_eq!(f.state.sys["canvas_id"], "c1");
        assert_eq!(f.state.sys["tenant_id"], "t1");
        assert_eq!(f.state.sys["history"], json!(["user: third"]));
        assert_eq!(f.state.sys["conversation_turns"], json!(2));
        let last = f.state.history.last().unwrap();
        assert_eq!(last["role"], "user");
        f.state
            .append_history("assistant", json!({"content": "the answer"}));
        Ok(())
    });
    let models = Arc::new(Recorder {
        calls: Mutex::new(Vec::new()),
        category: "ask".into(),
    });
    let bot = Bot::new(
        compile(&canvas(prior(), None).to_string()).unwrap(),
        &Registry::with_models(models),
    )
    .unwrap();
    let mut input = RunInput::new(
        "third",
        FixedOffset::east_opt(0)
            .unwrap()
            .with_ymd_and_hms(2026, 1, 1, 0, 0, 0)
            .unwrap(),
    );
    input.ids.user_id = "u1".into();
    input.ids.canvas_id = "c1".into();
    input.ids.session_id = "s1".into();
    input.ids.tenant_id = "t1".into();
    input.finish = Some(hook);
    let out = bot.run(input, &mut Names(seen.clone())).unwrap();
    let names = seen.lock().unwrap().clone();
    let hook_at = names.iter().position(|n| n == "hook").unwrap();
    assert!(names[..hook_at].contains(&"node_finished".to_string()));
    assert!(!names[..hook_at].contains(&"message_end".to_string()));
    assert!(names[hook_at..].contains(&"message_end".to_string()));
    assert_eq!(names.last().unwrap(), "workflow_finished");
    let encoded = ragflow_runtime::encode_history(&out.state.history);
    assert_eq!(
        encoded.as_array().unwrap().last().unwrap(),
        &json!(["assistant", {"content": "the answer"}])
    );
}

#[test]
fn a_raised_cancel_flag_ends_the_run_without_an_error_event() {
    let models = Arc::new(Recorder {
        calls: Mutex::new(Vec::new()),
        category: "ask".into(),
    });
    let bot = Bot::new(
        compile(&canvas(json!([]), None).to_string()).unwrap(),
        &Registry::with_models(models),
    )
    .unwrap();
    let mut input = RunInput::new(
        "x",
        FixedOffset::east_opt(0)
            .unwrap()
            .with_ymd_and_hms(2026, 1, 1, 0, 0, 0)
            .unwrap(),
    );
    input.cancel = Some(Arc::new(AtomicBool::new(true)));
    let mut events = Vec::new();
    assert_eq!(
        bot.run(input, &mut events).unwrap_err(),
        RunError::Cancelled
    );
    assert_eq!(
        events.iter().map(Event::name).collect::<Vec<_>>(),
        ["workflow_started"]
    );
}

#[test]
fn python_reprs_match_the_go_service() {
    assert_eq!(python_history_repr(&json!("it's\n")), "'it\\'s\\n'");
    assert_eq!(
        python_history_repr(
            &json!({"_created_time": "t", "extra": 1, "content": "a", "downloads": []})
        ),
        "{'content': 'a', 'downloads': [], 'extra': 1, '_created_time': 't'}"
    );
    assert_eq!(python_history_repr(&json!(null)), "None");
    assert_eq!(python_history_repr(&json!(true)), "True");
}

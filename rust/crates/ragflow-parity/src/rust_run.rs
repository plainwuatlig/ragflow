//! The Rust side: runs a canvas through `ragflow-runtime` with a `Models`
//! that replays recordings, and returns the events as the JSON lines
//! `oracle-go` prints (`{"event": <name>, "data": <payload>}`), including
//! an `llm_call` line per chat request.
//!
//! Agent and Categorize call `Models::chat_for` with their component id
//! (a deferred Agent calls while its Message runs, so the event stream
//! cannot tell); plain `chat` falls back to the last `node_started`.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use chrono::{FixedOffset, TimeZone};
use ragflow_runtime::models::{ChatMessage, ChatOptions, ModelRef, Models};
use ragflow_runtime::{compile, Bot, Event, EventSink, Registry, RunInput};
use serde_json::{json, Map, Value};

#[derive(Default)]
struct Shared {
    lines: Vec<Value>,
    current: String,
    calls: BTreeMap<String, usize>,
    missing: Vec<String>,
}

struct Recorded {
    answers: Map<String, Value>,
    shared: Arc<Mutex<Shared>>,
}

impl Models for Recorded {
    fn chat(
        &self,
        model: &ModelRef,
        messages: &[ChatMessage],
        options: &ChatOptions,
    ) -> Result<String, String> {
        let cpn = self
            .shared
            .lock()
            .expect("parity state lock")
            .current
            .clone();
        self.chat_for(&cpn, model, messages, options)
    }

    fn chat_for(
        &self,
        cpn: &str,
        model: &ModelRef,
        messages: &[ChatMessage],
        options: &ChatOptions,
    ) -> Result<String, String> {
        let mut shared = self.shared.lock().expect("parity state lock");
        let cpn = cpn.to_string();
        let n = *shared.calls.get(&cpn).unwrap_or(&0);
        shared.calls.insert(cpn.clone(), n + 1);
        let key = format!("{cpn}#{n}");
        let msgs: Vec<Value> = messages
            .iter()
            .map(|m| json!({ "role": m.role, "content": m.content }))
            .collect();
        shared.lines.push(json!({ "event": "llm_call", "data": {
            "component_id": cpn,
            "call_index": n,
            "key": key,
            "model": format!("{}@{}", model.model, model.provider),
            "messages": msgs,
            "options": {
                "temperature": options.temperature,
                "top_p": options.top_p,
                "max_tokens": options.max_tokens,
            },
        }}));
        match self.answers.get(&key).and_then(Value::as_str) {
            Some(answer) => Ok(answer.to_string()),
            None => {
                eprintln!(
                    "ragflow-parity: no recording for {key:?} (model {}@{}); messages:\n{}",
                    model.model,
                    model.provider,
                    serde_json::to_string_pretty(&msgs).unwrap_or_default()
                );
                shared.missing.push(key.clone());
                Err(format!("parity: no recording for {key:?}"))
            }
        }
    }

    fn embed(&self, _: &ModelRef, _: &[String]) -> Result<Vec<Vec<f32>>, String> {
        Err("parity: embeddings are not recorded".into())
    }
}

struct Sink(Arc<Mutex<Shared>>);

impl EventSink for Sink {
    fn emit(&mut self, event: Event) {
        let mut shared = self.0.lock().expect("parity state lock");
        if let Event::NodeStarted { component_id, .. } = &event {
            shared.current = component_id.clone();
        }
        shared
            .lines
            .push(json!({ "event": event.name(), "data": event.data() }));
    }
}

pub struct Outcome {
    pub lines: Vec<Value>,
    /// Recording keys the run asked for and did not find.
    pub missing: Vec<String>,
}

pub fn run(dsl: &str, query: &str, recordings: &Map<String, Value>) -> Outcome {
    let shared = Arc::new(Mutex::new(Shared::default()));
    let error = |shared: &Arc<Mutex<Shared>>, message: String| {
        shared
            .lock()
            .expect("parity state lock")
            .lines
            .push(json!({ "event": "error", "data": { "message": message } }));
    };
    match compile(dsl) {
        Err(e) => error(&shared, format!("compile: {e}")),
        Ok(plan) => {
            let models = Arc::new(Recorded {
                answers: recordings.clone(),
                shared: shared.clone(),
            });
            match Bot::new(plan, &Registry::with_models(models)) {
                Err(e) => error(&shared, format!("build: {e}")),
                Ok(bot) => {
                    let now = FixedOffset::east_opt(0)
                        .expect("utc offset")
                        .with_ymd_and_hms(2026, 1, 1, 0, 0, 0)
                        .single()
                        .expect("fixed date");
                    let input = RunInput {
                        query: query.to_string(),
                        now,
                        deadline: None,
                        ids: Default::default(),
                        user_input: None,
                        cancel: None,
                        finish: None,
                    };
                    // A run error is already in the stream as an `error` event.
                    let _ = bot.run(input, &mut Sink(shared.clone()));
                }
            }
        }
    }
    let shared = Arc::try_unwrap(shared)
        .ok()
        .expect("no other owners after the run")
        .into_inner()
        .expect("parity state lock");
    Outcome {
        lines: shared.lines,
        missing: shared.missing,
    }
}

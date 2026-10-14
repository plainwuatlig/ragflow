//! Categorize and Agent against a fake model that records what it was sent.

use std::sync::{Arc, Mutex};

use chrono::{FixedOffset, TimeZone};
use ragflow_runtime::components::categorize::go_quote;
use ragflow_runtime::models::{ChatMessage, ChatOptions, ModelRef, Models};
use ragflow_runtime::{compile, Bot, BuildError, Event, Registry, RunInput};
use serde_json::{json, Value};

#[derive(Default)]
struct Fake {
    answers: Mutex<Vec<String>>,
    sent: Mutex<Vec<(ModelRef, Vec<ChatMessage>, ChatOptions)>>,
}
impl Models for Fake {
    fn chat(
        &self,
        model: &ModelRef,
        messages: &[ChatMessage],
        options: &ChatOptions,
    ) -> Result<String, String> {
        self.sent
            .lock()
            .unwrap()
            .push((model.clone(), messages.to_vec(), options.clone()));
        Ok(self.answers.lock().unwrap().remove(0))
    }
    fn embed(&self, _: &ModelRef, _: &[String]) -> Result<Vec<Vec<f32>>, String> {
        unreachable!()
    }
}

fn parking_bot_canvas() -> Value {
    // A canvas using both Categorize forms: items carry uuids, the legacy
    // category_description carries the real routes.
    json!({ "components": {
        "begin": { "obj": { "component_name": "Begin" }, "upstream": [], "downstream": ["Categorize:i"] },
        "Categorize:i": { "obj": { "component_name": "Categorize", "params": {
            "llm_id": "gpt-4o-mini@default@OpenAI", "query": "sys.query",
            "items": [
                { "uuid": "u1", "name": "parking", "description": "停車場費率、優惠", "examples": [{ "value": "停一小時多少\n錢" }] },
                { "uuid": "u2", "name": "chitchat", "description": "", "examples": [] },
            ],
            "category_description": {
                "parking": { "description": "停車場費率、優惠", "examples": ["停一小時多少\n錢"], "to": ["Message:p"], "uuid": "u1" },
                "chitchat": { "description": "", "examples": [], "to": ["Agent:c"], "uuid": "u2" },
            },
        } }, "upstream": ["begin"], "downstream": ["Message:p", "Agent:c"] },
        "Message:p": { "obj": { "component_name": "Message", "params": { "content": ["parking"] } }, "upstream": ["Categorize:i"], "downstream": [] },
        "Agent:c": { "obj": { "component_name": "Agent", "params": {
            "llm_id": "gpt-4o-mini@default@OpenAI", "sys_prompt": "你是客服。現在：{sys.date}",
            "prompts": [{ "role": "user", "content": "{sys.query}" }], "temperature": 0.1, "cite": false, "tools": [], "mcp": [],
        } }, "upstream": ["Categorize:i"], "downstream": ["Message:c"] },
        "Message:c": { "obj": { "component_name": "Message", "params": { "content": ["{Agent:c@content}"] } }, "upstream": ["Agent:c"], "downstream": [] },
    }})
}

fn run(fake: Arc<Fake>, dsl: Value, query: &str) -> Vec<String> {
    let bot = Bot::new(
        compile(&dsl.to_string()).unwrap(),
        &Registry::with_models(fake),
    )
    .unwrap();
    let mut events = Vec::new();
    let input = RunInput {
        query: query.into(),
        now: FixedOffset::east_opt(8 * 3600)
            .unwrap()
            .with_ymd_and_hms(2026, 10, 13, 16, 21, 0)
            .unwrap(),
        deadline: None,
    };
    bot.run(input, &mut events).unwrap();
    events
        .into_iter()
        .filter_map(|e| match e {
            Event::Message { content } => Some(content),
            _ => None,
        })
        .collect()
}

#[test]
fn categorize_sends_upstreams_prompt_and_routes_by_category_description() {
    let fake = Arc::new(Fake::default());
    fake.answers
        .lock()
        .unwrap()
        .push("Category: Parking".into());
    assert_eq!(
        run(fake.clone(), parking_bot_canvas(), "停兩小時多少\n錢？"),
        ["parking"]
    );

    let sent = fake.sent.lock().unwrap();
    let (model, messages, options) = &sent[0];
    assert_eq!(
        model,
        &ModelRef {
            model: "gpt-4o-mini".into(),
            instance: Some("default".into()),
            provider: "OpenAI".into()
        }
    );
    assert_eq!(options, &ChatOptions::default());
    assert_eq!(
        messages[0].content,
        "You are an advanced classification system that categorizes user questions into specific types. Analyze the input question and classify it into ONE of the following categories:\n\
         - chitchat\n- parking\n\
         \nHere's description of each category:\n\
         \n------\nCategory: parking\nDescription: 停車場費率、優惠\n\
         \n---- Instructions ----\n\
         - Consider both explicit mentions and implied context\n\
         - Prioritize the most specific applicable category\n\
         - Return only the category name without explanations\n\
         \n---- Examples ----\n\
         USER: \"停一小時多少    錢\" -> parking\n"
    );
    assert_eq!(
        messages[1].content,
        "\n---- Real Data ----\nUSER: \"停兩小時多少錢？\" ->\n"
    );
}

#[test]
fn an_unmatched_answer_falls_back_to_the_first_category_and_the_agent_answers() {
    let fake = Arc::new(Fake::default());
    fake.answers
        .lock()
        .unwrap()
        .extend(["no idea".to_string(), "您好！".to_string()]);
    let mut dsl = parking_bot_canvas();
    // Put chitchat first so it is the default.
    let items = dsl["components"]["Categorize:i"]["obj"]["params"]["items"]
        .as_array_mut()
        .unwrap();
    items.reverse();
    assert_eq!(run(fake.clone(), dsl, "嗨"), ["您好！"]);

    let sent = fake.sent.lock().unwrap();
    let (_, messages, options) = &sent[1];
    assert_eq!(
        messages[0],
        ChatMessage {
            role: "system",
            content: "你是客服。現在：2026-10-13 16:21:00".into()
        }
    );
    assert_eq!(
        messages[1],
        ChatMessage {
            role: "user",
            content: "嗨".into()
        }
    );
    assert_eq!(options.temperature, Some(0.1));
}

#[test]
fn tools_and_citations_are_refused_at_build() {
    let mut dsl = parking_bot_canvas();
    dsl["components"]["Agent:c"]["obj"]["params"]["cite"] = json!(true);
    let err = Bot::new(
        compile(&dsl.to_string()).unwrap(),
        &Registry::with_models(Arc::new(Fake::default())),
    )
    .err()
    .unwrap();
    assert_eq!(
        err,
        BuildError::Params {
            id: "Agent:c".into(),
            message: "cite: not supported by this runtime yet".into()
        }
    );
}

#[test]
fn go_quote_keeps_cjk_and_escapes_controls() {
    assert_eq!(go_quote("停車\"費\"\\\t"), r#""停車\"費\"\\\t""#);
    assert_eq!(go_quote("\u{1}"), r#""\x01""#);
}

#[test]
fn model_ids_parse_with_and_without_instance() {
    assert_eq!(ModelRef::parse("bge-m3@Ollama").unwrap().instance, None);
    assert!(ModelRef::parse("gpt-4o-mini").is_err());
}

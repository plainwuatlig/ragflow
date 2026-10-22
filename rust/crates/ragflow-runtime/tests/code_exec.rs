//! CodeExec with `lang: lua` inside a canvas: arguments resolve from the run,
//! the business output feeds a Message, and other languages are refused.

use chrono::{FixedOffset, TimeZone};
use ragflow_runtime::{
    compile, Bot, BuildError, Component, Event, Node, NodeCtx, Outputs, Registry, RunInput,
};
use serde_json::{json, Value};

/// Emits fixed outputs, standing in for the Agent that extracts the inputs.
struct Fixed(Value);
impl Component for Fixed {
    fn invoke(&self, _: &mut NodeCtx<'_>) -> Result<Outputs, String> {
        Ok(self.0.as_object().unwrap().clone())
    }
}

const FEE: &str = r#"
local function car_fee(m)
  local h, rest = m // 60, m % 60
  local tail = (rest == 0) and 0 or (rest < 30 and 15 or 30)
  return h * 30 + tail
end
function main(args)
  if args.vehicle == "moto" then return 20 end
  return car_fee(tonumber(args.minutes))
end
"#;

fn canvas(extracted: Value, code: Value) -> Value {
    json!({ "components": {
        "begin": { "obj": { "component_name": "Begin" }, "upstream": [], "downstream": ["Agent:x"] },
        "Agent:x": { "obj": { "component_name": "Agent", "params": { "fixed": extracted } }, "upstream": ["begin"], "downstream": ["CodeExec:fee"] },
        "CodeExec:fee": { "obj": { "component_name": "CodeExec", "params": code }, "upstream": ["Agent:x"], "downstream": ["Message:m"] },
        "Message:m": { "obj": { "component_name": "Message", "params": { "content": ["共 {CodeExec:fee@fee} 元"] } }, "upstream": ["CodeExec:fee"], "downstream": [] },
    }})
}

fn registry() -> Registry {
    let mut registry = Registry::with_core();
    registry.register("Agent", |n: &Node| {
        Ok(Box::new(Fixed(n.params["fixed"].clone())) as Box<dyn Component>)
    });
    registry
}

fn messages(dsl: Value) -> Vec<String> {
    let bot = Bot::new(compile(&dsl.to_string()).unwrap(), &registry()).unwrap();
    let mut events = Vec::new();
    let input = RunInput {
        query: "how much for 140 minutes?".into(),
        now: FixedOffset::east_opt(8 * 3600)
            .unwrap()
            .with_ymd_and_hms(2026, 10, 22, 9, 0, 0)
            .unwrap(),
        deadline: None,
        ids: Default::default(),
        user_input: None,
        cancel: None,
        finish: None,
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

fn fee_node() -> Value {
    json!({
        "lang": "lua",
        "script": FEE,
        "arguments": { "vehicle": "Agent:x@vehicle", "minutes": "Agent:x@minutes" },
        "outputs": { "fee": { "type": "Number", "value": "" } },
    })
}

#[test]
fn the_fee_comes_from_the_script() {
    let out = messages(canvas(
        json!({ "vehicle": "car", "minutes": "140" }),
        fee_node(),
    ));
    assert_eq!(out.concat(), "共 75 元");
}

#[test]
fn a_script_error_reaches_the_message_and_the_run_goes_on() {
    let mut code = fee_node();
    code["script"] = json!("function main(args) error('no minutes') end");
    // The business output is null then, which a Message cannot reference
    // (as upstream); a canvas routes on `_ERROR` instead.
    let mut dsl = canvas(json!({ "vehicle": "car" }), code);
    dsl["components"]["Message:m"]["obj"]["params"]["content"] =
        json!(["失敗：{CodeExec:fee@_ERROR}"]);
    let out = messages(dsl).concat();
    assert!(out.starts_with("失敗："), "{out}");
    assert!(out.contains("no minutes"), "{out}");
}

#[test]
fn without_outputs_the_value_is_result() {
    let mut code = fee_node();
    code.as_object_mut().unwrap().remove("outputs");
    let mut dsl = canvas(json!({ "vehicle": "moto", "minutes": "30" }), code);
    dsl["components"]["Message:m"]["obj"]["params"]["content"] = json!(["{CodeExec:fee@result}"]);
    assert_eq!(messages(dsl).concat(), "20");
}

#[test]
fn other_languages_are_refused_when_the_bot_is_built() {
    let mut code = fee_node();
    code["lang"] = json!("python");
    let dsl = canvas(json!({}), code);
    let err = Bot::new(compile(&dsl.to_string()).unwrap(), &registry())
        .err()
        .expect("python must be refused");
    match err {
        BuildError::Params { id, message } => {
            assert_eq!(id, "CodeExec:fee");
            assert!(message.contains("\"python\" is not supported"), "{message}");
        }
        other => panic!("unexpected {other:?}"),
    }
}

//! CodeExec: runs the canvas's script and maps the value `main` returns to
//! the node's outputs.
//!
//! Upstream's `codeExecComponent` (agent/component/universe_a_wrappers.go and
//! agent/tool/code_exec*.go, main) hands `python`/`nodejs` scripts to a
//! sandbox provider. This runtime runs `lang: "lua"` only, in-process on
//! Luau: a fresh VM per run with the string, table, math, utf8, bit32 and
//! buffer libraries, no `require`, `loadstring`, `os` or `io`, library tables
//! read-only (Luau sandbox), a memory cap and a wall-clock budget. Lua is a
//! deliberate difference from upstream; a canvas asking for any other
//! language is refused when the Bot is built.
//!
//! The script defines `main(args)`; `args` is a table of the node's
//! `arguments`, each value a reference (`cpn@param`, `sys.query`) resolved
//! against the run, else the literal, and overridden by a data-upstream
//! output of the same name, as upstream merges them. The output mapping
//! follows upstream: with an `outputs` map, its one business output receives
//! the value, checked against its declared type; without one, `result` does.
//! A failing script sets `_ERROR` (and `exit_code`, `stderr`) and the run
//! goes on, as a sandboxed script's failure does upstream.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::{Duration, Instant};

use mlua::{Lua, LuaOptions, StdLib, Table, Value as LuaValue, VmState};
use serde_json::{Map, Number, Value};

use crate::graph::Node;
use crate::run::{Component, NodeCtx};
use crate::state::{Outputs, RunState};

const MEMORY_LIMIT: usize = 32 << 20;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_TIMEOUT_SECS: u64 = 30;
const MAX_STDOUT: usize = 64 << 10;
const MAX_DEPTH: usize = 64;

/// upstream's `codeExecSystemOutputKeys`.
const SYSTEM_OUTPUTS: &[&str] = &[
    "content",
    "actual_type",
    "attachments",
    "_ERROR",
    "_ARTIFACTS",
    "_ATTACHMENT_CONTENT",
    "raw_result",
    "_created_time",
    "_elapsed_time",
];

pub struct CodeExec {
    params: Map<String, Value>,
    script: String,
    outputs: Option<Map<String, Value>>,
}

pub fn factory(node: &Node) -> Result<Box<dyn Component>, String> {
    let params = node.params.as_object().cloned().unwrap_or_default();
    // upstream's normalizeCodeExecLang: `language` first, then `lang`.
    let lang = [params.get("language"), params.get("lang")]
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(|s| s.trim().to_ascii_lowercase())
        .find(|s| !s.is_empty())
        .unwrap_or_default();
    if lang != "lua" && lang != "luau" {
        return Err(format!(
            "CodeExec lang {lang:?} is not supported: this runtime runs lua scripts in-process; \
             python and javascript need upstream's sandbox"
        ));
    }
    let script = [params.get("script"), params.get("code")]
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .find(|s| !s.trim().is_empty())
        .ok_or("CodeExec script is empty")?
        .to_string();
    let outputs = params.get("outputs").and_then(Value::as_object).cloned();
    Ok(Box::new(CodeExec {
        params,
        script,
        outputs,
    }))
}

impl Component for CodeExec {
    fn invoke(&self, ctx: &mut NodeCtx<'_>) -> Result<Outputs, String> {
        let mut args = Map::new();
        if let Some(raw) = self.params.get("arguments").and_then(Value::as_object) {
            for (k, v) in raw {
                let value = match ctx.inputs.get(k) {
                    Some(input) => input.clone(),
                    None => resolve_argument(v, ctx.state),
                };
                args.insert(k.clone(), value);
            }
        }
        let budget = match self.params.get("timeout").and_then(Value::as_f64) {
            Some(s) if s > 0.0 => Duration::from_secs_f64(s.min(MAX_TIMEOUT_SECS as f64)),
            _ => DEFAULT_TIMEOUT,
        };

        let mut out = Outputs::new();
        let raw_result = match run_lua(&self.script, &args, budget) {
            Ok(run) => {
                if !run.stdout.is_empty() {
                    out.insert("stdout".into(), Value::String(run.stdout));
                }
                // upstream's tool envelope: raw_result and content are omitted
                // when empty, actual_type is always there.
                if !run.value.is_null() {
                    out.insert("raw_result".into(), run.value.clone());
                }
                out.insert("actual_type".into(), Value::String(actual_type(&run.value)));
                let content = render_content(&run.value);
                if !content.is_empty() {
                    out.insert("content".into(), Value::String(content));
                }
                run.value
            }
            Err(failure) => {
                if !failure.stdout.is_empty() {
                    out.insert("stdout".into(), Value::String(failure.stdout));
                }
                out.insert("_ERROR".into(), Value::String(failure.message.clone()));
                out.insert("exit_code".into(), Value::from(1));
                out.insert("stderr".into(), Value::String(failure.message));
                Value::Null
            }
        };

        match &self.outputs {
            Some(outputs) => apply_business_outputs(&mut out, outputs, &raw_result),
            None => {
                let result = out
                    .get("raw_result")
                    .or_else(|| out.get("content"))
                    .cloned();
                if let Some(result) = result {
                    out.insert("result".into(), result);
                    out.entry("_ERROR").or_insert(Value::String(String::new()));
                }
            }
        }
        Ok(out)
    }
}

/// upstream's `resolveCodeExecArgumentValue`: strings that name a variable
/// resolve to it, everything else is a literal.
fn resolve_argument(v: &Value, state: &RunState) -> Value {
    match v {
        Value::Object(m) => Value::Object(
            m.iter()
                .map(|(k, v)| (k.clone(), resolve_argument(v, state)))
                .collect(),
        ),
        Value::Array(items) => {
            Value::Array(items.iter().map(|v| resolve_argument(v, state)).collect())
        }
        Value::String(s) => {
            let r = s.trim();
            match (!r.is_empty()).then(|| state.lookup(r)).flatten() {
                Some(found) => found,
                None => v.clone(),
            }
        }
        _ => v.clone(),
    }
}

/// upstream's `applyCodeExecBusinessOutputs`.
fn apply_business_outputs(out: &mut Outputs, outputs: &Map<String, Value>, raw: &Value) {
    let failed = out
        .get("_ERROR")
        .and_then(Value::as_str)
        .is_some_and(|e| !e.trim().is_empty());
    let null_business = |out: &mut Outputs| {
        for name in outputs.keys() {
            if !SYSTEM_OUTPUTS.contains(&name.as_str()) {
                out.insert(name.clone(), Value::Null);
            }
        }
    };
    if failed {
        null_business(out);
        out.entry("actual_type")
            .or_insert_with(|| Value::String(actual_type(raw)));
        out.entry("content")
            .or_insert_with(|| Value::String(render_content(raw)));
        return;
    }
    match business_output(outputs).and_then(|(name, meta)| {
        check_expected_type(&expected_type(meta), raw, "")?;
        Ok(name)
    }) {
        Ok(name) => {
            out.insert("_ERROR".into(), Value::String(String::new()));
            out.insert("actual_type".into(), Value::String(actual_type(raw)));
            out.insert("content".into(), Value::String(render_content(raw)));
            out.insert(name, raw.clone());
        }
        Err(e) => {
            null_business(out);
            out.insert("actual_type".into(), Value::String(actual_type(raw)));
            out.insert("_ERROR".into(), Value::String(e));
            out.entry("content")
                .or_insert_with(|| Value::String(render_content(raw)));
        }
    }
}

/// upstream's `selectCodeExecBusinessOutput`.
fn business_output(outputs: &Map<String, Value>) -> Result<(String, &Value), String> {
    let (name, meta) = if outputs.len() == 1 {
        outputs.iter().next().unwrap()
    } else {
        let business: Vec<_> = outputs
            .iter()
            .filter(|(k, _)| !SYSTEM_OUTPUTS.contains(&k.as_str()))
            .collect();
        if business.len() != 1 {
            return Err(format!(
                "CodeExec contract must contain exactly one business output, got {}",
                business.len()
            ));
        }
        business[0]
    };
    let name = name.trim();
    if name.is_empty() {
        return Err("CodeExec business output name must not be empty".into());
    }
    if SYSTEM_OUTPUTS.contains(&name) {
        return Err(format!(
            "CodeExec reserved output name is not allowed: {name}"
        ));
    }
    if name.contains('.') {
        return Err(format!(
            "CodeExec business output name must not contain '.': {name}"
        ));
    }
    Ok((name.to_string(), meta))
}

/// upstream reads `fmt.Sprint(meta["type"])`, so an output object without a
/// `type` asks for the type `<nil>`, which it refuses.
fn expected_type(meta: &Value) -> String {
    match meta {
        Value::Object(m) => match m.get("type") {
            None | Some(Value::Null) => "<nil>".into(),
            Some(Value::String(s)) => s.trim().to_string(),
            Some(other) => other.to_string(),
        },
        _ => String::new(),
    }
}

fn normalize_type(t: &str) -> Result<String, String> {
    let t = t.trim();
    if t.is_empty() {
        return Ok(String::new());
    }
    let canonical = match t.to_ascii_lowercase().as_str() {
        "string" => "String",
        "number" => "Number",
        "boolean" => "Boolean",
        "object" => "Object",
        "null" => "Null",
        "any" => "Any",
        low if low.starts_with("array<") && t.ends_with('>') => {
            let inner = t["Array<".len()..t.len() - 1].trim();
            if inner.is_empty() {
                return Err(format!("unsupported expected type: {t}"));
            }
            return Ok(format!("Array<{}>", normalize_type(inner)?));
        }
        _ => return Ok(t.to_string()),
    };
    Ok(canonical.to_string())
}

/// upstream's `validateCodeExecExpectedType`.
fn check_expected_type(expected: &str, value: &Value, path: &str) -> Result<(), String> {
    let t = normalize_type(expected)?;
    if t.is_empty() || t.eq_ignore_ascii_case("Any") {
        return Ok(());
    }
    let at = if path.is_empty() { "value" } else { path };
    let mismatch = || {
        format!(
            "CodeExec contract mismatch at {at}: expected type {t}, got {}",
            actual_type(value)
        )
    };
    if let Some(inner) = t.strip_prefix("Array<").and_then(|r| r.strip_suffix('>')) {
        let Value::Array(items) = value else {
            return Err(mismatch());
        };
        for (i, item) in items.iter().enumerate() {
            check_expected_type(inner, item, &format!("{path}[{i}]"))?;
        }
        return Ok(());
    }
    let ok = match t.as_str() {
        "String" => value.is_string(),
        "Number" => value.is_number(),
        "Boolean" => value.is_boolean(),
        "Object" => value.is_object(),
        "Null" => value.is_null(),
        _ => return Err(format!("unsupported expected type: {expected}")),
    };
    if ok {
        Ok(())
    } else {
        Err(mismatch())
    }
}

/// upstream's `inferCodeExecActualType`.
fn actual_type(value: &Value) -> String {
    match value {
        Value::Null => "Null".into(),
        Value::Bool(_) => "Boolean".into(),
        Value::String(_) => "String".into(),
        Value::Number(_) => "Number".into(),
        Value::Object(_) => "Object".into(),
        Value::Array(items) => {
            let Some(first) = items.first().map(actual_type) else {
                return "Array<Any>".into();
            };
            if items[1..].iter().all(|i| actual_type(i) == first) {
                format!("Array<{first}>")
            } else {
                "Array<Any>".into()
            }
        }
    }
}

/// upstream's `renderCodeExecCanonicalContent`: strings as they are, objects
/// and arrays as Go's `json.MarshalIndent(v, "", "  ")` (keys sorted, HTML
/// characters escaped), anything else as `fmt.Sprint`.
fn render_content(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Object(_) | Value::Array(_) => {
            let pretty = serde_json::to_string_pretty(&sorted(value)).unwrap_or_default();
            let mut out = String::with_capacity(pretty.len());
            for c in pretty.chars() {
                match c {
                    '<' => out.push_str("\\u003c"),
                    '>' => out.push_str("\\u003e"),
                    '&' => out.push_str("\\u0026"),
                    '\u{2028}' => out.push_str("\\u2028"),
                    '\u{2029}' => out.push_str("\\u2029"),
                    c => out.push(c),
                }
            }
            out
        }
        other => other.to_string(),
    }
}

fn sorted(value: &Value) -> Value {
    match value {
        Value::Object(m) => {
            let mut keys: Vec<_> = m.keys().collect();
            keys.sort();
            Value::Object(
                keys.into_iter()
                    .map(|k| (k.clone(), sorted(&m[k])))
                    .collect(),
            )
        }
        Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
        other => other.clone(),
    }
}

struct Run {
    value: Value,
    stdout: String,
}

struct Failure {
    message: String,
    stdout: String,
}

fn run_lua(script: &str, args: &Map<String, Value>, budget: Duration) -> Result<Run, Failure> {
    let stdout = Rc::new(RefCell::new(String::new()));
    let result = execute(script, args, budget, stdout.clone());
    let stdout = stdout.take();
    match result {
        Ok(value) => Ok(Run { value, stdout }),
        Err(message) => Err(Failure { message, stdout }),
    }
}

fn execute(
    script: &str,
    args: &Map<String, Value>,
    budget: Duration,
    stdout: Rc<RefCell<String>>,
) -> Result<Value, String> {
    let libs =
        StdLib::STRING | StdLib::TABLE | StdLib::MATH | StdLib::UTF8 | StdLib::BIT | StdLib::BUFFER;
    let lua = Lua::new_with(libs, LuaOptions::default()).map_err(lua_message)?;
    lua.set_memory_limit(MEMORY_LIMIT).map_err(lua_message)?;
    let globals = lua.globals();
    for name in ["require", "loadstring", "collectgarbage"] {
        globals.raw_set(name, LuaValue::Nil).map_err(lua_message)?;
    }
    let print = lua
        .create_function(move |_, values: mlua::MultiValue| {
            let mut buf = stdout.borrow_mut();
            let line: Vec<String> = values.iter().map(display_lua).collect();
            if buf.len() < MAX_STDOUT {
                buf.push_str(&line.join("\t"));
                buf.push('\n');
            }
            Ok(())
        })
        .map_err(lua_message)?;
    globals.raw_set("print", print).map_err(lua_message)?;
    lua.sandbox(true).map_err(lua_message)?;

    let deadline = Instant::now() + budget;
    lua.set_interrupt(move |_| {
        if Instant::now() > deadline {
            Err(mlua::Error::runtime(format!(
                "CodeExec script timed out after {:.1}s",
                budget.as_secs_f64()
            )))
        } else {
            Ok(VmState::Continue)
        }
    });

    // The script's globals land in its own environment, which falls back to
    // the (read-only) libraries.
    let env = lua.create_table().map_err(lua_message)?;
    let meta = lua.create_table().map_err(lua_message)?;
    meta.raw_set("__index", lua.globals())
        .map_err(lua_message)?;
    env.set_metatable(Some(meta)).map_err(lua_message)?;
    lua.load(script)
        .set_name("=CodeExec")
        .set_environment(env.clone())
        .exec()
        .map_err(lua_message)?;
    let main = match env.raw_get::<LuaValue>("main").map_err(lua_message)? {
        LuaValue::Function(f) => f,
        _ => return Err("CodeExec script must define a function main(args)".into()),
    };
    let arg_table = to_lua(&lua, &Value::Object(args.clone())).map_err(lua_message)?;
    let returned: LuaValue = main.call(arg_table).map_err(lua_message)?;
    from_lua(&returned, 0)
}

/// The message of a Lua error, without mlua's wrapping or traceback.
fn lua_message(e: mlua::Error) -> String {
    match e {
        mlua::Error::RuntimeError(m) => m,
        mlua::Error::SyntaxError { message, .. } => message,
        mlua::Error::MemoryError(_) => "CodeExec script exceeded its memory limit".into(),
        mlua::Error::CallbackError { cause, .. } => lua_message((*cause).clone()),
        mlua::Error::WithContext { cause, .. } => lua_message((*cause).clone()),
        other => other.to_string(),
    }
}

fn display_lua(v: &LuaValue) -> String {
    match v {
        LuaValue::Nil => "nil".into(),
        LuaValue::Boolean(b) => b.to_string(),
        LuaValue::Integer(i) => i.to_string(),
        LuaValue::Number(n) => n.to_string(),
        LuaValue::String(s) => s.to_string_lossy(),
        other => other.type_name().into(),
    }
}

fn to_lua(lua: &Lua, v: &Value) -> mlua::Result<LuaValue> {
    Ok(match v {
        Value::Null => LuaValue::Nil,
        Value::Bool(b) => LuaValue::Boolean(*b),
        Value::Number(n) => LuaValue::Number(n.as_f64().unwrap_or(0.0)),
        Value::String(s) => LuaValue::String(lua.create_string(s)?),
        Value::Array(items) => {
            let t = lua.create_table()?;
            for (i, item) in items.iter().enumerate() {
                t.raw_set(i + 1, to_lua(lua, item)?)?;
            }
            LuaValue::Table(t)
        }
        Value::Object(m) => {
            let t = lua.create_table()?;
            for (k, item) in m {
                t.raw_set(k.as_str(), to_lua(lua, item)?)?;
            }
            LuaValue::Table(t)
        }
    })
}

/// A Lua value as JSON. Integral numbers become integers, as a JSON number
/// decoded by Go prints; a table whose keys are exactly 1..n is an array,
/// any other table an object (an empty table is `{}`).
fn from_lua(v: &LuaValue, depth: usize) -> Result<Value, String> {
    if depth > MAX_DEPTH {
        return Err("CodeExec result is nested too deeply or is cyclic".into());
    }
    Ok(match v {
        LuaValue::Nil => Value::Null,
        LuaValue::Boolean(b) => Value::Bool(*b),
        LuaValue::Integer(i) => Value::from(*i),
        LuaValue::Number(n) => number(*n)?,
        LuaValue::String(s) => Value::String(s.to_string_lossy()),
        LuaValue::Table(t) => table(t, depth)?,
        other => {
            return Err(format!(
                "unsupported result type: {}. Allowed values are String, Number, Boolean, Object, Array, or Null.",
                other.type_name()
            ))
        }
    })
}

fn number(n: f64) -> Result<Value, String> {
    if !n.is_finite() {
        return Err(format!("CodeExec result holds a non-finite number ({n})"));
    }
    if n.fract() == 0.0 && n.abs() < 9_007_199_254_740_992.0 {
        return Ok(Value::from(n as i64));
    }
    Ok(Value::Number(Number::from_f64(n).expect("finite")))
}

fn table(t: &Table, depth: usize) -> Result<Value, String> {
    let mut entries = Vec::new();
    for pair in t.clone().pairs::<LuaValue, LuaValue>() {
        entries.push(pair.map_err(lua_message)?);
    }
    let n = entries.len();
    let is_array = n > 0
        && entries.iter().all(|(k, _)| {
            let i = match k {
                LuaValue::Integer(i) => *i as f64,
                LuaValue::Number(f) => *f,
                _ => return false,
            };
            i.fract() == 0.0 && i >= 1.0 && i <= n as f64
        });
    if is_array {
        let mut items = vec![Value::Null; n];
        for (k, v) in &entries {
            let i = match k {
                LuaValue::Integer(i) => *i as usize,
                LuaValue::Number(f) => *f as usize,
                _ => unreachable!(),
            };
            items[i - 1] = from_lua(v, depth + 1)?;
        }
        return Ok(Value::Array(items));
    }
    let mut keyed = Vec::with_capacity(n);
    for (k, v) in &entries {
        let key = match k {
            LuaValue::String(s) => s.to_string_lossy(),
            LuaValue::Integer(i) => i.to_string(),
            LuaValue::Number(f) => number(*f)?.to_string(),
            LuaValue::Boolean(b) => b.to_string(),
            other => {
                return Err(format!(
                    "CodeExec result has a table key of type {}",
                    other.type_name()
                ))
            }
        };
        keyed.push((key, from_lua(v, depth + 1)?));
    }
    keyed.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(Value::Object(keyed.into_iter().collect()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn run(script: &str, args: Value) -> Result<Run, Failure> {
        run_lua(
            script,
            args.as_object().unwrap(),
            Duration::from_millis(500),
        )
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

    #[test]
    fn computes_the_parking_fee() {
        for (minutes, fee) in [(140, 75), (150, 90), (29, 15), (60, 30), (0, 0)] {
            let r = run(FEE, json!({"vehicle": "car", "minutes": minutes}))
                .ok()
                .unwrap();
            assert_eq!(r.value, json!(fee), "{minutes} min");
        }
        let r = run(FEE, json!({"vehicle": "moto", "minutes": "600"}))
            .ok()
            .unwrap();
        assert_eq!(r.value, json!(20));
    }

    #[test]
    fn converts_tables_numbers_and_strings() {
        let r = run(
            r#"function main(a) return {total = 2.5, items = {"a", "b"}, ok = true, n = 3} end"#,
            json!({}),
        )
        .ok()
        .unwrap();
        assert_eq!(
            r.value,
            json!({"items": ["a", "b"], "n": 3, "ok": true, "total": 2.5})
        );
        assert_eq!(
            render_content(&r.value),
            "{\n  \"items\": [\n    \"a\",\n    \"b\"\n  ],\n  \"n\": 3,\n  \"ok\": true,\n  \"total\": 2.5\n}"
        );
        assert_eq!(actual_type(&json!(["a", "b"])), "Array<String>");
        assert_eq!(actual_type(&json!([1, "b"])), "Array<Any>");
    }

    #[test]
    fn has_no_io_os_or_require() {
        for global in [
            "io",
            "os",
            "require",
            "loadstring",
            "debug",
            "dofile",
            "collectgarbage",
        ] {
            let r = run(
                &format!("function main() return {global} == nil end"),
                json!({}),
            )
            .ok()
            .unwrap();
            assert_eq!(r.value, json!(true), "{global} is reachable");
        }
    }

    #[test]
    fn cannot_change_the_libraries() {
        let e = run("function main() string.upper = nil return 1 end", json!({}))
            .err()
            .unwrap();
        assert!(e.message.contains("readonly"), "{}", e.message);
    }

    #[test]
    fn stops_a_runaway_script() {
        let e = run("function main() while true do end end", json!({}))
            .err()
            .unwrap();
        assert!(e.message.contains("timed out"), "{}", e.message);
        let e = run(
            "function main() local t = {} for i = 1, 1e9 do t[i] = string.rep('x', 1024) end end",
            json!({}),
        )
        .err()
        .unwrap();
        assert!(
            e.message.contains("memory") || e.message.contains("timed out"),
            "{}",
            e.message
        );
    }

    #[test]
    fn reports_script_errors_and_prints() {
        let e = run("function main() print('x', 1) error('boom') end", json!({}))
            .err()
            .unwrap();
        assert!(e.message.contains("boom"), "{}", e.message);
        assert_eq!(e.stdout, "x\t1\n");
        let e = run("local x = 1", json!({})).err().unwrap();
        assert_eq!(
            e.message,
            "CodeExec script must define a function main(args)"
        );
        let e = run("function main( end", json!({})).err().unwrap();
        assert!(!e.message.is_empty());
        let e = run("function main() return main end", json!({}))
            .err()
            .unwrap();
        assert!(
            e.message.starts_with("unsupported result type: function"),
            "{}",
            e.message
        );
    }

    #[test]
    fn maps_the_business_output_like_upstream() {
        let outputs = json!({"fee": {"type": "Number", "value": ""}});
        let mut out = Outputs::new();
        apply_business_outputs(&mut out, outputs.as_object().unwrap(), &json!(100));
        assert_eq!(out["fee"], json!(100));
        assert_eq!(out["_ERROR"], json!(""));
        assert_eq!(out["content"], json!("100"));
        assert_eq!(out["actual_type"], json!("Number"));

        let mut out = Outputs::new();
        apply_business_outputs(&mut out, outputs.as_object().unwrap(), &json!("100"));
        assert_eq!(out["fee"], Value::Null);
        assert_eq!(
            out["_ERROR"],
            json!("CodeExec contract mismatch at value: expected type Number, got String")
        );

        let untyped = json!({"fee": {}});
        let mut out = Outputs::new();
        apply_business_outputs(&mut out, untyped.as_object().unwrap(), &json!(1));
        assert_eq!(out["_ERROR"], json!("unsupported expected type: <nil>"));

        let two = json!({"a": {"type": "Any"}, "b": {"type": "Any"}});
        let mut out = Outputs::new();
        apply_business_outputs(&mut out, two.as_object().unwrap(), &json!(1));
        assert_eq!(
            out["_ERROR"],
            json!("CodeExec contract must contain exactly one business output, got 2")
        );
    }
}

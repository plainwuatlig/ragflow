//! Variable references inside prompts and params.
//!
//! The grammar is upstream's: `{Cpn:id@output[.path]}`, `{sys.x[.path]}`,
//! `{env.x[.path]}`, `{item}`, `{index}`, with any number of braces on
//! either side. Two render modes exist because upstream has two: strict
//! (prompts and params) fails on a reference that resolves to nothing;
//! display (Message bodies) renders it as an empty string.
//!
//! Values render as upstream renders them: strict mode with Go's `%v`
//! (`[a b]`, `map[k:v]`, floats in Go's shortest form), display mode as
//! JSON (upstream's `ResolveTemplateForDisplay`).

use std::sync::OnceLock;

use regex::Regex;
use serde_json::Value;

use crate::state::RunState;

fn pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"\{+\s*([a-zA-Z:0-9_]+@[A-Za-z0-9_.-]+|sys\.[A-Za-z0-9_.]+|env\.[A-Za-z0-9_.]+|item|index)\s*\}+",
        )
        .expect("reference pattern compiles")
    })
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum TemplateError {
    #[error("Can't find variable: '{0}'")]
    NotFound(String),
}

/// Renders `template`, failing on the first reference that resolves to
/// nothing.
pub fn render_strict(template: &str, state: &RunState) -> Result<String, TemplateError> {
    let mut missing = None;
    let out = pattern().replace_all(template, |caps: &regex::Captures| {
        let reference = &caps[1];
        match state.lookup(reference) {
            Some(value) => go_v(&value),
            None => {
                missing.get_or_insert_with(|| reference.to_string());
                String::new()
            }
        }
    });
    match missing {
        Some(reference) => Err(TemplateError::NotFound(reference)),
        None => Ok(out.into_owned()),
    }
}

/// Renders `template`, turning references that resolve to nothing into
/// empty strings.
pub fn render_display(template: &str, state: &RunState) -> String {
    pattern()
        .replace_all(template, |caps: &regex::Captures| {
            state
                .lookup(&caps[1])
                .map(|v| display_text(&v))
                .unwrap_or_default()
        })
        .into_owned()
}

/// A display-mode value: strings as they are, anything else as Go's
/// `encoding/json` writes it with HTML escaping off (upstream's
/// `stringifyDisplayValue`): object keys sorted, numbers in Go's float
/// format, Go's string escapes.
pub fn display_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => {
            let mut out = String::new();
            go_json(&mut out, other);
            out
        }
    }
}

fn go_json(out: &mut String, value: &Value) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => match (n.as_i64(), n.as_u64()) {
            (Some(i), _) => out.push_str(&i.to_string()),
            (None, Some(u)) => out.push_str(&u.to_string()),
            _ => out.push_str(&go_json_float(n.as_f64().unwrap_or(0.0))),
        },
        Value::String(s) => go_json_string(out, s),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                go_json(out, item);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, key) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                go_json_string(out, key);
                out.push(':');
                go_json(out, &map[key]);
            }
            out.push('}');
        }
    }
}

/// encoding/json's float64: shortest digits, plain notation unless the
/// magnitude is below 1e-6 or at least 1e21, where `1e+21` / `1e-7`.
fn go_json_float(f: f64) -> String {
    let abs = f.abs();
    if abs != 0.0 && !(1e-6..1e21).contains(&abs) {
        let sci = format!("{f:e}");
        let (mantissa, exp) = sci.split_once('e').expect("{:e} has an exponent");
        let exp: i32 = exp.parse().expect("exponent is an integer");
        let sign = if exp < 0 { '-' } else { '+' };
        let digits = if exp < 0 {
            exp.abs().to_string()
        } else {
            format!("{exp:02}")
        };
        format!("{mantissa}e{sign}{digits}")
    } else {
        format!("{f}")
    }
}

fn go_json_string(out: &mut String, s: &str) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{2028}' => out.push_str("\\u2028"),
            '\u{2029}' => out.push_str("\\u2029"),
            c if (c as u32) < 0x20 => {
                let b = c as u8;
                out.push_str("\\u00");
                out.push(HEX[(b >> 4) as usize] as char);
                out.push(HEX[(b & 0xf) as usize] as char);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// The reference names a template uses, in order of appearance.
pub fn references(template: &str) -> Vec<String> {
    pattern()
        .captures_iter(template)
        .map(|caps| caps[1].to_string())
        .collect()
}

/// A whole-string reference such as `{Retrieval:a@formalized_content}`,
/// returned without braces.
pub fn sole_reference(template: &str) -> Option<String> {
    let caps = pattern().captures(template.trim())?;
    (caps.get(0)?.as_str() == template.trim()).then(|| caps[1].to_string())
}

/// Go's `fmt.Sprintf("%v", v)` for a JSON-decoded value: numbers are
/// float64 (`3`, `0.1`, `1e+06`), maps print with sorted keys.
pub fn go_v(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => "<nil>".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => go_float(n.as_f64().unwrap_or(0.0)),
        Value::Array(items) => {
            let inner: Vec<String> = items.iter().map(go_v).collect();
            format!("[{}]", inner.join(" "))
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let inner: Vec<String> = keys
                .iter()
                .map(|k| format!("{k}:{}", go_v(&map[*k])))
                .collect();
            format!("map[{}]", inner.join(" "))
        }
    }
}

/// `%v` of a float64, as Go's `strconv` 'g' with shortest precision: the
/// shortest digits, in exponent form (`1e+06`) when the decimal exponent
/// is below -4 or at least 6.
fn go_float(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 {
            "+Inf".into()
        } else {
            "-Inf".into()
        };
    }
    if f == 0.0 {
        return if f.is_sign_negative() {
            "-0".into()
        } else {
            "0".into()
        };
    }
    let sci = format!("{f:e}");
    let (mantissa, exp) = sci.split_once('e').expect("{:e} has an exponent");
    let exp: i32 = exp.parse().expect("exponent is an integer");
    if !(-4..6).contains(&exp) {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{:02}", exp.abs())
    } else {
        format!("{f}")
    }
}

pub fn to_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

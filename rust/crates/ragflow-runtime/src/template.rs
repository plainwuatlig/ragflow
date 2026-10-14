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
                .map(|v| to_text(&v))
                .unwrap_or_default()
        })
        .into_owned()
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

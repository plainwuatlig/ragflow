//! Switch: routes to the `to` targets of the first condition group that
//! matches, else to `default` (or the first `end_cpn_ids` entry).
//!
//! Semantics follow upstream's `switch.go` at v1.0.0-rc1: groups are
//! `{op, clauses[{left, op, right}]}` or the legacy
//! `{logical_operator, items[{cpn_id, operator, value}]}`; an empty group
//! never matches; string comparisons ignore case; `>`/`<`/`>=`/`<=` on a
//! non-number fail the run; `right` is never template-resolved.

use serde_json::{json, Map, Value};

use crate::graph::Node;
use crate::run::{Component, NodeCtx};
use crate::state::{Outputs, RunState};
use crate::template::{references, render_strict};

pub struct Switch {
    params: Map<String, Value>,
}

pub fn factory(node: &Node) -> Result<Box<dyn Component>, String> {
    Ok(Box::new(Switch {
        params: node.params.as_object().cloned().unwrap_or_default(),
    }))
}

impl Component for Switch {
    fn invoke(&self, ctx: &mut NodeCtx<'_>) -> Result<Outputs, String> {
        let mut merged = self.params.clone();
        for (k, v) in ctx.inputs {
            merged.insert(k.clone(), v.clone());
        }
        let default = merged
            .get("default")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .or_else(|| first_string(merged.get("end_cpn_ids")))
            .unwrap_or_default();

        if let Some(Value::Array(groups)) = merged.get("conditions") {
            for (i, group) in groups.iter().enumerate() {
                let Some(group) = group.as_object() else {
                    continue;
                };
                let matched = evaluate_group(group, ctx.state)
                    .map_err(|e| format!("Switch: condition[{i}]: {e}"))?;
                if matched {
                    let mut targets = group_targets(group);
                    if targets.is_empty() {
                        targets.push(format!("matched_{i}"));
                    }
                    return Ok(next(targets));
                }
            }
        }
        Ok(next(vec![default]))
    }
}

fn next(targets: Vec<String>) -> Outputs {
    let mut out = Outputs::new();
    out.insert("_next".into(), json!(targets));
    out
}

fn first_string(value: Option<&Value>) -> Option<String> {
    value?
        .as_array()?
        .iter()
        .filter_map(Value::as_str)
        .find(|s| !s.is_empty())
        .map(str::to_string)
}

fn group_targets(group: &Map<String, Value>) -> Vec<String> {
    match group.get("to") {
        Some(Value::String(s)) if !s.is_empty() => vec![s.clone()],
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect(),
        _ => Vec::new(),
    }
}

fn evaluate_group(group: &Map<String, Value>, state: &RunState) -> Result<bool, String> {
    let (op, clauses) = normalize_group(group);
    let op = if op.is_empty() { "and".to_string() } else { op };
    if clauses.is_empty() {
        return Ok(false);
    }
    for (i, clause) in clauses.iter().enumerate() {
        let matched = evaluate_clause(clause, state).map_err(|e| format!("clause[{i}]: {e}"))?;
        if op == "or" && matched {
            return Ok(true);
        }
        if op == "and" && !matched {
            return Ok(false);
        }
    }
    Ok(op == "and")
}

struct Clause {
    left: String,
    op: String,
    right: Value,
}

fn normalize_group(group: &Map<String, Value>) -> (String, Vec<Clause>) {
    let text = |m: &Map<String, Value>, k: &str| {
        m.get(k)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    };
    if let Some(Value::Array(clauses)) = group.get("clauses") {
        let clauses = clauses
            .iter()
            .filter_map(Value::as_object)
            .map(|c| Clause {
                left: text(c, "left"),
                op: text(c, "op"),
                right: c.get("right").cloned().unwrap_or(Value::Null),
            })
            .collect();
        return (text(group, "op"), clauses);
    }
    let items = group
        .get("items")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let clauses = items
        .iter()
        .filter_map(Value::as_object)
        .map(|item| Clause {
            left: legacy_left(item),
            op: text(item, "operator"),
            right: item.get("value").cloned().unwrap_or(Value::Null),
        })
        .collect();
    (text(group, "logical_operator"), clauses)
}

fn legacy_left(item: &Map<String, Value>) -> String {
    if let Some(left) = item
        .get("left")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        return left.to_string();
    }
    let reference = item
        .get("cpn_id")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if reference.is_empty() {
        return String::new();
    }
    let looks_like_ref = !references(reference).is_empty()
        || reference.contains('@')
        || reference.starts_with("sys.")
        || reference.starts_with("env.");
    if looks_like_ref {
        format!("{{{{{reference}}}}}")
    } else {
        reference.to_string()
    }
}

fn normalize_op(op: &str) -> &str {
    match op {
        "" | "=" | "==" => "==",
        "<>" | "≠" | "!=" => "!=",
        "≥" => ">=",
        "≤" => "<=",
        other => other,
    }
}

fn evaluate_clause(clause: &Clause, state: &RunState) -> Result<bool, String> {
    let op = normalize_op(&clause.op);
    match op {
        "empty" => return Ok(is_empty(&left_for_empty(&clause.left, state))),
        "not empty" => return Ok(!is_empty(&left_for_empty(&clause.left, state))),
        _ => {}
    }
    let left = left_value(&clause.left, state);
    let right = &clause.right;
    let lower = |v: &Value| go_text(v).to_lowercase();
    Ok(match op {
        "==" => equal_fold(&left, right),
        "!=" => !equal_fold(&left, right),
        "contains" => lower(&left).contains(&lower(&null_to_empty(right))),
        "not contains" => !lower(&left).contains(&lower(&null_to_empty(right))),
        "start with" => lower(&left).starts_with(&lower(&null_to_empty(right))),
        "end with" => lower(&left).ends_with(&lower(&null_to_empty(right))),
        ">" | "<" | ">=" | "<=" => {
            let (Some(l), Some(r)) = (numeric(&left), numeric(right)) else {
                return Err(format!("operator {op:?} requires numeric operands"));
            };
            match op {
                ">" => l > r,
                "<" => l < r,
                ">=" => l >= r,
                _ => l <= r,
            }
        }
        other => return Err(format!("unknown operator {other:?}")),
    })
}

/// A template left side renders to a string; a literal stays as written; a
/// template that fails to resolve falls back to its own text.
fn left_value(left: &str, state: &RunState) -> Value {
    if left.is_empty() || references(left).is_empty() {
        return Value::String(left.to_string());
    }
    Value::String(render_strict(left, state).unwrap_or_else(|_| left.to_string()))
}

/// `empty` looks at the raw value of the first reference.
fn left_for_empty(left: &str, state: &RunState) -> Value {
    match references(left).first() {
        None => Value::String(left.to_string()),
        Some(reference) => {
            let known = reference == "item"
                || reference == "index"
                || reference.starts_with("sys.")
                || reference.starts_with("env.")
                || reference.contains('@');
            if known {
                state.lookup(reference).unwrap_or(Value::Null)
            } else {
                Value::String(left.to_string())
            }
        }
    }
}

fn is_empty(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(m) => m.is_empty(),
        _ => false,
    }
}

fn null_to_empty(value: &Value) -> Value {
    if value.is_null() {
        Value::String(String::new())
    } else {
        value.clone()
    }
}

fn equal_fold(a: &Value, b: &Value) -> bool {
    if a.is_null() || b.is_null() {
        return a.is_null() && b.is_null();
    }
    let (sa, sb) = (go_text(a), go_text(b));
    if let (Ok(na), Ok(nb)) = (sa.parse::<f64>(), sb.parse::<f64>()) {
        if na.is_finite() && nb.is_finite() {
            return na == nb;
        }
    }
    sa.to_lowercase() == sb.to_lowercase()
}

fn numeric(value: &Value) -> Option<f64> {
    let n = match value {
        Value::Number(n) => n.as_f64()?,
        Value::String(s) => s.parse::<f64>().ok()?,
        _ => return None,
    };
    n.is_finite().then_some(n)
}

/// Go's `%v` for the scalar values a condition compares.
fn go_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => "<nil>".into(),
        Value::Number(n) => match n.as_f64() {
            Some(f) if f.fract() == 0.0 && f.abs() < 1e15 => format!("{}", f as i64),
            _ => n.to_string(),
        },
        other => other.to_string(),
    }
}

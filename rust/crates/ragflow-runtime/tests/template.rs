use std::collections::BTreeMap;

use chrono::{FixedOffset, TimeZone};
use ragflow_runtime::template::{go_v, references, render_display, render_strict, sole_reference};
use ragflow_runtime::RunState;
use serde_json::json;

fn state() -> RunState {
    let now = FixedOffset::east_opt(8 * 3600)
        .unwrap()
        .with_ymd_and_hms(2026, 10, 13, 9, 0, 0)
        .unwrap();
    let mut s = RunState::seed(
        &BTreeMap::from([("env.lot".to_string(), json!("北區"))]),
        "停車費",
        now,
    );
    s.outputs.insert(
        "Retrieval:a".into(),
        json!({ "chunks": [{ "id": "c1" }], "meta": "{\"k\": {\"v\": 7}}", "n": 3 })
            .as_object()
            .unwrap()
            .clone(),
    );
    s
}

#[test]
fn resolves_sys_env_outputs_and_paths() {
    let s = state();
    assert_eq!(
        render_strict("{sys.query}@{env.lot}", &s).unwrap(),
        "停車費@北區"
    );
    assert_eq!(
        render_strict("{{Retrieval:a@chunks.0.id}}", &s).unwrap(),
        "c1"
    );
    assert_eq!(render_strict("{Retrieval:a@meta.k.v}", &s).unwrap(), "7");
    assert_eq!(render_strict("{Retrieval:a@n}", &s).unwrap(), "3");
}

#[test]
fn strict_renders_go_syntax_and_display_renders_json() {
    let s = state();
    assert_eq!(
        render_strict("{Retrieval:a@chunks}", &s).unwrap(),
        "[map[id:c1]]"
    );
    assert_eq!(
        render_display("{Retrieval:a@chunks}", &s),
        r#"[{"id":"c1"}]"#
    );
}

#[test]
fn numbers_render_as_go_float64() {
    for (v, want) in [
        (json!(3), "3"),
        (json!(100), "100"),
        (json!(0.1), "0.1"),
        (json!(1234567), "1.234567e+06"),
        (json!(0.00001), "1e-05"),
        (
            json!({ "b": 1, "a": [true, null] }),
            "map[a:[true <nil>] b:1]",
        ),
    ] {
        assert_eq!(go_v(&v), want);
    }
}

#[test]
fn sys_date_is_kept_when_the_globals_carry_one() {
    let now = FixedOffset::east_opt(0)
        .unwrap()
        .with_ymd_and_hms(2026, 1, 1, 0, 0, 0)
        .unwrap();
    let kept = RunState::seed(
        &BTreeMap::from([("sys.date".to_string(), json!("2025-12-31 23:59:59"))]),
        "q",
        now,
    );
    assert_eq!(kept.sys["date"], "2025-12-31 23:59:59");
    let blank = RunState::seed(
        &BTreeMap::from([("sys.date".to_string(), json!(" "))]),
        "q",
        now,
    );
    assert_eq!(blank.sys["date"], "2026-01-01 00:00:00");
}

#[test]
fn strict_fails_and_display_blanks_a_missing_reference() {
    let s = state();
    assert_eq!(
        render_strict("x{Agent:b@content}y", &s)
            .unwrap_err()
            .to_string(),
        "Can't find variable: 'Agent:b@content'"
    );
    assert_eq!(render_display("x{Agent:b@content}y", &s), "xy");
}

#[test]
fn lists_references_and_spots_a_sole_reference() {
    assert_eq!(
        references("{sys.query} and {Agent:a@content}"),
        ["sys.query", "Agent:a@content"]
    );
    assert_eq!(
        sole_reference(" {Retrieval:a@formalized_content} "),
        Some("Retrieval:a@formalized_content".into())
    );
    assert_eq!(sole_reference("see {sys.query}"), None);
}

#[test]
fn counts_turns_from_the_canvas_globals() {
    let now = FixedOffset::east_opt(0)
        .unwrap()
        .with_ymd_and_hms(2026, 1, 1, 0, 0, 0)
        .unwrap();
    let s = RunState::seed(
        &BTreeMap::from([("sys.conversation_turns".to_string(), json!(2))]),
        "q",
        now,
    );
    assert_eq!(s.sys["conversation_turns"], 3);
}

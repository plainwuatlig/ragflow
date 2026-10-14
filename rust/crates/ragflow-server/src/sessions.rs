//! Agent sessions: the storage trait the run path writes through, and the
//! list, get, create and delete endpoints as upstream's
//! `internal/service/agent_sessions.go` serves them.

use std::collections::BTreeMap;

use chrono::{Local, LocalResult, NaiveDateTime, TimeZone};
use serde_json::{json, Map, Value};

use crate::facts::{Facts, SessionRow, StoreError};
use crate::models::{Access, ModelError, Resolver, CHAT};
use crate::perm::{self, Op, PermError};
use crate::store::{Children, HistoryUpdate, ListParams, SessionUpdate, Store};

pub type Result<T> = std::result::Result<T, StoreError>;

/// Writes (and the paged reads) of `api_4_conversation*`.
pub trait SessionStore: Send + Sync {
    #[allow(clippy::too_many_arguments)]
    fn create_session(
        &self,
        id: &str,
        name: &str,
        canvas_id: &str,
        user_id: &str,
        source: Option<&str>,
        dsl: &Value,
        version_title: Option<&str>,
    ) -> Result<()>;
    fn update_history(
        &self,
        session_id: &str,
        canvas_id: &str,
        user_id: &str,
        update: &SessionUpdate,
        history: &HistoryUpdate,
    ) -> Result<()>;
    fn delete_session(&self, session_id: &str, canvas_id: &str) -> Result<u64>;
    fn session_ids_of_canvas(&self, canvas_id: &str) -> Result<Vec<String>>;
    fn messages_and_references(&self, ids: &[String]) -> Result<Children>;
    fn list_sessions(&self, p: &ListParams) -> Result<(i64, Vec<SessionRow>)>;
    fn list_session_names(
        &self,
        canvas_id: &str,
        exp_user_id: &str,
    ) -> Result<Vec<(String, Option<String>)>>;
}

impl SessionStore for Store {
    fn create_session(
        &self,
        id: &str,
        name: &str,
        canvas_id: &str,
        user_id: &str,
        source: Option<&str>,
        dsl: &Value,
        version_title: Option<&str>,
    ) -> Result<()> {
        Store::create_session(
            self,
            id,
            name,
            canvas_id,
            user_id,
            source,
            dsl,
            version_title,
        )
    }
    fn update_history(
        &self,
        session_id: &str,
        canvas_id: &str,
        user_id: &str,
        update: &SessionUpdate,
        history: &HistoryUpdate,
    ) -> Result<()> {
        Store::update_history(self, session_id, canvas_id, user_id, update, history)
    }
    fn delete_session(&self, session_id: &str, canvas_id: &str) -> Result<u64> {
        Store::delete_session(self, session_id, canvas_id)
    }
    fn session_ids_of_canvas(&self, canvas_id: &str) -> Result<Vec<String>> {
        Store::session_ids_of_canvas(self, canvas_id)
    }
    fn messages_and_references(&self, ids: &[String]) -> Result<Children> {
        Store::messages_and_references(self, ids)
    }
    fn list_sessions(&self, p: &ListParams) -> Result<(i64, Vec<SessionRow>)> {
        Store::list_sessions(self, p)
    }
    fn list_session_names(
        &self,
        canvas_id: &str,
        exp_user_id: &str,
    ) -> Result<Vec<(String, Option<String>)>> {
        Store::list_session_names(self, canvas_id, exp_user_id)
    }
}

/// A failed session call, as the envelope reports it.
#[derive(Debug, Clone, PartialEq)]
pub struct ApiError {
    pub code: i64,
    pub message: String,
}

fn api(code: i64, message: impl Into<String>) -> ApiError {
    ApiError {
        code,
        message: message.into(),
    }
}

fn server_error(e: impl std::fmt::Display) -> ApiError {
    api(500, e.to_string())
}

const NO_ACCESS: &str = "agent not found or no permission";

/// `CheckCanvasAccess` for a list or get: `103` unless the user may read
/// the canvas.
fn check_read(f: &dyn Facts, user_id: &str, canvas_id: &str) -> std::result::Result<(), ApiError> {
    match perm::canvas_for(f, user_id, canvas_id, Op::Read) {
        Ok(_) => Ok(()),
        Err(PermError::NotFound | PermError::Denied) => Err(api(103, NO_ACCESS)),
        Err(e) => Err(server_error(format!(
            "failed to check agent permission: {e}"
        ))),
    }
}

/// Go's `time.Time` JSON for a stored local `datetime`.
fn go_time(stored: &str) -> Value {
    let naive = NaiveDateTime::parse_from_str(stored, "%Y-%m-%d %H:%M:%S%.f")
        .or_else(|_| NaiveDateTime::parse_from_str(stored, "%Y-%m-%d %H:%M:%S"));
    let Ok(naive) = naive else {
        return Value::Null;
    };
    let local = match Local.from_local_datetime(&naive) {
        LocalResult::Single(t) | LocalResult::Ambiguous(t, _) => t,
        LocalResult::None => return Value::Null,
    };
    let mut fraction = format!("{:09}", local.timestamp_subsec_nanos());
    while fraction.ends_with('0') {
        fraction.pop();
    }
    let seconds = local.format("%Y-%m-%dT%H:%M:%S").to_string();
    let offset_secs = chrono::Offset::fix(local.offset()).local_minus_utc();
    let zone = if offset_secs == 0 {
        "Z".to_string()
    } else {
        let sign = if offset_secs < 0 { '-' } else { '+' };
        let a = offset_secs.abs();
        format!("{sign}{:02}:{:02}", a / 3600, (a % 3600) / 60)
    };
    let fraction = if fraction.is_empty() {
        String::new()
    } else {
        format!(".{fraction}")
    };
    Value::String(format!("{seconds}{fraction}{zone}"))
}

fn opt_string(v: &Option<String>) -> Value {
    v.as_ref().map_or(Value::Null, |s| Value::String(s.clone()))
}

fn opt_i64(v: Option<i64>) -> Value {
    v.map_or(Value::Null, Value::from)
}

fn duration_json(d: f64) -> Value {
    serde_json::Number::from_f64(d).map_or(Value::Null, Value::Number)
}

/// `normalizeAgentReferenceEntry`.
fn normalize_entry(r: &Value) -> Value {
    let list = |v: Option<&Value>| match v {
        None | Some(Value::Null) => json!([]),
        Some(o) => o.clone(),
    };
    let Some(map) = r.as_object() else {
        return json!({"chunks": [], "doc_aggs": []});
    };
    if map.contains_key("chunks") || map.contains_key("doc_aggs") {
        return json!({"chunks": list(map.get("chunks")), "doc_aggs": list(map.get("doc_aggs"))});
    }
    json!({"chunks": list(map.get("reference")), "doc_aggs": list(map.get("doc_aggs"))})
}

/// `normalizeAgentReferenceChunks`.
fn normalize_chunks(chunks: &Value) -> Value {
    let first = |m: &Map<String, Value>, a: &str, b: &str| {
        m.get(a)
            .filter(|v| !v.is_null())
            .or_else(|| m.get(b))
            .cloned()
            .unwrap_or(Value::Null)
    };
    Value::Array(
        chunks
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_object)
            .map(|c| {
                json!({
                    "id": first(c, "chunk_id", "id"),
                    "content": first(c, "content_with_weight", "content"),
                    "document_id": first(c, "doc_id", "document_id"),
                    "document_name": first(c, "docnm_kwd", "document_name"),
                    "dataset_id": first(c, "kb_id", "dataset_id"),
                    "image_id": first(c, "image_id", "img_id"),
                    "positions": first(c, "positions", "position_int"),
                })
            })
            .collect(),
    )
}

/// `normalizeAgentSession`: one entry of the list response.
fn normalize_session(
    session: &SessionRow,
    messages: Vec<Value>,
    references: &[Value],
    include_dsl: bool,
    with_messages: bool,
) -> Value {
    let mut messages: Vec<Value> = messages
        .into_iter()
        .map(|mut m| {
            if let Some(o) = m.as_object_mut() {
                o.remove("prompt");
            }
            m
        })
        .collect();
    let refs: Vec<Value> = references.iter().map(normalize_entry).collect();
    if !refs.is_empty() {
        let mut slot = 0;
        for (i, m) in messages.iter_mut().enumerate() {
            let role = m.get("role").and_then(Value::as_str).unwrap_or_default();
            if i != 0 && role != "user" {
                if slot < refs.len() {
                    m["reference"] = normalize_chunks(&refs[slot]["chunks"]);
                }
                slot += 1;
            }
        }
    }
    let mut out = Map::new();
    out.insert("id".into(), json!(session.id));
    out.insert("name".into(), opt_string(&session.name));
    out.insert("agent_id".into(), json!(session.dialog_id));
    out.insert("user_id".into(), json!(session.user_id));
    out.insert("exp_user_id".into(), opt_string(&session.exp_user_id));
    if with_messages {
        out.insert("message".into(), Value::Array(messages));
    }
    out.insert("tokens".into(), json!(session.tokens));
    out.insert("source".into(), opt_string(&session.source));
    out.insert("duration".into(), duration_json(session.duration));
    out.insert("round".into(), json!(session.round));
    out.insert("thumb_up".into(), json!(session.thumb_up));
    out.insert("errors".into(), opt_string(&session.errors));
    out.insert("version_title".into(), opt_string(&session.version_title));
    out.insert("create_time".into(), opt_i64(session.create_time));
    out.insert(
        "create_date".into(),
        session.create_date.as_deref().map_or(Value::Null, go_time),
    );
    out.insert("update_time".into(), opt_i64(session.update_time));
    out.insert(
        "update_date".into(),
        session.update_date.as_deref().map_or(Value::Null, go_time),
    );
    if include_dsl {
        out.insert(
            "dsl".into(),
            if session.dsl.is_null() {
                Value::Null
            } else {
                session.dsl.clone()
            },
        );
    }
    Value::Object(out)
}

/// The entity JSON of one session (`API4Conversation` with `omitempty`).
fn session_entity(session: &SessionRow, message: Vec<Value>, reference: Vec<Value>) -> Value {
    let mut out = Map::new();
    out.insert("id".into(), json!(session.id));
    if let Some(n) = &session.name {
        out.insert("name".into(), json!(n));
    }
    out.insert("dialog_id".into(), json!(session.dialog_id));
    out.insert("user_id".into(), json!(session.user_id));
    if let Some(e) = &session.exp_user_id {
        out.insert("exp_user_id".into(), json!(e));
    }
    out.insert("message".into(), Value::Array(message));
    out.insert("reference".into(), Value::Array(reference));
    out.insert("tokens".into(), json!(session.tokens));
    if let Some(s) = &session.source {
        out.insert("source".into(), json!(s));
    }
    if matches!(&session.dsl, Value::Object(m) if !m.is_empty()) {
        out.insert("dsl".into(), session.dsl.clone());
    }
    out.insert("duration".into(), duration_json(session.duration));
    out.insert("round".into(), json!(session.round));
    out.insert("thumb_up".into(), json!(session.thumb_up));
    if let Some(e) = &session.errors {
        out.insert("errors".into(), json!(e));
    }
    if let Some(v) = &session.version_title {
        out.insert("version_title".into(), json!(v));
    }
    if let Some(t) = session.create_time {
        out.insert("create_time".into(), json!(t));
    }
    if let Some(d) = &session.create_date {
        out.insert("create_date".into(), go_time(d));
    }
    if let Some(t) = session.update_time {
        out.insert("update_time".into(), json!(t));
    }
    if let Some(d) = &session.update_date {
        out.insert("update_date".into(), go_time(d));
    }
    Value::Object(out)
}

/// What `GET /agents/:id/sessions` accepts.
#[derive(Debug, Clone, Default)]
pub struct ListQuery {
    pub page: i64,
    pub page_size: i64,
    pub keywords: String,
    pub from_date: String,
    pub to_date: String,
    pub order_by: String,
    pub desc: bool,
    pub session_id: String,
    pub user_id: String,
    pub exp_user_id: String,
    pub include_dsl: bool,
    pub no_history: bool,
}

/// `parseAgentSessionDate`.
fn parse_date(value: &str, is_end: bool) -> std::result::Result<Option<String>, ApiError> {
    if value.is_empty() {
        return Ok(None);
    }
    let local_text = |t: chrono::DateTime<Local>| t.format("%Y-%m-%d %H:%M:%S").to_string();
    if value.contains('T') {
        let normalized = value
            .strip_suffix('Z')
            .map_or_else(|| value.to_string(), |v| format!("{v}+00:00"));
        if let Ok(t) = chrono::DateTime::parse_from_rfc3339(&normalized) {
            return Ok(Some(local_text(t.with_timezone(&Local))));
        }
        if let Ok(n) = NaiveDateTime::parse_from_str(&normalized, "%Y-%m-%dT%H:%M:%S") {
            if let LocalResult::Single(t) | LocalResult::Ambiguous(t, _) =
                Local.from_local_datetime(&n)
            {
                return Ok(Some(local_text(t)));
            }
        }
        return Err(api(
            101,
            format!("parsing time {value:?} as \"ISO 8601\": failed to parse as any supported ISO 8601 variant"),
        ));
    }
    let mut text = value.to_string();
    if text.len() == 10 {
        text.push_str(if is_end { " 23:59:59" } else { " 00:00:00" });
    }
    match NaiveDateTime::parse_from_str(&text, "%Y-%m-%d %H:%M:%S") {
        Ok(n) => Ok(Some(n.format("%Y-%m-%d %H:%M:%S").to_string())),
        Err(e) => Err(api(
            101,
            format!("parsing time {text:?} as \"2006-01-02 15:04:05\": {e}"),
        )),
    }
}

/// `ListAgentSessions`: the entries and the total.
pub fn list(
    f: &dyn Facts,
    s: &dyn SessionStore,
    user_id: &str,
    canvas_id: &str,
    q: &ListQuery,
) -> std::result::Result<(Vec<Value>, i64), ApiError> {
    if canvas_id.is_empty() {
        return Err(api(101, "agent_id is required"));
    }
    check_read(f, user_id, canvas_id)?;
    if !q.exp_user_id.is_empty() {
        let rows = s
            .list_session_names(canvas_id, &q.exp_user_id)
            .map_err(server_error)?;
        let total = rows.len() as i64;
        let data = rows
            .into_iter()
            .map(|(id, name)| json!({ "id": id, "name": name }))
            .collect();
        return Ok((data, total));
    }
    let from_date = parse_date(&q.from_date, false)?;
    let to_date = parse_date(&q.to_date, true)?;
    let (total, sessions) = s
        .list_sessions(&ListParams {
            canvas_id: canvas_id.to_string(),
            page: q.page,
            page_size: q.page_size,
            order_by: q.order_by.clone(),
            desc: q.desc,
            session_id: q.session_id.clone(),
            user_id: q.user_id.clone(),
            include_dsl: q.include_dsl,
            keywords: q.keywords.clone(),
            from_date,
            to_date,
            exp_user_id: String::new(),
        })
        .map_err(server_error)?;
    let ids: Vec<String> = sessions.iter().map(|x| x.id.clone()).collect();
    let mut children = if q.no_history {
        Children::new()
    } else {
        s.messages_and_references(&ids).map_err(server_error)?
    };
    let data = sessions
        .iter()
        .map(|session| {
            let (messages, references) = children.remove(&session.id).unwrap_or_default();
            normalize_session(session, messages, &references, q.include_dsl, !q.no_history)
        })
        .collect();
    Ok((data, total))
}

/// `GetAgentSession`.
pub fn get(
    f: &dyn Facts,
    s: &dyn SessionStore,
    user_id: &str,
    canvas_id: &str,
    session_id: &str,
) -> std::result::Result<Value, ApiError> {
    if session_id.is_empty() {
        return Err(api(101, "session_id is required"));
    }
    check_read(f, user_id, canvas_id)?;
    let Some(session) = f
        .session_in_canvas(session_id, canvas_id)
        .map_err(|e| server_error(format!("failed to fetch session: {e}")))?
    else {
        return Err(api(404, "agent session not found"));
    };
    let mut children = s
        .messages_and_references(std::slice::from_ref(&session.id))
        .map_err(|e| server_error(format!("failed to fetch session: {e}")))?;
    let (messages, references) = children.remove(&session.id).unwrap_or_default();
    Ok(session_entity(&session, messages, references))
}

/// `CreateAgentSession`.
#[allow(clippy::too_many_arguments)]
pub fn create(
    f: &dyn Facts,
    s: &dyn SessionStore,
    catalog: &crate::models::Catalog,
    user_id: &str,
    canvas_id: &str,
    name: &str,
    source: &str,
    request_dsl: Option<Value>,
) -> std::result::Result<Value, ApiError> {
    if canvas_id.is_empty() {
        return Err(api(101, "create agent session: agent_id is required"));
    }
    let canvas = match perm::canvas_for(f, user_id, canvas_id, Op::Run) {
        Ok(c) => c,
        Err(PermError::NotFound | PermError::Denied) => return Err(api(103, NO_ACCESS)),
        Err(e) => return Err(server_error(format!("check canvas access: {e}"))),
    };
    let dsl = match request_dsl {
        Some(Value::Object(m)) if !m.is_empty() => Value::Object(m),
        _ => canvas.dsl.clone(),
    };
    // validateAgentChatModels
    if let Some(components) = dsl.get("components").and_then(Value::as_object) {
        let resolver = Resolver { facts: f, catalog };
        let access = Access {
            user_id: user_id.to_string(),
            tenant_id: canvas.user_id.clone(),
        };
        for comp in components.values() {
            let (cname, params) = crate::service::component_name_and_params(comp);
            if !cname.eq_ignore_ascii_case("Agent") {
                continue;
            }
            let model_ref = match params.get("model_id") {
                Some(Value::String(s)) => s.as_str(),
                _ => params.get("llm_id").and_then(Value::as_str).unwrap_or(""),
            };
            match resolver.resolve(&access, CHAT, model_ref) {
                Ok(_) => {}
                Err(ModelError::Unavailable(_)) => {
                    return Err(api(
                        102,
                        "The configured chat model is missing or unavailable. Please select a valid model.",
                    ))
                }
                Err(ModelError::Store(_)) => {
                    return Err(api(500, crate::service::INTERNAL_ERROR));
                }
            }
        }
    }
    let version_title = f
        .latest_version(canvas_id)
        .map_err(|e| server_error(format!("load latest version title: {e}")))?
        .and_then(|v| v.title);
    let name = match name.trim() {
        "" => "session",
        n => n,
    };
    let id = uuid::Uuid::new_v4().simple().to_string();
    let source = (!source.is_empty()).then_some(source);
    let stored_dsl = if dsl.is_null() {
        json!({})
    } else {
        dsl.clone()
    };
    s.create_session(
        &id,
        name,
        canvas_id,
        user_id,
        source,
        &stored_dsl,
        version_title.as_deref(),
    )
    .map_err(|e| server_error(format!("create agent session: {e}")))?;
    let row = f
        .session_in_canvas(&id, canvas_id)
        .map_err(server_error)?
        .ok_or_else(|| server_error("create agent session: row not found after insert"))?;
    Ok(session_entity(&row, Vec::new(), Vec::new()))
}

/// `DeleteAgentSessionItem`: `Ok(true)` when a session was deleted.
pub fn delete_one(
    f: &dyn Facts,
    s: &dyn SessionStore,
    user_id: &str,
    canvas_id: &str,
    session_id: &str,
) -> std::result::Result<bool, ApiError> {
    if session_id.is_empty() {
        return Err(api(101, "session_id is required"));
    }
    match perm::canvas_for(f, user_id, canvas_id, Op::Use) {
        Ok(_) => {}
        Err(PermError::NotFound | PermError::Denied) => return Err(api(103, NO_ACCESS)),
        Err(e) => return Err(server_error(format!("failed to check agent access: {e}"))),
    }
    if f.session_in_canvas(session_id, canvas_id)
        .map_err(server_error)?
        .is_none()
    {
        return Ok(false);
    }
    let rows = s
        .delete_session(session_id, canvas_id)
        .map_err(server_error)?;
    Ok(rows > 0)
}

/// The data and message of `DeleteAgentSessions`.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchDelete {
    pub data: Value,
    pub message: String,
}

/// `DeleteAgentSessions`: `ids`, or every session when `delete_all`.
pub fn delete_many(
    f: &dyn Facts,
    s: &dyn SessionStore,
    user_id: &str,
    canvas_id: &str,
    ids: Vec<String>,
    delete_all: bool,
) -> std::result::Result<BatchDelete, ApiError> {
    if canvas_id.is_empty() {
        return Err(api(101, "agent_id is required"));
    }
    match perm::canvas_for(f, user_id, canvas_id, Op::Use) {
        Ok(_) => {}
        Err(PermError::NotFound | PermError::Denied) => {
            return Err(api(
                102,
                format!("you don't have permission to delete sessions for agent {canvas_id}"),
            ))
        }
        Err(e) => {
            return Err(server_error(format!(
                "check agent deletion permission: {e}"
            )))
        }
    }
    let mut ids = ids;
    let nothing = BatchDelete {
        data: Value::Null,
        message: "success".into(),
    };
    if ids.is_empty() {
        if !delete_all {
            return Ok(nothing);
        }
        ids = s.session_ids_of_canvas(canvas_id).map_err(server_error)?;
        if ids.is_empty() {
            return Ok(nothing);
        }
    }
    // checkDuplicateSessionIDs: de-duplicated, with a message per repeat.
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut unique = Vec::new();
    for id in &ids {
        let id = id.trim().to_string();
        let n = counts.entry(id.clone()).or_default();
        *n += 1;
        if *n == 1 {
            unique.push(id);
        }
    }
    let duplicates: Vec<String> = unique
        .iter()
        .filter(|id| counts[*id] > 1)
        .map(|id| format!("Duplicate session ids: {id}"))
        .collect();
    let mut errors = Vec::new();
    let mut success = 0usize;
    for id in unique {
        let id = id.trim().to_string();
        if id.is_empty() {
            errors.push("Session ID is empty".to_string());
            continue;
        }
        if f.session_in_canvas(&id, canvas_id)
            .map_err(server_error)?
            .is_none()
        {
            errors.push(format!("The agent doesn't own the session {id}"));
            continue;
        }
        s.delete_session(&id, canvas_id).map_err(server_error)?;
        success += 1;
    }
    if !errors.is_empty() {
        if success > 0 {
            return Ok(BatchDelete {
                message: format!(
                    "Partially deleted {success} sessions with {} errors",
                    errors.len()
                ),
                data: json!({ "success_count": success, "errors": errors }),
            });
        }
        return Err(api(102, errors.join("; ")));
    }
    if !duplicates.is_empty() {
        return Ok(BatchDelete {
            message: format!(
                "Partially deleted {success} sessions with {} errors",
                duplicates.len()
            ),
            data: json!({ "success_count": success, "errors": duplicates }),
        });
    }
    Ok(nothing)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn references_pair_with_the_assistant_messages_after_the_first() {
        let session = SessionRow {
            id: "s".into(),
            name: Some("n".into()),
            dialog_id: "c".into(),
            user_id: "u".into(),
            exp_user_id: Some("u".into()),
            tokens: 0,
            source: Some("agent".into()),
            dsl: Value::Null,
            duration: 0.0,
            round: 2,
            thumb_up: 0,
            errors: None,
            version_title: None,
            create_time: Some(1),
            create_date: Some("2026-10-13 16:21:25.000".into()),
            update_time: Some(2),
            update_date: None,
        };
        let messages = vec![
            json!({"role": "user", "content": "q1", "id": "m1"}),
            json!({"role": "assistant", "content": "a1", "id": "m1", "prompt": "x"}),
            json!({"role": "user", "content": "q2", "id": "m2"}),
            json!({"role": "assistant", "content": "a2", "id": "m2"}),
        ];
        let refs = vec![
            json!({"chunks": [{"chunk_id": "c1", "content_with_weight": "text", "doc_id": "d", "docnm_kwd": "doc", "kb_id": "k"}], "doc_aggs": []}),
            json!({"chunks": [], "doc_aggs": []}),
        ];
        let out = normalize_session(&session, messages, &refs, false, true);
        let m = out["message"].as_array().unwrap();
        assert!(m[0].get("reference").is_none());
        assert_eq!(m[1]["reference"][0]["id"], "c1");
        assert_eq!(m[1]["reference"][0]["content"], "text");
        assert_eq!(m[1]["reference"][0]["document_name"], "doc");
        assert!(m[1].get("prompt").is_none());
        assert_eq!(m[3]["reference"], json!([]));
        assert_eq!(out["agent_id"], "c");
        assert!(out.get("dsl").is_none());
        assert_eq!(out["update_date"], Value::Null);
    }

    #[test]
    fn dates_read_like_gos_json() {
        // Whatever the zone of the test machine, the shape is RFC 3339 with
        // the fraction trimmed.
        let v = go_time("2026-10-13 16:21:25.000");
        let s = v.as_str().unwrap();
        assert!(s.starts_with("2026-10-13T16:21:25"), "{s}");
        assert!(s.ends_with('Z') || s[19..].contains(['+', '-']), "{s}");
        assert!(!s.contains('.'), "{s}");
        let v = go_time("2026-10-13 16:21:25.250");
        assert!(v.as_str().unwrap().contains(".25"));
    }

    #[test]
    fn date_filters_parse_like_upstream() {
        assert_eq!(
            parse_date("2026-10-01", false).unwrap().as_deref(),
            Some("2026-10-01 00:00:00")
        );
        assert_eq!(
            parse_date("2026-10-01", true).unwrap().as_deref(),
            Some("2026-10-01 23:59:59")
        );
        assert_eq!(
            parse_date("2026-10-01 05:06:07", false).unwrap().as_deref(),
            Some("2026-10-01 05:06:07")
        );
        assert!(parse_date("yesterday", false).is_err());
        assert!(parse_date("", false).unwrap().is_none());
    }
}

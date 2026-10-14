//! The HTTP surface: RAGFlow's agent endpoints, with the request parsing,
//! authentication and response envelopes of upstream's Go handlers
//! (`internal/handler/{auth,agent,bot}.go`).
//!
//! | endpoint | auth |
//! |---|---|
//! | `POST /api/v1/agents/chat/completions` | session token or API token |
//! | `GET/POST /api/v1/agents/{id}/sessions`, `GET/DELETE .../sessions/{sid}`, `DELETE .../sessions` | same |
//! | `GET /api/v1/agents/{id}/logs/{mid}` | same |
//! | `POST /api/v1/tasks/{sid}/cancel` | same |
//! | `POST /api/v1/agentbots/{id}/completions`, `GET .../inputs`, `GET .../logs/{mid}` | also a beta token |

#![allow(clippy::result_large_err)]

use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::Response;
use axum::routing::{get, post};
use axum::Router;
use serde_json::{json, Map, Value};

use crate::auth;
use crate::facts::{Facts, User};
use crate::gojson;
use crate::kv::Kv;
use crate::models::Catalog;
use crate::perm::{self, Op, PermError};
use crate::service::{Agent, AgentError, Run, RunEvent, RunParams, INTERNAL_ERROR};
use crate::sessions::{self, ApiError, ListQuery, SessionStore};

pub struct App {
    pub agent: Arc<Agent>,
    pub facts: Arc<dyn Facts>,
    pub sessions: Arc<dyn SessionStore>,
    pub kv: Arc<dyn Kv>,
    pub catalog: Arc<Catalog>,
    /// `RAGFLOW_SECRET_KEY`; Kvrocks is read when it is unset.
    pub secret_env: Option<String>,
}

pub fn router(app: Arc<App>) -> Router {
    Router::new()
        .route("/api/v1/agents/chat/completions", post(chat_completions))
        .route(
            "/api/v1/agents/{canvas_id}/sessions",
            get(list_sessions)
                .post(create_session)
                .delete(delete_sessions),
        )
        .route(
            "/api/v1/agents/{canvas_id}/sessions/{session_id}",
            get(get_session).delete(delete_session),
        )
        .route(
            "/api/v1/agents/{canvas_id}/logs/{message_id}",
            get(agent_logs),
        )
        .route("/api/v1/tasks/{session_id}/cancel", post(cancel_task))
        .route(
            "/api/v1/agentbots/{agent_id}/completions",
            post(agentbot_completions),
        )
        .route("/api/v1/agentbots/{agent_id}/inputs", get(agentbot_inputs))
        .route(
            "/api/v1/agentbots/{agent_id}/logs/{message_id}",
            get(agentbot_logs),
        )
        .with_state(app)
}

// ---------------------------------------------------------------------------
// Envelopes

fn respond(status: u16, body: String) -> Response {
    Response::builder()
        .status(StatusCode::from_u16(status).unwrap_or(StatusCode::OK))
        .header(header::CONTENT_TYPE, "application/json; charset=utf-8")
        .body(Body::from(body))
        .expect("a response")
}

/// `{"code", "data", "message"}` (`common.ResponseWithCodeData`).
fn envelope(code: i64, data: Value, message: &str) -> Response {
    envelope_status(200, code, data, message)
}

fn envelope_status(status: u16, code: i64, data: Value, message: &str) -> Response {
    respond(
        status,
        gojson::marshal(&json!({ "code": code, "data": data, "message": message })),
    )
}

/// `{"code", "message"}` (`common.ErrorWithCode`).
fn error_with_code(code: i64, message: &str) -> Response {
    respond(
        200,
        gojson::marshal(&json!({ "code": code, "message": message })),
    )
}

fn success(data: Value) -> Response {
    envelope(0, data, "success")
}

fn api_error(e: ApiError) -> Response {
    error_with_code(e.code, &e.message)
}

// ---------------------------------------------------------------------------
// Authentication

#[derive(Clone)]
struct Authed {
    user: User,
    /// The agent a beta token is bound to.
    bound_agent: Option<String>,
    beta: bool,
}

impl App {
    fn secret(&self) -> Option<String> {
        if let Some(s) = self.secret_env.as_ref().filter(|s| !s.is_empty()) {
            return Some(s.clone());
        }
        self.kv.secret_key().ok().flatten()
    }

    /// `GetUserByToken`.
    fn user_by_session_token(&self, authorization: &str) -> Option<User> {
        let secret = self.secret()?;
        let now = chrono::Utc::now().timestamp();
        let access = auth::extract_access_token(authorization, &secret, now).ok()?;
        if access.len() < 32 {
            return None;
        }
        self.facts.user_by_access_token(&access).ok().flatten()
    }

    fn user_of_tenant(&self, tenant_id: &str) -> Option<User> {
        let user = self.facts.user_by_id(tenant_id).ok().flatten()?;
        user.access_token
            .as_deref()
            .is_some_and(|t| !t.is_empty())
            .then_some(user)
    }

    /// `GetUserByAPIToken`.
    fn user_by_api_token(&self, authorization: &str) -> Option<User> {
        let token = auth::api_token_from_header(authorization)?;
        let row = self.facts.api_token(token).ok().flatten()?;
        self.user_of_tenant(&row.tenant_id)
    }

    /// `GetUserByBetaAPIToken` and the agent the token is bound to.
    fn user_by_beta_token(&self, authorization: &str) -> Option<(User, Option<String>)> {
        let token = auth::beta_token_from_header(authorization.trim())?;
        let rows = self.facts.beta_tokens(token).ok()?;
        let row = rows.into_iter().next()?;
        let user = self.user_of_tenant(&row.tenant_id)?;
        Some((user, row.dialog_id))
    }

    /// `AuthMiddleware`.
    fn authenticate(&self, headers: &HeaderMap) -> Result<Authed, Response> {
        let authorization = headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if authorization.is_empty() {
            return Err(envelope_status(
                401,
                401,
                Value::Null,
                "Missing Authorization header",
            ));
        }
        let user = self
            .user_by_session_token(authorization)
            .or_else(|| self.user_by_api_token(authorization));
        let Some(user) = user else {
            return Err(envelope_status(
                401,
                401,
                Value::Null,
                "Invalid access token",
            ));
        };
        if user.is_superuser {
            return Err(envelope_status(
                403,
                403,
                Value::Null,
                "Super user shouldn't access the URL",
            ));
        }
        Ok(Authed {
            user,
            bound_agent: None,
            beta: false,
        })
    }

    /// `BetaAuthMiddleware`: a session token, an API token or a beta token,
    /// with the `ragflow_auth` cookie as a fallback; every failure is a
    /// business error, not an HTTP one.
    fn authenticate_beta(&self, headers: &HeaderMap) -> Result<Authed, Response> {
        let mut authorization = headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        if authorization.is_empty() {
            authorization = cookie(headers, "ragflow_auth").unwrap_or_default();
        }
        let invalid = || envelope(102, Value::Null, "Authorization is not valid!");
        if authorization.is_empty() {
            return Err(invalid());
        }
        if let Some(user) = self.user_by_session_token(&authorization) {
            return Ok(Authed {
                user,
                bound_agent: None,
                beta: false,
            });
        }
        if let Some(user) = self.user_by_api_token(&authorization) {
            return Ok(Authed {
                user,
                bound_agent: None,
                beta: false,
            });
        }
        if let Some((user, bound)) = self.user_by_beta_token(&authorization) {
            return Ok(Authed {
                user,
                bound_agent: bound.filter(|b| !b.is_empty()),
                beta: true,
            });
        }
        Err(invalid())
    }
}

/// A cookie's value, URL-decoded as `gin.Context.Cookie` does.
fn cookie(headers: &HeaderMap, name: &str) -> Option<String> {
    for value in headers.get_all(header::COOKIE) {
        for pair in value.to_str().ok()?.split(';') {
            if let Some((k, v)) = pair.trim().split_once('=') {
                if k == name {
                    return Some(percent_decode(v));
                }
            }
        }
    }
    None
}

/// `url.QueryUnescape`, with `gin.Context.Cookie`'s error handling (a bad
/// escape yields an empty value).
fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let hex = |c: u8| (c as char).to_digit(16).map(|d| d as u8);
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'%' => {
                let (Some(h), Some(l)) = (
                    b.get(i + 1).and_then(|c| hex(*c)),
                    b.get(i + 2).and_then(|c| hex(*c)),
                ) else {
                    return String::new();
                };
                out.push(h << 4 | l);
                i += 3;
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

// ---------------------------------------------------------------------------
// Request bodies

/// `ShouldBindJSON` that tolerates an empty body (`io.EOF`).
fn parse_body(body: &Bytes) -> Result<Map<String, Value>, Response> {
    if body.iter().all(u8::is_ascii_whitespace) {
        return Ok(Map::new());
    }
    match serde_json::from_slice::<Value>(body) {
        Ok(Value::Object(m)) => Ok(m),
        Ok(_) => Err(envelope(
            101,
            Value::Null,
            "Invalid request: json: cannot unmarshal into Go value of type map",
        )),
        Err(e) => Err(envelope(101, Value::Null, &format!("Invalid request: {e}"))),
    }
}

fn string_field(m: &Map<String, Value>, key: &str) -> String {
    m.get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// `NormalizeOpenAIMessageContent`.
fn message_content(content: Option<&Value>) -> Result<String, String> {
    match content {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Array(parts)) => Ok(parts
            .iter()
            .filter_map(Value::as_object)
            .filter(|p| p.get("type").and_then(Value::as_str) == Some("text"))
            .map(|p| string_field(p, "text"))
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join("\n")),
        Some(_) => Err("messages[].content must be a string or an array of content parts.".into()),
    }
}

/// `ResolveCompletionQuestion`.
fn resolve_question(question: &str, query: &str, messages: &[Value]) -> Result<String, String> {
    if !question.is_empty() {
        return Ok(question.to_string());
    }
    if !query.is_empty() {
        return Ok(query.to_string());
    }
    let Some(last) = messages.last() else {
        return Ok(String::new());
    };
    if last.get("role").and_then(Value::as_str) != Some("user") {
        return Err("The last content of this conversation is not from user.".into());
    }
    message_content(last.get("content"))
}

/// `unwrapFormInput`: a form field's submitted value.
fn unwrap_form_input(raw: &Value) -> Value {
    let Some(field) = raw.as_object() else {
        return raw.clone();
    };
    if let Some(v) = field.get("value") {
        return v.clone();
    }
    if field.contains_key("type") {
        return Value::Null;
    }
    raw.clone()
}

// ---------------------------------------------------------------------------
// SSE

/// One frame, as `WriteChatbotRunEvent` writes it.
fn frame(ev: &RunEvent) -> String {
    if ev.kind == "done" {
        return "data:[DONE]\n\n".to_string();
    }
    if ev.kind == "error" {
        let internal = ev.data.get("kind").and_then(Value::as_str) == Some("internal");
        let message = if internal {
            INTERNAL_ERROR.to_string()
        } else {
            ev.data
                .get("message")
                .and_then(Value::as_str)
                .filter(|m| !m.is_empty())
                .unwrap_or("an internal error occurred")
                .to_string()
        };
        let mut payload = json!({ "code": 500, "message": message, "data": false });
        if !ev.session_id.is_empty() {
            payload["task_id"] = json!(ev.session_id);
            payload["session_id"] = json!(ev.session_id);
        }
        return format!("data:{}\n\n", gojson::marshal(&payload));
    }
    let mut payload: BTreeMap<&str, Value> = BTreeMap::new();
    payload.insert("data", ev.data.clone());
    payload.insert("created_at", json!(ev.created_at));
    if !ev.kind.is_empty() {
        payload.insert("event", json!(ev.kind));
    }
    if !ev.message_id.is_empty() {
        payload.insert("message_id", json!(ev.message_id));
    }
    if !ev.session_id.is_empty() {
        payload.insert("task_id", json!(ev.session_id));
        payload.insert("session_id", json!(ev.session_id));
    }
    format!(
        "data:{}\n\n",
        gojson::marshal(&Value::Object(
            payload
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect()
        ))
    )
}

/// The chat endpoint's stream handler: an `error` event is sent as a
/// `message` (`convertAgentRunErrorToMessage`); the rest pass through.
fn errors_as_messages(ev: RunEvent) -> RunEvent {
    if ev.kind == "error" {
        error_as_message(ev)
    } else {
        ev
    }
}

/// `convertAgentRunErrorToMessage`.
fn error_as_message(mut ev: RunEvent) -> RunEvent {
    let internal = ev.data.get("kind").and_then(Value::as_str) == Some("internal");
    let content = if internal {
        INTERNAL_ERROR.to_string()
    } else {
        match ev.data.get("message").and_then(Value::as_str) {
            Some(m) if !m.is_empty() => m.to_string(),
            _ => "Agent run failed.".to_string(),
        }
    };
    ev.kind = "message".into();
    ev.data = json!({ "content": content });
    ev
}

/// Streams a run as SSE: every event through `map`, then `data:[DONE]`
/// unless the run sent one. Dropping the response cancels the run.
fn sse_response(run: Run, map: fn(RunEvent) -> RunEvent, empty_frame: bool) -> Response {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();
    let cancel = run.cancel.clone();
    let session_id = run.session_id.clone();
    let mut events = run.events;
    tokio::spawn(async move {
        let mut emitted = false;
        let mut done_sent = false;
        loop {
            let ev = tokio::select! {
                ev = events.recv() => match ev {
                    Some(ev) => map(ev),
                    None => break,
                },
                // The client went away: stop the run now rather than at its
                // next event.
                _ = tx.closed() => {
                    cancel.store(true, Ordering::SeqCst);
                    return;
                }
            };
            emitted = true;
            done_sent |= ev.kind == "done";
            if tx.send(Ok(Bytes::from(frame(&ev)))).is_err() {
                cancel.store(true, Ordering::SeqCst);
                return;
            }
        }
        if !emitted && empty_frame {
            let ev = RunEvent {
                kind: String::new(),
                data: json!({}),
                message_id: String::new(),
                created_at: chrono::Utc::now().timestamp(),
                session_id: session_id.clone(),
            };
            let _ = tx.send(Ok(Bytes::from(frame(&ev))));
        }
        if !done_sent {
            let done = RunEvent {
                kind: "done".into(),
                data: Value::Null,
                message_id: String::new(),
                created_at: 0,
                session_id: String::new(),
            };
            let _ = tx.send(Ok(Bytes::from(frame(&done))));
        }
    });
    Response::builder()
        .status(200)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(Body::from_stream(
            tokio_stream::wrappers::UnboundedReceiverStream::new(rx),
        ))
        .expect("a response")
}

/// Drains a run into one JSON reply (`stream` off), as the handler does.
async fn collect_reply(mut run: Run) -> Response {
    let mut full_content = String::new();
    let mut reference = Map::new();
    let mut structured = Map::new();
    let mut usage = Value::Null;
    let mut final_ans: Option<RunEvent> = None;
    let mut had_events = false;
    while let Some(mut ev) = run.events.recv().await {
        had_events = true;
        if ev.kind == "error" {
            ev = error_as_message(ev);
            final_ans = Some(ev.clone());
        }
        if ev.kind == "message" {
            if let Some(c) = ev.data.get("content").and_then(Value::as_str) {
                full_content.push_str(c);
            }
            if ev.data.get("start_to_think").and_then(Value::as_bool) == Some(true) {
                full_content.push_str("<think>");
            } else if ev.data.get("end_to_think").and_then(Value::as_bool) == Some(true) {
                full_content.push_str("</think>");
            }
        }
        if let Some(r) = ev.data.get("reference").and_then(Value::as_object) {
            for (k, v) in r {
                reference.insert(k.clone(), v.clone());
            }
        }
        if ev.kind == "node_finished" {
            if let (Some(s), Some(id)) = (
                ev.data.pointer("/outputs/structured"),
                ev.data.get("component_id").and_then(Value::as_str),
            ) {
                structured.insert(id.to_string(), s.clone());
            }
        }
        if ev.kind == "workflow_finished" {
            if let Some(u) = ev.data.get("usage").filter(|u| !u.is_null()) {
                usage = u.clone();
            }
        }
        if ev.kind == "message_end" {
            final_ans = Some(ev.clone());
        }
        if ev.kind == "workflow_finished" {
            continue;
        }
        if final_ans.is_none() {
            final_ans = Some(ev);
        }
    }
    let Some(final_ans) = final_ans.filter(|_| had_events) else {
        return success(json!({ "session_id": run.session_id }));
    };
    let mut data = match final_ans.data.clone() {
        Value::Object(m) => m,
        _ => Map::new(),
    };
    data.insert("content".into(), Value::String(full_content));
    if !reference.is_empty() {
        data.insert("reference".into(), Value::Object(reference));
    }
    if !usage.is_null() {
        data.insert("usage".into(), usage);
    }
    if !structured.is_empty() {
        data.insert("structured".into(), Value::Object(structured));
    }
    success(json!({
        "event": final_ans.kind,
        "data": data,
        "message_id": final_ans.message_id,
        "created_at": final_ans.created_at,
        "task_id": final_ans.session_id,
        "session_id": final_ans.session_id,
    }))
}

// ---------------------------------------------------------------------------
// Handlers

/// Runs blocking work (database, Kvrocks, the run's preparation) off the
/// async runtime.
async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> T + Send + 'static,
) -> Result<T, Response> {
    tokio::task::spawn_blocking(f).await.map_err(|e| {
        eprintln!("handler panicked: {e}");
        envelope(500, Value::Null, INTERNAL_ERROR)
    })
}

/// What a completions handler's blocking half decided.
enum Prep {
    Reply(Response),
    /// A started run, to stream or to collect.
    Run {
        run: Run,
        stream: bool,
        convert_errors: bool,
    },
}

async fn finish(prep: Result<Prep, Response>) -> Response {
    match prep {
        Err(r) | Ok(Prep::Reply(r)) => r,
        Ok(Prep::Run {
            run,
            stream: true,
            convert_errors,
        }) => sse_response(
            run,
            if convert_errors {
                errors_as_messages
            } else {
                identity
            },
            convert_errors,
        ),
        Ok(Prep::Run {
            run, stream: false, ..
        }) => collect_reply(run).await,
    }
}

fn identity(ev: RunEvent) -> RunEvent {
    ev
}

fn has_files(req: &Map<String, Value>) -> bool {
    match req.get("files") {
        None | Some(Value::Null) => false,
        Some(Value::Array(a)) => !a.is_empty(),
        Some(_) => true,
    }
}

async fn chat_completions(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    finish(blocking(move || chat_prepare(&app, &headers, &body)).await).await
}

fn chat_prepare(app: &Arc<App>, headers: &HeaderMap, body: &Bytes) -> Prep {
    let authed = match app.authenticate(headers) {
        Ok(a) => a,
        Err(r) => return Prep::Reply(r),
    };
    let req = match parse_body(body) {
        Ok(r) => r,
        Err(r) => return Prep::Reply(r),
    };
    let messages: Vec<Value> = req
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let (question, query) = (string_field(&req, "question"), string_field(&req, "query"));
    let question = match resolve_question(&question, &query, &messages) {
        Ok(q) => q,
        Err(e) => return Prep::Reply(envelope(101, Value::Null, &e)),
    };
    let query = question;
    let messages = if !query.is_empty() {
        vec![json!({ "role": "user", "content": query })]
    } else if let Some(last) = messages.last() {
        vec![last.clone()]
    } else {
        Vec::new()
    };
    let agent_id = string_field(&req, "agent_id");
    if agent_id.is_empty() {
        return Prep::Reply(envelope(101, Value::Null, "`agent_id` is required."));
    }
    let openai = req
        .get("openai-compatible")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if openai && messages.is_empty() {
        return Prep::Reply(envelope(
            102,
            Value::Null,
            "at least one message is required in openai-compatible mode.",
        ));
    }
    if openai {
        return Prep::Reply(envelope(
            102,
            Value::Null,
            "openai-compatible mode is not supported by this server",
        ));
    }
    if has_files(&req) {
        return Prep::Reply(envelope(
            102,
            Value::Null,
            "`files` are not supported by this server: file inputs need RAGFlow's document parsers",
        ));
    }
    let inputs = req.get("inputs").and_then(Value::as_object).cloned();
    let has_inputs = inputs.as_ref().is_some_and(|i| !i.is_empty());
    let user_input = if !query.is_empty() && has_inputs {
        let mut values: Map<String, Value> = inputs
            .iter()
            .flatten()
            .map(|(k, v)| (k.clone(), unwrap_form_input(v)))
            .collect();
        values.insert("query".into(), Value::String(query.clone()));
        Value::Object(values)
    } else if !query.is_empty() {
        Value::String(query.clone())
    } else if has_inputs {
        Value::Object(
            inputs
                .iter()
                .flatten()
                .map(|(k, v)| (k.clone(), unwrap_form_input(v)))
                .collect(),
        )
    } else {
        match messages
            .last()
            .filter(|m| m.get("role").and_then(Value::as_str) == Some("user"))
            .and_then(|m| message_content(m.get("content")).ok())
        {
            Some(c) if !c.is_empty() => Value::String(c),
            _ => Value::Null,
        }
    };
    let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let client_session = string_field(&req, "session_id");
    let trusted_first_touch = client_session.is_empty();
    let session_id = if trusted_first_touch {
        uuid::Uuid::new_v4().simple().to_string()
    } else {
        client_session
    };
    let started = app.agent.start(RunParams {
        user_id: authed.user.id.clone(),
        canvas_id: agent_id,
        session_id: Some(session_id),
        trusted_first_touch,
        version: None,
        released: false,
        user_input,
    });
    match started {
        Err(e) => Prep::Reply(start_failure(&e, false)),
        Ok(run) => Prep::Run {
            run,
            stream,
            convert_errors: true,
        },
    }
}

/// `mapAgentError` for the chat endpoint; the agentbot endpoint reports the
/// Go error text instead, except where that would leak storage detail.
fn start_failure(e: &AgentError, agentbot: bool) -> Response {
    if let AgentError::Storage(cause) = e {
        eprintln!("agent run failed to start: {cause}");
    }
    if agentbot {
        let message = match e {
            AgentError::NotFound => "agent: not found or access denied".to_string(),
            AgentError::SessionNotFound(id) => {
                format!("RunAgent: session {id:?} not found: agent: not found or access denied")
            }
            AgentError::NotOwner => "agent not owned by user".to_string(),
            AgentError::Storage(_) => INTERNAL_ERROR.to_string(),
            AgentError::Message(m) => m.clone(),
        };
        return envelope(102, Value::Null, &message);
    }
    let (code, message) = e.code_and_message();
    envelope(code, Value::Null, &message)
}

async fn agentbot_completions(
    State(app): State<Arc<App>>,
    Path(agent_id): Path<String>,
    Query(query): Query<std::collections::HashMap<String, String>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    finish(blocking(move || agentbot_prepare(&app, &agent_id, &query, &headers, &body)).await).await
}

fn agentbot_prepare(
    app: &Arc<App>,
    agent_id: &str,
    query: &std::collections::HashMap<String, String>,
    headers: &HeaderMap,
    body: &Bytes,
) -> Prep {
    let authed = match app.authenticate_beta(headers) {
        Ok(a) => a,
        Err(r) => return Prep::Reply(r),
    };
    // Unlike upstream, whose agentbot completions let a beta token run any
    // agent of its tenant, a beta token runs only the agent it is bound to.
    if authed.beta {
        if let Some(bound) = &authed.bound_agent {
            if bound != agent_id {
                return Prep::Reply(envelope(
                    401,
                    Value::Null,
                    "API token is not authorized for this agent.",
                ));
            }
        }
    }
    if agent_id.is_empty() {
        return Prep::Reply(envelope(101, Value::Null, "`agent_id` is required."));
    }
    let req = match parse_body(body) {
        Ok(r) => r,
        Err(r) => return Prep::Reply(r),
    };
    let messages: Vec<Value> = req
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let question = match resolve_question(
        &string_field(&req, "question"),
        &string_field(&req, "query"),
        &messages,
    ) {
        Ok(q) => q,
        Err(e) => return Prep::Reply(envelope(101, Value::Null, &e)),
    };
    if has_files(&req) {
        return Prep::Reply(envelope(
            102,
            Value::Null,
            "`files` are not supported by this server: file inputs need RAGFlow's document parsers",
        ));
    }
    let mut release = req.get("release").and_then(Value::as_bool);
    if release.is_none() {
        if let Some(raw) = query.get("release") {
            match parse_bool(raw) {
                Some(b) => release = Some(b),
                None => {
                    return Prep::Reply(envelope(
                        101,
                        Value::Null,
                        &format!(
                        "Invalid release value: strconv.ParseBool: parsing {raw:?}: invalid syntax"
                    ),
                    ))
                }
            }
        }
    }
    // agentbotUserInput: the question, else the begin-form inputs (one
    // field lifts its value, several become a name/value map).
    let user_input = if !question.is_empty() {
        Value::String(question)
    } else {
        let inputs = req
            .get("inputs")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let lift = |raw: &Value| match raw.as_object().and_then(|f| f.get("value")) {
            Some(v) => v.clone(),
            None => raw.clone(),
        };
        match inputs.len() {
            0 => Value::Null,
            1 => lift(inputs.values().next().expect("one input")),
            _ => Value::Object(inputs.iter().map(|(k, v)| (k.clone(), lift(v))).collect()),
        }
    };
    let session_id = string_field(&req, "session_id");
    let started = app.agent.start(RunParams {
        user_id: authed.user.id.clone(),
        canvas_id: agent_id.to_string(),
        session_id: (!session_id.is_empty()).then_some(session_id),
        trusted_first_touch: false,
        version: None,
        released: release == Some(true),
        user_input,
    });
    match started {
        Err(e) => Prep::Reply(start_failure(&e, true)),
        Ok(run) => Prep::Run {
            run,
            stream: true,
            convert_errors: false,
        },
    }
}

/// `strconv.ParseBool`.
fn parse_bool(s: &str) -> Option<bool> {
    match s {
        "1" | "t" | "T" | "TRUE" | "true" | "True" => Some(true),
        "0" | "f" | "F" | "FALSE" | "false" | "False" => Some(false),
        _ => None,
    }
}

async fn agentbot_inputs(
    State(app): State<Arc<App>>,
    Path(agent_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    let r = blocking(move || {
        let authed = match app.authenticate_beta(&headers) {
            Ok(a) => a,
            Err(r) => return r,
        };
        let canvas =
            match perm::canvas_for(app.facts.as_ref(), &authed.user.id, &agent_id, Op::Read) {
                Ok(c) => c,
                Err(PermError::NotFound | PermError::Denied) => {
                    return envelope(102, Value::Null, "agent: not found or access denied")
                }
                Err(e) => {
                    return envelope(
                        102,
                        Value::Null,
                        &format!("bot: check canvas {agent_id} permission: {e}"),
                    )
                }
            };
        // The begin form, prologue and mode come from the canvas's own DSL.
        let begin = canvas
            .dsl
            .get("components")
            .and_then(Value::as_object)
            .and_then(|c| {
                c.values().find(|comp| {
                    comp.pointer("/obj/component_name").and_then(Value::as_str) == Some("Begin")
                })
            });
        let params = begin.and_then(|b| b.pointer("/obj/params"));
        let text = |k: &str| {
            params
                .and_then(|p| p.get(k))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
        };
        let inputs = params
            .and_then(|p| p.get("inputs"))
            .filter(|i| i.is_object())
            .cloned()
            .unwrap_or(Value::Null);
        success(json!({
            "title": canvas.title.clone().unwrap_or_default(),
            "avatar": canvas.avatar.clone().unwrap_or_default(),
            "inputs": inputs,
            "prologue": text("prologue"),
            "mode": text("mode"),
        }))
    })
    .await;
    r.unwrap_or_else(|e| e)
}

/// The logs a run left under `{agent}-{message}-logs`: nothing for agent
/// canvases, so `{}`.
fn logs_reply(app: &App, agent_id: &str, message_id: &str, lenient: bool) -> Response {
    let key = format!("{agent_id}-{message_id}-logs");
    match app.kv.get_raw(&key) {
        Err(_) if lenient => success(json!({})),
        Err(_) => envelope(500, Value::Null, "failed to read agent logs"),
        Ok(None) => success(json!({})),
        Ok(Some(raw)) if raw.trim().is_empty() => success(json!({})),
        Ok(Some(raw)) => match serde_json::from_str::<Value>(&raw) {
            Ok(v) => success(v),
            Err(_) if lenient => success(json!({})),
            Err(_) => envelope(500, Value::Null, "failed to decode agent logs"),
        },
    }
}

async fn agent_logs(
    State(app): State<Arc<App>>,
    Path((canvas_id, message_id)): Path<(String, String)>,
    headers: HeaderMap,
) -> Response {
    let r = blocking(move || {
        let authed = match app.authenticate(&headers) {
            Ok(a) => a,
            Err(r) => return r,
        };
        match perm::canvas_for(app.facts.as_ref(), &authed.user.id, &canvas_id, Op::Read) {
            Ok(_) => logs_reply(&app, &canvas_id, &message_id, true),
            Err(PermError::NotFound | PermError::Denied) => envelope(
                103,
                Value::Null,
                "Make sure you have permission to access the agent.",
            ),
            Err(e) => envelope(500, Value::Null, &e.to_string()),
        }
    })
    .await;
    r.unwrap_or_else(|e| e)
}

async fn agentbot_logs(
    State(app): State<Arc<App>>,
    Path((agent_id, message_id)): Path<(String, String)>,
    headers: HeaderMap,
) -> Response {
    let r = blocking(move || {
        let authed = match app.authenticate_beta(&headers) {
            Ok(a) => a,
            Err(r) => return r,
        };
        if let Some(bound) = &authed.bound_agent {
            if bound != &agent_id {
                return envelope(
                    401,
                    Value::Null,
                    "API token is not authorized for this agent.",
                );
            }
        }
        match perm::canvas_for(app.facts.as_ref(), &authed.user.id, &agent_id, Op::Read) {
            Ok(_) => logs_reply(&app, &agent_id, &message_id, false),
            Err(PermError::NotFound | PermError::Denied) => {
                envelope(102, Value::Null, "agent: not found or access denied")
            }
            Err(e) => envelope(
                102,
                Value::Null,
                &format!("bot: check canvas {agent_id} permission: {e}"),
            ),
        }
    })
    .await;
    r.unwrap_or_else(|e| e)
}

type QueryMap = Query<std::collections::HashMap<String, String>>;

async fn list_sessions(
    State(app): State<Arc<App>>,
    Path(canvas_id): Path<String>,
    Query(q): QueryMap,
    headers: HeaderMap,
) -> Response {
    let r = blocking(move || {
        let authed = match app.authenticate(&headers) {
            Ok(a) => a,
            Err(r) => return r,
        };
        let get = |k: &str| q.get(k).cloned().unwrap_or_default();
        let int = |k: &str| get(k).parse::<i64>().unwrap_or(0);
        let falsy = |k: &str| matches!(q.get(k).map(String::as_str), Some("false" | "False"));
        let query = ListQuery {
            page: int("page"),
            page_size: int("page_size"),
            keywords: get("keywords"),
            from_date: get("from_date"),
            to_date: get("to_date"),
            order_by: q
                .get("orderby")
                .cloned()
                .unwrap_or_else(|| "update_time".into()),
            desc: !falsy("desc"),
            session_id: get("id"),
            user_id: get("user_id"),
            exp_user_id: get("exp_user_id"),
            include_dsl: !falsy("dsl"),
            no_history: falsy("include_history"),
        };
        match sessions::list(
            app.facts.as_ref(),
            app.sessions.as_ref(),
            &authed.user.id,
            &canvas_id,
            &query,
        ) {
            Ok((data, total)) => respond(
                200,
                gojson::marshal(&json!({
                    "code": 0, "data": data, "message": "success", "total": total
                })),
            ),
            Err(e) => api_error(e),
        }
    })
    .await;
    r.unwrap_or_else(|e| e)
}

async fn create_session(
    State(app): State<Arc<App>>,
    Path(canvas_id): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let r = blocking(move || {
        let authed = match app.authenticate(&headers) {
            Ok(a) => a,
            Err(r) => return r,
        };
        let req = match parse_body(&body) {
            Ok(r) => r,
            Err(r) => return r,
        };
        match sessions::create(
            app.facts.as_ref(),
            app.sessions.as_ref(),
            &app.catalog,
            &authed.user.id,
            &canvas_id,
            &string_field(&req, "name"),
            &string_field(&req, "source"),
            req.get("dsl").cloned(),
        ) {
            Ok(row) => success(row),
            Err(e) => api_error(e),
        }
    })
    .await;
    r.unwrap_or_else(|e| e)
}

async fn get_session(
    State(app): State<Arc<App>>,
    Path((canvas_id, session_id)): Path<(String, String)>,
    headers: HeaderMap,
) -> Response {
    let r = blocking(move || {
        let authed = match app.authenticate(&headers) {
            Ok(a) => a,
            Err(r) => return r,
        };
        match sessions::get(
            app.facts.as_ref(),
            app.sessions.as_ref(),
            &authed.user.id,
            &canvas_id,
            &session_id,
        ) {
            Ok(row) => success(row),
            Err(e) => api_error(e),
        }
    })
    .await;
    r.unwrap_or_else(|e| e)
}

async fn delete_session(
    State(app): State<Arc<App>>,
    Path((canvas_id, session_id)): Path<(String, String)>,
    headers: HeaderMap,
) -> Response {
    let r = blocking(move || {
        let authed = match app.authenticate(&headers) {
            Ok(a) => a,
            Err(r) => return r,
        };
        match sessions::delete_one(
            app.facts.as_ref(),
            app.sessions.as_ref(),
            &authed.user.id,
            &canvas_id,
            &session_id,
        ) {
            Ok(deleted) => success(json!(deleted)),
            Err(e) => api_error(e),
        }
    })
    .await;
    r.unwrap_or_else(|e| e)
}

async fn delete_sessions(
    State(app): State<Arc<App>>,
    Path(canvas_id): Path<String>,
    Query(q): QueryMap,
    headers: HeaderMap,
) -> Response {
    let r = blocking(move || {
        let authed = match app.authenticate(&headers) {
            Ok(a) => a,
            Err(r) => return r,
        };
        let ids: Vec<String> = q
            .get("ids")
            .map(|raw| {
                raw.split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        let delete_all = q.get("delete_all").map(String::as_str) == Some("true");
        match sessions::delete_many(
            app.facts.as_ref(),
            app.sessions.as_ref(),
            &authed.user.id,
            &canvas_id,
            ids,
            delete_all,
        ) {
            // upstream returns its result struct, which has no JSON tags.
            Ok(done) => {
                let message = if done.data.is_null() {
                    ""
                } else {
                    done.message.as_str()
                };
                success(json!({ "Data": done.data, "Message": message }))
            }
            Err(e) => api_error(e),
        }
    })
    .await;
    r.unwrap_or_else(|e| e)
}

async fn cancel_task(
    State(app): State<Arc<App>>,
    Path(session_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    let r = blocking(move || {
        let authed = match app.authenticate(&headers) {
            Ok(a) => a,
            Err(r) => return r,
        };
        match app.agent.cancel(&authed.user.id, &session_id) {
            Ok(()) => success(json!(true)),
            Err(e) => {
                if let AgentError::Storage(cause) = &e {
                    eprintln!("cancel failed: {cause}");
                }
                let (code, message) = e.code_and_message();
                envelope(code, Value::Null, &message)
            }
        }
    })
    .await;
    r.unwrap_or_else(|e| e)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn question_resolution() {
        let user = vec![json!({"role": "user", "content": "hi"})];
        assert_eq!(resolve_question("q", "x", &user).unwrap(), "q");
        assert_eq!(resolve_question("", "x", &user).unwrap(), "x");
        assert_eq!(resolve_question("", "", &user).unwrap(), "hi");
        assert_eq!(resolve_question("", "", &[]).unwrap(), "");
        let assistant = vec![json!({"role": "assistant", "content": "hi"})];
        assert!(resolve_question("", "", &assistant).is_err());
        let parts = vec![json!({"role": "user", "content": [
            {"type": "text", "text": "a"}, {"type": "image_url"}, {"type": "text", "text": "b"}]})];
        assert_eq!(resolve_question("", "", &parts).unwrap(), "a\nb");
    }

    #[test]
    fn form_values_unwrap() {
        assert_eq!(
            unwrap_form_input(&json!({"value": 3, "type": "line"})),
            json!(3)
        );
        assert_eq!(unwrap_form_input(&json!({"type": "line"})), Value::Null);
        assert_eq!(unwrap_form_input(&json!({"a": 1})), json!({"a": 1}));
        assert_eq!(unwrap_form_input(&json!("x")), json!("x"));
    }

    #[test]
    fn frames_follow_the_go_envelope() {
        let ev = RunEvent {
            kind: "message".into(),
            data: json!({"content": "a<b"}),
            message_id: "m".into(),
            created_at: 7,
            session_id: "s".into(),
        };
        assert_eq!(
            frame(&ev),
            concat!(
                r#"data:{"created_at":7,"data":{"content":"a"#,
                "\\u003c",
                r#"b"},"event":"message","message_id":"m","session_id":"s","task_id":"s"}"#,
                "\n\n"
            )
        );
        let done = RunEvent {
            kind: "done".into(),
            data: Value::Null,
            message_id: String::new(),
            created_at: 0,
            session_id: String::new(),
        };
        assert_eq!(frame(&done), "data:[DONE]\n\n");
        let err = RunEvent {
            kind: "error".into(),
            data: json!({"message": "boom"}),
            message_id: "m".into(),
            created_at: 7,
            session_id: "s".into(),
        };
        assert_eq!(
            frame(&err),
            "data:{\"code\":500,\"data\":false,\"message\":\"boom\",\"session_id\":\"s\",\"task_id\":\"s\"}\n\n"
        );
        let internal = RunEvent {
            data: json!({"message": "secret", "kind": "internal"}),
            ..err
        };
        assert!(frame(&internal).contains(INTERNAL_ERROR));
        assert!(!frame(&internal).contains("secret"));
    }

    #[test]
    fn errors_become_messages_for_the_chat_endpoint() {
        let ev = RunEvent {
            kind: "error".into(),
            data: json!({"message": "boom"}),
            message_id: "m".into(),
            created_at: 7,
            session_id: "s".into(),
        };
        let m = error_as_message(ev.clone());
        assert_eq!(
            (m.kind.as_str(), &m.data),
            ("message", &json!({"content": "boom"}))
        );
        let empty = error_as_message(RunEvent {
            data: json!({}),
            ..ev
        });
        assert_eq!(empty.data, json!({"content": "Agent run failed."}));
    }

    #[test]
    fn cookies_and_bools() {
        let mut h = HeaderMap::new();
        h.insert(
            header::COOKIE,
            "a=1; ragflow_auth=Bearer%20abc; b=2".parse().unwrap(),
        );
        assert_eq!(cookie(&h, "ragflow_auth").unwrap(), "Bearer abc");
        assert_eq!(cookie(&h, "nope"), None);
        assert_eq!(percent_decode("a+b%C3%A9"), "a b\u{e9}");
        assert_eq!(percent_decode("bad%zz"), "");
        assert_eq!(percent_decode("%é"), "");
        assert_eq!(parse_bool("True"), Some(true));
        assert_eq!(parse_bool("0"), Some(false));
        assert_eq!(parse_bool("yes"), None);
    }

    // -----------------------------------------------------------------
    // The router, end to end over in-memory stores.

    use axum::body::to_bytes;
    use axum::http::Request;
    use tower::ServiceExt;

    use crate::facts::{ApiToken, User};
    use crate::kv::fake::FakeKv;
    use crate::testutil::{agent, world};

    const SECRET: &str = "test-secret";
    const ACCESS: &str = "0123456789abcdef0123456789abcdef";

    fn app() -> (Arc<App>, Arc<crate::testutil::World>) {
        let mut w = world();
        let base = &mut Arc::get_mut(&mut w).expect("a fresh world").base;
        base.users = vec![
            User {
                id: "alice".into(),
                access_token: Some(ACCESS.into()),
                is_superuser: false,
            },
            User {
                id: "root".into(),
                access_token: Some("f".repeat(32)),
                is_superuser: true,
            },
            User {
                id: "bob".into(),
                access_token: Some("b".repeat(32)),
                is_superuser: false,
            },
            User {
                id: "tokenless".into(),
                access_token: None,
                is_superuser: false,
            },
        ];
        base.tokens = vec![
            ApiToken {
                tenant_id: "alice".into(),
                token: "ragflow-alice".into(),
                dialog_id: None,
                beta: None,
            },
            ApiToken {
                tenant_id: "alice".into(),
                token: "ragflow-beta-row".into(),
                dialog_id: Some("another-agent".into()),
                beta: Some("beta-bound".into()),
            },
            ApiToken {
                tenant_id: "alice".into(),
                token: "ragflow-beta-own".into(),
                dialog_id: Some("c1".into()),
                beta: Some("beta-own".into()),
            },
            ApiToken {
                tenant_id: "tokenless".into(),
                token: "ragflow-tokenless".into(),
                dialog_id: None,
                beta: None,
            },
        ];
        let kv = Arc::new(FakeKv::default());
        let agent = agent(&w, kv.clone());
        let app = Arc::new(App {
            agent,
            facts: w.clone(),
            sessions: w.clone(),
            kv,
            catalog: Arc::new(Catalog::default()),
            secret_env: Some(SECRET.into()),
        });
        (app, w)
    }

    async fn send(
        app: &Arc<App>,
        method: &str,
        path: &str,
        token: Option<&str>,
        body: Option<Value>,
    ) -> (u16, String) {
        let mut req = Request::builder().method(method).uri(path);
        if let Some(t) = token {
            req = req.header("authorization", t);
        }
        let req = req
            .body(match body {
                Some(b) => Body::from(b.to_string()),
                None => Body::empty(),
            })
            .unwrap();
        let resp = router(app.clone()).oneshot(req).await.unwrap();
        let status = resp.status().as_u16();
        let bytes = to_bytes(resp.into_body(), 1 << 24).await.unwrap();
        (status, String::from_utf8(bytes.to_vec()).unwrap())
    }

    fn frames(text: &str) -> Vec<Value> {
        text.split("\n\n")
            .filter(|f| f.starts_with("data:"))
            .map(|f| match &f[5..] {
                "[DONE]" => json!("[DONE]"),
                d => serde_json::from_str(d).unwrap(),
            })
            .collect()
    }

    #[tokio::test]
    async fn bad_credentials_get_upstreams_envelopes() {
        let (app, _) = app();
        let path = "/api/v1/agents/c1/sessions";
        let (s, b) = send(&app, "GET", path, None, None).await;
        assert_eq!(s, 401);
        assert_eq!(
            b,
            r#"{"code":401,"data":null,"message":"Missing Authorization header"}"#
        );
        for token in ["Bearer garbage", "Bearer ragflow-nope", "ragflow-tokenless"] {
            let (s, b) = send(&app, "GET", path, Some(token), None).await;
            assert_eq!(s, 401, "{token}");
            assert_eq!(
                b, r#"{"code":401,"data":null,"message":"Invalid access token"}"#,
                "{token}"
            );
        }
        let root = auth::dump_access_token(&"f".repeat(32), SECRET, chrono::Utc::now().timestamp());
        let (s, b) = send(&app, "GET", path, Some(&format!("Bearer {root}")), None).await;
        assert_eq!(s, 403);
        assert!(b.contains("Super user shouldn't access the URL"));
        let old = auth::dump_access_token(ACCESS, SECRET, chrono::Utc::now().timestamp() - 90_000);
        assert_eq!(
            send(&app, "GET", path, Some(&format!("Bearer {old}")), None)
                .await
                .0,
            401
        );
        let good = auth::dump_access_token(ACCESS, SECRET, chrono::Utc::now().timestamp());
        let (s, b) = send(&app, "GET", path, Some(&format!("Bearer {good}")), None).await;
        assert_eq!((s, b.contains(r#""code":0"#)), (200, true), "{b}");
        let (s, _) = send(&app, "GET", path, Some("Bearer ragflow-alice"), None).await;
        assert_eq!(s, 200);
        // A beta token is not accepted on the standard endpoints.
        assert_eq!(send(&app, "GET", path, Some("beta-own"), None).await.0, 401);
    }

    #[tokio::test]
    async fn the_beta_endpoints_report_failures_as_business_errors() {
        let (app, _) = app();
        let path = "/api/v1/agentbots/c1/inputs";
        for token in [None, Some("Bearer garbage")] {
            let (s, b) = send(&app, "GET", path, token, None).await;
            assert_eq!(s, 200);
            assert_eq!(
                b,
                r#"{"code":102,"data":null,"message":"Authorization is not valid!"}"#
            );
        }
        let (_, b) = send(&app, "GET", path, Some("Bearer beta-own"), None).await;
        assert!(b.contains(r#""code":0"#), "{b}");
    }

    #[tokio::test]
    async fn a_streamed_chat_completion_is_sse_ending_with_done() {
        let (app, w) = app();
        let (s, text) = send(
            &app,
            "POST",
            "/api/v1/agents/chat/completions",
            Some("Bearer ragflow-alice"),
            Some(json!({"agent_id": "c1", "query": "hello", "stream": true})),
        )
        .await;
        assert_eq!(s, 200);
        assert!(text.ends_with("data:[DONE]\n\n"), "{text}");
        let f = frames(&text);
        let events: Vec<_> = f.iter().filter_map(|x| x["event"].as_str()).collect();
        assert_eq!(events.first(), Some(&"workflow_started"));
        assert_eq!(events.last(), Some(&"workflow_finished"));
        assert!(events.contains(&"message"));
        let sid = f[0]["session_id"].as_str().unwrap();
        assert_eq!(f[0]["task_id"], f[0]["session_id"]);
        assert_eq!(f[0]["data"]["inputs"], "hello");
        // The session was written before the last frame went out.
        assert_eq!(w.sessions.lock().unwrap()[0].id, sid);
    }

    #[tokio::test]
    async fn without_stream_one_json_reply_carries_the_whole_answer() {
        let (app, _) = app();
        let (s, text) = send(
            &app,
            "POST",
            "/api/v1/agents/chat/completions",
            Some("Bearer ragflow-alice"),
            Some(json!({"agent_id": "c1", "query": "hello"})),
        )
        .await;
        assert_eq!(s, 200);
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["code"], 0);
        assert_eq!(v["data"]["event"], "message_end");
        assert_eq!(v["data"]["data"]["content"], "you said hello (turn 1)");
        assert_eq!(v["data"]["data"]["usage"]["calls"], 0);
        assert_eq!(v["data"]["session_id"], v["data"]["task_id"]);
    }

    #[tokio::test]
    async fn a_failing_run_streams_as_a_message_on_the_chat_endpoint_and_as_an_error_on_the_bot_endpoint(
    ) {
        let (_, w) = app();
        {
            // The newest version fails when its Message renders.
            let mut broken = crate::testutil::dsl();
            broken["components"]["Message:a"]["obj"]["params"]["content"] = json!(["{sys.nope}"]);
            // World's rows are shared through Arc; rebuild an app over a changed one.
            let mut changed = world();
            Arc::get_mut(&mut changed).unwrap().base.versions[0].dsl = broken;
            let base = &mut Arc::get_mut(&mut changed).unwrap().base;
            base.users = w.base.users.clone();
            base.tokens = w.base.tokens.clone();
            let kv = Arc::new(FakeKv::default());
            let app2 = Arc::new(App {
                agent: agent(&changed, kv.clone()),
                facts: changed.clone(),
                sessions: changed.clone(),
                kv,
                catalog: Arc::new(Catalog::default()),
                secret_env: Some(SECRET.into()),
            });
            let (_, text) = send(
                &app2,
                "POST",
                "/api/v1/agents/chat/completions",
                Some("Bearer ragflow-alice"),
                Some(json!({"agent_id": "c1", "query": "hi", "stream": true})),
            )
            .await;
            let f = frames(&text);
            assert!(
                f.iter().any(|x| x["event"] == "message"
                    && x["data"]["content"]
                        .as_str()
                        .is_some_and(|c| c.contains("sys.nope"))),
                "{text}"
            );
            assert!(!f.iter().any(|x| x["code"] == 500));
            assert_eq!(f.last().unwrap(), &json!("[DONE]"));
            let (_, text) = send(
                &app2,
                "POST",
                "/api/v1/agentbots/c1/completions",
                Some("Bearer ragflow-alice"),
                Some(json!({"query": "hi"})),
            )
            .await;
            let f = frames(&text);
            let err = f.iter().find(|x| x["code"] == 500).expect("an error frame");
            assert!(err["message"].as_str().unwrap().contains("sys.nope"));
            assert_eq!(err["data"], false);
        }
    }

    #[tokio::test]
    async fn a_beta_token_runs_only_its_own_agent() {
        let (app, _) = app();
        let body = json!({"query": "hello"});
        let (_, bound) = send(
            &app,
            "POST",
            "/api/v1/agentbots/c1/completions",
            Some("Bearer beta-bound"),
            Some(body.clone()),
        )
        .await;
        assert_eq!(
            bound,
            r#"{"code":401,"data":null,"message":"API token is not authorized for this agent."}"#
        );
        let (_, own) = send(
            &app,
            "POST",
            "/api/v1/agentbots/c1/completions",
            Some("Bearer beta-own"),
            Some(body.clone()),
        )
        .await;
        assert!(own.contains("workflow_finished"), "{own}");
        // An API token has no binding.
        let (_, api) = send(
            &app,
            "POST",
            "/api/v1/agentbots/c1/completions",
            Some("Bearer ragflow-alice"),
            Some(body),
        )
        .await;
        assert!(api.contains("workflow_finished"), "{api}");
    }

    #[tokio::test]
    async fn requests_this_version_does_not_serve_are_refused_plainly() {
        let (app, w) = app();
        for body in [
            json!({"agent_id": "c1", "query": "x", "files": [{"id": "1"}]}),
            json!({"agent_id": "c1", "messages": [{"role": "user", "content": "x"}], "openai-compatible": true}),
        ] {
            let (s, text) = send(
                &app,
                "POST",
                "/api/v1/agents/chat/completions",
                Some("Bearer ragflow-alice"),
                Some(body),
            )
            .await;
            let v: Value = serde_json::from_str(&text).unwrap();
            assert_eq!((s, v["code"].as_i64()), (200, Some(102)), "{text}");
        }
        let (_, text) = send(
            &app,
            "POST",
            "/api/v1/agents/chat/completions",
            Some("Bearer ragflow-alice"),
            Some(json!({"query": "x"})),
        )
        .await;
        assert_eq!(
            text,
            r#"{"code":101,"data":null,"message":"`agent_id` is required."}"#
        );
        assert!(w.sessions.lock().unwrap().is_empty(), "nothing was written");
    }

    #[tokio::test]
    async fn cancel_replies_true_even_when_nothing_runs() {
        let (app, _) = app();
        let (s, text) = send(
            &app,
            "POST",
            "/api/v1/tasks/nothing/cancel",
            Some("Bearer ragflow-alice"),
            None,
        )
        .await;
        assert_eq!(
            (s, text.as_str()),
            (200, r#"{"code":0,"data":true,"message":"success"}"#)
        );
    }

    #[tokio::test]
    async fn sessions_can_be_made_listed_read_and_deleted() {
        let (app, _) = app();
        let t = Some("Bearer ragflow-alice");
        let (_, text) = send(
            &app,
            "POST",
            "/api/v1/agents/c1/sessions",
            t,
            Some(json!({"name": "mine"})),
        )
        .await;
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["code"], 0, "{text}");
        let id = v["data"]["id"].as_str().unwrap().to_string();
        assert_eq!(v["data"]["name"], "mine");
        let (_, text) = send(
            &app,
            "GET",
            &format!("/api/v1/agents/c1/sessions/{id}"),
            t,
            None,
        )
        .await;
        assert_eq!(
            serde_json::from_str::<Value>(&text).unwrap()["data"]["id"],
            id.as_str()
        );
        let (_, text) = send(&app, "GET", "/api/v1/agents/c1/sessions/nope", t, None).await;
        assert_eq!(text, r#"{"code":404,"message":"agent session not found"}"#);
        let (_, text) = send(
            &app,
            "DELETE",
            &format!("/api/v1/agents/c1/sessions/{id}"),
            t,
            None,
        )
        .await;
        assert_eq!(text, r#"{"code":0,"data":true,"message":"success"}"#);
        let (_, text) = send(&app, "GET", "/api/v1/agents/none/sessions", t, None).await;
        assert_eq!(
            text,
            r#"{"code":103,"message":"agent not found or no permission"}"#
        );
    }
}

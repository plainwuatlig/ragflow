//! `AgentService.RunAgent` and `CancelSessionRun`: everything between an
//! authenticated request and the stream of run events, following upstream's
//! `internal/service/agent.go` step for step.
//!
//! [`Agent::start`] does the synchronous part (access checks, the run lease,
//! the DSL to run, model validation, the session row, the saved question)
//! and returns a [`Run`] whose events arrive on a channel while a thread
//! executes the canvas. The thread persists the turn before the closing
//! events go out, releases the lease, and stops the watchers.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::{FixedOffset, Local, Offset};
use ragflow_runtime::retrieval::RetrievalService;
use ragflow_runtime::{
    compile, python_history_repr, Bot, Event, EventSink, Registry, RunError, RunInput,
};
use serde_json::{json, Map, Value};

use crate::dslops;
use crate::facts::{Canvas, Facts, StoreError, Version};
use crate::kv::{Active, Kv};
use crate::llm::ServerModels;
use crate::models::{Access, Catalog, ModelError, Resolver, CHAT};
use crate::perm::{self, Op, PermError};
use crate::retrieval::GuardedRetrieval;
use crate::sessions::SessionStore;
use crate::store::{HistoryUpdate, MessageRow, SessionUpdate};

/// Shown for a failure whose cause must not leave the server.
pub const INTERNAL_ERROR: &str = "Internal storage error while accessing the agent.";
const MODEL_UNAVAILABLE: &str =
    "The configured chat model is missing or unavailable. Please select a valid model.";
pub const CANCELLED: &str = "Agent run was cancelled.";

/// Why a run could not start (`mapAgentError`'s inputs).
#[derive(Debug, Clone, PartialEq)]
pub enum AgentError {
    /// `ErrUserCanvasNotFound` / `ErrUserCanvasVersionNotFound`.
    NotFound,
    /// A session id the caller may not run, or that does not exist.
    SessionNotFound(String),
    /// `ErrAgentNotOwner`.
    NotOwner,
    /// `ErrAgentStorageError`: logged, never shown.
    Storage(String),
    /// Anything else, shown as is.
    Message(String),
}

impl AgentError {
    /// `mapAgentError`: the response code and message.
    pub fn code_and_message(&self) -> (i64, String) {
        match self {
            AgentError::NotOwner => (
                103,
                "Only the owner of the agent is authorized for this operation.".into(),
            ),
            AgentError::NotFound | AgentError::SessionNotFound(_) => (
                103,
                "Make sure you have permission to access the agent.".into(),
            ),
            AgentError::Storage(_) => (500, INTERNAL_ERROR.into()),
            AgentError::Message(m) => (102, m.clone()),
        }
    }
}

impl From<StoreError> for AgentError {
    fn from(e: StoreError) -> Self {
        AgentError::Storage(e.0)
    }
}

impl From<crate::kv::KvError> for AgentError {
    fn from(e: crate::kv::KvError) -> Self {
        AgentError::Storage(e.0)
    }
}

fn perm_to_agent(e: PermError) -> AgentError {
    match e {
        PermError::NotFound | PermError::Denied => AgentError::NotFound,
        PermError::Invalid(m) | PermError::Store(m) => AgentError::Storage(m),
    }
}

/// One event of a run, in the shape the SSE layer writes.
#[derive(Debug, Clone, PartialEq)]
pub struct RunEvent {
    pub kind: String,
    pub data: Value,
    pub message_id: String,
    pub created_at: i64,
    pub session_id: String,
}

/// What a request asks of `RunAgent`.
#[derive(Debug, Clone)]
pub struct RunParams {
    pub user_id: String,
    pub canvas_id: String,
    /// The client's session id, if it sent one.
    pub session_id: Option<String>,
    /// Set when the handler minted `session_id` itself: the session row may
    /// not exist yet (`WithAgentSessionID`).
    pub trusted_first_touch: bool,
    /// A specific `user_canvas_version` to run.
    pub version: Option<String>,
    /// Run the newest released version (`release=true`).
    pub released: bool,
    /// `Null`, a string, or a form-style object.
    pub user_input: Value,
}

/// A started run.
pub struct Run {
    pub session_id: String,
    pub message_id: String,
    pub events: tokio::sync::mpsc::UnboundedReceiver<RunEvent>,
    /// Raised when the consumer goes away: the run is cancelled.
    pub cancel: Arc<AtomicBool>,
}

struct ActiveRun {
    user_id: String,
    token: String,
    cancel: Arc<AtomicBool>,
}

pub struct Agent {
    pub facts: Arc<dyn Facts>,
    pub sessions: Arc<dyn SessionStore>,
    pub kv: Arc<dyn Kv>,
    pub catalog: Arc<Catalog>,
    /// The retrieval backend, if this server was built with one.
    pub retrieval: Option<Arc<dyn RetrievalService>>,
    active: Mutex<HashMap<String, ActiveRun>>,
    /// How often a run looks for a competing lease holder to go away.
    pub lease_poll: Duration,
    /// How often the cancel marker is polled.
    pub cancel_poll: Duration,
    /// How often the lease is refreshed.
    pub lease_refresh: Duration,
}

fn new_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

fn now_secs() -> f64 {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    d.as_secs_f64()
}

fn now_unix() -> i64 {
    chrono::Utc::now().timestamp()
}

impl Agent {
    pub fn new(
        facts: Arc<dyn Facts>,
        sessions: Arc<dyn SessionStore>,
        kv: Arc<dyn Kv>,
        catalog: Arc<Catalog>,
        retrieval: Option<Arc<dyn RetrievalService>>,
    ) -> Self {
        Agent {
            facts,
            sessions,
            kv,
            catalog,
            retrieval,
            active: Mutex::new(HashMap::new()),
            lease_poll: Duration::from_millis(100),
            cancel_poll: crate::kv::CANCEL_POLL,
            lease_refresh: crate::kv::LEASE_TTL / 3,
        }
    }

    /// `validateAgentChatModels`: every Agent node's model must resolve for
    /// the caller inside the canvas owner's tenant.
    fn validate_models(
        &self,
        user_id: &str,
        tenant_id: &str,
        dsl: &Value,
    ) -> Result<(), AgentError> {
        let Some(components) = dsl.get("components").and_then(Value::as_object) else {
            return Ok(());
        };
        let resolver = Resolver {
            facts: self.facts.as_ref(),
            catalog: &self.catalog,
        };
        let access = Access {
            user_id: user_id.to_string(),
            tenant_id: tenant_id.to_string(),
        };
        for comp in components.values() {
            let (name, params) = component_name_and_params(comp);
            if !name.eq_ignore_ascii_case("Agent") {
                continue;
            }
            let model_ref = match params.get("model_id") {
                Some(Value::String(s)) => s.as_str(),
                _ => params.get("llm_id").and_then(Value::as_str).unwrap_or(""),
            };
            match resolver.resolve(&access, CHAT, model_ref) {
                Ok(_) => {}
                Err(ModelError::Unavailable(_)) => {
                    return Err(AgentError::Message(MODEL_UNAVAILABLE.into()))
                }
                Err(ModelError::Store(m)) => return Err(AgentError::Storage(m)),
            }
        }
        Ok(())
    }

    /// `RunAgent`.
    pub fn start(self: &Arc<Self>, p: RunParams) -> Result<Run, AgentError> {
        let received_at = now_secs();
        let canvas = perm::canvas_for(self.facts.as_ref(), &p.user_id, &p.canvas_id, Op::Run)
            .map_err(perm_to_agent)?;
        let tenant_id = canvas.user_id.clone();
        let new_session = p.session_id.as_deref().unwrap_or("").is_empty();
        let session_id = match &p.session_id {
            Some(s) if !s.is_empty() => s.clone(),
            _ => new_id(),
        };
        let query_text = dslops::stringify_input(&agent_run_query(&p.user_input));

        if !new_session {
            match self.facts.session_in_canvas(&session_id, &p.canvas_id)? {
                None if !p.trusted_first_touch => {
                    return Err(AgentError::SessionNotFound(session_id.clone()))
                }
                None => {}
                Some(existing) => {
                    perm::session_run(self.facts.as_ref(), &p.user_id, &existing)
                        .map_err(perm_to_agent)?;
                }
            }
        }

        let message_id = new_id();
        let persist_question = {
            let this = self.clone();
            let (canvas_id, user_id, session_id, message_id, text) = (
                p.canvas_id.clone(),
                p.user_id.clone(),
                session_id.clone(),
                message_id.clone(),
                query_text.clone(),
            );
            let saved = Mutex::new(false);
            move || -> Result<(), AgentError> {
                let mut saved = saved.lock().expect("saved flag");
                if *saved {
                    return Ok(());
                }
                let Some(session) = this.facts.session_in_canvas(&session_id, &canvas_id)? else {
                    return Ok(());
                };
                perm::session_run(this.facts.as_ref(), &user_id, &session)
                    .map_err(perm_to_agent)?;
                if !text.is_empty() {
                    this.sessions
                        .update_history(
                            &session_id,
                            &canvas_id,
                            &user_id,
                            &SessionUpdate::default(),
                            &HistoryUpdate {
                                message: Some(MessageRow {
                                    message_id: message_id.clone(),
                                    role: "user",
                                    content: text.clone(),
                                    created_at: received_at,
                                }),
                                ..Default::default()
                            },
                        )
                        .map_err(|e| AgentError::Storage(format!("persist question: {e}")))?;
                }
                *saved = true;
                Ok(())
            }
        };
        persist_question()?;

        // One run per session across every process: take the lease, and
        // while another run holds it keep trying every 100 ms.
        let token = new_id();
        let cancel = Arc::new(AtomicBool::new(false));
        let active = Active {
            session_id: session_id.clone(),
            token: token.clone(),
            user_id: p.user_id.clone(),
            canvas_id: p.canvas_id.clone(),
            run_id: format!("{}-{}", p.canvas_id, session_id),
        };
        loop {
            let registered = match self.kv.register(&active) {
                Ok(r) => r,
                Err(e) => {
                    let _ = self.kv.release(&session_id, &token);
                    return Err(AgentError::Storage(format!("register active session: {e}")));
                }
            };
            if registered {
                let mut local = self.active.lock().expect("active runs");
                if !local.contains_key(&session_id) {
                    local.insert(
                        session_id.clone(),
                        ActiveRun {
                            user_id: p.user_id.clone(),
                            token: token.clone(),
                            cancel: cancel.clone(),
                        },
                    );
                    break;
                }
                drop(local);
                let _ = self.kv.release(&session_id, &token);
            }
            persist_question()?;
            std::thread::sleep(self.lease_poll);
        }

        // From here every early return must give the lease back.
        let guard = LeaseGuard {
            agent: self.clone(),
            session_id: session_id.clone(),
            token: token.clone(),
            watchers: Arc::new(AtomicBool::new(false)),
            armed: true,
        };
        self.spawn_watchers(&guard, &cancel);
        if let Ok(true) = self.kv.cancel_requested(&session_id) {
            cancel.store(true, Ordering::SeqCst);
        }

        // The DSL to run: the requested version, else the newest saved one
        // (or the canvas's own when it has none), else the session's own
        // snapshot.
        let mut version_row: Option<Version> = None;
        let mut dsl: Value = Value::Null;
        if p.released {
            version_row = Some(
                self.facts
                    .latest_released_version(&p.canvas_id)?
                    .ok_or_else(|| {
                        AgentError::Message(format!(
                            "load latest released version for canvas {:?}: user_canvas_version: not found",
                            p.canvas_id
                        ))
                    })?,
            );
        } else if let Some(version_id) = p.version.as_deref().filter(|v| !v.is_empty()) {
            let row = self
                .facts
                .version(version_id)?
                .ok_or(AgentError::NotFound)?;
            if row.canvas_id != p.canvas_id {
                return Err(AgentError::NotFound);
            }
            version_row = Some(row);
        }
        if version_row.is_none() {
            match self.facts.latest_version(&p.canvas_id)? {
                Some(row) => version_row = Some(row),
                None => {
                    if is_non_empty(&canvas.dsl) {
                        dsl = dslops::normalize_for_run(&canvas.dsl);
                    }
                }
            }
        }
        if dsl.is_null() {
            if let Some(v) = version_row.as_ref().filter(|v| is_non_empty(&v.dsl)) {
                dsl = dslops::normalize_for_run(&v.dsl);
            }
        }
        let existing = self.facts.session_in_canvas(&session_id, &p.canvas_id)?;
        if let Some(session) = &existing {
            perm::session_run(self.facts.as_ref(), &p.user_id, session).map_err(perm_to_agent)?;
            if is_non_empty(&session.dsl) {
                dsl = dslops::normalize_for_run(&session.dsl);
            }
        }
        let session_found = existing.is_some();

        let fresh = !session_found || new_session;
        // upstream resets a new session's conversation state before it runs;
        // an empty DSL (a canvas never saved) takes the placeholder path.
        let run_dsl = if fresh {
            dslops::reset_for_new_session(&dsl)
        } else {
            dsl.clone()
        };
        // This server runs the components it implements; refuse the rest
        // before anything is written.
        let plan = if dsl.is_null() {
            None
        } else {
            Some(compile(&run_dsl.to_string()).map_err(|e| AgentError::Message(e.to_string()))?)
        };

        self.validate_models(&p.user_id, &tenant_id, &dsl)?;

        if fresh {
            self.sessions
                .create_session(
                    &session_id,
                    &dslops::session_name(&p.user_input),
                    &p.canvas_id,
                    &p.user_id,
                    Some("agent"),
                    &run_dsl,
                    version_row.as_ref().and_then(|v| v.title.as_deref()),
                )
                .map_err(|e| AgentError::Storage(format!("create session: {e}")))?;
        }
        persist_question()?;

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let job = Job {
            agent: self.clone(),
            tx,
            session_id: session_id.clone(),
            message_id: message_id.clone(),
            canvas: canvas.clone(),
            user_id: p.user_id.clone(),
            user_input: p.user_input.clone(),
            run_dsl,
            drop_on_failure: fresh && query_text.is_empty(),
            cancel: cancel.clone(),
            guard,
            plan,
            tenant_id,
        };
        std::thread::spawn(move || job.run());
        Ok(Run {
            session_id,
            message_id,
            events: rx,
            cancel,
        })
    }

    /// The lease keeper and the cancel poller of one run.
    fn spawn_watchers(self: &Arc<Self>, guard: &LeaseGuard, cancel: &Arc<AtomicBool>) {
        let stop = guard.watchers.clone();
        {
            let (this, stop, cancel) = (self.clone(), stop.clone(), cancel.clone());
            let sid = guard.session_id.clone();
            std::thread::spawn(move || {
                while !stop.load(Ordering::SeqCst) {
                    sleep_unless(&stop, this.cancel_poll);
                    if stop.load(Ordering::SeqCst) {
                        break;
                    }
                    // A failed read is skipped; the next poll tries again.
                    if let Ok(true) = this.kv.cancel_requested(&sid) {
                        cancel.store(true, Ordering::SeqCst);
                        break;
                    }
                }
            });
        }
        let (this, cancel) = (self.clone(), cancel.clone());
        let (sid, token) = (guard.session_id.clone(), guard.token.clone());
        std::thread::spawn(move || {
            let lease = crate::kv::LEASE_TTL;
            let mut last_confirmed = std::time::Instant::now();
            while !stop.load(Ordering::SeqCst) {
                sleep_unless(&stop, this.lease_refresh);
                if stop.load(Ordering::SeqCst) {
                    break;
                }
                match this.kv.refresh(&sid, &token) {
                    Ok(true) => last_confirmed = std::time::Instant::now(),
                    Ok(false) => {
                        cancel.store(true, Ordering::SeqCst);
                        break;
                    }
                    Err(_) if last_confirmed.elapsed() >= lease * 2 / 3 => {
                        cancel.store(true, Ordering::SeqCst);
                        break;
                    }
                    Err(_) => {}
                }
            }
        });
    }

    /// `CancelSessionRun`.
    pub fn cancel(&self, user_id: &str, session_id: &str) -> Result<(), AgentError> {
        let local = {
            let active = self.active.lock().expect("active runs");
            active
                .get(session_id)
                .map(|a| (a.user_id.clone(), a.token.clone(), a.cancel.clone()))
        };
        if let Some((owner, token, cancel)) = local {
            if owner != user_id {
                return Err(AgentError::NotOwner);
            }
            cancel.store(true, Ordering::SeqCst);
            // The marker lets other processes' watchers see it; the local
            // flag is already effective.
            let _ = self.kv.request_cancel(session_id, &token);
            return Ok(());
        }
        let remote = self
            .kv
            .active(session_id)
            .map_err(|e| AgentError::Storage(format!("read active session: {e}")))?;
        if let Some(remote) = remote {
            if !remote.user_id.is_empty() && remote.user_id != user_id {
                return Err(AgentError::NotOwner);
            }
            self.kv
                .request_cancel(session_id, &remote.token)
                .map_err(|e| AgentError::Storage(format!("publish remote marker: {e}")))?;
        }
        Ok(())
    }
}

fn sleep_unless(stop: &AtomicBool, total: Duration) {
    let step = Duration::from_millis(20).min(total);
    let mut slept = Duration::ZERO;
    while slept < total && !stop.load(Ordering::SeqCst) {
        std::thread::sleep(step);
        slept += step;
    }
}

/// `component_name` and `params` of a component in either stored shape.
pub fn component_name_and_params(comp: &Value) -> (String, Map<String, Value>) {
    let obj = comp.get("obj").unwrap_or(comp);
    let name = obj
        .get("component_name")
        .and_then(Value::as_str)
        .or_else(|| comp.get("name").and_then(Value::as_str))
        .unwrap_or_default()
        .to_string();
    let params = obj
        .get("params")
        .or_else(|| comp.get("params"))
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    (name, params)
}

fn is_non_empty(v: &Value) -> bool {
    matches!(v, Value::Object(m) if !m.is_empty())
}

/// `agentRunQuery`: a form input's `query`, else the input itself.
fn agent_run_query(user_input: &Value) -> Value {
    match user_input {
        Value::Object(m) => m.get("query").cloned().unwrap_or(Value::Null),
        other => other.clone(),
    }
}

/// Gives the lease back and stops the watchers, however the run ends.
struct LeaseGuard {
    agent: Arc<Agent>,
    session_id: String,
    token: String,
    watchers: Arc<AtomicBool>,
    armed: bool,
}

impl Drop for LeaseGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        self.watchers.store(true, Ordering::SeqCst);
        let _ = self.agent.kv.release(&self.session_id, &self.token);
        let mut local = self.agent.active.lock().expect("active runs");
        if local
            .get(&self.session_id)
            .is_some_and(|a| a.token == self.token)
        {
            local.remove(&self.session_id);
        }
    }
}

/// The thread that runs a canvas.
struct Job {
    agent: Arc<Agent>,
    tx: tokio::sync::mpsc::UnboundedSender<RunEvent>,
    session_id: String,
    message_id: String,
    canvas: Canvas,
    user_id: String,
    user_input: Value,
    run_dsl: Value,
    drop_on_failure: bool,
    cancel: Arc<AtomicBool>,
    #[allow(dead_code)]
    guard: LeaseGuard,
    plan: Option<ragflow_runtime::Plan>,
    tenant_id: String,
}

/// Forwards runtime events to the stream, adding the run's usage to
/// `workflow_finished`.
struct Forward {
    tx: tokio::sync::mpsc::UnboundedSender<RunEvent>,
    session_id: String,
    message_id: String,
    cancel: Arc<AtomicBool>,
    models: Arc<ServerModels>,
}

impl EventSink for Forward {
    fn emit(&mut self, event: Event) {
        let kind = event.name().to_string();
        let mut data = event.data();
        if matches!(event, Event::WorkflowFinished { .. }) {
            data["usage"] = self.models.usage().to_json();
        }
        let sent = self.tx.send(RunEvent {
            kind,
            data,
            message_id: self.message_id.clone(),
            created_at: now_unix(),
            session_id: self.session_id.clone(),
        });
        if sent.is_err() {
            // The client went away: stop the run.
            self.cancel.store(true, Ordering::SeqCst);
        }
    }
}

impl Job {
    fn send(&self, kind: &str, data: Value) {
        let _ = self.tx.send(RunEvent {
            kind: kind.into(),
            data,
            message_id: self.message_id.clone(),
            created_at: now_unix(),
            session_id: self.session_id.clone(),
        });
    }

    fn mark_failed(&self, reason: &str) {
        let _ = self.agent.sessions.update_history(
            &self.session_id,
            &self.canvas.id,
            &self.user_id,
            &SessionUpdate {
                errors: Some(Some(reason.to_string())),
                ..Default::default()
            },
            &HistoryUpdate::default(),
        );
    }

    fn drop_session(&self) {
        if self.drop_on_failure {
            let _ = self
                .agent
                .sessions
                .delete_session(&self.session_id, &self.canvas.id);
        }
    }

    fn run(mut self) {
        let Some(plan) = self.plan.take() else {
            // No DSL at all: the placeholder answer upstream gives.
            self.placeholder();
            return;
        };
        let models = Arc::new(ServerModels::new(
            self.agent.facts.clone(),
            self.agent.catalog.clone(),
            self.tenant_id.clone(),
            self.cancel.clone(),
        ));
        let mut registry = Registry::with_models(models.clone());
        if let Some(inner) = &self.agent.retrieval {
            registry = registry.with_retrieval(Arc::new(GuardedRetrieval {
                inner: inner.clone(),
                facts: self.agent.facts.clone(),
                user_id: self.user_id.clone(),
                canvas_id: self.canvas.id.clone(),
            }));
        }
        let bot = match Bot::new(plan, &registry) {
            Ok(b) => b,
            Err(e) => {
                self.mark_failed(&format!("compile: {e}"));
                self.send("error", json!({ "message": e.to_string() }));
                self.drop_session();
                return;
            }
        };

        let now = Local::now();
        let offset = FixedOffset::east_opt(now.offset().fix().local_minus_utc())
            .unwrap_or_else(|| FixedOffset::east_opt(0).expect("utc"));
        let mut input = RunInput::new("", now.with_timezone(&offset));
        input.query = match &self.user_input {
            Value::String(s) => s.clone(),
            _ => String::new(),
        };
        if let Value::Object(form) = &self.user_input {
            input.user_input = Some(form.clone());
        }
        input.ids = ragflow_runtime::SysIds {
            user_id: self.user_id.clone(),
            canvas_id: self.canvas.id.clone(),
            session_id: self.session_id.clone(),
            tenant_id: self.tenant_id.clone(),
        };
        input.cancel = Some(self.cancel.clone());

        // Persist the turn before the closing events, as upstream does.
        let persist = {
            let (agent, sid, cid, uid, mid) = (
                self.agent.clone(),
                self.session_id.clone(),
                self.canvas.id.clone(),
                self.user_id.clone(),
                self.message_id.clone(),
            );
            let (run_dsl, user_input) = (self.run_dsl.clone(), self.user_input.clone());
            move |f: ragflow_runtime::Finish<'_>| -> Result<(), String> {
                let answer = f
                    .answer
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let append_assistant = f.deferred_failure.is_none() || !f.visible_answer.is_empty();
                let stored = if f.deferred_failure.is_some() {
                    f.visible_answer.clone()
                } else {
                    answer.clone()
                };
                if f.deferred_failure.is_none() {
                    f.state
                        .append_history("assistant", partial_assistant_output(f.answer, &answer));
                    f.state.append_sys_history(format!(
                        "assistant: {}",
                        python_history_repr(&partial_assistant_output(f.answer, &answer))
                    ));
                }
                let has_question =
                    !dslops::stringify_input(&agent_run_query(&user_input)).is_empty();
                let reference = normalize_reference(f.reference.as_ref());
                let mut built = run_dsl.clone();
                dslops::apply_build_effects(&mut built);
                let update = SessionUpdate {
                    bump_round: true,
                    errors: Some(None),
                    dsl: Some(dslops::persisted_dsl(&built, f.state)),
                };
                let history = HistoryUpdate {
                    message: append_assistant.then(|| MessageRow {
                        message_id: mid.clone(),
                        role: "assistant",
                        content: stored,
                        created_at: now_secs(),
                    }),
                    question_id: has_question.then(|| mid.clone()),
                    reference: Some(reference),
                };
                agent
                    .sessions
                    .update_history(&sid, &cid, &uid, &update, &history)
                    .map_err(|e| e.to_string())
            }
        };
        input.finish = Some(Box::new(persist));

        let mut sink = Forward {
            tx: self.tx.clone(),
            session_id: self.session_id.clone(),
            message_id: self.message_id.clone(),
            cancel: self.cancel.clone(),
            models: models.clone(),
        };
        // The runtime reports its own failures as `error` events through
        // the sink; the text of each is kept for the session's `errors`.
        let result = bot.run(input, &mut sink);
        match result {
            Ok(outcome) => {
                if let Some(failure) = outcome.deferred_failure {
                    self.mark_failed(&format!("invoke: {failure}"));
                }
            }
            Err(RunError::Cancelled) => {
                self.send("cancelled", json!({ "message": CANCELLED }));
            }
            Err(RunError::Persist(reason)) => {
                self.mark_failed(&format!("persist session: {reason}"));
                self.send(
                    "error",
                    json!({ "message": INTERNAL_ERROR, "kind": "internal" }),
                );
            }
            Err(other) => {
                self.mark_failed(&format!("invoke: {other}"));
                self.drop_session();
            }
        }
    }

    /// A run with no DSL at all: the canned answer upstream gives.
    fn placeholder(&self) {
        let answer = format!(
            "No published version found for canvas {:?} \u{2014} publish a version before running.",
            self.canvas.id
        );
        let mut state = ragflow_runtime::RunState::default();
        state.set_conversation(&Value::Null, &Value::Null);
        let update = SessionUpdate {
            bump_round: true,
            errors: Some(None),
            dsl: Some(dslops::persisted_dsl(&self.run_dsl, &state)),
        };
        let has_question = !dslops::stringify_input(&agent_run_query(&self.user_input)).is_empty();
        let history = HistoryUpdate {
            message: Some(MessageRow {
                message_id: self.message_id.clone(),
                role: "assistant",
                content: answer.clone(),
                created_at: now_secs(),
            }),
            question_id: has_question.then(|| self.message_id.clone()),
            reference: Some(normalize_reference(None)),
        };
        if self
            .agent
            .sessions
            .update_history(
                &self.session_id,
                &self.canvas.id,
                &self.user_id,
                &update,
                &history,
            )
            .is_err()
        {
            self.send(
                "error",
                json!({ "message": INTERNAL_ERROR, "kind": "internal" }),
            );
            return;
        }
        self.send("message", json!({ "content": answer }));
        self.send("message_end", json!({}));
        self.send(
            "workflow_finished",
            json!({ "outputs": answer, "usage": crate::llm::Usage::default().to_json() }),
        );
    }
}

/// `partialAssistantOutput`: `{content, downloads?, attachment?}`.
fn partial_assistant_output(answer: &Map<String, Value>, content: &str) -> Value {
    let mut out = Map::new();
    out.insert("content".into(), Value::String(content.to_string()));
    let non_empty = |v: &Value| match v {
        Value::Null => false,
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
        _ => true,
    };
    if let Some(d) = answer.get("downloads").filter(|v| non_empty(v)) {
        out.insert("downloads".into(), d.clone());
    }
    if let Some(a) = answer.get("attachment").filter(|v| non_empty(v)) {
        out.insert("attachment".into(), a.clone());
    }
    Value::Object(out)
}

/// `normalizeAgentReferenceEntry`: only `chunks` and `doc_aggs` are stored.
pub fn normalize_reference(reference: Option<&Value>) -> Value {
    let list = |v: Option<&Value>| match v {
        None | Some(Value::Null) => Value::Array(Vec::new()),
        Some(other) => other.clone(),
    };
    match reference {
        None | Some(Value::Null) => json!({ "chunks": [], "doc_aggs": [] }),
        Some(r) => json!({ "chunks": list(r.get("chunks")), "doc_aggs": list(r.get("doc_aggs")) }),
    }
}

/// A channel receiver for blocking callers (tests).
pub fn collect_blocking(mut run: Run) -> Vec<RunEvent> {
    let mut out = Vec::new();
    while let Some(ev) = run.events.blocking_recv() {
        out.push(ev);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::facts::*;
    use crate::kv::fake::FakeKv;
    use crate::testutil::{agent, dsl, world, World};

    fn params(user: &str, session: Option<&str>, trusted: bool, query: &str) -> RunParams {
        RunParams {
            user_id: user.into(),
            canvas_id: "c1".into(),
            session_id: session.map(Into::into),
            trusted_first_touch: trusted,
            version: None,
            released: false,
            user_input: if query.is_empty() {
                Value::Null
            } else {
                json!(query)
            },
        }
    }

    #[test]
    fn a_first_touch_run_creates_the_session_saves_both_turns_and_closes_after_saving() {
        let w = world();
        let kv = Arc::new(FakeKv::default());
        let a = agent(&w, kv.clone());
        let run = a.start(params("alice", Some("s1"), true, "hello")).unwrap();
        assert_eq!(run.session_id, "s1");
        let events = collect_blocking(run);
        let kinds: Vec<_> = events.iter().map(|e| e.kind.as_str()).collect();
        assert_eq!(kinds.first(), Some(&"workflow_started"));
        assert_eq!(
            &kinds[kinds.len() - 3..],
            ["node_finished", "message_end", "workflow_finished"]
        );
        let finished = events.last().unwrap();
        assert_eq!(
            finished.data["usage"],
            json!({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0})
        );
        assert_eq!(finished.data["outputs"], "you said hello (turn 1)");
        // The question was saved before the run, the answer before the closing events.
        assert_eq!(*w.calls.lock().unwrap(), ["create", "question", "answer"]);
        let s = w.sessions.lock().unwrap()[0].clone();
        assert_eq!(
            (s.round, s.source.as_deref(), s.version_title.as_deref()),
            (1, Some("agent"), Some("v-title"))
        );
        assert_eq!(s.name.as_deref(), Some("hello"));
        assert_eq!(
            s.dsl["history"],
            json!([["user", "hello"], ["assistant", {"content": "you said hello (turn 1)"}]])
        );
        assert_eq!(s.dsl["globals"]["sys.conversation_turns"], 1);
        assert_eq!(
            s.dsl["globals"]["sys.history"],
            json!([
                "user: hello",
                "assistant: {'content': 'you said hello (turn 1)'}"
            ])
        );
        let m = w.messages.lock().unwrap().clone();
        assert_eq!(
            m.iter().map(|(_, p, r)| (*p, r.role)).collect::<Vec<_>>(),
            [(0, "user"), (1, "assistant")]
        );
        assert_eq!(m[0].2.message_id, m[1].2.message_id);
        let r = w.references.lock().unwrap().clone();
        assert_eq!((r[0].1, r[0].3), (1, Some(1)));
        assert_eq!(r[0].2, json!({"chunks": [], "doc_aggs": []}));
        // The lease is gone.
        std::thread::sleep(Duration::from_millis(50));
        assert!(kv.leases.lock().unwrap().is_empty());
    }

    #[test]
    fn a_second_turn_continues_the_snapshot_and_positions() {
        let w = world();
        let a = agent(&w, Arc::new(FakeKv::default()));
        collect_blocking(a.start(params("alice", Some("s1"), true, "one")).unwrap());
        std::thread::sleep(Duration::from_millis(50));
        let events = collect_blocking(a.start(params("alice", Some("s1"), false, "two")).unwrap());
        assert_eq!(
            events.last().unwrap().data["outputs"],
            "you said two (turn 2)"
        );
        let m = w.messages.lock().unwrap().clone();
        assert_eq!(
            m.iter().map(|(_, p, r)| (*p, r.role)).collect::<Vec<_>>(),
            [(0, "user"), (1, "assistant"), (3, "user"), (4, "assistant")]
        );
        let s = w.sessions.lock().unwrap()[0].clone();
        assert_eq!(s.round, 2);
        assert_eq!(s.dsl["history"].as_array().unwrap().len(), 4);
    }

    #[test]
    fn nobody_runs_a_session_that_is_not_theirs_or_does_not_exist() {
        let w = world();
        let a = agent(&w, Arc::new(FakeKv::default()));
        collect_blocking(a.start(params("alice", Some("s1"), true, "one")).unwrap());
        std::thread::sleep(Duration::from_millis(50));
        // A team member may run the canvas but not alice's session.
        assert!(a.start(params("bob", None, false, "hi")).is_ok());
        assert_eq!(
            a.start(params("bob", Some("s1"), false, "hi"))
                .err()
                .unwrap(),
            AgentError::NotFound
        );
        // A client cannot invent a session id.
        assert_eq!(
            a.start(params("alice", Some("made-up"), false, "hi"))
                .err()
                .unwrap(),
            AgentError::SessionNotFound("made-up".into())
        );
        // Strangers see no canvas.
        assert_eq!(
            a.start(params("eve", None, false, "hi")).err().unwrap(),
            AgentError::NotFound
        );
    }

    #[test]
    fn a_second_run_waits_for_the_lease() {
        let w = world();
        let kv = Arc::new(FakeKv::default());
        let a = agent(&w, kv.clone());
        kv.leases.lock().unwrap().insert(
            "s1".into(),
            Active {
                session_id: "s1".into(),
                token: "other".into(),
                user_id: "alice".into(),
                canvas_id: "c1".into(),
                run_id: "r".into(),
            },
        );
        let release = kv.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(150));
            release.leases.lock().unwrap().clear();
        });
        let started = std::time::Instant::now();
        let run = a.start(params("alice", Some("s1"), true, "hi")).unwrap();
        assert!(started.elapsed() >= Duration::from_millis(140));
        collect_blocking(run);
    }

    #[test]
    fn a_cancel_stops_a_run_waiting_on_a_model_and_leaves_only_the_question() {
        use std::net::TcpListener;
        // A model that never answers.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            let _held = listener.accept();
            std::thread::sleep(Duration::from_secs(10));
        });
        let w = world();
        let mut base = w.base.clone();
        base.member("alice", "alice", "owner");
        base.providers = vec![ModelProvider {
            id: "p".into(),
            tenant_id: "alice".into(),
            provider_name: "OpenAI".into(),
        }];
        base.instances = vec![ModelInstance {
            id: "i".into(),
            provider_id: "p".into(),
            instance_name: "dev".into(),
            api_key: "k".into(),
            status: "active".into(),
            extra: format!(r#"{{"base_url":"{url}"}}"#),
        }];
        let mut with_agent = dsl();
        with_agent["components"]["Agent:a"] = json!({"obj": {"component_name": "Agent", "params": {
            "llm_id": "m@dev@OpenAI", "sys_prompt": "be brief", "prompts": [{"role": "user", "content": "{sys.query}"}]}},
            "upstream": ["begin"], "downstream": ["Message:b"]});
        with_agent["components"]["Message:b"] = json!({"obj": {"component_name": "Message", "params": {"content": ["{Agent:a@content}"]}},
            "upstream": ["Agent:a"], "downstream": []});
        with_agent["components"]["begin"]["downstream"] = json!(["Message:a", "Agent:a"]);
        base.versions[0].dsl = with_agent;
        let w = Arc::new(World {
            base,
            ..Default::default()
        });
        let catalog = Catalog::from_values(&[json!({"name": "OpenAI", "url": {"default": url},
            "url_suffix": {"chat": "chat/completions"}, "class": "local",
            "models": [{"name": "m", "model_types": ["chat"]}]})])
        .unwrap();
        let kv = Arc::new(FakeKv::default());
        let mut a = Agent::new(w.clone(), w.clone(), kv.clone(), Arc::new(catalog), None);
        a.lease_poll = Duration::from_millis(10);
        a.cancel_poll = Duration::from_millis(10);
        let a = Arc::new(a);
        let run = a.start(params("alice", Some("s1"), true, "hello")).unwrap();
        let canceller = {
            let a = a.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(300));
                assert_eq!(a.cancel("bob", "s1"), Err(AgentError::NotOwner));
                a.cancel("alice", "s1").unwrap();
            })
        };
        let started = std::time::Instant::now();
        let events = collect_blocking(run);
        canceller.join().unwrap();
        assert!(started.elapsed() < Duration::from_secs(5));
        let last = events.last().unwrap();
        assert_eq!(
            (last.kind.as_str(), &last.data),
            ("cancelled", &json!({"message": CANCELLED}))
        );
        assert!(!events.iter().any(|e| e.kind == "workflow_finished"));
        let s = w.sessions.lock().unwrap()[0].clone();
        assert_eq!(s.round, 0, "a cancelled turn is not counted");
        assert_eq!(
            w.messages.lock().unwrap().len(),
            1,
            "only the question was saved"
        );
        std::thread::sleep(Duration::from_millis(100));
        assert!(kv.leases.lock().unwrap().is_empty());
        assert!(
            a.cancel("alice", "s1").is_ok(),
            "nothing is running any more"
        );
    }

    #[test]
    fn a_failed_save_ends_the_run_with_the_internal_message() {
        let w = world();
        *w.fail_persist.lock().unwrap() = true;
        let a = agent(&w, Arc::new(FakeKv::default()));
        let events = collect_blocking(a.start(params("alice", Some("s1"), true, "hi")).unwrap());
        let last = events.last().unwrap();
        assert_eq!(last.kind, "error");
        assert_eq!(last.data["kind"], "internal");
        assert!(!events.iter().any(|e| e.kind == "message_end"));
        // The question stays, marked failed.
        let s = w.sessions.lock().unwrap()[0].clone();
        assert!(s.errors.unwrap().starts_with("persist session:"));
    }

    #[test]
    fn an_empty_first_touch_that_fails_leaves_no_session() {
        let w = world();
        // A canvas whose Message cannot render: the run fails.
        let mut broken = dsl();
        broken["components"]["Message:a"]["obj"]["params"]["content"] =
            json!(["{Agent:nope@content}"]);
        // (unknown references render empty in display mode; use a strict failure instead)
        broken["components"]["Message:a"]["obj"]["params"]["content"] = json!(["{sys.nope}"]);
        let w = Arc::new(World {
            base: {
                let mut b = w.base.clone();
                b.versions[0].dsl = broken;
                b
            },
            ..Default::default()
        });
        let a = agent(&w, Arc::new(FakeKv::default()));
        let events = collect_blocking(a.start(params("alice", None, false, "")).unwrap());
        assert!(events.iter().any(|e| e.kind == "error"), "{events:?}");
        std::thread::sleep(Duration::from_millis(50));
        assert!(
            w.sessions.lock().unwrap().is_empty(),
            "the session was dropped"
        );
        assert!(w.calls.lock().unwrap().contains(&"delete".to_string()));
    }

    #[test]
    fn a_canvas_with_components_this_server_lacks_is_refused_up_front() {
        let w = world();
        let mut unsupported = dsl();
        unsupported["components"]["Code:x"] = json!({"obj": {"component_name": "CodeExec", "params": {}}, "upstream": ["begin"], "downstream": []});
        let w = Arc::new(World {
            base: {
                let mut b = w.base.clone();
                b.versions[0].dsl = unsupported;
                b
            },
            ..Default::default()
        });
        let a = agent(&w, Arc::new(FakeKv::default()));
        let err = a.start(params("alice", None, false, "hi")).err().unwrap();
        assert!(
            matches!(&err, AgentError::Message(m) if m.contains("CodeExec")),
            "{err:?}"
        );
        assert!(w.sessions.lock().unwrap().is_empty(), "nothing was written");
        assert!(w.calls.lock().unwrap().is_empty());
    }

    #[test]
    fn references_are_normalised_to_chunks_and_doc_aggs() {
        assert_eq!(
            normalize_reference(None),
            json!({"chunks": [], "doc_aggs": []})
        );
        assert_eq!(
            normalize_reference(Some(&json!({"chunks": [1], "doc_aggs": [2], "total": 1}))),
            json!({"chunks": [1], "doc_aggs": [2]})
        );
        assert_eq!(
            normalize_reference(Some(&json!({"chunks": [1]}))),
            json!({"chunks": [1], "doc_aggs": []})
        );
    }
}

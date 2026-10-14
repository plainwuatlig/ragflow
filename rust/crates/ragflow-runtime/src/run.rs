//! Runs a compiled canvas once.
//!
//! Scheduling follows upstream's eino workflow as far as it is observable:
//! a node runs once every upstream has finished or been skipped, and is
//! itself skipped when no upstream reached it. A Switch or Categorize
//! gates the children in its own `downstream`: of those, only the ones it
//! names in `_next` run (any other route is dropped); a child that is not
//! in its `downstream` is not gated. A run where no terminal component ran
//! fails, as upstream's workflow does when it runs out of tasks before END. A node's inputs are its data
//! upstream's outputs (see `Node::data_upstream`). Any component error ends
//! the run.
//!
//! Deferred Agents follow upstream's `DeferAgentToMessage`
//! (canvas/scheduler.go, component/agent.go and message.go on main): such
//! an Agent only starts when scheduled; the Message that references its
//! `content` runs it while rendering, and the Agent's `node_finished` is
//! emitted then, in template order. Agents nobody consumed finish at the end
//! of the run without calling a model. If a deferred Agent fails, the run
//! ends the way upstream's service does (`deferredAgentStreamFailureText`,
//! service/agent.go): the failure text becomes the reply, then
//! `message_end`, with no `error` and no `workflow_finished`.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use chrono::{DateTime, FixedOffset};
use serde_json::{Map, Value};

use crate::event::{Event, EventSink};
use crate::graph::{Graph, Node};
use crate::state::{Outputs, RunState, RESERVED_OUTPUT_KEYS};
use crate::Plan;

/// What a component sees while it runs.
pub struct NodeCtx<'a> {
    pub node: &'a Node,
    /// The data upstream's outputs, or the run input.
    pub inputs: &'a Outputs,
    pub state: &'a mut RunState,
    pub events: &'a mut dyn EventSink,
    /// Deferred Agents waiting for a Message to run them.
    pub deferred: &'a mut dyn Deferred,
    /// Set while a Message runs this Agent in place: the Agent streams its
    /// answer as `message` events as the model produces it.
    pub stream_messages: bool,
    /// Set by an Agent that streamed its answer, so the Message does not
    /// say it again.
    pub streamed: bool,
}

pub trait Component: Send + Sync {
    fn invoke(&self, ctx: &mut NodeCtx<'_>) -> Result<Outputs, String>;
}

/// The deferred Agents of one run.
pub trait Deferred {
    /// Whether `id` is a deferred Agent that has started but not run.
    fn is_pending(&self, id: &str) -> bool;
    /// Runs the pending Agent `id` now and finishes its node: its outputs
    /// land in `state` and its `node_finished` is emitted.
    fn open(
        &mut self,
        id: &str,
        state: &mut RunState,
        events: &mut dyn EventSink,
    ) -> Result<Opened, String>;
}

/// A deferred Agent that has run.
pub struct Opened {
    pub outputs: Outputs,
    /// Its answer was already emitted as `message` events while it arrived.
    pub streamed: bool,
}

/// No deferred Agents (what a deferred Agent itself sees).
pub struct NoDeferred;

impl Deferred for NoDeferred {
    fn is_pending(&self, _: &str) -> bool {
        false
    }
    fn open(
        &mut self,
        id: &str,
        _: &mut RunState,
        _: &mut dyn EventSink,
    ) -> Result<Opened, String> {
        Err(format!("{id} is not a deferred Agent"))
    }
}

struct Pending {
    inputs: Outputs,
    started: Instant,
}

struct DeferredAgents<'b> {
    graph: &'b Graph,
    components: &'b BTreeMap<String, Box<dyn Component>>,
    pending: BTreeMap<String, Pending>,
    /// The first deferred Agent error of the run.
    failure: Option<String>,
}

impl Deferred for DeferredAgents<'_> {
    fn is_pending(&self, id: &str) -> bool {
        self.pending.contains_key(id)
    }

    fn open(
        &mut self,
        id: &str,
        state: &mut RunState,
        events: &mut dyn EventSink,
    ) -> Result<Opened, String> {
        let Some(pending) = self.pending.get(id) else {
            return Err(format!("{id} is not a pending deferred Agent"));
        };
        let (inputs, started) = (pending.inputs.clone(), pending.started);
        let node = &self.graph.nodes[id];
        let streamed;
        let result = {
            let mut ctx = NodeCtx {
                node,
                inputs: &inputs,
                state: &mut *state,
                events: &mut *events,
                deferred: &mut NoDeferred,
                stream_messages: true,
                streamed: false,
            };
            let r = self.components[id].invoke(&mut ctx);
            streamed = ctx.streamed;
            r
        };
        match result {
            Err(e) => {
                // Stays pending: upstream finishes it with the leftovers.
                self.failure.get_or_insert_with(|| e.clone());
                Err(e)
            }
            Ok(outputs) => {
                self.pending.remove(id);
                let kept = sanitized(&outputs);
                state.outputs.insert(id.to_string(), kept.clone());
                events.emit(Event::NodeFinished {
                    component_id: id.to_string(),
                    component_name: node.name.to_string(),
                    inputs,
                    outputs: kept.clone(),
                    error: None,
                    elapsed_time: started.elapsed().as_secs_f64(),
                    created_at: unix_now(),
                });
                Ok(Opened {
                    outputs: kept,
                    streamed,
                })
            }
        }
    }
}

impl DeferredAgents<'_> {
    /// `CompleteAllDeferredNodes`: the Agents nobody ran finish with no
    /// outputs (in id order; upstream's order is its map order).
    fn flush(&mut self, events: &mut dyn EventSink) {
        for (id, pending) in std::mem::take(&mut self.pending) {
            events.emit(Event::NodeFinished {
                component_id: id.clone(),
                component_name: self.graph.nodes[&id].name.to_string(),
                inputs: pending.inputs,
                outputs: Outputs::new(),
                error: None,
                elapsed_time: pending.started.elapsed().as_secs_f64(),
                created_at: unix_now(),
            });
        }
    }
}

/// Counts the `message` events that pass through.
struct Tracking<'a> {
    inner: &'a mut dyn EventSink,
    messages: usize,
}

impl EventSink for Tracking<'_> {
    fn emit(&mut self, event: Event) {
        if matches!(event, Event::Message { .. }) {
            self.messages += 1;
        }
        self.inner.emit(event);
    }
}

pub type Factory = Box<dyn Fn(&Node) -> Result<Box<dyn Component>, String> + Send + Sync>;

/// Component implementations by name.
#[derive(Default)]
pub struct Registry {
    factories: BTreeMap<&'static str, Factory>,
}

impl Registry {
    pub fn register<F>(&mut self, name: &'static str, factory: F) -> &mut Self
    where
        F: Fn(&Node) -> Result<Box<dyn Component>, String> + Send + Sync + 'static,
    {
        self.factories.insert(name, Box::new(factory));
        self
    }

    /// The components that need nothing from outside the runtime.
    pub fn with_core() -> Self {
        let mut r = Registry::default();
        r.register("Begin", crate::components::begin::factory)
            .register("Message", crate::components::message::factory)
            .register("Switch", crate::components::switch::factory);
        r
    }

    /// The core components plus Categorize and Agent, which call `models`.
    pub fn with_models(models: std::sync::Arc<dyn crate::models::Models>) -> Self {
        let mut r = Registry::with_core();
        let m = models.clone();
        r.register("Categorize", move |n: &Node| {
            crate::components::categorize::factory(n, m.clone())
        });
        let m = models;
        r.register("Agent", move |n: &Node| {
            crate::components::agent::factory(n, m.clone())
        });
        r
    }

    /// Adds the Retrieval component, searching through `service`.
    pub fn with_retrieval(
        mut self,
        service: std::sync::Arc<dyn crate::retrieval::RetrievalService>,
    ) -> Self {
        self.register("Retrieval", move |n: &Node| {
            crate::components::retrieval::factory(n, service.clone())
        });
        self
    }
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum BuildError {
    #[error("no implementation registered for {name} (component {id})")]
    NotRegistered { id: String, name: String },
    #[error("component {id}: {message}")]
    Params { id: String, message: String },
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum RunError {
    #[error("canvas: component {id:?} invoke: {message}")]
    Component { id: String, message: String },
    /// The run's cancel flag was raised; no event is emitted for it.
    #[error("context canceled")]
    Cancelled,
    /// The finish hook (the service's persistence) failed.
    #[error("{0}")]
    Persist(String),
    #[error("turn budget exhausted before component {next:?}")]
    Deadline { next: String },
    /// No terminal component ran: upstream's workflow runs out of tasks
    /// before END (eino `no tasks to execute`), and the service rewrites
    /// the error for a Switch (`canvasInvokeError`, service/agent.go).
    #[error("{message}")]
    NoTerminal { last: String, message: String },
}

/// The identities the service puts into `sys.*` before a run
/// (`sys.user_id`, `sys.canvas_id`, `sys.session_id`, `sys.tenant_id`).
#[derive(Debug, Default, Clone)]
pub struct SysIds {
    pub user_id: String,
    pub canvas_id: String,
    pub session_id: String,
    pub tenant_id: String,
}

/// What the service sees when the components are done and the closing events
/// are still unsent (upstream persists the turn at this point).
pub struct Finish<'a> {
    /// The terminal component's outputs (`{}` when none).
    pub answer: &'a Outputs,
    /// The run's state; the hook records the assistant turn in its history.
    pub state: &'a mut RunState,
    /// `agentRunReferencePayload`: the citation payload, when there is one.
    pub reference: Option<Value>,
    /// The reply as the client reads it: the answer, or a failed deferred
    /// Agent's error text when that is what is shown.
    pub visible_answer: String,
    /// Set when a deferred Agent failed and its text became the reply.
    pub deferred_failure: Option<&'a str>,
}

pub type FinishHook = Box<dyn FnOnce(Finish<'_>) -> Result<(), String> + Send>;

pub struct RunInput {
    pub query: String,
    pub now: DateTime<FixedOffset>,
    pub deadline: Option<Instant>,
    pub ids: SysIds,
    /// A form-style user input (`{"query": ..., <field>: ...}`): it replaces
    /// the `{"query": <query>}` workflow input, and its `query` becomes
    /// `sys.query`.
    pub user_input: Option<Map<String, Value>>,
    /// Raised by whoever cancels the run; checked between components.
    pub cancel: Option<Arc<AtomicBool>>,
    /// Runs once before the closing events (`message_end`,
    /// `workflow_finished`) are emitted.
    pub finish: Option<FinishHook>,
}

impl RunInput {
    pub fn new(query: &str, now: DateTime<FixedOffset>) -> Self {
        RunInput {
            query: query.to_string(),
            now,
            deadline: None,
            ids: SysIds::default(),
            user_input: None,
            cancel: None,
            finish: None,
        }
    }
}

#[derive(Debug)]
pub struct RunOutcome {
    /// The first terminal component (by id) that produced outputs.
    pub answer: Outputs,
    pub state: RunState,
    /// A deferred Agent failed and its error text was shown as the reply
    /// (upstream ends such a run without `workflow_finished`).
    pub deferred_failure: Option<String>,
}

/// A canvas with every component instantiated, ready to serve turns.
pub struct Bot {
    plan: Plan,
    components: BTreeMap<String, Box<dyn Component>>,
}

#[derive(Clone, Copy, PartialEq)]
enum Status {
    Done,
    Skipped,
}

impl Bot {
    pub fn new(plan: Plan, registry: &Registry) -> Result<Self, BuildError> {
        let mut components = BTreeMap::new();
        for node in plan.graph.nodes.values() {
            let factory =
                registry
                    .factories
                    .get(node.name)
                    .ok_or_else(|| BuildError::NotRegistered {
                        id: node.id.clone(),
                        name: node.name.to_string(),
                    })?;
            let component = factory(node).map_err(|message| BuildError::Params {
                id: node.id.clone(),
                message,
            })?;
            components.insert(node.id.clone(), component);
        }
        Ok(Bot { plan, components })
    }

    pub fn graph(&self) -> &Graph {
        &self.plan.graph
    }

    /// `agentRunReferencePayload`: the run's citations, without the document
    /// aggregates when an Agent ran without `cite` (upstream drops them).
    fn reference(&self, state: &RunState) -> Option<Value> {
        let mut reference = crate::components::retrieval::retrieval_reference(state)?;
        let agent_ran =
            self.plan.graph.nodes.values().any(|n| {
                n.name == "Agent" && state.outputs.get(&n.id).is_some_and(|o| !o.is_empty())
            });
        let cite = self.plan.graph.nodes.values().any(|n| {
            n.name == "Agent"
                && state.outputs.get(&n.id).is_some_and(|o| !o.is_empty())
                && n.params.get("cite").and_then(Value::as_bool) == Some(true)
        });
        if agent_ran && !cite {
            if let Some(map) = reference.as_object_mut() {
                map.remove("doc_aggs");
            }
        }
        Some(reference)
    }

    pub fn run(
        &self,
        mut input: RunInput,
        events: &mut dyn EventSink,
    ) -> Result<RunOutcome, RunError> {
        let started = Instant::now();
        let finish = input.finish.take();
        let graph = &self.plan.graph;
        let mut events = Tracking {
            inner: events,
            messages: 0,
        };
        let cancelled = || {
            input
                .cancel
                .as_ref()
                .is_some_and(|c| c.load(Ordering::SeqCst))
        };
        let mut state = RunState::seed(&self.plan.dsl.globals, &input.query, input.now);
        // service/agent.go buildRunFunc: the conversation comes from the
        // session DSL, the current turn is appended to it, and the service
        // identities land in sys.*.
        state.set_conversation(&self.plan.dsl.history, &self.plan.dsl.memory);
        let (user_input, query_value, mut start_input) = match &input.user_input {
            Some(form) => (
                Value::Object(form.clone()),
                form.get("query").cloned().unwrap_or(Value::Null),
                form.clone(),
            ),
            None => {
                let mut start = Map::new();
                start.insert("query".into(), Value::String(input.query.clone()));
                (
                    Value::String(input.query.clone()),
                    Value::String(input.query.clone()),
                    start,
                )
            }
        };
        state.sys.insert("query".into(), query_value.clone());
        state.append_current_user(query_value.clone());
        state.append_sys_history(format!(
            "user: {}",
            crate::state::render_user_history_value(&query_value)
        ));
        for (key, value) in [
            ("user_id", &input.ids.user_id),
            ("canvas_id", &input.ids.canvas_id),
            ("session_id", &input.ids.session_id),
            ("tenant_id", &input.ids.tenant_id),
        ] {
            if !value.is_empty() {
                state.sys.insert(key.into(), Value::String(value.clone()));
            }
        }
        let _ = &mut start_input;
        events.emit(Event::WorkflowStarted {
            inputs: user_input.clone(),
        });

        let mut deferred = DeferredAgents {
            graph,
            components: &self.components,
            pending: BTreeMap::new(),
            failure: None,
        };
        // An eager Agent's answer is never streamed (upstream's
        // SuppressAgentMessageEvents); it decides whether the answer is
        // re-emitted at the end.
        let mut suppressed = false;
        // The last scheduled component to finish (eino's "last completed").
        let mut last_finished = String::new();
        let mut status: BTreeMap<&str, Status> = BTreeMap::new();
        let mut raw: BTreeMap<&str, Outputs> = BTreeMap::new();
        for id in &graph.order {
            let node = &graph.nodes[id];
            let reached = node.upstream.is_empty()
                || node.upstream.iter().any(|up| {
                    status.get(up.as_str()) == Some(&Status::Done)
                        && routes_to(&graph.nodes[up], &raw[up.as_str()], id)
                });
            if !reached {
                status.insert(id, Status::Skipped);
                continue;
            }
            if cancelled() {
                return Err(RunError::Cancelled);
            }
            if let Some(deadline) = input.deadline {
                if Instant::now() >= deadline {
                    let err = RunError::Deadline { next: id.clone() };
                    events.emit(Event::Error {
                        message: err.to_string(),
                    });
                    return Err(err);
                }
            }

            let empty = Outputs::new();
            let inputs = match &node.data_upstream {
                None => &start_input,
                Some(up) => raw.get(up.as_str()).unwrap_or(&empty),
            };
            let inputs = sanitized(inputs);
            events.emit(Event::NodeStarted {
                component_id: id.clone(),
                component_name: node.name.to_string(),
                inputs: inputs.clone(),
                created_at: unix_now(),
            });
            if node.defers_to_message {
                deferred.pending.insert(
                    id.clone(),
                    Pending {
                        inputs,
                        started: Instant::now(),
                    },
                );
                raw.insert(id, Outputs::new());
                status.insert(id, Status::Done);
                continue;
            }
            let node_started = Instant::now();
            let result = {
                let mut ctx = NodeCtx {
                    node,
                    inputs: &inputs,
                    state: &mut state,
                    events: &mut events,
                    deferred: &mut deferred,
                    stream_messages: false,
                    streamed: false,
                };
                self.components[id].invoke(&mut ctx)
            };
            let outputs = match result {
                Ok(outputs) => outputs,
                Err(_) if cancelled() => return Err(RunError::Cancelled),
                Err(message) => {
                    if let Some(failure) = deferred.failure.take() {
                        let answer = terminal_answer(graph, &state);
                        deferred.flush(&mut events);
                        let emitted = events.messages > 0;
                        let shown = emitted || !suppressed;
                        let mut visible = answer_text(&answer);
                        if visible.is_empty() && !emitted && shown {
                            visible = failure.trim().to_string();
                            events.emit(Event::Message {
                                content: visible.clone(),
                            });
                        }
                        let reference = self.reference(&state);
                        if let Some(hook) = finish {
                            hook(Finish {
                                answer: &answer,
                                state: &mut state,
                                reference: reference.clone(),
                                visible_answer: visible,
                                deferred_failure: Some(failure.as_str()),
                            })
                            .map_err(RunError::Persist)?;
                        }
                        if shown {
                            events.emit(Event::MessageEnd { reference });
                        }
                        return Ok(RunOutcome {
                            answer,
                            state,
                            deferred_failure: Some(failure),
                        });
                    }
                    let err = RunError::Component {
                        id: id.clone(),
                        message,
                    };
                    // The service completes leftover deferred Agents before
                    // it reports the error.
                    deferred.flush(&mut events);
                    events.emit(Event::Error {
                        message: err.to_string(),
                    });
                    return Err(err);
                }
            };
            if node.name == "Agent" {
                suppressed = true;
            }
            let kept = sanitized(&outputs);
            state.outputs.insert(id.clone(), kept.clone());
            events.emit(Event::NodeFinished {
                component_id: id.clone(),
                component_name: node.name.to_string(),
                inputs,
                outputs: kept,
                error: None,
                elapsed_time: node_started.elapsed().as_secs_f64(),
                created_at: unix_now(),
            });
            raw.insert(id, outputs);
            status.insert(id, Status::Done);
            last_finished = id.clone();
        }

        let answer = terminal_answer(graph, &state);
        deferred.flush(&mut events);
        let terminal_ran = graph
            .nodes
            .values()
            .any(|n| n.downstream.is_empty() && status.get(n.id.as_str()) == Some(&Status::Done));
        if !terminal_ran {
            let message = if last_finished.starts_with("Switch:") {
                "agent invoke: Switch routing stopped because no connected branch matched the condition; check the Switch branches".to_string()
            } else {
                format!(
                    "agent invoke: [GraphRunError] no tasks to execute, last completed nodes: [{last_finished}]"
                )
            };
            events.emit(Event::Error {
                message: message.clone(),
            });
            return Err(RunError::NoTerminal {
                last: last_finished,
                message,
            });
        }
        let reference = self.reference(&state);
        if let Some(hook) = finish {
            hook(Finish {
                answer: &answer,
                state: &mut state,
                reference: reference.clone(),
                visible_answer: answer_text(&answer),
                deferred_failure: None,
            })
            .map_err(RunError::Persist)?;
        }
        // service/agent.go: the answer is emitted after the fact only when
        // nothing was shown live and no eager Agent suppressed it.
        let emitted = events.messages > 0;
        if !emitted && !suppressed {
            events.emit(Event::Message {
                content: answer_text(&answer),
            });
        }
        if emitted || !suppressed {
            events.emit(Event::MessageEnd { reference });
        }
        events.emit(Event::WorkflowFinished {
            query: user_input,
            outputs: workflow_outputs(&answer),
            elapsed_time: started.elapsed().as_secs_f64(),
            created_at: unix_now(),
        });
        Ok(RunOutcome {
            answer,
            state,
            deferred_failure: None,
        })
    }
}

/// The first terminal (by id) with outputs.
fn terminal_answer(graph: &Graph, state: &RunState) -> Outputs {
    graph
        .nodes
        .values()
        .filter(|n| n.downstream.is_empty())
        .find_map(|n| state.outputs.get(&n.id).filter(|o| !o.is_empty()).cloned())
        .unwrap_or_default()
}

fn answer_text(answer: &Outputs) -> String {
    answer
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// `workflow_finished.outputs` as upstream shapes it
/// (`workflowOutputsFromTerminal`, service/agent.go): the answer's `content`
/// as a bare string, or `{content, downloads?, attachment?}` when either of
/// those is non-empty; `""` when there is no answer.
fn workflow_outputs(answer: &Outputs) -> Value {
    if answer.is_empty() {
        return Value::String(String::new());
    }
    let content = answer
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let present = |key: &str| {
        answer.get(key).filter(|v| match v {
            Value::Null => false,
            Value::Array(a) => !a.is_empty(),
            Value::Object(m) => !m.is_empty(),
            _ => true,
        })
    };
    let (downloads, attachment) = (present("downloads"), present("attachment"));
    if downloads.is_none() && attachment.is_none() {
        return Value::String(content);
    }
    let mut out = Map::new();
    out.insert("content".into(), Value::String(content));
    if let Some(d) = downloads {
        out.insert("downloads".into(), d.clone());
    }
    if let Some(a) = attachment {
        out.insert("attachment".into(), a.clone());
    }
    Value::Object(out)
}

/// Whether a finished `parent` reaches `child`.
fn routes_to(parent: &Node, outputs: &Outputs, child: &str) -> bool {
    if !parent.branching {
        return true;
    }
    let chosen = match outputs.get("_next") {
        Some(Value::String(one)) => one == child,
        Some(Value::Array(many)) => many.iter().any(|v| v.as_str() == Some(child)),
        _ => false,
    };
    // Only children in the router's `downstream` sit behind its branch
    // (multibranch.go); a child that lists the router as upstream without
    // being in its `downstream` has a plain edge and always runs.
    chosen || !parent.downstream.iter().any(|d| d == child)
}

fn sanitized(outputs: &Outputs) -> Outputs {
    outputs
        .iter()
        .filter(|(k, _)| !RESERVED_OUTPUT_KEYS.contains(&k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or_default()
}

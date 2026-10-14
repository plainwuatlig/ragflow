//! Runs a compiled canvas once.
//!
//! Scheduling follows upstream's eino workflow as far as it is observable:
//! a node runs once every upstream has finished or been skipped, and is
//! itself skipped when no upstream reached it. A Switch or Categorize
//! reaches only the children it names in `_next` that are also in its own
//! `downstream`; any other route is dropped. A node's inputs are its data
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
    ) -> Result<Outputs, String>;
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
    ) -> Result<Outputs, String> {
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
    ) -> Result<Outputs, String> {
        let Some(pending) = self.pending.get(id) else {
            return Err(format!("{id} is not a pending deferred Agent"));
        };
        let (inputs, started) = (pending.inputs.clone(), pending.started);
        let node = &self.graph.nodes[id];
        let result = {
            let mut ctx = NodeCtx {
                node,
                inputs: &inputs,
                state: &mut *state,
                events: &mut *events,
                deferred: &mut NoDeferred,
            };
            self.components[id].invoke(&mut ctx)
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
                Ok(kept)
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
    #[error("turn budget exhausted before component {next:?}")]
    Deadline { next: String },
}

pub struct RunInput {
    pub query: String,
    pub now: DateTime<FixedOffset>,
    pub deadline: Option<Instant>,
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

    pub fn run(&self, input: RunInput, events: &mut dyn EventSink) -> Result<RunOutcome, RunError> {
        let started = Instant::now();
        let graph = &self.plan.graph;
        let mut events = Tracking {
            inner: events,
            messages: 0,
        };
        let mut state = RunState::seed(&self.plan.dsl.globals, &input.query, input.now);
        let mut start_input = Map::new();
        start_input.insert("query".into(), Value::String(input.query.clone()));
        events.emit(Event::WorkflowStarted {
            inputs: Value::String(input.query.clone()),
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
                };
                self.components[id].invoke(&mut ctx)
            };
            let outputs = match result {
                Ok(outputs) => outputs,
                Err(message) => {
                    if let Some(failure) = deferred.failure.take() {
                        let answer = terminal_answer(graph, &state);
                        deferred.flush(&mut events);
                        let emitted = events.messages > 0;
                        let shown = emitted || !suppressed;
                        if answer_text(&answer).is_empty() && !emitted && shown {
                            events.emit(Event::Message {
                                content: failure.trim().to_string(),
                            });
                        }
                        if shown {
                            events.emit(Event::MessageEnd);
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
        }

        let answer = terminal_answer(graph, &state);
        deferred.flush(&mut events);
        // service/agent.go: the answer is emitted after the fact only when
        // nothing was shown live and no eager Agent suppressed it.
        let emitted = events.messages > 0;
        if !emitted && !suppressed {
            events.emit(Event::Message {
                content: answer_text(&answer),
            });
        }
        if emitted || !suppressed {
            events.emit(Event::MessageEnd);
        }
        events.emit(Event::WorkflowFinished {
            query: Value::String(input.query),
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
    chosen && parent.downstream.iter().any(|d| d == child)
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

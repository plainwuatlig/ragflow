//! Events one run emits, in upstream's names and `data` shapes, so the
//! conformance suite can compare them with the goldens recorded from the Go runtime.

use serde_json::{json, Value};

use crate::state::Outputs;

#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    WorkflowStarted {
        inputs: Value,
    },
    NodeStarted {
        component_id: String,
        component_name: String,
        inputs: Outputs,
        created_at: f64,
    },
    NodeFinished {
        component_id: String,
        component_name: String,
        inputs: Outputs,
        outputs: Outputs,
        error: Option<String>,
        elapsed_time: f64,
        created_at: f64,
    },
    Message {
        content: String,
    },
    /// `reference` is the run's citation payload (`{chunks, doc_aggs,
    /// total}`) when Retrieval recorded any, else `null` on the wire.
    MessageEnd {
        reference: Option<Value>,
    },
    WorkflowFinished {
        query: Value,
        outputs: Value,
        elapsed_time: f64,
        created_at: f64,
    },
    Error {
        message: String,
    },
}

impl Event {
    pub fn name(&self) -> &'static str {
        match self {
            Event::WorkflowStarted { .. } => "workflow_started",
            Event::NodeStarted { .. } => "node_started",
            Event::NodeFinished { .. } => "node_finished",
            Event::Message { .. } => "message",
            Event::MessageEnd { .. } => "message_end",
            Event::WorkflowFinished { .. } => "workflow_finished",
            Event::Error { .. } => "error",
        }
    }

    /// The `data` object of the event's frame.
    pub fn data(&self) -> Value {
        match self {
            Event::WorkflowStarted { inputs } => json!({ "inputs": inputs }),
            Event::NodeStarted {
                component_id,
                component_name,
                inputs,
                created_at,
            } => json!({
                "inputs": inputs,
                "created_at": created_at,
                "component_id": component_id,
                "component_name": component_name,
                "component_type": component_name,
                "thoughts": "",
            }),
            Event::NodeFinished {
                component_id,
                component_name,
                inputs,
                outputs,
                error,
                elapsed_time,
                created_at,
            } => json!({
                "inputs": inputs,
                "outputs": if outputs.is_empty() { Value::Null } else { Value::Object(outputs.clone()) },
                "component_id": component_id,
                "component_name": component_name,
                "component_type": component_name,
                "error": error,
                "elapsed_time": elapsed_time,
                "created_at": created_at,
            }),
            Event::Message { content } => json!({ "content": content }),
            // upstream stores the payload in an interface field, so a run
            // without citations still writes `"reference": null`.
            Event::MessageEnd { reference } => json!({ "reference": reference }),
            Event::WorkflowFinished {
                query,
                outputs,
                elapsed_time,
                created_at,
            } => json!({
                "inputs": { "query": query },
                "outputs": outputs,
                "elapsed_time": elapsed_time,
                "created_at": created_at,
            }),
            Event::Error { message } => json!({ "message": message }),
        }
    }
}

/// Receives events as the run produces them.
pub trait EventSink {
    fn emit(&mut self, event: Event);
}

impl EventSink for Vec<Event> {
    fn emit(&mut self, event: Event) {
        self.push(event);
    }
}

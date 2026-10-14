//! A Rust runtime for RAGFlow agent canvases (DSL v2).
//!
//! [`compile`] turns a canvas into a [`Plan`], refusing any component
//! outside [`SUPPORTED`] by name; [`Bot`] instantiates the plan's
//! components and runs turns.

pub mod components;
mod dsl;
mod event;
mod graph;
pub mod models;
pub mod retrieval;
mod run;
mod state;
pub mod template;

pub use dsl::{Component as DslComponent, ComponentObj, Dsl};
pub use event::{Event, EventSink};
pub use graph::{CompileError, Graph, Node, SUPPORTED};
pub use run::{
    Bot, BuildError, Component, Deferred, Factory, Finish, FinishHook, NoDeferred, NodeCtx, Opened,
    Registry, RunError, RunInput, RunOutcome, SysIds,
};
pub use state::{
    decode_history, decode_memory, encode_history, encode_memory, history_content,
    python_history_repr, render_user_history_value, Outputs, RunState,
};

/// A canvas that passed compilation.
#[derive(Debug)]
pub struct Plan {
    pub begin: String,
    pub dsl: Dsl,
    pub graph: Graph,
}

pub fn compile(json: &str) -> Result<Plan, CompileError> {
    let dsl: Dsl =
        serde_json::from_str(json).map_err(|e| CompileError::Malformed(e.to_string()))?;
    let graph = graph::build(&dsl)?;
    Ok(Plan {
        begin: graph.begin.clone(),
        dsl,
        graph,
    })
}

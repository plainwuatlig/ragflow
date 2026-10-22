//! Compiles a canvas into a DAG the scheduler can run.
//!
//! Edges come from each node's `upstream` list, as upstream's
//! `BuildWorkflow` (canvas/scheduler.go, main) builds them: the first
//! upstream that is not a Message carries data into the node, any further
//! upstreams only make it wait. An edge leaving a Message is ordering only
//! (`messageEdgeIsOrderingOnly`); a node whose upstreams are all Messages
//! reads the workflow input. `downstream` decides terminals, which children
//! a Switch or Categorize may route to, and whether an Agent defers its
//! answer to a Message (`directMessageDownstream`).

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

use crate::dsl::Dsl;

/// Components this runtime implements.
pub const SUPPORTED: &[&str] = &[
    "Begin",
    "Categorize",
    "Retrieval",
    "Switch",
    "Agent",
    "Message",
    "CodeExec",
];

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum CompileError {
    #[error("dsl is not valid JSON for a canvas: {0}")]
    Malformed(String),
    #[error("component {id} is {name}, which this runtime does not implement")]
    Unsupported { id: String, name: String },
    #[error("component {from} points to {to}, which is not in the canvas")]
    DanglingEdge { from: String, to: String },
    #[error("component {id} lists {upstream} as upstream, which is not in the canvas")]
    UnknownUpstream { id: String, upstream: String },
    #[error("component {0} lists itself as upstream")]
    SelfEdge(String),
    #[error("canvas has a cycle through {0:?}")]
    Cycle(Vec<String>),
    #[error("canvas has no Begin component")]
    NoBegin,
}

#[derive(Debug)]
pub struct Node {
    pub id: String,
    /// What run errors call the component (see [`Dsl::display_name`]).
    pub label: String,
    /// The component name in this runtime's spelling (upstream matches
    /// names case-insensitively).
    pub name: &'static str,
    pub params: Value,
    /// Deduplicated, in DSL order.
    pub upstream: Vec<String>,
    /// The upstream whose outputs are this node's inputs: the first one
    /// that is not a Message. `None` means the workflow input.
    pub data_upstream: Option<String>,
    pub downstream: Vec<String>,
    /// Switch and Categorize choose which children run.
    pub branching: bool,
    /// An Agent whose every child is a terminal Message: it does not run
    /// when scheduled; the Message that references its `content` runs it
    /// and it finishes then.
    pub defers_to_message: bool,
}

#[derive(Debug)]
pub struct Graph {
    pub nodes: BTreeMap<String, Node>,
    /// A topological order; ties broken by id so runs are reproducible.
    pub order: Vec<String>,
    pub begin: String,
}

pub fn build(dsl: &Dsl) -> Result<Graph, CompileError> {
    let mut nodes = BTreeMap::new();
    for (id, c) in &dsl.components {
        let given = &c.obj.component_name;
        let name = SUPPORTED
            .iter()
            .find(|s| s.eq_ignore_ascii_case(given))
            .ok_or_else(|| CompileError::Unsupported {
                id: id.clone(),
                name: given.clone(),
            })?;
        let mut upstream: Vec<String> = Vec::new();
        for up in &c.upstream {
            if up == id {
                return Err(CompileError::SelfEdge(id.clone()));
            }
            if !dsl.components.contains_key(up) {
                return Err(CompileError::UnknownUpstream {
                    id: id.clone(),
                    upstream: up.clone(),
                });
            }
            if !upstream.contains(up) {
                upstream.push(up.clone());
            }
        }
        if let Some(to) = c
            .downstream
            .iter()
            .find(|to| !dsl.components.contains_key(*to))
        {
            return Err(CompileError::DanglingEdge {
                from: id.clone(),
                to: to.clone(),
            });
        }
        nodes.insert(
            id.clone(),
            Node {
                id: id.clone(),
                label: dsl.display_name(id),
                name,
                params: c.obj.params.clone(),
                upstream,
                data_upstream: None,
                downstream: c.downstream.clone(),
                branching: matches!(*name, "Switch" | "Categorize"),
                defers_to_message: false,
            },
        );
    }

    let deferring: Vec<String> = nodes
        .values()
        .filter(|n| {
            n.name == "Agent"
                && !n.downstream.is_empty()
                && n.downstream
                    .iter()
                    .all(|d| nodes[d].name == "Message" && nodes[d].downstream.is_empty())
        })
        .map(|n| n.id.clone())
        .collect();
    for id in deferring {
        nodes.get_mut(&id).expect("node exists").defers_to_message = true;
    }
    let data: Vec<(String, Option<String>)> = nodes
        .values()
        .map(|n| {
            let up = n
                .upstream
                .iter()
                .find(|u| nodes[*u].name != "Message")
                .cloned();
            (n.id.clone(), up)
        })
        .collect();
    for (id, up) in data {
        nodes.get_mut(&id).expect("node exists").data_upstream = up;
    }

    let begin = nodes
        .values()
        .find(|n| n.name == "Begin")
        .map(|n| n.id.clone())
        .ok_or(CompileError::NoBegin)?;
    let order = topological(&nodes)?;
    Ok(Graph {
        nodes,
        order,
        begin,
    })
}

fn topological(nodes: &BTreeMap<String, Node>) -> Result<Vec<String>, CompileError> {
    let mut waiting: BTreeMap<&str, usize> = nodes
        .values()
        .map(|n| (n.id.as_str(), n.upstream.len()))
        .collect();
    let mut ready: BTreeSet<&str> = waiting
        .iter()
        .filter(|(_, n)| **n == 0)
        .map(|(id, _)| *id)
        .collect();
    let mut order = Vec::with_capacity(nodes.len());
    while let Some(id) = ready.pop_first() {
        order.push(id.to_string());
        for child in nodes
            .values()
            .filter(|n| n.upstream.iter().any(|u| u == id))
        {
            let left = waiting.get_mut(child.id.as_str()).expect("node exists");
            *left -= 1;
            if *left == 0 {
                ready.insert(child.id.as_str());
            }
        }
    }
    if order.len() < nodes.len() {
        let stuck = waiting
            .into_iter()
            .filter(|(_, n)| *n > 0)
            .map(|(id, _)| id.to_string())
            .collect();
        return Err(CompileError::Cycle(stuck));
    }
    Ok(order)
}

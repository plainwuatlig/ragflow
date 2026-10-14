//! The retrieval backend the Retrieval component searches through, with
//! upstream's dependency check in front of it: the caller and the canvas
//! owner must both be allowed to use every dataset the node selects
//! (`checkCanvasRetrievalDependencies`).

use std::sync::Arc;

use ragflow_runtime::retrieval::{RetrievalChunk, RetrievalRequest, RetrievalService};

use crate::facts::Facts;
use crate::perm;

pub struct GuardedRetrieval {
    pub inner: Arc<dyn RetrievalService>,
    pub facts: Arc<dyn Facts>,
    pub user_id: String,
    pub canvas_id: String,
}

impl RetrievalService for GuardedRetrieval {
    fn search(&self, req: &RetrievalRequest) -> Result<Vec<RetrievalChunk>, String> {
        if !req.retrieval_from.eq_ignore_ascii_case("memory") {
            perm::canvas_datasets(
                self.facts.as_ref(),
                &self.user_id,
                &self.canvas_id,
                &req.dataset_ids,
            )
            .map_err(|e| format!("agent Retrieval dependency permission: {e}"))?;
        }
        self.inner.search(req)
    }

    fn resolve_dataset_id(
        &self,
        name_or_id: &str,
        tenant_id: &str,
        user_id: &str,
    ) -> Option<String> {
        self.inner
            .resolve_dataset_id(name_or_id, tenant_id, user_id)
    }
}

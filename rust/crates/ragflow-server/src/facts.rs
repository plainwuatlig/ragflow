//! The rows of RAGFlow's MySQL that the server reads, as plain values, behind
//! one trait so the access rules, canvas selection and model resolution can
//! be tested without a database. [`crate::store::Store`] is the MySQL
//! implementation.

use serde_json::Value;

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
#[error("{0}")]
pub struct StoreError(pub String);

pub type Result<T> = std::result::Result<T, StoreError>;

impl From<mysql::Error> for StoreError {
    fn from(e: mysql::Error) -> Self {
        StoreError(e.to_string())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct User {
    pub id: String,
    pub access_token: Option<String>,
    pub is_superuser: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ApiToken {
    pub tenant_id: String,
    pub token: String,
    pub dialog_id: Option<String>,
    pub beta: Option<String>,
}

/// A `user_tenant` row with `status = '1'`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Membership {
    pub role: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Canvas {
    pub id: String,
    pub user_id: String,
    pub permission: String,
    pub title: Option<String>,
    pub avatar: Option<String>,
    pub canvas_category: String,
    /// `Null` when the column is empty.
    pub dsl: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Version {
    pub id: String,
    pub canvas_id: String,
    pub title: Option<String>,
    pub release: bool,
    pub dsl: Value,
}

/// An `api_4_conversation` row, children not loaded.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionRow {
    pub id: String,
    pub name: Option<String>,
    pub dialog_id: String,
    pub user_id: String,
    pub exp_user_id: Option<String>,
    pub tokens: i64,
    pub source: Option<String>,
    pub dsl: Value,
    pub duration: f64,
    pub round: i64,
    pub thumb_up: i64,
    pub errors: Option<String>,
    pub version_title: Option<String>,
    pub create_time: Option<i64>,
    pub create_date: Option<String>,
    pub update_time: Option<i64>,
    pub update_date: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TenantRow {
    pub id: String,
    pub llm_id: String,
    pub tenant_llm_id: Option<String>,
    pub embd_id: String,
    pub tenant_embd_id: Option<String>,
    pub rerank_id: String,
    pub tenant_rerank_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelProvider {
    pub id: String,
    pub tenant_id: String,
    pub provider_name: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelInstance {
    pub id: String,
    pub provider_id: String,
    pub instance_name: String,
    pub api_key: String,
    pub status: String,
    pub extra: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TenantModel {
    pub id: String,
    pub model_name: String,
    pub provider_id: String,
    pub instance_id: String,
    pub model_type: i64,
    pub status: String,
    pub extra: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    pub id: String,
    pub tenant_id: String,
    pub permission: String,
    pub status: String,
    pub embd_id: String,
    pub tenant_embd_id: Option<String>,
}

/// Read access to RAGFlow's tables. Every method is one lookup with the
/// semantics of the Go DAO it mirrors.
pub trait Facts: Send + Sync {
    /// `WHERE access_token = ?`, first row by primary key.
    fn user_by_access_token(&self, token: &str) -> Result<Option<User>>;
    fn user_by_id(&self, id: &str) -> Result<Option<User>>;
    /// `WHERE token = ?`.
    fn api_token(&self, token: &str) -> Result<Option<ApiToken>>;
    /// `WHERE beta = ?`, in table order.
    fn beta_tokens(&self, beta: &str) -> Result<Vec<ApiToken>>;
    /// `user_tenant` for the pair with `status = '1'`, first row by id.
    fn membership(&self, user_id: &str, tenant_id: &str) -> Result<Option<Membership>>;
    /// Users with an active `owner` row in the tenant.
    fn tenant_owners(&self, tenant_id: &str) -> Result<Vec<String>>;

    fn canvas(&self, id: &str) -> Result<Option<Canvas>>;
    fn version(&self, id: &str) -> Result<Option<Version>>;
    /// Newest by `create_time`, released or not.
    fn latest_version(&self, canvas_id: &str) -> Result<Option<Version>>;
    /// Newest released one by `update_time`.
    fn latest_released_version(&self, canvas_id: &str) -> Result<Option<Version>>;

    /// `WHERE id = ? AND dialog_id = ?`.
    fn session_in_canvas(&self, id: &str, canvas_id: &str) -> Result<Option<SessionRow>>;
    fn session(&self, id: &str) -> Result<Option<SessionRow>>;

    fn tenant(&self, id: &str) -> Result<Option<TenantRow>>;
    fn tenant_model(&self, id: &str) -> Result<Option<TenantModel>>;
    fn model_provider(&self, id: &str) -> Result<Option<ModelProvider>>;
    fn model_provider_by_name(
        &self,
        tenant_id: &str,
        provider_name: &str,
    ) -> Result<Option<ModelProvider>>;
    fn model_instance(&self, id: &str) -> Result<Option<ModelInstance>>;
    fn model_instance_by_name(
        &self,
        provider_id: &str,
        instance_name: &str,
    ) -> Result<Option<ModelInstance>>;
    fn models_by_name(
        &self,
        provider_id: &str,
        instance_id: &str,
        model_name: &str,
    ) -> Result<Vec<TenantModel>>;
    fn instances_of_provider(&self, provider_id: &str) -> Result<Vec<ModelInstance>>;
    fn models_of_instance(&self, instance_id: &str) -> Result<Vec<TenantModel>>;

    /// `knowledgebase WHERE id = ? AND status = '1'`.
    fn dataset(&self, id: &str) -> Result<Option<Dataset>>;
}

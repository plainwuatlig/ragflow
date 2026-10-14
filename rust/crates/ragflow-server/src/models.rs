//! Model references to endpoints, by the two paths upstream's Go server has:
//!
//! * [`Resolver::resolve`] is `service.ModelFactory.ResolveInfo`/`New*Model`:
//!   tenant-model ids and `model@instance@provider` names, with the
//!   permission checks of `internal/permission`. The pre-run validation of
//!   every Agent's model and the retrieval embedding use it.
//! * [`component_chat_credentials`] is `resolveChatModelRef` of the Agent
//!   and Categorize components: provider, then instance, then credentials,
//!   for the canvas owner's tenant, with no status or permission checks.
//!
//! The provider catalog is upstream's `conf/models/*.json`.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::Value;

use crate::facts::{Facts, ModelInstance, ModelProvider, StoreError, TenantModel};
use crate::perm;

pub const CHAT: i64 = 1;
pub const EMBEDDING: i64 = 2;
pub const RERANK: i64 = 16;

#[derive(Debug, Clone, PartialEq)]
pub struct CatalogModel {
    pub name: String,
    pub model_types: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CatalogProvider {
    pub name: String,
    /// `url`: region (or `default`) to base URL.
    pub url: BTreeMap<String, String>,
    /// `url_suffix`: `chat`, `embedding`, ...
    pub url_suffix: BTreeMap<String, String>,
    pub class: Option<String>,
    pub models: Vec<CatalogModel>,
}

/// Upstream's `ProviderManager`: the provider files of `conf/models`.
#[derive(Debug, Clone, Default)]
pub struct Catalog {
    providers: Vec<CatalogProvider>,
}

fn string_map(v: Option<&Value>) -> BTreeMap<String, String> {
    v.and_then(Value::as_object)
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect()
        })
        .unwrap_or_default()
}

impl Catalog {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .collect();
        files.sort();
        let mut providers = Vec::new();
        for path in files {
            let text = std::fs::read_to_string(&path).map_err(|e| format!("{path:?}: {e}"))?;
            let v: Value = serde_json::from_str(&text).map_err(|e| format!("{path:?}: {e}"))?;
            providers.push(Self::provider_from(&v)?);
        }
        Ok(Catalog { providers })
    }

    pub fn from_values(values: &[Value]) -> Result<Self, String> {
        let providers = values
            .iter()
            .map(Self::provider_from)
            .collect::<Result<_, _>>()?;
        Ok(Catalog { providers })
    }

    fn provider_from(v: &Value) -> Result<CatalogProvider, String> {
        let name = v
            .get("name")
            .and_then(Value::as_str)
            .ok_or("a provider file without a name")?
            .to_string();
        let models = v
            .get("models")
            .and_then(Value::as_array)
            .map(|ms| {
                ms.iter()
                    .filter_map(|m| {
                        Some(CatalogModel {
                            name: m.get("name")?.as_str()?.to_string(),
                            model_types: m
                                .get("model_types")
                                .and_then(Value::as_array)
                                .map(|a| {
                                    a.iter()
                                        .filter_map(|t| t.as_str().map(str::to_string))
                                        .collect()
                                })
                                .unwrap_or_default(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(CatalogProvider {
            name,
            url: string_map(v.get("url")),
            url_suffix: string_map(v.get("url_suffix").or_else(|| v.get("url-suffix"))),
            class: v.get("class").and_then(Value::as_str).map(str::to_string),
            models,
        })
    }

    /// `FindProvider`: case-insensitive.
    pub fn provider(&self, name: &str) -> Option<&CatalogProvider> {
        self.providers
            .iter()
            .find(|p| p.name.eq_ignore_ascii_case(name))
    }
}

/// `entity.ModelTypeFromString` for the catalog's `model_types`.
pub fn type_bits(names: &[String]) -> i64 {
    names.iter().fold(0, |bits, n| {
        bits | match n.as_str() {
            "chat" => 1,
            "embedding" => 2,
            "asr" | "speech2text" => 4,
            "vision" | "image2text" => 8,
            "rerank" => 16,
            "tts" => 32,
            "ocr" => 64,
            _ => 0,
        }
    })
}

pub fn type_name(bit: i64) -> &'static str {
    match bit {
        CHAT => "chat",
        EMBEDDING => "embedding",
        4 => "speech2text",
        8 => "image2text",
        RERANK => "rerank",
        32 => "tts",
        64 => "ocr",
        _ => "",
    }
}

/// Why a model could not be resolved.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ModelError {
    /// `errModelConfigUnavailable` or `gorm.ErrRecordNotFound`: the caller
    /// is told the model is missing or unavailable.
    #[error("model configuration unavailable: {0}")]
    Unavailable(String),
    /// Anything else: a storage failure.
    #[error("{0}")]
    Store(String),
}

impl From<StoreError> for ModelError {
    fn from(e: StoreError) -> Self {
        ModelError::Store(e.0)
    }
}

impl From<perm::PermError> for ModelError {
    fn from(e: perm::PermError) -> Self {
        match e {
            perm::PermError::Store(s) => ModelError::Store(s),
            other => ModelError::Unavailable(other.to_string()),
        }
    }
}

fn unavailable<T>(msg: impl Into<String>) -> Result<T, ModelError> {
    Err(ModelError::Unavailable(msg.into()))
}

/// Who asks: a user inside a tenant, or (empty `user_id`) background work
/// scoped to a tenant.
#[derive(Debug, Clone, Default)]
pub struct Access {
    pub user_id: String,
    pub tenant_id: String,
}

/// A resolved model: what a driver needs to call it.
#[derive(Debug, Clone, PartialEq)]
pub struct Target {
    pub model_id: String,
    pub model_name: String,
    pub model_type: i64,
    pub provider: String,
    pub instance_id: String,
    pub instance_name: String,
    pub api_key: String,
    /// The instance's `base_url`, or empty.
    pub base_url: String,
    pub region: String,
}

pub struct Resolver<'a> {
    pub facts: &'a dyn Facts,
    pub catalog: &'a Catalog,
}

/// The instance's `{region, base_url}` (`decodeModelInstanceExtra`); other
/// keys are ignored, but a field of the wrong type is an error.
fn instance_extra(raw: &str) -> Result<(String, String), ModelError> {
    if raw.trim().is_empty() {
        return Ok((String::new(), String::new()));
    }
    let v: Value = serde_json::from_str(raw).map_err(|e| {
        ModelError::Unavailable(format!("decode model instance configuration: {e}"))
    })?;
    let field = |k: &str| -> Result<String, ModelError> {
        match v.get(k) {
            None | Some(Value::Null) => Ok(String::new()),
            Some(Value::String(s)) => Ok(s.clone()),
            Some(_) => unavailable(format!(
                "decode model instance configuration: field {k:?} is not a string"
            )),
        }
    };
    if !v.is_object() {
        return unavailable("decode model instance configuration: not an object");
    }
    Ok((field("region")?, field("base_url")?))
}

/// `parseModelName`: right-anchored `model@instance@provider`.
pub fn parse_model_name(name: &str) -> Result<(String, String, String), String> {
    let parts: Vec<&str> = name.split('@').collect();
    match parts.len() {
        1 => Err(format!("provider name missing in model name: {name}")),
        2 => Ok((parts[0].into(), "default".into(), parts[1].into())),
        n => Ok((
            parts[..n - 2].join("@"),
            parts[n - 2].into(),
            parts[n - 1].into(),
        )),
    }
}

impl Resolver<'_> {
    /// `ModelFactory.authorize`.
    fn authorize(&self, access: &Access) -> Result<(), ModelError> {
        if access.tenant_id.is_empty() {
            return unavailable("tenant id is required");
        }
        if access.user_id.is_empty() || access.user_id == access.tenant_id {
            return Ok(());
        }
        if perm::is_tenant_member(self.facts, &access.user_id, &access.tenant_id)? {
            Ok(())
        } else {
            unavailable(format!(
                "user {} has no access to tenant {}",
                access.user_id, access.tenant_id
            ))
        }
    }

    /// `ResolveInfo` / `New*Model`'s resolution: the target for `model_ref`
    /// as a `model_type` model.
    pub fn resolve(
        &self,
        access: &Access,
        model_type: i64,
        model_ref: &str,
    ) -> Result<Target, ModelError> {
        self.authorize(access)?;
        self.resolve_authorized(access, model_type, model_ref)
    }

    /// `resolveDefaultConfig`: the tenant's default model of a type.
    pub fn resolve_default(&self, access: &Access, model_type: i64) -> Result<Target, ModelError> {
        self.authorize(access)?;
        let tenant = self
            .facts
            .tenant(&access.tenant_id)?
            .ok_or_else(|| ModelError::Unavailable("record not found".into()))?;
        let (name, id) = match model_type {
            CHAT => (tenant.llm_id, tenant.tenant_llm_id),
            EMBEDDING => (tenant.embd_id, tenant.tenant_embd_id),
            RERANK => (tenant.rerank_id, tenant.tenant_rerank_id),
            _ => (String::new(), None),
        };
        if let Some(id) = id.filter(|i| !i.trim().is_empty()) {
            return self.resolve_authorized(access, model_type, &id);
        }
        if name.trim().is_empty() {
            return unavailable(format!("no default {} model is set", type_name(model_type)));
        }
        self.resolve_authorized(access, model_type, &name)
    }

    fn resolve_authorized(
        &self,
        access: &Access,
        model_type: i64,
        model_ref: &str,
    ) -> Result<Target, ModelError> {
        let model_ref = model_ref.trim();
        if model_ref.is_empty() {
            return unavailable("model ref is required");
        }
        if model_type == 0 {
            return unavailable("model type is required");
        }
        match self.facts.tenant_model(model_ref)? {
            Some(model) => self.tenant_model_target(access, model_type, model_ref, model),
            None => self.composite_target(access, model_type, model_ref),
        }
    }

    fn tenant_model_target(
        &self,
        access: &Access,
        model_type: i64,
        model_ref: &str,
        model: TenantModel,
    ) -> Result<Target, ModelError> {
        if !model.status.eq_ignore_ascii_case("active") {
            return unavailable(format!("tenant model {model_ref:?} is disabled"));
        }
        if model.model_type == 0 {
            return unavailable(format!("tenant model {model_ref:?} has no model type"));
        }
        let Some(provider) = self.facts.model_provider(&model.provider_id)? else {
            return unavailable(format!(
                "provider id={} not found for model {model_ref:?}",
                model.provider_id
            ));
        };
        if !access.user_id.is_empty() {
            perm::model_resource_usable(self.facts, &access.user_id, &provider.tenant_id)?;
        } else if provider.tenant_id != access.tenant_id {
            return unavailable(format!(
                "tenant {} has no access to provider owned by tenant {}",
                access.tenant_id, provider.tenant_id
            ));
        }
        let Some(instance) = self.facts.model_instance(&model.instance_id)? else {
            return unavailable(format!(
                "instance id={} not found for model {model_ref:?}",
                model.instance_id
            ));
        };
        if instance.provider_id != provider.id {
            return unavailable(format!(
                "instance id={} does not belong to provider id={}",
                model.instance_id, provider.id
            ));
        }
        if !instance.status.eq_ignore_ascii_case("active") {
            return unavailable(format!(
                "model instance {:?} is disabled",
                instance.instance_name
            ));
        }
        if model.model_type & model_type == 0 {
            return unavailable(format!(
                "tenant model {model_ref:?} cannot be used as {} model",
                type_name(model_type)
            ));
        }
        self.target(
            &provider,
            &instance,
            Some(&model),
            &model.model_name,
            model_type,
        )
    }

    fn target(
        &self,
        provider: &ModelProvider,
        instance: &ModelInstance,
        model: Option<&TenantModel>,
        model_name: &str,
        model_type: i64,
    ) -> Result<Target, ModelError> {
        if self.catalog.provider(&provider.provider_name).is_none() {
            return unavailable(format!(
                "provider {:?} driver not found",
                provider.provider_name
            ));
        }
        if let Some(m) = model {
            if !m.extra.trim().is_empty() {
                serde_json::from_str::<Value>(&m.extra)
                    .map_err(|e| ModelError::Unavailable(format!("read model metadata: {e}")))?;
            }
        }
        let (region, base_url) = instance_extra(&instance.extra)?;
        Ok(Target {
            model_id: model.map(|m| m.id.clone()).unwrap_or_default(),
            model_name: model_name.to_string(),
            model_type,
            provider: provider.provider_name.clone(),
            instance_id: instance.id.clone(),
            instance_name: instance.instance_name.clone(),
            api_key: instance.api_key.clone(),
            base_url,
            region,
        })
    }

    fn composite_target(
        &self,
        access: &Access,
        model_type: i64,
        model_ref: &str,
    ) -> Result<Target, ModelError> {
        if model_type == EMBEDDING
            && (model_ref.ends_with("@default@Builtin") || model_ref.ends_with("@Builtin"))
        {
            return unavailable("builtin embedding models are not supported by this server");
        }
        let (pure, instance_name, provider_name) =
            parse_model_name(model_ref).map_err(ModelError::Unavailable)?;
        let Some(provider) = self
            .facts
            .model_provider_by_name(&access.tenant_id, &provider_name)?
        else {
            return unavailable(format!(
                "provider {provider_name:?} not found for model {model_ref:?}"
            ));
        };
        if !access.user_id.is_empty() {
            perm::model_resource_usable(self.facts, &access.user_id, &provider.tenant_id)?;
        }
        let Some(instance) = self
            .facts
            .model_instance_by_name(&provider.id, &instance_name)?
        else {
            return unavailable(format!(
                "instance {instance_name:?} not found for model {model_ref:?}"
            ));
        };
        if !instance.status.eq_ignore_ascii_case("active") {
            return unavailable(format!("model instance {instance_name:?} is disabled"));
        }
        if !access.user_id.is_empty() {
            perm::model_resource_usable(self.facts, &access.user_id, &provider.tenant_id)?;
        }
        instance_extra(&instance.extra)?;
        let rows = self
            .facts
            .models_by_name(&provider.id, &instance.id, &pure)?;
        let mut available = 0i64;
        let mut chosen: Option<&TenantModel> = None;
        let mut disabled_for_type = false;
        for row in &rows {
            if row.status.eq_ignore_ascii_case("active") {
                available |= row.model_type;
                if row.model_type & model_type != 0 && chosen.is_none() {
                    chosen = Some(row);
                }
            } else if row.model_type & model_type != 0 {
                disabled_for_type = true;
            }
        }
        if available != 0 && available & model_type == 0 {
            return unavailable(format!(
                "model {model_ref:?} cannot be used as {} model",
                type_name(model_type)
            ));
        }
        if chosen.is_none() && available == 0 && disabled_for_type {
            return unavailable(format!("model {model_ref:?} is disabled"));
        }
        if available != 0 {
            let model = chosen.expect("an active row with the requested type exists");
            return self.target(
                &provider,
                &instance,
                Some(model),
                &model.model_name,
                model_type,
            );
        }
        // No tenant_model row: the catalog entry stands in.
        let Some(catalog) = self.catalog.provider(&provider_name) else {
            return unavailable(format!("model provider config not found: {provider_name}"));
        };
        let Some(entry) = catalog
            .models
            .iter()
            .find(|m| m.name.eq_ignore_ascii_case(&pure))
        else {
            return unavailable(format!("model config not found: {model_ref}"));
        };
        if type_bits(&entry.model_types) & model_type == 0 {
            return unavailable(format!(
                "model {model_ref:?} cannot be used as {} model",
                type_name(model_type)
            ));
        }
        self.target(&provider, &instance, None, &entry.name, model_type)
    }
}

/// What the Agent and Categorize components call a chat model with:
/// `resolveChatModelRef` followed by the driver lookup. A composite
/// `model@instance@provider` is split right-anchored; a bare tenant model id
/// or instance id is looked up first. Credentials come from the canvas
/// owner's `tenant_model_provider` + `tenant_model_instance` rows; nothing
/// is checked beyond their existence (no status, no permission).
#[derive(Debug, Clone, PartialEq)]
pub struct ChatCredentials {
    pub model: String,
    pub provider: String,
    pub api_key: String,
    pub base_url: String,
}

fn is_bare_tenant_model_id(s: &str) -> bool {
    let s = s.trim();
    s.len() == 32 && !s.contains('@') && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// `fillInstanceCredentials`: the base URL comes from `extra` only when it
/// decodes as a string map.
fn fill_instance(instance: &ModelInstance, api_key: &str, base_url: &str) -> (String, String) {
    let key = if api_key.is_empty() {
        instance.api_key.clone()
    } else {
        api_key.to_string()
    };
    let mut base = base_url.to_string();
    if base.is_empty() && !instance.extra.is_empty() {
        if let Ok(m) = serde_json::from_str::<BTreeMap<String, String>>(&instance.extra) {
            base = m.get("base_url").cloned().unwrap_or_default();
        }
    }
    (key, base)
}

pub fn component_chat_credentials(
    facts: &dyn Facts,
    tenant_id: &str,
    model_ref: &ragflow_runtime::models::ModelRef,
) -> Result<ChatCredentials, String> {
    // A bare 32-hex id is a tenant_model id, else a tenant_model_instance id.
    if model_ref.provider.is_empty() && is_bare_tenant_model_id(&model_ref.model) {
        let id = model_ref.model.as_str();
        if tenant_id.is_empty() {
            return Err(format!("tenant chat model id {id:?} not found"));
        }
        if let Some(model) = facts.tenant_model(id).map_err(|e| e.0)? {
            if !model.status.trim().eq_ignore_ascii_case("active") {
                return Err(format!("tenant model id {id} is disabled"));
            }
            if model.model_type & CHAT == 0 {
                return Err(format!("tenant model id {id} cannot be used as chat model"));
            }
            let provider = facts
                .model_provider(&model.provider_id)
                .map_err(|e| e.0)?
                .ok_or_else(|| {
                    format!("resolve provider for tenant model id {id:?}: record not found")
                })?;
            if provider.tenant_id != tenant_id {
                return Err(format!("tenant {tenant_id} has no access to model id {id}"));
            }
            let instance = facts
                .model_instance(&model.instance_id)
                .map_err(|e| e.0)?
                .ok_or_else(|| {
                    format!("resolve instance for tenant model id {id:?}: record not found")
                })?;
            let (key, base) = fill_instance(&instance, "", "");
            return Ok(ChatCredentials {
                model: model.model_name,
                provider: provider.provider_name,
                api_key: key,
                base_url: base,
            });
        }
        if let Some(instance) = facts.model_instance(id).map_err(|e| e.0)? {
            let provider = facts
                .model_provider(&instance.provider_id)
                .map_err(|e| e.0)?
                .ok_or_else(|| {
                    format!(
                        "resolve provider for tenant model instance id {id:?}: record not found"
                    )
                })?;
            if provider.tenant_id != tenant_id {
                return Err(format!(
                    "tenant {tenant_id} has no access to model instance id {id}"
                ));
            }
            let models: Vec<_> = facts
                .models_of_instance(&instance.id)
                .map_err(|e| e.0)?
                .into_iter()
                .filter(|m| {
                    m.provider_id == provider.id
                        && m.status.trim().eq_ignore_ascii_case("active")
                        && m.model_type & CHAT != 0
                })
                .collect();
            return match models.as_slice() {
                [] => Err(format!("tenant model instance id {id} has no active chat model")),
                [one] => {
                    let (key, base) = fill_instance(&instance, "", "");
                    Ok(ChatCredentials {
                        model: one.model_name.clone(),
                        provider: provider.provider_name,
                        api_key: key,
                        base_url: base,
                    })
                }
                many => Err(format!(
                    "tenant model instance id {id} has {} active chat models; use tenant_model.id or model@instance@provider",
                    many.len()
                )),
            };
        }
        return Err(format!("tenant chat model id {id:?} not found"));
    }

    let mut creds = ChatCredentials {
        model: model_ref.model.clone(),
        provider: model_ref.provider.clone(),
        api_key: String::new(),
        base_url: String::new(),
    };
    if creds.provider.is_empty() || tenant_id.is_empty() {
        return Ok(creds);
    }
    let instance_name = model_ref.instance.as_deref().unwrap_or("default");
    let Some(provider) = facts
        .model_provider_by_name(tenant_id, &creds.provider)
        .map_err(|e| e.0)?
    else {
        return Ok(creds);
    };
    let mut instance = facts
        .model_instance_by_name(&provider.id, instance_name)
        .map_err(|e| e.0)?;
    if instance.is_none() && instance_name == "default" {
        // A missing `default` instance maps to the provider's only active one.
        let active: Vec<_> = facts
            .instances_of_provider(&provider.id)
            .map_err(|e| e.0)?
            .into_iter()
            .filter(|i| !i.status.trim().eq_ignore_ascii_case("inactive"))
            .collect();
        if let [only] = active.as_slice() {
            instance = Some(only.clone());
        }
    }
    if let Some(instance) = instance {
        // resolveTenantModelInstanceCredentials only counts when the key is set.
        if !instance.api_key.is_empty() {
            let (key, base) = fill_instance(&instance, "", "");
            creds.api_key = key;
            creds.base_url = base;
        }
    }
    Ok(creds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::facts::*;
    use crate::perm::fake::Fake;

    fn catalog() -> Catalog {
        Catalog::from_values(&[
            serde_json::json!({"name": "OpenAI", "url": {"default": "https://api.openai.com/v1"},
                "url_suffix": {"chat": "chat/completions", "embedding": "embeddings"}, "class": "gpt",
                "models": [{"name": "gpt-4o-mini", "model_types": ["chat", "vision"]}]}),
            serde_json::json!({"name": "Ollama", "url_suffix": {"chat": "api/chat", "embedding": "api/embed"},
                "class": "local", "models": []}),
        ])
        .unwrap()
    }

    fn world() -> Fake {
        let mut f = Fake::default();
        f.member("alice", "alice", "owner")
            .member("bob", "alice", "normal");
        f.providers = vec![
            ModelProvider {
                id: "p-openai".into(),
                tenant_id: "alice".into(),
                provider_name: "OpenAI".into(),
            },
            ModelProvider {
                id: "p-ollama".into(),
                tenant_id: "alice".into(),
                provider_name: "Ollama".into(),
            },
        ];
        f.instances = vec![
            ModelInstance {
                id: "i-dev".into(),
                provider_id: "p-openai".into(),
                instance_name: "dev".into(),
                api_key: "sk-1".into(),
                status: "active".into(),
                extra: r#"{"base_url":"http://stub/v1","region":""}"#.into(),
            },
            ModelInstance {
                id: "i-off".into(),
                provider_id: "p-openai".into(),
                instance_name: "off".into(),
                api_key: "sk-2".into(),
                status: "inactive".into(),
                extra: "{}".into(),
            },
            ModelInstance {
                id: "i-oll".into(),
                provider_id: "p-ollama".into(),
                instance_name: "local".into(),
                api_key: String::new(),
                status: "active".into(),
                extra: r#"{"base_url":"http://oll:11434"}"#.into(),
            },
        ];
        f.models = vec![
            TenantModel {
                id: "m-4o".into(),
                model_name: "gpt-4o-mini".into(),
                provider_id: "p-openai".into(),
                instance_id: "i-dev".into(),
                model_type: 9,
                status: "active".into(),
                extra: r#"{"max_tokens":16384}"#.into(),
            },
            TenantModel {
                id: "m-bge".into(),
                model_name: "bge-m3".into(),
                provider_id: "p-ollama".into(),
                instance_id: "i-oll".into(),
                model_type: 2,
                status: "active".into(),
                extra: "{}".into(),
            },
        ];
        f
    }

    fn resolver<'a>(f: &'a Fake, c: &'a Catalog) -> Resolver<'a> {
        Resolver {
            facts: f,
            catalog: c,
        }
    }

    fn access(user: &str, tenant: &str) -> Access {
        Access {
            user_id: user.into(),
            tenant_id: tenant.into(),
        }
    }

    #[test]
    fn a_tenant_model_id_resolves_with_its_instance() {
        let (f, c) = (world(), catalog());
        let t = resolver(&f, &c)
            .resolve(&access("alice", "alice"), CHAT, "m-4o")
            .unwrap();
        assert_eq!(
            (
                t.model_name.as_str(),
                t.provider.as_str(),
                t.base_url.as_str()
            ),
            ("gpt-4o-mini", "OpenAI", "http://stub/v1")
        );
        assert_eq!(t.api_key, "sk-1");
        let e = resolver(&f, &c)
            .resolve(&access("alice", "alice"), EMBEDDING, "m-4o")
            .unwrap_err();
        assert!(
            matches!(e, ModelError::Unavailable(m) if m.contains("cannot be used as embedding"))
        );
    }

    #[test]
    fn a_composite_name_resolves_through_provider_and_instance() {
        let (f, c) = (world(), catalog());
        let r = resolver(&f, &c);
        let t = r
            .resolve(&access("bob", "alice"), CHAT, "gpt-4o-mini@dev@OpenAI")
            .unwrap();
        assert_eq!(t.model_id, "m-4o");
        assert!(
            matches!(r.resolve(&access("bob", "alice"), CHAT, "gpt-4o-mini@off@OpenAI"),
            Err(ModelError::Unavailable(m)) if m.contains("disabled"))
        );
        assert!(matches!(
            r.resolve(&access("bob", "alice"), CHAT, "gpt-4o-mini@nope@OpenAI"),
            Err(ModelError::Unavailable(_))
        ));
        assert!(
            matches!(r.resolve(&access("bob", "alice"), CHAT, "gpt-4o-mini"),
            Err(ModelError::Unavailable(m)) if m.contains("provider name missing"))
        );
        // No tenant_model row: the catalog entry stands in, by type.
        let t = r.resolve(&access("alice", "alice"), CHAT, "GPT-4O-MINI@dev@openai");
        assert!(
            matches!(t, Err(ModelError::Unavailable(_))),
            "provider names match exactly in the tenant table"
        );
    }

    #[test]
    fn people_outside_the_tenant_get_nothing() {
        let (mut f, c) = (world(), catalog());
        f.member("eve", "eve", "owner");
        let r = resolver(&f, &c);
        assert!(matches!(
            r.resolve(&access("eve", "alice"), CHAT, "m-4o"),
            Err(ModelError::Unavailable(_))
        ));
        // A user inside their own tenant still may not use another tenant's provider.
        assert!(matches!(
            r.resolve(&access("eve", "eve"), CHAT, "m-4o"),
            Err(ModelError::Unavailable(_))
        ));
        // Background work is scoped to the tenant that owns the provider.
        assert!(r.resolve(&access("", "alice"), CHAT, "m-4o").is_ok());
        assert!(matches!(
            r.resolve(&access("", "eve"), CHAT, "m-4o"),
            Err(ModelError::Unavailable(_))
        ));
    }

    #[test]
    fn the_default_model_comes_from_the_tenant_row() {
        let (mut f, c) = (world(), catalog());
        f.tenants = vec![TenantRow {
            id: "alice".into(),
            llm_id: String::new(),
            tenant_llm_id: None,
            embd_id: "bge-m3@local@Ollama".into(),
            tenant_embd_id: None,
            rerank_id: String::new(),
            tenant_rerank_id: None,
        }];
        let r = resolver(&f, &c);
        assert_eq!(
            r.resolve_default(&access("", "alice"), EMBEDDING)
                .unwrap()
                .model_id,
            "m-bge"
        );
        assert!(
            matches!(r.resolve_default(&access("", "alice"), CHAT), Err(ModelError::Unavailable(m)) if m.contains("no default chat"))
        );
    }

    #[test]
    fn the_component_path_reads_credentials_from_the_owner_tenant() {
        let f = world();
        let r = ragflow_runtime::models::ModelRef {
            model: "gpt-4o-mini".into(),
            instance: Some("dev".into()),
            provider: "OpenAI".into(),
        };
        let c = component_chat_credentials(&f, "alice", &r).unwrap();
        assert_eq!(
            (c.api_key.as_str(), c.base_url.as_str()),
            ("sk-1", "http://stub/v1")
        );
        // Another tenant has no such provider: no credentials, not an error.
        let c = component_chat_credentials(&f, "bob", &r).unwrap();
        assert_eq!(c.api_key, "");
        // A disabled instance still yields its key on this path.
        let r2 = ragflow_runtime::models::ModelRef {
            model: "x".into(),
            instance: Some("off".into()),
            provider: "OpenAI".into(),
        };
        assert_eq!(
            component_chat_credentials(&f, "alice", &r2)
                .unwrap()
                .api_key,
            "sk-2"
        );
        // A bare tenant model id.
        let bare = ragflow_runtime::models::ModelRef {
            model: "0123456789abcdef0123456789abcdef".into(),
            instance: None,
            provider: String::new(),
        };
        assert!(component_chat_credentials(&f, "alice", &bare)
            .unwrap_err()
            .contains("not found"));
        let mut f2 = world();
        f2.models[0].id = "0123456789abcdef0123456789abcdef".into();
        let c = component_chat_credentials(&f2, "alice", &bare).unwrap();
        assert_eq!(
            (c.model.as_str(), c.provider.as_str()),
            ("gpt-4o-mini", "OpenAI")
        );
    }
}

//! Who may do what, as upstream's `internal/permission` community policy
//! decides it for the resources the agent endpoints touch: canvases, agent
//! sessions, datasets and the model rows. The facts come from [`Facts`].
//!
//! The rules (policy_ce.go, database_source.go):
//! * the owner of a resource may do whatever its owner operations list;
//! * a resource shared with the tenant (`permission = 'team'`) lets an
//!   active `owner`/`admin`/`normal` member of that tenant use its tenant
//!   operations; an `invite` row, or none, does not;
//! * everything else is denied, and a missing row is "not found".

use crate::facts::{Canvas, Facts, SessionRow, StoreError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Read,
    Update,
    Delete,
    Run,
    Use,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PermError {
    #[error("permission: resource not found")]
    NotFound,
    #[error("permission: access denied")]
    Denied,
    #[error("permission: invalid permission request: {0}")]
    Invalid(String),
    #[error("{0}")]
    Store(String),
}

impl From<StoreError> for PermError {
    fn from(e: StoreError) -> Self {
        PermError::Store(e.0)
    }
}

fn is_member_role(role: &str) -> bool {
    matches!(role, "owner" | "admin" | "normal")
}

/// `CheckTenant(user, tenant, TenantMember)`: an active owner, admin or
/// normal member.
pub fn is_tenant_member(f: &dyn Facts, user_id: &str, tenant_id: &str) -> Result<bool, PermError> {
    Ok(f.membership(user_id, tenant_id)?
        .is_some_and(|m| is_member_role(&m.role)))
}

/// `CheckTenant(user, tenant, TenantNormalMember)`: exactly `normal`.
fn is_normal_member(f: &dyn Facts, user_id: &str, tenant_id: &str) -> Result<bool, PermError> {
    Ok(f.membership(user_id, tenant_id)?
        .is_some_and(|m| m.role == "normal"))
}

/// The single active owner of a tenant (`loadTenantOwnerIDs`).
fn tenant_owner(f: &dyn Facts, tenant_id: &str) -> Result<String, PermError> {
    let mut owners = f.tenant_owners(tenant_id)?;
    owners.sort();
    owners.dedup();
    match owners.len() {
        1 => Ok(owners.remove(0)),
        0 => Err(PermError::Invalid(format!(
            "tenant {tenant_id:?} has no active owner"
        ))),
        _ => Err(PermError::Invalid(format!(
            "tenant {tenant_id:?} has multiple active owners"
        ))),
    }
}

fn canvas_ops(f: &dyn Facts, user_id: &str, canvas: &Canvas) -> Result<&'static [Op], PermError> {
    if user_id == canvas.user_id {
        return Ok(&[Op::Read, Op::Update, Op::Delete, Op::Run, Op::Use]);
    }
    if canvas.permission == "team" && is_tenant_member(f, user_id, &canvas.user_id)? {
        return Ok(&[Op::Read, Op::Update, Op::Run, Op::Use]);
    }
    Err(PermError::Denied)
}

/// `CheckCanvasPermission` then `canvasDAO.GetByID`: the canvas, if `user_id`
/// may do `op` with it.
pub fn canvas_for(
    f: &dyn Facts,
    user_id: &str,
    canvas_id: &str,
    op: Op,
) -> Result<Canvas, PermError> {
    if user_id.trim().is_empty() {
        return Err(PermError::NotFound);
    }
    let canvas = f.canvas(canvas_id)?.ok_or(PermError::NotFound)?;
    if !canvas_ops(f, user_id, &canvas)?.contains(&op) {
        return Err(PermError::Denied);
    }
    Ok(canvas)
}

/// `checkAgentSessionRunPermission`: a session is private to the user who
/// created it. A session whose canvas is gone is not found.
pub fn session_run(f: &dyn Facts, user_id: &str, session: &SessionRow) -> Result<(), PermError> {
    if f.canvas(&session.dialog_id)?.is_none() {
        return Err(PermError::NotFound);
    }
    if session.user_id == user_id {
        Ok(())
    } else {
        Err(PermError::Denied)
    }
}

/// What a user may do with a dataset: `Use` for its owner (the tenant's
/// owner user), or for a `normal` member of its tenant when it is shared
/// with the team.
fn dataset_usable_by(
    f: &dyn Facts,
    user_id: &str,
    dataset: &crate::facts::Dataset,
) -> Result<(), PermError> {
    let owner = tenant_owner(f, &dataset.tenant_id)?;
    if user_id == owner {
        return Ok(());
    }
    if dataset.permission == "team" && is_normal_member(f, user_id, &dataset.tenant_id)? {
        return Ok(());
    }
    Err(PermError::Denied)
}

/// `CheckCanvasDependencies` for datasets: the caller may run the canvas and
/// use each dataset, and so may the canvas owner. Errors read like
/// upstream's `dataset "<id>": permission: ...`.
pub fn canvas_datasets(
    f: &dyn Facts,
    user_id: &str,
    canvas_id: &str,
    dataset_ids: &[String],
) -> Result<(), String> {
    let wrap = |id: &str, e: PermError| format!("dataset {id:?}: {e}");
    let mut seen = std::collections::BTreeSet::new();
    for id in dataset_ids {
        let id = id.trim();
        if id.is_empty() || !seen.insert(id.to_string()) {
            continue;
        }
        let canvas = f
            .canvas(canvas_id)
            .map_err(|e| wrap(id, e.into()))?
            .ok_or_else(|| wrap(id, PermError::NotFound))?;
        if !canvas_ops(f, user_id, &canvas)
            .map_err(|e| wrap(id, e))?
            .contains(&Op::Run)
        {
            return Err(wrap(id, PermError::Denied));
        }
        let dataset = f
            .dataset(id)
            .map_err(|e| wrap(id, e.into()))?
            .ok_or_else(|| wrap(id, PermError::NotFound))?;
        dataset_usable_by(f, user_id, &dataset).map_err(|e| wrap(id, e))?;
        dataset_usable_by(f, &canvas.user_id, &dataset).map_err(|e| wrap(id, e))?;
    }
    Ok(())
}

/// The model resources (provider, instance, model) of a tenant are readable
/// and usable by the tenant's owner and by its members.
pub fn model_resource_usable(
    f: &dyn Facts,
    user_id: &str,
    provider_tenant_id: &str,
) -> Result<(), PermError> {
    let owner = tenant_owner(f, provider_tenant_id)?;
    if user_id == owner || is_tenant_member(f, user_id, provider_tenant_id)? {
        Ok(())
    } else {
        Err(PermError::Denied)
    }
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) mod fake {
    //! An in-memory [`Facts`] for tests.
    use std::collections::BTreeMap;

    use crate::facts::*;

    #[derive(Default, Clone)]
    pub struct Fake {
        pub users: Vec<User>,
        pub tokens: Vec<ApiToken>,
        pub memberships: BTreeMap<(String, String), String>,
        pub canvases: Vec<Canvas>,
        pub versions: Vec<Version>,
        pub sessions: Vec<SessionRow>,
        pub tenants: Vec<TenantRow>,
        pub models: Vec<TenantModel>,
        pub providers: Vec<ModelProvider>,
        pub instances: Vec<ModelInstance>,
        pub datasets: Vec<Dataset>,
    }

    impl Fake {
        pub fn member(&mut self, user: &str, tenant: &str, role: &str) -> &mut Self {
            self.memberships
                .insert((user.into(), tenant.into()), role.into());
            self
        }
    }

    impl Facts for Fake {
        fn user_by_access_token(&self, token: &str) -> Result<Option<User>> {
            Ok(self
                .users
                .iter()
                .find(|u| u.access_token.as_deref() == Some(token))
                .cloned())
        }
        fn user_by_id(&self, id: &str) -> Result<Option<User>> {
            Ok(self.users.iter().find(|u| u.id == id).cloned())
        }
        fn api_token(&self, token: &str) -> Result<Option<ApiToken>> {
            Ok(self.tokens.iter().find(|t| t.token == token).cloned())
        }
        fn beta_tokens(&self, beta: &str) -> Result<Vec<ApiToken>> {
            Ok(self
                .tokens
                .iter()
                .filter(|t| t.beta.as_deref() == Some(beta))
                .cloned()
                .collect())
        }
        fn membership(&self, user: &str, tenant: &str) -> Result<Option<Membership>> {
            Ok(self
                .memberships
                .get(&(user.to_string(), tenant.to_string()))
                .map(|role| Membership { role: role.clone() }))
        }
        fn tenant_owners(&self, tenant: &str) -> Result<Vec<String>> {
            Ok(self
                .memberships
                .iter()
                .filter(|((_, t), role)| t == tenant && *role == "owner")
                .map(|((u, _), _)| u.clone())
                .collect())
        }
        fn canvas(&self, id: &str) -> Result<Option<Canvas>> {
            Ok(self.canvases.iter().find(|c| c.id == id).cloned())
        }
        fn version(&self, id: &str) -> Result<Option<Version>> {
            Ok(self.versions.iter().find(|v| v.id == id).cloned())
        }
        fn latest_version(&self, canvas: &str) -> Result<Option<Version>> {
            Ok(self
                .versions
                .iter()
                .rev()
                .find(|v| v.canvas_id == canvas)
                .cloned())
        }
        fn latest_released_version(&self, canvas: &str) -> Result<Option<Version>> {
            Ok(self
                .versions
                .iter()
                .rev()
                .find(|v| v.canvas_id == canvas && v.release)
                .cloned())
        }
        fn session_in_canvas(&self, id: &str, canvas: &str) -> Result<Option<SessionRow>> {
            Ok(self
                .sessions
                .iter()
                .find(|s| s.id == id && s.dialog_id == canvas)
                .cloned())
        }
        fn session(&self, id: &str) -> Result<Option<SessionRow>> {
            Ok(self.sessions.iter().find(|s| s.id == id).cloned())
        }
        fn tenant(&self, id: &str) -> Result<Option<TenantRow>> {
            Ok(self.tenants.iter().find(|t| t.id == id).cloned())
        }
        fn tenant_model(&self, id: &str) -> Result<Option<TenantModel>> {
            Ok(self.models.iter().find(|m| m.id == id).cloned())
        }
        fn model_provider(&self, id: &str) -> Result<Option<ModelProvider>> {
            Ok(self.providers.iter().find(|p| p.id == id).cloned())
        }
        fn model_provider_by_name(
            &self,
            tenant: &str,
            name: &str,
        ) -> Result<Option<ModelProvider>> {
            Ok(self
                .providers
                .iter()
                .find(|p| p.tenant_id == tenant && p.provider_name == name)
                .cloned())
        }
        fn model_instance(&self, id: &str) -> Result<Option<ModelInstance>> {
            Ok(self.instances.iter().find(|i| i.id == id).cloned())
        }
        fn model_instance_by_name(
            &self,
            provider: &str,
            name: &str,
        ) -> Result<Option<ModelInstance>> {
            Ok(self
                .instances
                .iter()
                .find(|i| i.provider_id == provider && i.instance_name == name)
                .cloned())
        }
        fn models_by_name(
            &self,
            provider: &str,
            instance: &str,
            name: &str,
        ) -> Result<Vec<TenantModel>> {
            Ok(self
                .models
                .iter()
                .filter(|m| {
                    m.provider_id == provider && m.instance_id == instance && m.model_name == name
                })
                .cloned()
                .collect())
        }
        fn instances_of_provider(&self, provider: &str) -> Result<Vec<ModelInstance>> {
            Ok(self
                .instances
                .iter()
                .filter(|i| i.provider_id == provider)
                .cloned()
                .collect())
        }
        fn models_of_instance(&self, instance: &str) -> Result<Vec<TenantModel>> {
            Ok(self
                .models
                .iter()
                .filter(|m| m.instance_id == instance)
                .cloned()
                .collect())
        }
        fn dataset(&self, id: &str) -> Result<Option<Dataset>> {
            Ok(self.datasets.iter().find(|d| d.id == id).cloned())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fake::Fake;
    use super::*;
    use crate::facts::Dataset;
    use serde_json::Value;

    fn canvas(id: &str, owner: &str, permission: &str) -> Canvas {
        Canvas {
            id: id.into(),
            user_id: owner.into(),
            permission: permission.into(),
            title: None,
            avatar: None,
            canvas_category: "agent_canvas".into(),
            dsl: Value::Null,
        }
    }

    fn world() -> Fake {
        let mut f = Fake::default();
        f.member("alice", "alice", "owner")
            .member("bob", "bob", "owner")
            .member("bob", "alice", "normal")
            .member("carol", "carol", "owner")
            .member("carol", "alice", "invite")
            .member("dan", "alice", "admin");
        f.canvases = vec![
            canvas("team", "alice", "team"),
            canvas("mine", "alice", "me"),
        ];
        f.datasets = vec![
            Dataset {
                id: "ds-team".into(),
                tenant_id: "alice".into(),
                permission: "team".into(),
                status: "1".into(),
                embd_id: String::new(),
                tenant_embd_id: None,
            },
            Dataset {
                id: "ds-me".into(),
                tenant_id: "alice".into(),
                permission: "me".into(),
                status: "1".into(),
                embd_id: String::new(),
                tenant_embd_id: None,
            },
        ];
        f
    }

    #[test]
    fn the_owner_and_team_members_run_a_team_canvas() {
        let f = world();
        assert!(canvas_for(&f, "alice", "team", Op::Run).is_ok());
        assert!(
            canvas_for(&f, "bob", "team", Op::Run).is_ok(),
            "normal member"
        );
        assert!(
            canvas_for(&f, "dan", "team", Op::Run).is_ok(),
            "admin member"
        );
        assert_eq!(
            canvas_for(&f, "carol", "team", Op::Run).unwrap_err(),
            PermError::Denied,
            "an invite row is not membership"
        );
        assert_eq!(
            canvas_for(&f, "bob", "mine", Op::Run).unwrap_err(),
            PermError::Denied
        );
        assert_eq!(
            canvas_for(&f, "bob", "nope", Op::Run).unwrap_err(),
            PermError::NotFound
        );
        assert_eq!(
            canvas_for(&f, "bob", "team", Op::Delete).unwrap_err(),
            PermError::Denied,
            "a member may not delete"
        );
    }

    #[test]
    fn a_session_belongs_to_the_user_who_made_it() {
        let f = world();
        let session = SessionRow {
            id: "s".into(),
            name: None,
            dialog_id: "team".into(),
            user_id: "bob".into(),
            exp_user_id: None,
            tokens: 0,
            source: None,
            dsl: Value::Null,
            duration: 0.0,
            round: 0,
            thumb_up: 0,
            errors: None,
            version_title: None,
            create_time: None,
            create_date: None,
            update_time: None,
            update_date: None,
        };
        assert!(session_run(&f, "bob", &session).is_ok());
        assert_eq!(session_run(&f, "alice", &session), Err(PermError::Denied));
        let orphan = SessionRow {
            dialog_id: "gone".into(),
            ..session
        };
        assert_eq!(session_run(&f, "bob", &orphan), Err(PermError::NotFound));
    }

    #[test]
    fn datasets_need_both_the_caller_and_the_canvas_owner() {
        let f = world();
        let ids = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(canvas_datasets(&f, "alice", "team", &ids(&["ds-team", "ds-me"])).is_ok());
        assert!(canvas_datasets(&f, "bob", "team", &ids(&["ds-team"])).is_ok());
        let err = canvas_datasets(&f, "bob", "team", &ids(&["ds-me"])).unwrap_err();
        assert_eq!(err, r#"dataset "ds-me": permission: access denied"#);
        let err = canvas_datasets(&f, "alice", "team", &ids(&["missing"])).unwrap_err();
        assert_eq!(err, r#"dataset "missing": permission: resource not found"#);
        assert!(canvas_datasets(&f, "bob", "team", &ids(&["", "  "])).is_ok());
    }

    #[test]
    fn model_rows_are_usable_by_the_tenant_and_its_members() {
        let f = world();
        assert!(model_resource_usable(&f, "alice", "alice").is_ok());
        assert!(model_resource_usable(&f, "bob", "alice").is_ok());
        assert!(model_resource_usable(&f, "dan", "alice").is_ok());
        assert_eq!(
            model_resource_usable(&f, "carol", "alice"),
            Err(PermError::Denied)
        );
        assert!(matches!(
            model_resource_usable(&f, "alice", "nobody"),
            Err(PermError::Invalid(_))
        ));
    }
}

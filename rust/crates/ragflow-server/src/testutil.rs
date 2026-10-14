//! Test doubles shared by the tests of the service and the HTTP layer.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};

use crate::facts::*;
use crate::kv::fake::FakeKv;
use crate::models::Catalog;
use crate::perm::fake::Fake;
use crate::service::Agent;
use crate::sessions::{Result as SResult, SessionStore};
use crate::store::{next_position, Children, HistoryUpdate, ListParams, MessageRow, SessionUpdate};

/// Static rows from [`Fake`], and `api_4_conversation*` kept in memory
/// with the same position arithmetic as the SQL.
/// `(session, position, reference, message_position)`.
pub(crate) type RefRow = (String, i64, Value, Option<i64>);

#[derive(Default)]
pub(crate) struct World {
    pub(crate) base: Fake,
    pub(crate) sessions: Mutex<Vec<SessionRow>>,
    pub(crate) messages: Mutex<Vec<(String, i64, MessageRow)>>,
    pub(crate) references: Mutex<Vec<RefRow>>,
    pub(crate) calls: Mutex<Vec<String>>,
    pub(crate) fail_persist: Mutex<bool>,
}

impl Facts for World {
    fn user_by_access_token(&self, t: &str) -> crate::facts::Result<Option<User>> {
        self.base.user_by_access_token(t)
    }
    fn user_by_id(&self, id: &str) -> crate::facts::Result<Option<User>> {
        self.base.user_by_id(id)
    }
    fn api_token(&self, t: &str) -> crate::facts::Result<Option<ApiToken>> {
        self.base.api_token(t)
    }
    fn beta_tokens(&self, t: &str) -> crate::facts::Result<Vec<ApiToken>> {
        self.base.beta_tokens(t)
    }
    fn membership(&self, u: &str, t: &str) -> crate::facts::Result<Option<Membership>> {
        self.base.membership(u, t)
    }
    fn tenant_owners(&self, t: &str) -> crate::facts::Result<Vec<String>> {
        self.base.tenant_owners(t)
    }
    fn canvas(&self, id: &str) -> crate::facts::Result<Option<Canvas>> {
        self.base.canvas(id)
    }
    fn version(&self, id: &str) -> crate::facts::Result<Option<Version>> {
        self.base.version(id)
    }
    fn latest_version(&self, c: &str) -> crate::facts::Result<Option<Version>> {
        self.base.latest_version(c)
    }
    fn latest_released_version(&self, c: &str) -> crate::facts::Result<Option<Version>> {
        self.base.latest_released_version(c)
    }
    fn session_in_canvas(&self, id: &str, c: &str) -> crate::facts::Result<Option<SessionRow>> {
        Ok(self
            .sessions
            .lock()
            .unwrap()
            .iter()
            .find(|s| s.id == id && s.dialog_id == c)
            .cloned())
    }
    fn session(&self, id: &str) -> crate::facts::Result<Option<SessionRow>> {
        Ok(self
            .sessions
            .lock()
            .unwrap()
            .iter()
            .find(|s| s.id == id)
            .cloned())
    }
    fn tenant(&self, id: &str) -> crate::facts::Result<Option<TenantRow>> {
        self.base.tenant(id)
    }
    fn tenant_model(&self, id: &str) -> crate::facts::Result<Option<TenantModel>> {
        self.base.tenant_model(id)
    }
    fn model_provider(&self, id: &str) -> crate::facts::Result<Option<ModelProvider>> {
        self.base.model_provider(id)
    }
    fn model_provider_by_name(
        &self,
        t: &str,
        n: &str,
    ) -> crate::facts::Result<Option<ModelProvider>> {
        self.base.model_provider_by_name(t, n)
    }
    fn model_instance(&self, id: &str) -> crate::facts::Result<Option<ModelInstance>> {
        self.base.model_instance(id)
    }
    fn model_instance_by_name(
        &self,
        p: &str,
        n: &str,
    ) -> crate::facts::Result<Option<ModelInstance>> {
        self.base.model_instance_by_name(p, n)
    }
    fn models_by_name(&self, p: &str, i: &str, n: &str) -> crate::facts::Result<Vec<TenantModel>> {
        self.base.models_by_name(p, i, n)
    }
    fn instances_of_provider(&self, p: &str) -> crate::facts::Result<Vec<ModelInstance>> {
        self.base.instances_of_provider(p)
    }
    fn models_of_instance(&self, i: &str) -> crate::facts::Result<Vec<TenantModel>> {
        self.base.models_of_instance(i)
    }
    fn dataset(&self, id: &str) -> crate::facts::Result<Option<Dataset>> {
        self.base.dataset(id)
    }
}

impl SessionStore for World {
    fn create_session(
        &self,
        id: &str,
        name: &str,
        canvas_id: &str,
        user_id: &str,
        source: Option<&str>,
        dsl: &Value,
        version_title: Option<&str>,
    ) -> SResult<()> {
        self.calls.lock().unwrap().push("create".into());
        self.sessions.lock().unwrap().push(SessionRow {
            id: id.into(),
            name: Some(name.into()),
            dialog_id: canvas_id.into(),
            user_id: user_id.into(),
            exp_user_id: Some(user_id.into()),
            tokens: 0,
            source: source.map(Into::into),
            dsl: dsl.clone(),
            duration: 0.0,
            round: 0,
            thumb_up: 0,
            errors: None,
            version_title: version_title.map(Into::into),
            create_time: Some(1),
            create_date: None,
            update_time: Some(1),
            update_date: None,
        });
        Ok(())
    }
    fn update_history(
        &self,
        session_id: &str,
        canvas_id: &str,
        user_id: &str,
        update: &SessionUpdate,
        history: &HistoryUpdate,
    ) -> SResult<()> {
        let kind = if history
            .message
            .as_ref()
            .is_some_and(|m| m.role == "assistant")
        {
            "answer"
        } else if history.message.is_some() {
            "question"
        } else {
            "update"
        };
        self.calls.lock().unwrap().push(kind.into());
        if kind == "answer" && *self.fail_persist.lock().unwrap() {
            return Err(StoreError("disk full".into()));
        }
        let mut sessions = self.sessions.lock().unwrap();
        let s = sessions
            .iter_mut()
            .find(|s| s.id == session_id && s.dialog_id == canvas_id && s.user_id == user_id)
            .ok_or_else(|| StoreError("record not found".into()))?;
        if update.bump_round {
            s.round += 1;
        }
        if let Some(e) = &update.errors {
            s.errors = e.clone();
        }
        if let Some(d) = &update.dsl {
            s.dsl = d.clone();
        }
        let mut messages = self.messages.lock().unwrap();
        let mut references = self.references.lock().unwrap();
        let position = match &history.question_id {
            Some(q) => {
                messages
                    .iter()
                    .filter(|(sid, _, m)| {
                        sid == session_id && m.message_id == *q && m.role == "user"
                    })
                    .map(|(_, p, _)| *p)
                    .max()
                    .ok_or_else(|| StoreError("record not found".into()))?
                    + 1
            }
            None => next_position(
                messages
                    .iter()
                    .filter(|(sid, _, _)| sid == session_id)
                    .map(|(_, p, _)| *p)
                    .max(),
                references
                    .iter()
                    .filter(|(sid, _, _, _)| sid == session_id)
                    .map(|(_, p, _, _)| *p)
                    .max(),
            ),
        };
        if let Some(m) = &history.message {
            messages.push((session_id.into(), position, m.clone()));
        }
        if let Some(r) = &history.reference {
            references.push((
                session_id.into(),
                position,
                r.clone(),
                history.message.as_ref().map(|_| position),
            ));
        }
        Ok(())
    }
    fn delete_session(&self, session_id: &str, canvas_id: &str) -> SResult<u64> {
        self.calls.lock().unwrap().push("delete".into());
        let mut sessions = self.sessions.lock().unwrap();
        let before = sessions.len();
        sessions.retain(|s| !(s.id == session_id && s.dialog_id == canvas_id));
        self.messages
            .lock()
            .unwrap()
            .retain(|(sid, _, _)| sid != session_id);
        self.references
            .lock()
            .unwrap()
            .retain(|(sid, _, _, _)| sid != session_id);
        Ok((before - sessions.len()) as u64)
    }
    fn session_ids_of_canvas(&self, c: &str) -> SResult<Vec<String>> {
        Ok(self
            .sessions
            .lock()
            .unwrap()
            .iter()
            .filter(|s| s.dialog_id == c)
            .map(|s| s.id.clone())
            .collect())
    }
    fn messages_and_references(&self, _: &[String]) -> SResult<Children> {
        Ok(BTreeMap::new())
    }
    fn list_sessions(&self, _: &ListParams) -> SResult<(i64, Vec<SessionRow>)> {
        Ok((0, Vec::new()))
    }
    fn list_session_names(&self, _: &str, _: &str) -> SResult<Vec<(String, Option<String>)>> {
        Ok(Vec::new())
    }
}

pub(crate) fn dsl() -> Value {
    json!({
        "globals": {"sys.history": [], "sys.query": "", "sys.conversation_turns": 0},
        "history": [], "path": [], "memory": [],
        "components": {
            "begin": {"obj": {"component_name": "Begin", "params": {}}, "upstream": [], "downstream": ["Message:a"]},
            "Message:a": {"obj": {"component_name": "Message", "params": {"content": ["you said {sys.query} (turn {sys.conversation_turns})"]}},
                "upstream": ["begin"], "downstream": []}
        }
    })
}

pub(crate) fn world() -> Arc<World> {
    let mut base = Fake::default();
    base.member("alice", "alice", "owner")
        .member("bob", "alice", "normal");
    base.canvases = vec![Canvas {
        id: "c1".into(),
        user_id: "alice".into(),
        permission: "team".into(),
        title: Some("t".into()),
        avatar: None,
        canvas_category: "agent_canvas".into(),
        dsl: Value::Null,
    }];
    base.versions = vec![Version {
        id: "v1".into(),
        canvas_id: "c1".into(),
        title: Some("v-title".into()),
        release: false,
        dsl: dsl(),
    }];
    Arc::new(World {
        base,
        ..Default::default()
    })
}

pub(crate) fn agent(w: &Arc<World>, kv: Arc<FakeKv>) -> Arc<Agent> {
    let mut a = Agent::new(w.clone(), w.clone(), kv, Arc::new(Catalog::default()), None);
    a.lease_poll = Duration::from_millis(10);
    a.cancel_poll = Duration::from_millis(10);
    a.lease_refresh = Duration::from_millis(20);
    Arc::new(a)
}

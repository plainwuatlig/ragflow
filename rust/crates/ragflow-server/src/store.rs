//! RAGFlow's MySQL, read through [`Facts`] and written through the session
//! methods at the bottom. Reads mirror the Go DAOs (`internal/dao`); writes
//! are the transactions of `API4ConversationDAO.UpdateHistory`, `Create` and
//! `DeleteBySessionIDAndAgentID`.

use chrono::{Datelike, Local, Timelike};
use mysql::prelude::Queryable;
use mysql::{Opts, Pool, Row, Transaction, TxOpts, Value as My};
use serde_json::Value;

use crate::facts::*;
use crate::gojson;

pub struct Store {
    pool: Pool,
}

fn err(e: impl std::fmt::Display) -> StoreError {
    StoreError(e.to_string())
}

fn text(row: &mut Row, col: &str) -> Option<String> {
    row.take_opt::<Option<String>, _>(col)
        .and_then(|r| r.ok())
        .flatten()
}

fn int(row: &mut Row, col: &str) -> Option<i64> {
    row.take_opt::<Option<i64>, _>(col)
        .and_then(|r| r.ok())
        .flatten()
}

fn float(row: &mut Row, col: &str) -> Option<f64> {
    row.take_opt::<Option<f64>, _>(col)
        .and_then(|r| r.ok())
        .flatten()
}

fn json_col(raw: Option<String>) -> Result<Value> {
    match raw {
        None => Ok(Value::Null),
        Some(s) if s.trim().is_empty() => Ok(Value::Null),
        Some(s) => serde_json::from_str(&s).map_err(err),
    }
}

/// A local wall-clock `datetime` value for `now`, truncated to the second
/// when `truncate` (Go's `Truncate(time.Second)`).
pub fn local_datetime(truncate: bool) -> (i64, My) {
    let now = Local::now();
    let micros = if truncate {
        0
    } else {
        now.timestamp_subsec_micros()
    };
    (
        now.timestamp_millis(),
        My::Date(
            now.year() as u16,
            now.month() as u8,
            now.day() as u8,
            now.hour() as u8,
            now.minute() as u8,
            now.second() as u8,
            micros,
        ),
    )
}

impl Store {
    pub fn connect(url: &str) -> std::result::Result<Self, String> {
        let opts = Opts::from_url(url).map_err(|e| e.to_string())?;
        Ok(Store {
            pool: Pool::new(opts).map_err(|e| e.to_string())?,
        })
    }

    fn one<P: Into<mysql::Params>>(&self, sql: &str, params: P) -> Result<Option<Row>> {
        let mut conn = self.pool.get_conn().map_err(err)?;
        conn.exec_first(sql, params).map_err(err)
    }

    fn all<P: Into<mysql::Params>>(&self, sql: &str, params: P) -> Result<Vec<Row>> {
        let mut conn = self.pool.get_conn().map_err(err)?;
        conn.exec(sql, params).map_err(err)
    }

    fn session_row(mut row: Row) -> Result<SessionRow> {
        let date = |row: &mut Row, col: &str| {
            row.take_opt::<Option<String>, _>(col)
                .and_then(|r| r.ok())
                .flatten()
        };
        Ok(SessionRow {
            id: text(&mut row, "id").unwrap_or_default(),
            name: text(&mut row, "name"),
            dialog_id: text(&mut row, "dialog_id").unwrap_or_default(),
            user_id: text(&mut row, "user_id").unwrap_or_default(),
            exp_user_id: text(&mut row, "exp_user_id"),
            tokens: int(&mut row, "tokens").unwrap_or(0),
            source: text(&mut row, "source"),
            dsl: json_col(text(&mut row, "dsl"))?,
            duration: float(&mut row, "duration").unwrap_or(0.0),
            round: int(&mut row, "round").unwrap_or(0),
            thumb_up: int(&mut row, "thumb_up").unwrap_or(0),
            errors: text(&mut row, "errors"),
            version_title: text(&mut row, "version_title"),
            create_time: int(&mut row, "create_time"),
            create_date: date(&mut row, "create_date"),
            update_time: int(&mut row, "update_time"),
            update_date: date(&mut row, "update_date"),
        })
    }
}

const SESSION_COLUMNS: &str = "id, name, dialog_id, user_id, exp_user_id, tokens, source, \
    CAST(dsl AS CHAR) AS dsl, duration, `round`, thumb_up, errors, version_title, create_time, \
    CAST(create_date AS CHAR) AS create_date, update_time, CAST(update_date AS CHAR) AS update_date";

fn model_row(mut row: Row) -> TenantModel {
    TenantModel {
        id: text(&mut row, "id").unwrap_or_default(),
        model_name: text(&mut row, "model_name").unwrap_or_default(),
        provider_id: text(&mut row, "provider_id").unwrap_or_default(),
        instance_id: text(&mut row, "instance_id").unwrap_or_default(),
        model_type: int(&mut row, "model_type").unwrap_or(0),
        status: text(&mut row, "status").unwrap_or_default(),
        extra: text(&mut row, "extra").unwrap_or_default(),
    }
}

fn instance_row(mut row: Row) -> ModelInstance {
    ModelInstance {
        id: text(&mut row, "id").unwrap_or_default(),
        provider_id: text(&mut row, "provider_id").unwrap_or_default(),
        instance_name: text(&mut row, "instance_name").unwrap_or_default(),
        api_key: text(&mut row, "api_key").unwrap_or_default(),
        status: text(&mut row, "status").unwrap_or_default(),
        extra: text(&mut row, "extra").unwrap_or_default(),
    }
}

fn provider_row(mut row: Row) -> ModelProvider {
    ModelProvider {
        id: text(&mut row, "id").unwrap_or_default(),
        tenant_id: text(&mut row, "tenant_id").unwrap_or_default(),
        provider_name: text(&mut row, "provider_name").unwrap_or_default(),
    }
}

fn token_row(mut row: Row) -> ApiToken {
    ApiToken {
        tenant_id: text(&mut row, "tenant_id").unwrap_or_default(),
        token: text(&mut row, "token").unwrap_or_default(),
        dialog_id: text(&mut row, "dialog_id"),
        beta: text(&mut row, "beta"),
    }
}

fn user_row(mut row: Row) -> User {
    User {
        id: text(&mut row, "id").unwrap_or_default(),
        access_token: text(&mut row, "access_token"),
        is_superuser: int(&mut row, "is_superuser").is_some_and(|v| v != 0),
    }
}

impl Facts for Store {
    fn user_by_access_token(&self, token: &str) -> Result<Option<User>> {
        Ok(self
            .one(
                "SELECT id, access_token, is_superuser FROM `user` WHERE access_token = ? ORDER BY id LIMIT 1",
                (token,),
            )?
            .map(user_row))
    }

    fn user_by_id(&self, id: &str) -> Result<Option<User>> {
        Ok(self
            .one(
                "SELECT id, access_token, is_superuser FROM `user` WHERE id = ? ORDER BY id LIMIT 1",
                (id,),
            )?
            .map(user_row))
    }

    fn api_token(&self, token: &str) -> Result<Option<ApiToken>> {
        Ok(self
            .one(
                "SELECT tenant_id, token, dialog_id, beta FROM api_token WHERE token = ? LIMIT 1",
                (token,),
            )?
            .map(token_row))
    }

    fn beta_tokens(&self, beta: &str) -> Result<Vec<ApiToken>> {
        Ok(self
            .all(
                "SELECT tenant_id, token, dialog_id, beta FROM api_token WHERE beta = ?",
                (beta,),
            )?
            .into_iter()
            .map(token_row)
            .collect())
    }

    fn membership(&self, user_id: &str, tenant_id: &str) -> Result<Option<Membership>> {
        Ok(self
            .one(
                "SELECT role FROM user_tenant WHERE user_id = ? AND tenant_id = ? AND status = '1' \
                 ORDER BY id LIMIT 1",
                (user_id, tenant_id),
            )?
            .map(|mut r| Membership {
                role: text(&mut r, "role").unwrap_or_default(),
            }))
    }

    fn tenant_owners(&self, tenant_id: &str) -> Result<Vec<String>> {
        Ok(self
            .all(
                "SELECT user_id FROM user_tenant WHERE tenant_id = ? AND role = 'owner' AND status = '1'",
                (tenant_id,),
            )?
            .into_iter()
            .filter_map(|mut r| text(&mut r, "user_id"))
            .collect())
    }

    fn canvas(&self, id: &str) -> Result<Option<Canvas>> {
        let Some(mut row) = self.one(
            "SELECT id, user_id, permission, title, avatar, canvas_category, CAST(dsl AS CHAR) AS dsl \
             FROM user_canvas WHERE id = ? LIMIT 1",
            (id,),
        )?
        else {
            return Ok(None);
        };
        Ok(Some(Canvas {
            id: text(&mut row, "id").unwrap_or_default(),
            user_id: text(&mut row, "user_id").unwrap_or_default(),
            permission: text(&mut row, "permission").unwrap_or_default(),
            title: text(&mut row, "title"),
            avatar: text(&mut row, "avatar"),
            canvas_category: text(&mut row, "canvas_category").unwrap_or_default(),
            dsl: json_col(text(&mut row, "dsl"))?,
        }))
    }

    fn version(&self, id: &str) -> Result<Option<Version>> {
        self.version_query("WHERE id = ?", id)
    }

    fn latest_version(&self, canvas_id: &str) -> Result<Option<Version>> {
        self.version_query(
            "WHERE user_canvas_id = ? ORDER BY create_time DESC",
            canvas_id,
        )
    }

    fn latest_released_version(&self, canvas_id: &str) -> Result<Option<Version>> {
        self.version_query(
            "WHERE user_canvas_id = ? AND `release` = 1 ORDER BY update_time DESC",
            canvas_id,
        )
    }

    fn session_in_canvas(&self, id: &str, canvas_id: &str) -> Result<Option<SessionRow>> {
        self.one(
            &format!("SELECT {SESSION_COLUMNS} FROM api_4_conversation WHERE id = ? AND dialog_id = ? LIMIT 1"),
            (id, canvas_id),
        )?
        .map(Self::session_row)
        .transpose()
    }

    fn session(&self, id: &str) -> Result<Option<SessionRow>> {
        self.one(
            &format!("SELECT {SESSION_COLUMNS} FROM api_4_conversation WHERE id = ? LIMIT 1"),
            (id,),
        )?
        .map(Self::session_row)
        .transpose()
    }

    fn tenant(&self, id: &str) -> Result<Option<TenantRow>> {
        Ok(self
            .one(
                "SELECT id, llm_id, tenant_llm_id, embd_id, tenant_embd_id, rerank_id, tenant_rerank_id \
                 FROM tenant WHERE id = ? LIMIT 1",
                (id,),
            )?
            .map(|mut r| TenantRow {
                id: text(&mut r, "id").unwrap_or_default(),
                llm_id: text(&mut r, "llm_id").unwrap_or_default(),
                tenant_llm_id: text(&mut r, "tenant_llm_id"),
                embd_id: text(&mut r, "embd_id").unwrap_or_default(),
                tenant_embd_id: text(&mut r, "tenant_embd_id"),
                rerank_id: text(&mut r, "rerank_id").unwrap_or_default(),
                tenant_rerank_id: text(&mut r, "tenant_rerank_id"),
            }))
    }

    fn tenant_model(&self, id: &str) -> Result<Option<TenantModel>> {
        Ok(self
            .one(
                "SELECT id, model_name, provider_id, instance_id, model_type, status, extra \
                 FROM tenant_model WHERE id = ? LIMIT 1",
                (id,),
            )?
            .map(model_row))
    }

    fn model_provider(&self, id: &str) -> Result<Option<ModelProvider>> {
        Ok(self
            .one(
                "SELECT id, tenant_id, provider_name FROM tenant_model_provider WHERE id = ? LIMIT 1",
                (id,),
            )?
            .map(provider_row))
    }

    fn model_provider_by_name(
        &self,
        tenant_id: &str,
        provider_name: &str,
    ) -> Result<Option<ModelProvider>> {
        Ok(self
            .one(
                "SELECT id, tenant_id, provider_name FROM tenant_model_provider \
                 WHERE tenant_id = ? AND provider_name = ? ORDER BY id LIMIT 1",
                (tenant_id, provider_name),
            )?
            .map(provider_row))
    }

    fn model_instance(&self, id: &str) -> Result<Option<ModelInstance>> {
        Ok(self
            .one(
                "SELECT id, provider_id, instance_name, api_key, status, extra \
                 FROM tenant_model_instance WHERE id = ? LIMIT 1",
                (id,),
            )?
            .map(instance_row))
    }

    fn model_instance_by_name(
        &self,
        provider_id: &str,
        instance_name: &str,
    ) -> Result<Option<ModelInstance>> {
        Ok(self
            .one(
                "SELECT id, provider_id, instance_name, api_key, status, extra \
                 FROM tenant_model_instance WHERE provider_id = ? AND instance_name = ? ORDER BY id LIMIT 1",
                (provider_id, instance_name),
            )?
            .map(instance_row))
    }

    fn models_by_name(
        &self,
        provider_id: &str,
        instance_id: &str,
        model_name: &str,
    ) -> Result<Vec<TenantModel>> {
        Ok(self
            .all(
                "SELECT id, model_name, provider_id, instance_id, model_type, status, extra \
                 FROM tenant_model WHERE provider_id = ? AND instance_id = ? AND model_name = ?",
                (provider_id, instance_id, model_name),
            )?
            .into_iter()
            .map(model_row)
            .collect())
    }

    fn instances_of_provider(&self, provider_id: &str) -> Result<Vec<ModelInstance>> {
        Ok(self
            .all(
                "SELECT id, provider_id, instance_name, api_key, status, extra \
                 FROM tenant_model_instance WHERE provider_id = ? ORDER BY create_time DESC",
                (provider_id,),
            )?
            .into_iter()
            .map(instance_row)
            .collect())
    }

    fn models_of_instance(&self, instance_id: &str) -> Result<Vec<TenantModel>> {
        Ok(self
            .all(
                "SELECT id, model_name, provider_id, instance_id, model_type, status, extra \
                 FROM tenant_model WHERE instance_id = ?",
                (instance_id,),
            )?
            .into_iter()
            .map(model_row)
            .collect())
    }

    fn dataset(&self, id: &str) -> Result<Option<Dataset>> {
        Ok(self
            .one(
                "SELECT id, tenant_id, permission, status, embd_id, tenant_embd_id \
                 FROM knowledgebase WHERE id = ? AND status = '1' LIMIT 1",
                (id,),
            )?
            .map(|mut r| Dataset {
                id: text(&mut r, "id").unwrap_or_default(),
                tenant_id: text(&mut r, "tenant_id").unwrap_or_default(),
                permission: text(&mut r, "permission").unwrap_or_default(),
                status: text(&mut r, "status").unwrap_or_default(),
                embd_id: text(&mut r, "embd_id").unwrap_or_default(),
                tenant_embd_id: text(&mut r, "tenant_embd_id"),
            }))
    }
}

impl Store {
    fn version_query(&self, tail: &str, arg: &str) -> Result<Option<Version>> {
        let sql = format!(
            "SELECT id, user_canvas_id, title, `release` AS rel, CAST(dsl AS CHAR) AS dsl \
             FROM user_canvas_version {tail} LIMIT 1"
        );
        let Some(mut row) = self.one(&sql, (arg,))? else {
            return Ok(None);
        };
        Ok(Some(Version {
            id: text(&mut row, "id").unwrap_or_default(),
            canvas_id: text(&mut row, "user_canvas_id").unwrap_or_default(),
            title: text(&mut row, "title"),
            release: int(&mut row, "rel").is_some_and(|v| v != 0),
            dsl: json_col(text(&mut row, "dsl"))?,
        }))
    }
}

/// A session's message rows and reference rows, hydrated, by session id.
pub type Children = std::collections::BTreeMap<String, (Vec<Value>, Vec<Value>)>;

/// One message row to write (`api_4_conversation_message`).
#[derive(Debug, Clone, PartialEq)]
pub struct MessageRow {
    pub message_id: String,
    pub role: &'static str,
    pub content: String,
    pub created_at: f64,
}

/// What `ConversationHistoryUpdate` carries.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HistoryUpdate {
    pub message: Option<MessageRow>,
    /// The user turn the answer belongs to: its position + 1 is the slot.
    pub question_id: Option<String>,
    /// The normalised reference to append; `None` appends none.
    pub reference: Option<Value>,
}

/// Column updates of the session row itself.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SessionUpdate {
    /// `round = COALESCE(round, 0) + 1`.
    pub bump_round: bool,
    /// `errors = NULL` (`Some(None)`) or `errors = <reason>`.
    pub errors: Option<Option<String>>,
    pub dsl: Option<Value>,
}

/// `max(max(message position), max(reference position)) + 2`, the first
/// being -2 + 2 = 0.
pub fn next_position(max_message: Option<i64>, max_reference: Option<i64>) -> i64 {
    max_message.unwrap_or(-2).max(max_reference.unwrap_or(-2)) + 2
}

impl Store {
    /// `API4ConversationDAO.Create`: the session row (children empty).
    #[allow(clippy::too_many_arguments)]
    pub fn create_session(
        &self,
        id: &str,
        name: &str,
        canvas_id: &str,
        user_id: &str,
        source: Option<&str>,
        dsl: &Value,
        version_title: Option<&str>,
    ) -> Result<()> {
        let (ms, date) = local_datetime(true);
        let mut conn = self.pool.get_conn().map_err(err)?;
        conn.exec_drop(
            "INSERT INTO api_4_conversation (id, name, dialog_id, user_id, exp_user_id, tokens, source, \
             dsl, duration, `round`, thumb_up, errors, version_title, create_time, create_date, \
             update_time, update_date) VALUES (?, ?, ?, ?, ?, 0, ?, ?, 0, 0, 0, NULL, ?, ?, ?, ?, ?)",
            (
                id,
                name,
                canvas_id,
                user_id,
                user_id,
                source,
                gojson::marshal(dsl),
                version_title,
                ms,
                date.clone(),
                ms,
                date,
            ),
        )
        .map_err(err)
    }

    /// `API4ConversationDAO.UpdateHistory`: lock the session row, update it,
    /// then write the addressed child rows, all in one transaction.
    pub fn update_history(
        &self,
        session_id: &str,
        canvas_id: &str,
        user_id: &str,
        update: &SessionUpdate,
        history: &HistoryUpdate,
    ) -> Result<()> {
        let mut conn = self.pool.get_conn().map_err(err)?;
        let mut tx = conn.start_transaction(TxOpts::default()).map_err(err)?;
        let locked: Option<String> = tx
            .exec_first(
                "SELECT id FROM api_4_conversation WHERE id = ? AND dialog_id = ? AND user_id = ? \
                 LIMIT 1 FOR UPDATE",
                (session_id, canvas_id, user_id),
            )
            .map_err(err)?;
        if locked.is_none() {
            return Err(StoreError("record not found".into()));
        }
        let (ms, date) = local_datetime(true);
        let mut sets: Vec<String> = Vec::new();
        let mut params: Vec<My> = Vec::new();
        if update.bump_round {
            sets.push("`round` = COALESCE(`round`, 0) + 1".into());
        }
        match &update.errors {
            Some(None) => sets.push("errors = NULL".into()),
            Some(Some(reason)) => {
                sets.push("errors = ?".into());
                params.push(reason.as_str().into());
            }
            None => {}
        }
        if let Some(dsl) = &update.dsl {
            sets.push("dsl = ?".into());
            params.push(gojson::marshal(dsl).into());
        }
        sets.push("update_time = ?".into());
        params.push(ms.into());
        sets.push("update_date = ?".into());
        params.push(date);
        params.extend([session_id.into(), canvas_id.into(), user_id.into()]);
        tx.exec_drop(
            format!(
                "UPDATE api_4_conversation SET {} WHERE id = ? AND dialog_id = ? AND user_id = ?",
                sets.join(", ")
            ),
            params,
        )
        .map_err(err)?;
        write_history(&mut tx, session_id, history)?;
        tx.commit().map_err(err)
    }

    /// `DeleteBySessionIDAndAgentID`: the session and its child rows.
    pub fn delete_session(&self, session_id: &str, canvas_id: &str) -> Result<u64> {
        let mut conn = self.pool.get_conn().map_err(err)?;
        let mut tx = conn.start_transaction(TxOpts::default()).map_err(err)?;
        let ids: Vec<String> = tx
            .exec(
                "SELECT id FROM api_4_conversation WHERE id = ? AND dialog_id = ? ORDER BY id FOR UPDATE",
                (session_id, canvas_id),
            )
            .map_err(err)?;
        for id in &ids {
            for table in ["api_4_conversation_message", "api_4_conversation_reference"] {
                tx.exec_drop(
                    format!("DELETE FROM {table} WHERE conversation_id = ?"),
                    (id,),
                )
                .map_err(err)?;
            }
        }
        tx.exec_drop(
            "DELETE FROM api_4_conversation WHERE id = ? AND dialog_id = ?",
            (session_id, canvas_id),
        )
        .map_err(err)?;
        let affected = tx.affected_rows();
        tx.commit().map_err(err)?;
        Ok(affected)
    }

    /// Session ids of one canvas (`ListIDsByAgentID`).
    pub fn session_ids_of_canvas(&self, canvas_id: &str) -> Result<Vec<String>> {
        let mut conn = self.pool.get_conn().map_err(err)?;
        conn.exec(
            "SELECT id FROM api_4_conversation WHERE dialog_id = ?",
            (canvas_id,),
        )
        .map_err(err)
    }
}

/// `updateConversationHistory` for the two shapes the run path uses: a
/// message with an optional reference, positioned either after the last
/// existing row or in the slot after its question.
fn write_history(tx: &mut Transaction<'_>, session_id: &str, h: &HistoryUpdate) -> Result<()> {
    if h.message.is_none() && h.reference.is_none() {
        return Ok(());
    }
    let position = match &h.question_id {
        Some(question_id) => {
            let q: Option<i64> = tx
                .exec_first(
                    "SELECT position FROM api_4_conversation_message \
                     WHERE conversation_id = ? AND message_id = ? AND role = 'user' \
                     ORDER BY position DESC LIMIT 1",
                    (session_id, question_id),
                )
                .map_err(err)?;
            q.ok_or_else(|| StoreError("record not found".into()))? + 1
        }
        None => {
            let m: Option<i64> = tx
                .exec_first(
                    "SELECT COALESCE(MAX(position), -2) FROM api_4_conversation_message WHERE conversation_id = ?",
                    (session_id,),
                )
                .map_err(err)?;
            let r: Option<i64> = tx
                .exec_first(
                    "SELECT COALESCE(MAX(position), -2) FROM api_4_conversation_reference WHERE conversation_id = ?",
                    (session_id,),
                )
                .map_err(err)?;
            next_position(m, r)
        }
    };
    if let Some(m) = &h.message {
        tx.exec_drop(
            "INSERT INTO api_4_conversation_message (conversation_id, position, message_id, role, \
             content, content_type, status, thumb_up, feedback, created_at, metadata) \
             VALUES (?, ?, ?, ?, ?, 'text', NULL, NULL, NULL, ?, '{}')",
            (
                session_id,
                position,
                m.message_id.as_str(),
                m.role,
                m.content.as_str(),
                m.created_at,
            ),
        )
        .map_err(err)?;
    }
    if let Some(reference) = &h.reference {
        let message_position: Option<i64> = h.message.as_ref().map(|_| position);
        tx.exec_drop(
            "INSERT INTO api_4_conversation_reference (conversation_id, position, reference, message_position) \
             VALUES (?, ?, ?, ?)",
            (session_id, position, gojson::marshal(reference), message_position),
        )
        .map_err(err)?;
    }
    Ok(())
}

/// `ListAgentSessionsParams`.
#[derive(Debug, Clone, Default)]
pub struct ListParams {
    pub canvas_id: String,
    pub page: i64,
    pub page_size: i64,
    pub order_by: String,
    pub desc: bool,
    pub session_id: String,
    pub user_id: String,
    pub include_dsl: bool,
    pub keywords: String,
    /// Naive local `YYYY-MM-DD HH:MM:SS` bounds on the date column.
    pub from_date: Option<String>,
    pub to_date: Option<String>,
    pub exp_user_id: String,
}

/// `normalizeAgentSessionOrderBy`.
fn order_column(order_by: &str) -> &'static str {
    match order_by {
        "id" => "id",
        "name" => "name",
        "create_time" => "create_time",
        "create_date" => "create_date",
        "update_time" => "update_time",
        "update_date" => "update_date",
        "tokens" => "tokens",
        "duration" => "duration",
        "round" => "`round`",
        "thumb_up" => "thumb_up",
        _ => "update_time",
    }
}

impl Store {
    /// `ChatSessionDAO.ListAgentSessions`: the page of sessions and the
    /// total before paging.
    pub fn list_sessions(&self, p: &ListParams) -> Result<(i64, Vec<SessionRow>)> {
        let mut wheres = vec!["dialog_id = ?".to_string()];
        let mut params: Vec<My> = vec![p.canvas_id.as_str().into()];
        if !p.session_id.is_empty() {
            wheres.push("id = ?".into());
            params.push(p.session_id.as_str().into());
        }
        if !p.user_id.is_empty() {
            wheres.push("user_id = ?".into());
            params.push(p.user_id.as_str().into());
        }
        if !p.keywords.is_empty() {
            let pattern = format!("%{}%", p.keywords.to_lowercase());
            wheres.push(
                "(LOWER(id) LIKE ? OR LOWER(name) LIKE ? OR EXISTS (SELECT 1 FROM \
                 api_4_conversation_message AS cm WHERE cm.conversation_id = api_4_conversation.id \
                 AND LOWER(cm.content) LIKE ?))"
                    .into(),
            );
            for _ in 0..3 {
                params.push(pattern.as_str().into());
            }
        }
        let date_column = if p.order_by.starts_with("update_") {
            "update_date"
        } else {
            "create_date"
        };
        if let Some(from) = &p.from_date {
            wheres.push(format!("{date_column} >= ?"));
            params.push(from.as_str().into());
        }
        if let Some(to) = &p.to_date {
            wheres.push(format!("{date_column} <= ?"));
            params.push(to.as_str().into());
        }
        if !p.exp_user_id.is_empty() {
            wheres.push("exp_user_id = ?".into());
            params.push(p.exp_user_id.as_str().into());
        }
        let clause = wheres.join(" AND ");
        let total: Option<i64> = self
            .one(
                &format!("SELECT COUNT(*) FROM api_4_conversation WHERE {clause}"),
                params.clone(),
            )?
            .and_then(|mut r| r.take_opt::<i64, usize>(0).and_then(|v| v.ok()));
        let page = if p.page <= 0 { 1 } else { p.page };
        let size = if p.page_size <= 0 { 30 } else { p.page_size };
        let mut columns = SESSION_COLUMNS.to_string();
        if !p.include_dsl {
            columns = columns.replace("CAST(dsl AS CHAR) AS dsl", "NULL AS dsl");
        }
        let sql = format!(
            "SELECT {columns} FROM api_4_conversation WHERE {clause} ORDER BY {} {} LIMIT {} OFFSET {}",
            order_column(&p.order_by),
            if p.desc { "DESC" } else { "ASC" },
            size,
            (page - 1) * size
        );
        let rows = self.all(&sql, params)?;
        let sessions = rows
            .into_iter()
            .map(Self::session_row)
            .collect::<Result<Vec<_>>>()?;
        Ok((total.unwrap_or(0), sessions))
    }

    /// `ListAgentSessionNames`: `(id, name)` of one user's sessions.
    pub fn list_session_names(
        &self,
        canvas_id: &str,
        exp_user_id: &str,
    ) -> Result<Vec<(String, Option<String>)>> {
        Ok(self
            .all(
                "SELECT id, name FROM api_4_conversation WHERE dialog_id = ? AND exp_user_id = ? \
                 ORDER BY create_date DESC",
                (canvas_id, exp_user_id),
            )?
            .into_iter()
            .map(|mut r| (text(&mut r, "id").unwrap_or_default(), text(&mut r, "name")))
            .collect())
    }
}

/// The message and reference rows of one or more sessions, hydrated the way
/// `hydrateAPIConversations` does: ordered by position, a message's stored
/// metadata overlaid by its typed columns.
impl Store {
    pub fn messages_and_references(&self, session_ids: &[String]) -> Result<Children> {
        let mut out: Children = session_ids
            .iter()
            .map(|id| (id.clone(), (Vec::new(), Vec::new())))
            .collect();
        if session_ids.is_empty() {
            return Ok(out);
        }
        let marks = vec!["?"; session_ids.len()].join(",");
        let params: Vec<My> = session_ids.iter().map(|s| s.as_str().into()).collect();
        let mut conn = self.pool.get_conn().map_err(err)?;
        let rows: Vec<Row> = conn
            .exec(
                format!(
                    "SELECT conversation_id, message_id, role, content, content_type, status, thumb_up, \
                     feedback, created_at, metadata FROM api_4_conversation_message \
                     WHERE conversation_id IN ({marks}) ORDER BY conversation_id, position"
                ),
                params.clone(),
            )
            .map_err(err)?;
        for mut row in rows {
            let id = text(&mut row, "conversation_id").unwrap_or_default();
            let message = hydrate_message(&mut row)?;
            if let Some(slot) = out.get_mut(&id) {
                slot.0.push(message);
            }
        }
        let rows: Vec<Row> = conn
            .exec(
                format!(
                    "SELECT conversation_id, reference FROM api_4_conversation_reference \
                     WHERE conversation_id IN ({marks}) ORDER BY conversation_id, position"
                ),
                params,
            )
            .map_err(err)?;
        for mut row in rows {
            let id = text(&mut row, "conversation_id").unwrap_or_default();
            let raw = text(&mut row, "reference").unwrap_or_default();
            let value = serde_json::from_str(&raw).unwrap_or(Value::Null);
            if let Some(slot) = out.get_mut(&id) {
                slot.1.push(value);
            }
        }
        Ok(out)
    }
}

/// `marshalMessage`: the typed columns over the stored metadata.
fn hydrate_message(row: &mut Row) -> Result<Value> {
    let metadata: serde_json::Map<String, Value> = text(row, "metadata")
        .filter(|m| !m.is_empty())
        .map(|m| serde_json::from_str(&m).map_err(err))
        .transpose()?
        .unwrap_or_default();
    let mut message = metadata;
    let content_type = text(row, "content_type").unwrap_or_default();
    let content = text(row, "content");
    if let Some(v) = text(row, "message_id") {
        message.insert("id".into(), Value::String(v));
    }
    if let Some(v) = text(row, "role") {
        message.insert("role".into(), Value::String(v));
    }
    if let Some(c) = content {
        let value = if content_type == "json" {
            serde_json::from_str(&c).unwrap_or(Value::String(c))
        } else {
            Value::String(c)
        };
        message.insert("content".into(), value);
    }
    if let Some(v) = text(row, "status") {
        message.insert("status".into(), Value::String(v));
    }
    if let Some(v) = int(row, "thumb_up") {
        message.insert("thumbup".into(), Value::Bool(v != 0));
    }
    if let Some(v) = text(row, "feedback") {
        message.insert("feedback".into(), Value::String(v));
    }
    if let Some(v) = float(row, "created_at") {
        if let Some(n) = serde_json::Number::from_f64(v) {
            message.insert("created_at".into(), Value::Number(n));
        }
    }
    Ok(Value::Object(message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_follow_the_larger_of_the_two_tables() {
        assert_eq!(next_position(None, None), 0);
        assert_eq!(next_position(Some(-2), Some(-2)), 0);
        assert_eq!(next_position(Some(0), None), 2);
        assert_eq!(next_position(Some(1), Some(1)), 3);
        assert_eq!(next_position(Some(4), Some(1)), 6);
        assert_eq!(next_position(Some(0), Some(1)), 3);
    }
}

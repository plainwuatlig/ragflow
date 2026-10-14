//! Kvrocks (Redis protocol, database 1, no key prefix): the session secret
//! (read only) and the run lease and cancel marker that let several server
//! processes, Go or Rust, agree on who runs a session and how to stop it.
//! The key names, the Lua scripts and the timings are upstream's
//! (`internal/agent/canvas/run_tracker.go`, `cancel.go`).

use std::sync::Mutex;
use std::time::Duration;

use redis::{Client, Connection, Script};

/// Kvrocks key of the secret that signs UI session tokens.
pub const SECRET_KEY: &str = "ragflow:system:secret_key";
/// The lease lives this long and is refreshed by its owner.
pub const LEASE_TTL: Duration = Duration::from_secs(30);
/// How long a cancel marker lives.
pub const CANCEL_TTL: Duration = Duration::from_secs(24 * 60 * 60);
/// How often a run looks for a cancel marker.
pub const CANCEL_POLL: Duration = Duration::from_millis(500);

pub fn active_key(session_id: &str) -> String {
    format!("agent:active-session:{session_id}")
}

pub fn cancel_key(session_id: &str) -> String {
    format!("{session_id}-cancel")
}

/// `canvas.ActiveSession`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Active {
    pub session_id: String,
    pub token: String,
    pub user_id: String,
    pub canvas_id: String,
    pub run_id: String,
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
#[error("{0}")]
pub struct KvError(pub String);

impl From<redis::RedisError> for KvError {
    fn from(e: redis::RedisError) -> Self {
        KvError(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, KvError>;

/// The operations the server needs from Kvrocks.
pub trait Kv: Send + Sync {
    /// `GET ragflow:system:secret_key`; never creates it.
    fn secret_key(&self) -> Result<Option<String>>;
    /// `RegisterActiveSession`: take the session's lease unless held, and
    /// drop a cancel marker left by an older run.
    fn register(&self, active: &Active) -> Result<bool>;
    /// `RefreshActiveSession`: extend the lease while `token` owns it.
    fn refresh(&self, session_id: &str, token: &str) -> Result<bool>;
    /// `ReleaseActiveSession`: delete lease and cancel marker if `token` owns it.
    fn release(&self, session_id: &str, token: &str) -> Result<bool>;
    /// `RequestCancelActiveSession`: set the cancel marker if `token` owns the lease.
    fn request_cancel(&self, session_id: &str, token: &str) -> Result<bool>;
    /// `GetActiveSession`.
    fn active(&self, session_id: &str) -> Result<Option<Active>>;
    /// `CancelRequested`: a non-empty `{session_id}-cancel`.
    fn cancel_requested(&self, session_id: &str) -> Result<bool>;
    /// `canvas.RequestCancel`: set the marker unconditionally.
    fn set_cancel(&self, session_id: &str) -> Result<()>;
    /// A plain `GET`.
    fn get_raw(&self, key: &str) -> Result<Option<String>>;
}

const REGISTER: &str = r#"
if redis.call("EXISTS", KEYS[1]) == 1 then return 0 end
redis.call("DEL", KEYS[2])
redis.call("HSET", KEYS[1],
  "session_id", ARGV[1], "token", ARGV[2], "user_id", ARGV[3],
  "canvas_id", ARGV[4], "run_id", ARGV[5], "started_at", ARGV[6])
redis.call("PEXPIRE", KEYS[1], ARGV[7])
return 1"#;

const REFRESH: &str = r#"if redis.call("HGET", KEYS[1], "token") == ARGV[1] then return redis.call("PEXPIRE", KEYS[1], ARGV[2]) end return 0"#;

const RELEASE: &str = r#"
if redis.call("HGET", KEYS[1], "token") ~= ARGV[1] then return 0 end
redis.call("DEL", KEYS[2])
redis.call("DEL", KEYS[1])
return 1"#;

const REQUEST_CANCEL: &str = r#"
if redis.call("HGET", KEYS[1], "token") ~= ARGV[1] then return 0 end
redis.call("SET", KEYS[2], "x", "PX", ARGV[2])
return 1"#;

/// A Redis-protocol client with a small pool of connections.
pub struct Kvrocks {
    client: Client,
    pool: Mutex<Vec<Connection>>,
}

impl Kvrocks {
    /// `url` as `redis://[:password@]host:port/1`.
    pub fn connect(url: &str) -> Result<Self> {
        let client = Client::open(url)?;
        let kv = Kvrocks {
            client,
            pool: Mutex::new(Vec::new()),
        };
        kv.with(|c| redis::cmd("PING").query::<String>(c))?;
        Ok(kv)
    }

    fn with<T>(&self, f: impl FnOnce(&mut Connection) -> redis::RedisResult<T>) -> Result<T> {
        let pooled = self.pool.lock().expect("pool lock").pop();
        let mut conn = match pooled {
            Some(c) => c,
            None => self.client.get_connection()?,
        };
        let out = f(&mut conn)?;
        let mut pool = self.pool.lock().expect("pool lock");
        if pool.len() < 16 {
            pool.push(conn);
        }
        Ok(out)
    }
}

fn millis(d: Duration) -> String {
    d.as_millis().to_string()
}

impl Kv for Kvrocks {
    fn secret_key(&self) -> Result<Option<String>> {
        self.with(|c| redis::cmd("GET").arg(SECRET_KEY).query::<Option<String>>(c))
            .map(|v| v.filter(|s| !s.is_empty()))
    }

    fn register(&self, a: &Active) -> Result<bool> {
        let started = chrono::Utc::now().timestamp_millis().to_string();
        let n: i64 = self.with(|c| {
            Script::new(REGISTER)
                .key(active_key(&a.session_id))
                .key(cancel_key(&a.session_id))
                .arg(&a.session_id)
                .arg(&a.token)
                .arg(&a.user_id)
                .arg(&a.canvas_id)
                .arg(&a.run_id)
                .arg(&started)
                .arg(millis(LEASE_TTL))
                .invoke(c)
        })?;
        Ok(n == 1)
    }

    fn refresh(&self, session_id: &str, token: &str) -> Result<bool> {
        let n: i64 = self.with(|c| {
            Script::new(REFRESH)
                .key(active_key(session_id))
                .arg(token)
                .arg(millis(LEASE_TTL))
                .invoke(c)
        })?;
        Ok(n == 1)
    }

    fn release(&self, session_id: &str, token: &str) -> Result<bool> {
        let n: i64 = self.with(|c| {
            Script::new(RELEASE)
                .key(active_key(session_id))
                .key(cancel_key(session_id))
                .arg(token)
                .invoke(c)
        })?;
        Ok(n == 1)
    }

    fn request_cancel(&self, session_id: &str, token: &str) -> Result<bool> {
        let n: i64 = self.with(|c| {
            Script::new(REQUEST_CANCEL)
                .key(active_key(session_id))
                .key(cancel_key(session_id))
                .arg(token)
                .arg(millis(CANCEL_TTL))
                .invoke(c)
        })?;
        Ok(n == 1)
    }

    fn active(&self, session_id: &str) -> Result<Option<Active>> {
        let fields: std::collections::HashMap<String, String> =
            self.with(|c| redis::cmd("HGETALL").arg(active_key(session_id)).query(c))?;
        if fields.is_empty() {
            return Ok(None);
        }
        let get = |k: &str| fields.get(k).cloned().unwrap_or_default();
        Ok(Some(Active {
            session_id: get("session_id"),
            token: get("token"),
            user_id: get("user_id"),
            canvas_id: get("canvas_id"),
            run_id: get("run_id"),
        }))
    }

    fn cancel_requested(&self, session_id: &str) -> Result<bool> {
        let v: Option<String> =
            self.with(|c| redis::cmd("GET").arg(cancel_key(session_id)).query(c))?;
        Ok(v.is_some_and(|s| !s.is_empty()))
    }

    fn get_raw(&self, key: &str) -> Result<Option<String>> {
        self.with(|c| redis::cmd("GET").arg(key).query::<Option<String>>(c))
    }

    fn set_cancel(&self, session_id: &str) -> Result<()> {
        self.with(|c| {
            redis::cmd("SET")
                .arg(cancel_key(session_id))
                .arg("x")
                .arg("PX")
                .arg(millis(CANCEL_TTL))
                .query::<()>(c)
        })
    }
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) mod fake {
    //! An in-memory [`Kv`] with the scripts' semantics, for tests of the
    //! code around the lease.
    use std::collections::HashMap;
    use std::sync::Mutex;

    use super::*;

    #[derive(Default)]
    pub struct FakeKv {
        pub secret: Mutex<Option<String>>,
        pub leases: Mutex<HashMap<String, Active>>,
        pub cancels: Mutex<std::collections::HashSet<String>>,
        /// When set, every call fails (a Kvrocks outage).
        pub down: Mutex<bool>,
    }

    impl FakeKv {
        fn check(&self) -> Result<()> {
            if *self.down.lock().unwrap() {
                Err(KvError("connection refused".into()))
            } else {
                Ok(())
            }
        }
    }

    impl Kv for FakeKv {
        fn secret_key(&self) -> Result<Option<String>> {
            self.check()?;
            Ok(self.secret.lock().unwrap().clone())
        }
        fn register(&self, a: &Active) -> Result<bool> {
            self.check()?;
            let mut leases = self.leases.lock().unwrap();
            if leases.contains_key(&a.session_id) {
                return Ok(false);
            }
            self.cancels.lock().unwrap().remove(&a.session_id);
            leases.insert(a.session_id.clone(), a.clone());
            Ok(true)
        }
        fn refresh(&self, session_id: &str, token: &str) -> Result<bool> {
            self.check()?;
            Ok(self
                .leases
                .lock()
                .unwrap()
                .get(session_id)
                .is_some_and(|a| a.token == token))
        }
        fn release(&self, session_id: &str, token: &str) -> Result<bool> {
            self.check()?;
            let mut leases = self.leases.lock().unwrap();
            if leases.get(session_id).is_some_and(|a| a.token == token) {
                leases.remove(session_id);
                self.cancels.lock().unwrap().remove(session_id);
                return Ok(true);
            }
            Ok(false)
        }
        fn request_cancel(&self, session_id: &str, token: &str) -> Result<bool> {
            self.check()?;
            if self
                .leases
                .lock()
                .unwrap()
                .get(session_id)
                .is_some_and(|a| a.token == token)
            {
                self.cancels.lock().unwrap().insert(session_id.to_string());
                return Ok(true);
            }
            Ok(false)
        }
        fn active(&self, session_id: &str) -> Result<Option<Active>> {
            self.check()?;
            Ok(self.leases.lock().unwrap().get(session_id).cloned())
        }
        fn cancel_requested(&self, session_id: &str) -> Result<bool> {
            self.check()?;
            Ok(self.cancels.lock().unwrap().contains(session_id))
        }
        fn get_raw(&self, _key: &str) -> Result<Option<String>> {
            self.check()?;
            Ok(None)
        }
        fn set_cancel(&self, session_id: &str) -> Result<()> {
            self.check()?;
            self.cancels.lock().unwrap().insert(session_id.to_string());
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_names_are_upstreams() {
        assert_eq!(active_key("abc"), "agent:active-session:abc");
        assert_eq!(cancel_key("abc"), "abc-cancel");
        assert_eq!(SECRET_KEY, "ragflow:system:secret_key");
    }
}

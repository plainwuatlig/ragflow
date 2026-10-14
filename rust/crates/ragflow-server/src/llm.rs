//! Chat calls for the Agent and Categorize components: credentials by the
//! components' rule ([`crate::models::component_chat_credentials`]), then an
//! HTTP call to the provider's endpoint.
//!
//! Two wire protocols are spoken, chosen by the provider's catalog entry:
//! the OpenAI `chat/completions` protocol (any provider whose chat
//! `url_suffix` ends in `chat/completions`) and Ollama's `api/chat`. Any
//! other provider is refused with an error naming it.

use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ragflow_runtime::models::{ChatMessage, ChatOptions, ModelRef, Models};
use serde_json::{json, Value};

use crate::facts::Facts;
use crate::models::{component_chat_credentials, Catalog, CatalogProvider};
use crate::net;

/// Upstream's stream and non-stream call limits are provider constants;
/// twenty minutes covers the longest of them.
const CALL_TIMEOUT: Duration = Duration::from_secs(20 * 60);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub total_tokens: i64,
    pub calls: i64,
}

impl Usage {
    /// The `usage` object of `workflow_finished`.
    pub fn to_json(self) -> Value {
        json!({
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Wire {
    OpenAi,
    Ollama,
}

fn wire_of(provider: &CatalogProvider) -> Option<Wire> {
    let chat = provider.url_suffix.get("chat")?;
    if provider.name.eq_ignore_ascii_case("ollama") {
        Some(Wire::Ollama)
    } else if chat.trim_matches('/').ends_with("chat/completions") {
        Some(Wire::OpenAi)
    } else {
        None
    }
}

/// The endpoint root: the instance's `base_url`, else the catalog default.
pub fn base_url(provider: &CatalogProvider, configured: &str, region: &str) -> Option<String> {
    let configured = configured.trim();
    if !configured.is_empty() {
        return Some(configured.trim_end_matches('/').to_string());
    }
    let key = if region.is_empty() { "default" } else { region };
    provider
        .url
        .get(key)
        .or_else(|| {
            if region.is_empty() {
                provider.url.get("default")
            } else {
                None
            }
        })
        .filter(|u| !u.is_empty())
        .map(|u| u.trim_end_matches('/').to_string())
}

/// One run's chat client. It is bound to the canvas owner's tenant, counts
/// its calls' token usage, and gives up when `cancel` is raised.
pub struct ServerModels {
    pub facts: Arc<dyn Facts>,
    pub catalog: Arc<Catalog>,
    pub tenant_id: String,
    pub cancel: Arc<AtomicBool>,
    usage: Mutex<Usage>,
}

impl ServerModels {
    pub fn new(
        facts: Arc<dyn Facts>,
        catalog: Arc<Catalog>,
        tenant_id: String,
        cancel: Arc<AtomicBool>,
    ) -> Self {
        ServerModels {
            facts,
            catalog,
            tenant_id,
            cancel,
            usage: Mutex::new(Usage::default()),
        }
    }

    pub fn usage(&self) -> Usage {
        *self.usage.lock().expect("usage lock")
    }

    fn call(
        &self,
        model: &ModelRef,
        messages: &[ChatMessage],
        options: &ChatOptions,
        mut on_delta: Option<&mut dyn FnMut(&str)>,
    ) -> Result<String, String> {
        let creds = component_chat_credentials(self.facts.as_ref(), &self.tenant_id, model)?;
        if creds.provider.is_empty() {
            return Err(format!(
                "resolve driver \"\": no chat driver for model {:?}",
                model.model
            ));
        }
        let provider = self.catalog.provider(&creds.provider).ok_or_else(|| {
            format!(
                "resolve driver {:?}: provider {:?} is not configured",
                creds.provider, creds.provider
            )
        })?;
        let wire = wire_of(provider).ok_or_else(|| {
            format!(
                "provider {:?} has no chat driver in this server (OpenAI-compatible and Ollama endpoints are supported)",
                provider.name
            )
        })?;
        let local = provider.class.as_deref() == Some("local");
        if creds.api_key.trim().is_empty() && !local {
            return Err("api key is required".into());
        }
        let root = base_url(provider, &creds.base_url, "")
            .ok_or_else(|| "no base URL configured for region \"\"".to_string())?;
        let suffix = provider.url_suffix["chat"].trim_start_matches('/');
        let url = format!("{root}/{suffix}");
        let model_name = creds.model.clone();
        let request = Request {
            url,
            api_key: creds.api_key,
            body: match wire {
                Wire::OpenAi => openai_body(&creds.model, &provider.name, messages, options),
                Wire::Ollama => ollama_body(&creds.model, messages, options),
            },
            wire,
            stream: options.stream,
            strict: !local,
        };
        let (tx, rx) = mpsc::channel();
        let deltas = on_delta.is_some();
        // The call runs on its own thread so a cancel can stop waiting for
        // a model that is slow to answer.
        std::thread::spawn(move || {
            let sender = tx.clone();
            let result = request.send(&mut |delta: &str| {
                if deltas {
                    let _ = sender.send(Msg::Delta(delta.to_string()));
                }
            });
            let _ = tx.send(Msg::Done(result));
        });
        loop {
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(Msg::Delta(d)) => {
                    if let Some(f) = on_delta.as_mut() {
                        f(&d);
                    }
                }
                Ok(Msg::Done(result)) => {
                    // The model client leads a driver's failure with the
                    // model's name (`EinoChatModel`'s `"%s: %w"`).
                    let (text, usage) = result.map_err(|e| format!("{model_name}: {e}"))?;
                    let mut total = self.usage.lock().expect("usage lock");
                    total.calls += 1;
                    total.prompt_tokens += usage.0;
                    total.completion_tokens += usage.1;
                    total.total_tokens += usage.2;
                    return Ok(text);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if self.cancel.load(Ordering::SeqCst) {
                        return Err(ragflow_runtime::models::CANCELLED.into());
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err("the model call ended without a result".into())
                }
            }
        }
    }
}

enum Msg {
    Delta(String),
    Done(Result<Reply, String>),
}

impl Models for ServerModels {
    fn chat(
        &self,
        model: &ModelRef,
        messages: &[ChatMessage],
        options: &ChatOptions,
    ) -> Result<String, String> {
        self.call(model, messages, options, None)
    }

    fn chat_stream_for(
        &self,
        _component_id: &str,
        model: &ModelRef,
        messages: &[ChatMessage],
        options: &ChatOptions,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String, String> {
        self.call(model, messages, options, Some(on_delta))
    }

    fn embed(&self, _: &ModelRef, _: &[String]) -> Result<Vec<Vec<f32>>, String> {
        Err("embeddings are resolved by the retrieval service, not the components".into())
    }
}

fn wire_messages(messages: &[ChatMessage]) -> Vec<Value> {
    messages
        .iter()
        .map(|m| json!({ "role": m.role, "content": m.content }))
        .collect()
}

/// `buildRequestBody` of upstream's OpenAI driver.
fn openai_body(model: &str, provider: &str, messages: &[ChatMessage], o: &ChatOptions) -> Value {
    let mut body = json!({
        "model": model,
        "messages": wire_messages(messages),
        "stream": o.stream,
    });
    if let Some(t) = o.temperature {
        body["temperature"] = json!(t);
    }
    if let Some(p) = o.top_p {
        body["top_p"] = json!(p);
    }
    if let Some(m) = o.max_tokens {
        body["max_tokens"] = json!(m);
    }
    if o.stream {
        body["stream_options"] = json!({ "include_usage": true });
    }
    if provider.eq_ignore_ascii_case("openai") && model.to_lowercase().contains("qwen3") {
        body["enable_thinking"] = json!(false);
    }
    body
}

fn ollama_body(model: &str, messages: &[ChatMessage], o: &ChatOptions) -> Value {
    let mut options = serde_json::Map::new();
    if let Some(t) = o.temperature {
        options.insert("temperature".into(), json!(t));
    }
    if let Some(p) = o.top_p {
        options.insert("top_p".into(), json!(p));
    }
    if let Some(m) = o.max_tokens {
        options.insert("num_predict".into(), json!(m));
    }
    let mut body = json!({
        "model": model,
        "messages": wire_messages(messages),
        "stream": o.stream,
    });
    if !options.is_empty() {
        body["options"] = Value::Object(options);
    }
    body
}

struct Request {
    url: String,
    api_key: String,
    body: Value,
    wire: Wire,
    stream: bool,
    strict: bool,
}

type Reply = (String, (i64, i64, i64));

impl Request {
    fn send(self, on_delta: &mut dyn FnMut(&str)) -> Result<Reply, String> {
        let agent = net::agent(self.strict, CALL_TIMEOUT);
        let mut req = agent
            .post(&self.url)
            .set("Content-Type", "application/json");
        if !self.api_key.trim().is_empty() {
            req = req.set("Authorization", &format!("Bearer {}", self.api_key));
        }
        let resp = match req.send_string(&self.body.to_string()) {
            Ok(r) => r,
            Err(ureq::Error::Status(code, r)) => {
                return Err(status_error(code, r));
            }
            Err(e) => {
                return Err(format!(
                    "failed to send request: Post {:?}: {}",
                    self.url,
                    transport_cause(&e)
                ))
            }
        };
        match (self.wire, self.stream) {
            (Wire::OpenAi, false) => {
                let v: Value = resp
                    .into_json()
                    .map_err(|e| format!("failed to parse response: {e}"))?;
                let text = v
                    .pointer("/choices/0/message/content")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                Ok((text, openai_usage(v.get("usage"))))
            }
            (Wire::OpenAi, true) => read_openai_stream(resp, on_delta),
            (Wire::Ollama, false) => {
                let v: Value = resp
                    .into_json()
                    .map_err(|e| format!("failed to parse response: {e}"))?;
                Ok((
                    v.pointer("/message/content")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    ollama_usage(&v),
                ))
            }
            (Wire::Ollama, true) => read_ollama_stream(resp, on_delta),
        }
    }
}

/// The cap on a provider's error page (`maxModelErrorBodyBytes`).
const ERROR_BODY_LIMIT: u64 = 1 << 20;

/// The driver's `API request failed with status %d: %s`.
fn status_error(code: u16, response: ureq::Response) -> String {
    use std::io::Read;
    let mut body = Vec::new();
    let read = response
        .into_reader()
        .take(ERROR_BODY_LIMIT + 1)
        .read_to_end(&mut body);
    match read {
        Err(e) => format!("API request failed with status {code}; failed to read error response: {e}"),
        Ok(_) if body.len() as u64 > ERROR_BODY_LIMIT => format!(
            "API request failed with status {code}; failed to read error response: response body exceeds {ERROR_BODY_LIMIT} bytes"
        ),
        Ok(_) => format!(
            "API request failed with status {code}: {}",
            String::from_utf8_lossy(&body)
        ),
    }
}

/// What Go's `url.Error` carries after the URL; the text of a transport
/// failure differs in detail between the two stacks.
fn transport_cause(e: &ureq::Error) -> String {
    match e {
        ureq::Error::Transport(t) => t.to_string(),
        other => other.to_string(),
    }
}

fn openai_usage(usage: Option<&Value>) -> (i64, i64, i64) {
    let Some(u) = usage.filter(|u| u.is_object()) else {
        return (0, 0, 0);
    };
    let n = |k: &str| u.get(k).and_then(Value::as_i64).unwrap_or(0);
    let (p, c) = (n("prompt_tokens"), n("completion_tokens"));
    let t = u
        .get("total_tokens")
        .and_then(Value::as_i64)
        .unwrap_or(p + c);
    (p, c, t)
}

fn ollama_usage(v: &Value) -> (i64, i64, i64) {
    let n = |k: &str| v.get(k).and_then(Value::as_i64).unwrap_or(0);
    let (p, c) = (n("prompt_eval_count"), n("eval_count"));
    (p, c, p + c)
}

/// An OpenAI `data:` stream: content deltas in order, usage from the chunk
/// that carries it.
fn read_openai_stream(
    resp: ureq::Response,
    on_delta: &mut dyn FnMut(&str),
) -> Result<Reply, String> {
    let mut text = String::new();
    let mut usage = (0, 0, 0);
    for line in BufReader::new(resp.into_reader()).lines() {
        let line = line.map_err(|e| format!("failed to read response: {e}"))?;
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() {
            continue;
        }
        if data == "[DONE]" {
            break;
        }
        let v: Value = serde_json::from_str(data).map_err(|e| format!("invalid SSE event: {e}"))?;
        if let Some(delta) = v
            .pointer("/choices/0/delta/content")
            .and_then(Value::as_str)
        {
            if !delta.is_empty() {
                on_delta(delta);
            }
            text.push_str(delta);
        }
        if v.get("usage").is_some_and(|u| u.is_object()) {
            usage = openai_usage(v.get("usage"));
        }
    }
    Ok((text, usage))
}

fn read_ollama_stream(
    resp: ureq::Response,
    on_delta: &mut dyn FnMut(&str),
) -> Result<Reply, String> {
    let mut text = String::new();
    let mut usage = (0, 0, 0);
    for line in BufReader::new(resp.into_reader()).lines() {
        let line = line.map_err(|e| format!("failed to read response: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Value =
            serde_json::from_str(&line).map_err(|e| format!("invalid stream line: {e}"))?;
        if let Some(delta) = v.pointer("/message/content").and_then(Value::as_str) {
            if !delta.is_empty() {
                on_delta(delta);
            }
            text.push_str(delta);
        }
        if v.get("done").and_then(Value::as_bool) == Some(true) {
            usage = ollama_usage(&v);
            break;
        }
    }
    Ok((text, usage))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::facts::*;
    use crate::perm::fake::Fake;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    fn catalog(url: &str) -> Catalog {
        Catalog::from_values(&[json!({"name": "OpenAI", "url": {"default": url},
            "url_suffix": {"chat": "chat/completions"}, "class": "local", "models": []})])
        .unwrap()
    }

    fn world() -> Fake {
        let mut f = Fake::default();
        f.member("alice", "alice", "owner");
        f.providers = vec![ModelProvider {
            id: "p".into(),
            tenant_id: "alice".into(),
            provider_name: "OpenAI".into(),
        }];
        f.instances = vec![ModelInstance {
            id: "i".into(),
            provider_id: "p".into(),
            instance_name: "dev".into(),
            api_key: "sk-test".into(),
            status: "active".into(),
            extra: "{}".into(),
        }];
        f
    }

    /// Serves one canned HTTP response and returns the request it got.
    fn serve_once(response: String) -> (String, std::thread::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let mut buf = vec![0u8; 65536];
            let mut got = Vec::new();
            loop {
                let n = s.read(&mut buf).unwrap();
                got.extend_from_slice(&buf[..n]);
                let text = String::from_utf8_lossy(&got).to_string();
                if let Some(i) = text.find("\r\n\r\n") {
                    let len = text
                        .lines()
                        .find_map(|l| {
                            l.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .map(|v| v.trim().parse::<usize>().unwrap())
                        })
                        .unwrap_or(0);
                    if got.len() >= i + 4 + len {
                        break;
                    }
                }
            }
            s.write_all(response.as_bytes()).unwrap();
            String::from_utf8_lossy(&got).to_string()
        });
        (format!("http://{addr}"), handle)
    }

    fn models(url: &str, cancel: Arc<AtomicBool>) -> ServerModels {
        ServerModels::new(
            Arc::new(world()),
            Arc::new(catalog(url)),
            "alice".into(),
            cancel,
        )
    }

    fn http(body: &str, content_type: &str) -> String {
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
    }

    #[test]
    fn a_streamed_openai_call_joins_deltas_and_counts_usage() {
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n\
                   data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":2,\"total_tokens\":9}}\n\n\
                   data: [DONE]\n\n";
        let (url, server) = serve_once(http(sse, "text/event-stream"));
        let m = models(&format!("{url}/v1"), Arc::new(AtomicBool::new(false)));
        let reply = m
            .chat(
                &ModelRef::parse("gpt-x@dev@OpenAI").unwrap(),
                &[ChatMessage {
                    role: "user",
                    content: "hi".into(),
                }],
                &ChatOptions {
                    temperature: Some(0.1),
                    stream: true,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(reply, "hello");
        assert_eq!(
            m.usage(),
            Usage {
                prompt_tokens: 7,
                completion_tokens: 2,
                total_tokens: 9,
                calls: 1
            }
        );
        let request = server.join().unwrap();
        assert!(
            request.starts_with("POST /v1/chat/completions "),
            "{request}"
        );
        assert!(request
            .to_lowercase()
            .contains("authorization: bearer sk-test"));
        let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
        assert_eq!(body["model"], "gpt-x");
        assert_eq!(body["stream"], true);
        assert_eq!(body["stream_options"]["include_usage"], true);
        assert_eq!(body["temperature"], 0.1);
        assert!(body.get("top_p").is_none());
    }

    #[test]
    fn deltas_reach_the_caller_as_they_arrive() {
        let sse = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"bc\"}}]}\n\n\
                   data: [DONE]\n\n";
        let (url, server) = serve_once(http(sse, "text/event-stream"));
        let m = models(&url, Arc::new(AtomicBool::new(false)));
        let mut got = Vec::new();
        let reply = m
            .chat_stream_for(
                "Agent:x",
                &ModelRef::parse("gpt-x@dev@OpenAI").unwrap(),
                &[ChatMessage {
                    role: "user",
                    content: "hi".into(),
                }],
                &ChatOptions {
                    stream: true,
                    ..Default::default()
                },
                &mut |d| got.push(d.to_string()),
            )
            .unwrap();
        server.join().unwrap();
        assert_eq!(reply, "abc");
        assert_eq!(got, ["a", "bc"]);
    }

    #[test]
    fn a_plain_call_reads_the_message() {
        let body = r#"{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":3,"completion_tokens":1}}"#;
        let (url, server) = serve_once(http(body, "application/json"));
        let m = models(&url, Arc::new(AtomicBool::new(false)));
        let reply = m
            .chat(
                &ModelRef::parse("gpt-x@dev@OpenAI").unwrap(),
                &[ChatMessage {
                    role: "user",
                    content: "hi".into(),
                }],
                &ChatOptions::default(),
            )
            .unwrap();
        assert_eq!(reply, "ok");
        assert_eq!(m.usage().total_tokens, 4);
        let request = server.join().unwrap();
        assert!(!request.contains("stream_options"));
    }

    #[test]
    fn a_cancel_stops_the_wait() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            let (_s, _) = listener.accept().unwrap();
            std::thread::sleep(Duration::from_secs(5));
        });
        let cancel = Arc::new(AtomicBool::new(false));
        let m = models(&url, cancel.clone());
        let flag = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            flag.store(true, Ordering::SeqCst);
        });
        let started = std::time::Instant::now();
        let err = m
            .chat(
                &ModelRef::parse("gpt-x@dev@OpenAI").unwrap(),
                &[ChatMessage {
                    role: "user",
                    content: "hi".into(),
                }],
                &ChatOptions::default(),
            )
            .unwrap_err();
        assert_eq!(err, "context canceled");
        assert!(started.elapsed() < Duration::from_secs(3));
    }

    #[test]
    fn a_missing_key_or_provider_is_reported() {
        let mut f = world();
        f.instances[0].api_key.clear();
        let strict =
            Catalog::from_values(&[json!({"name": "OpenAI", "url": {"default": "https://x"},
            "url_suffix": {"chat": "chat/completions"}, "class": "gpt", "models": []})])
            .unwrap();
        let m = ServerModels::new(
            Arc::new(f),
            Arc::new(strict),
            "alice".into(),
            Arc::new(AtomicBool::new(false)),
        );
        let msgs = [ChatMessage {
            role: "user",
            content: "hi".into(),
        }];
        let err = m
            .chat(
                &ModelRef::parse("x@dev@OpenAI").unwrap(),
                &msgs,
                &ChatOptions::default(),
            )
            .unwrap_err();
        assert_eq!(err, "api key is required");
        let err = m
            .chat(
                &ModelRef::parse("x@dev@Nope").unwrap(),
                &msgs,
                &ChatOptions::default(),
            )
            .unwrap_err();
        assert!(err.contains("Nope"), "{err}");
    }

    #[test]
    fn base_urls_prefer_the_instance() {
        let c = catalog("https://api.example/v1/");
        let p = c.provider("openai").unwrap();
        assert_eq!(base_url(p, "http://x/", "").unwrap(), "http://x");
        assert_eq!(base_url(p, "", "").unwrap(), "https://api.example/v1");
        assert_eq!(base_url(p, "", "eu"), None);
    }

    #[test]
    fn a_status_error_reads_like_the_go_driver() {
        let r = ureq::Response::new(500, "Internal Server Error", "{\"error\": \"boom\"}").unwrap();
        assert_eq!(
            status_error(500, r),
            "API request failed with status 500: {\"error\": \"boom\"}"
        );
        let big = "x".repeat((ERROR_BODY_LIMIT + 1) as usize);
        let r = ureq::Response::new(502, "Bad Gateway", &big).unwrap();
        assert_eq!(
            status_error(502, r),
            "API request failed with status 502; failed to read error response: response body exceeds 1048576 bytes"
        );
    }
}

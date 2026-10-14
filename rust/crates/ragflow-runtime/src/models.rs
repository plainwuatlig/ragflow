//! The model calls components make, behind a trait so the runtime itself
//! does no network I/O. `ragflow-serve` supplies an OpenAI-compatible
//! implementation.

/// A RAGFlow model id: `model@provider`, `model@instance@provider` (the
/// model name may itself contain `@`: the split is right-anchored), or a bare
/// 32-character hex `tenant_model` id, which has no provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRef {
    pub model: String,
    pub instance: Option<String>,
    pub provider: String,
}

impl ModelRef {
    pub fn parse(id: &str) -> Result<Self, String> {
        let id = id.trim();
        let parts: Vec<&str> = id.split('@').collect();
        if parts.len() == 1 && id.len() == 32 && id.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Ok(ModelRef {
                model: id.to_string(),
                instance: None,
                provider: String::new(),
            });
        }
        let n = parts.len();
        if n < 2 || parts[n - 1].is_empty() {
            return Err(format!(
                "model id {id:?} is not model@provider or model@instance@provider"
            ));
        }
        let (model, instance) = if n == 2 {
            (parts[0].to_string(), None)
        } else {
            (parts[..n - 2].join("@"), Some(parts[n - 2].to_string()))
        };
        if model.is_empty() || instance.as_deref() == Some("") {
            return Err(format!(
                "model id {id:?} is not model@provider or model@instance@provider"
            ));
        }
        Ok(ModelRef {
            model,
            instance,
            provider: parts[n - 1].to_string(),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChatMessage {
    pub role: &'static str,
    pub content: String,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChatOptions {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_tokens: Option<u64>,
    /// The caller reads the reply as a stream (upstream's Agent does;
    /// Categorize does not), which only changes the wire call.
    pub stream: bool,
}

pub trait Models: Send + Sync {
    fn chat(
        &self,
        model: &ModelRef,
        messages: &[ChatMessage],
        options: &ChatOptions,
    ) -> Result<String, String>;

    fn embed(&self, model: &ModelRef, texts: &[String]) -> Result<Vec<Vec<f32>>, String>;

    /// `chat` on behalf of a named component. A deferred Agent makes its
    /// call while the consuming Message runs, so the caller cannot be
    /// inferred from the event stream; implementations that care (the
    /// parity oracle keys recordings by it) override this.
    fn chat_for(
        &self,
        component_id: &str,
        model: &ModelRef,
        messages: &[ChatMessage],
        options: &ChatOptions,
    ) -> Result<String, String> {
        let _ = component_id;
        self.chat(model, messages, options)
    }
}

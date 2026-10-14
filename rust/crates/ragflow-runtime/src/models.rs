//! The model calls components make, behind a trait so the runtime itself
//! does no network I/O. `ragflow-serve` supplies an OpenAI-compatible
//! implementation.

/// A RAGFlow model id: `model@provider` or `model@instance@provider`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRef {
    pub model: String,
    pub instance: Option<String>,
    pub provider: String,
}

impl ModelRef {
    pub fn parse(id: &str) -> Result<Self, String> {
        let parts: Vec<&str> = id.split('@').collect();
        match parts.as_slice() {
            [model, provider] if !model.is_empty() && !provider.is_empty() => Ok(ModelRef {
                model: model.to_string(),
                instance: None,
                provider: provider.to_string(),
            }),
            [model, instance, provider]
                if !model.is_empty() && !instance.is_empty() && !provider.is_empty() =>
            {
                Ok(ModelRef {
                    model: model.to_string(),
                    instance: Some(instance.to_string()),
                    provider: provider.to_string(),
                })
            }
            _ => Err(format!(
                "model id {id:?} is not model@provider or model@instance@provider"
            )),
        }
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
}

pub trait Models: Send + Sync {
    fn chat(
        &self,
        model: &ModelRef,
        messages: &[ChatMessage],
        options: &ChatOptions,
    ) -> Result<String, String>;

    fn embed(&self, model: &ModelRef, texts: &[String]) -> Result<Vec<Vec<f32>>, String>;
}

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;
use thiserror::Error;

pub const PROFILE_ID: &str = "NS-LLM";
pub const RESULT_SCHEMA: &str = "qsol-harness/headless-result/1";
pub const RECEIPT_SCHEMA: &str = "qsol-harness-receipt/1";

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("configuration error: {0}")]
    Config(String),
    #[error("provider error: {0}")]
    Provider(String),
    #[error("provider requires seed but the request omitted one")]
    MissingSeed,
    #[error("provider adapter is unavailable: {0}")]
    UnsupportedProvider(String),
    #[error("required provider credential environment variable is missing: {0}")]
    MissingAuthEnv(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[cfg(feature = "openai-compatible")]
    #[error(transparent)]
    Http(#[from] reqwest::Error),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderTransport {
    LocalProcess,
    Ollama,
    Vllm,
    OpenaiCompatible,
    Native,
}

impl fmt::Display for ProviderTransport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::LocalProcess => "local-process",
            Self::Ollama => "ollama",
            Self::Vllm => "vllm",
            Self::OpenaiCompatible => "openai-compatible",
            Self::Native => "native",
        };
        f.write_str(value)
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct ProviderCapabilities {
    #[serde(default)]
    pub tools: bool,
    #[serde(default)]
    pub streaming: bool,
    #[serde(default)]
    pub seed: bool,
    #[serde(default)]
    pub structured_output: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ProviderConfig {
    pub provider_id: String,
    pub model_id: String,
    pub transport: ProviderTransport,
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub auth_env: Option<String>,
    #[serde(default)]
    pub capabilities: ProviderCapabilities,
}

impl ProviderConfig {
    fn validate(&self) -> Result<(), RuntimeError> {
        if self.provider_id.trim().is_empty() {
            return Err(RuntimeError::Config("provider_id must not be empty".into()));
        }
        if self.model_id.trim().is_empty() {
            return Err(RuntimeError::Config("model_id must not be empty".into()));
        }
        if self.auth_env.as_deref().is_some_and(|value| value.trim().is_empty()) {
            return Err(RuntimeError::Config("auth_env must be omitted or non-empty".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuntimeConfig {
    pub profile: String,
    pub provider: ProviderConfig,
}

impl RuntimeConfig {
    pub fn validate(&self) -> Result<(), RuntimeError> {
        if self.profile != PROFILE_ID {
            return Err(RuntimeError::Config(format!(
                "profile must be {PROFILE_ID}, got {}",
                self.profile
            )));
        }
        self.provider.validate()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct InferenceRequest {
    pub experiment_id: String,
    pub prompt: String,
    #[serde(default)]
    pub seed: Option<u64>,
}

impl InferenceRequest {
    fn validate(&self) -> Result<(), RuntimeError> {
        if self.experiment_id.trim().is_empty() {
            return Err(RuntimeError::Config("experiment_id must not be empty".into()));
        }
        if self.prompt.is_empty() {
            return Err(RuntimeError::Config("prompt must not be empty".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ProviderIdentity {
    pub provider_id: String,
    pub model_id: String,
    pub transport: ProviderTransport,
}

impl From<&ProviderConfig> for ProviderIdentity {
    fn from(config: &ProviderConfig) -> Self {
        Self {
            provider_id: config.provider_id.clone(),
            model_id: config.model_id.clone(),
            transport: config.transport.clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ReceiptActor {
    pub kind: &'static str,
    pub id: String,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct EvidenceReceipt {
    pub schema_version: &'static str,
    pub receipt_id: String,
    pub experiment_id: String,
    pub action: &'static str,
    pub input_sha256: String,
    pub output_sha256: String,
    pub evidence_class: &'static str,
    pub actor: ReceiptActor,
    pub status: &'static str,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct HeadlessResult {
    pub schema: &'static str,
    pub profile: &'static str,
    pub provider_identity: ProviderIdentity,
    pub output: String,
    pub receipt: EvidenceReceipt,
}

pub trait ModelProvider: Send + Sync {
    fn identity(&self) -> ProviderIdentity;
    fn supports_seed(&self) -> bool;
    fn infer(&self, request: &InferenceRequest) -> Result<String, RuntimeError>;
}

#[derive(Clone, Debug)]
struct FixtureProvider {
    config: ProviderConfig,
}

impl ModelProvider for FixtureProvider {
    fn identity(&self) -> ProviderIdentity {
        ProviderIdentity::from(&self.config)
    }

    fn supports_seed(&self) -> bool {
        self.config.capabilities.seed
    }

    fn infer(&self, request: &InferenceRequest) -> Result<String, RuntimeError> {
        let canonical = serde_json::to_vec(request)?;
        Ok(format!(
            "fixture:{}:{}",
            self.config.model_id,
            sha256_hex(&canonical)
        ))
    }
}

#[cfg(feature = "openai-compatible")]
#[derive(Clone, Debug)]
struct OpenAiCompatibleProvider {
    config: ProviderConfig,
    endpoint: String,
}

#[cfg(feature = "openai-compatible")]
impl OpenAiCompatibleProvider {
    fn new(config: ProviderConfig) -> Result<Self, RuntimeError> {
        let endpoint = if config.provider_id.eq_ignore_ascii_case("xai") {
            #[cfg(not(feature = "xai-compat"))]
            {
                return Err(RuntimeError::UnsupportedProvider(
                    "xAI is compatibility-only; rebuild with --features xai-compat".into(),
                ));
            }
            #[cfg(feature = "xai-compat")]
            {
                config
                    .endpoint
                    .clone()
                    .unwrap_or_else(|| "https://api.x.ai/v1/chat/completions".into())
            }
        } else if matches!(config.transport, ProviderTransport::Ollama) {
            config
                .endpoint
                .clone()
                .unwrap_or_else(|| "http://127.0.0.1:11434/v1/chat/completions".into())
        } else {
            config.endpoint.clone().ok_or_else(|| {
                RuntimeError::Config(format!(
                    "endpoint is required for provider {} over {}",
                    config.provider_id, config.transport
                ))
            })?
        };
        Ok(Self { config, endpoint })
    }
}

#[cfg(feature = "openai-compatible")]
impl ModelProvider for OpenAiCompatibleProvider {
    fn identity(&self) -> ProviderIdentity {
        ProviderIdentity::from(&self.config)
    }

    fn supports_seed(&self) -> bool {
        self.config.capabilities.seed
    }

    fn infer(&self, request: &InferenceRequest) -> Result<String, RuntimeError> {
        let client = reqwest::blocking::Client::new();
        let mut body = serde_json::json!({
            "model": self.config.model_id,
            "messages": [{"role": "user", "content": request.prompt}],
            "stream": false
        });
        if let Some(seed) = request.seed {
            body["seed"] = serde_json::json!(seed);
        }

        let mut outgoing = client.post(&self.endpoint).json(&body);
        if let Some(name) = &self.config.auth_env {
            let token = std::env::var(name)
                .map_err(|_| RuntimeError::MissingAuthEnv(name.clone()))?;
            outgoing = outgoing.bearer_auth(token);
        } else if self.config.provider_id.eq_ignore_ascii_case("xai") {
            #[cfg(feature = "xai-compat")]
            {
                let token = std::env::var("XAI_API_KEY")
                    .map_err(|_| RuntimeError::MissingAuthEnv("XAI_API_KEY".into()))?;
                outgoing = outgoing.bearer_auth(token);
            }
        }

        let value: serde_json::Value = outgoing.send()?.error_for_status()?.json()?;
        value
            .pointer("/choices/0/message/content")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| RuntimeError::Provider("response omitted choices[0].message.content".into()))
    }
}

fn build_provider(config: ProviderConfig) -> Result<Box<dyn ModelProvider>, RuntimeError> {
    config.validate()?;
    if matches!(config.transport, ProviderTransport::Native)
        && config.provider_id.eq_ignore_ascii_case("fixture")
    {
        return Ok(Box::new(FixtureProvider { config }));
    }

    #[cfg(feature = "openai-compatible")]
    if matches!(
        config.transport,
        ProviderTransport::Ollama | ProviderTransport::Vllm | ProviderTransport::OpenaiCompatible
    ) {
        return Ok(Box::new(OpenAiCompatibleProvider::new(config)?));
    }

    Err(RuntimeError::UnsupportedProvider(format!(
        "{} via {}",
        config.provider_id, config.transport
    )))
}

pub struct CompositionRoot {
    provider: Box<dyn ModelProvider>,
}

impl CompositionRoot {
    pub fn from_config(config: RuntimeConfig) -> Result<Self, RuntimeError> {
        config.validate()?;
        let provider = build_provider(config.provider)?;
        Ok(Self { provider })
    }

    pub fn run(&self, request: InferenceRequest) -> Result<HeadlessResult, RuntimeError> {
        request.validate()?;
        if self.provider.supports_seed() && request.seed.is_none() {
            return Err(RuntimeError::MissingSeed);
        }

        let identity = self.provider.identity();
        let input = serde_json::to_vec(&request)?;
        let input_sha256 = sha256_hex(&input);
        let output = self.provider.infer(&request)?;
        let output_sha256 = sha256_hex(output.as_bytes());
        let actor_id = format!("{}/{}", identity.provider_id, identity.model_id);
        let receipt_material = format!(
            "{}\0{}\0{}\0{}",
            request.experiment_id, actor_id, input_sha256, output_sha256
        );
        let receipt = EvidenceReceipt {
            schema_version: RECEIPT_SCHEMA,
            receipt_id: sha256_hex(receipt_material.as_bytes()),
            experiment_id: request.experiment_id,
            action: "model_inference",
            input_sha256,
            output_sha256,
            evidence_class: "MODEL_INFERENCE",
            actor: ReceiptActor {
                kind: "model",
                id: actor_id,
            },
            status: "ok",
        };

        Ok(HeadlessResult {
            schema: RESULT_SCHEMA,
            profile: PROFILE_ID,
            provider_identity: identity,
            output,
            receipt,
        })
    }
}

pub fn load_config(path: impl AsRef<std::path::Path>) -> Result<RuntimeConfig, RuntimeError> {
    let bytes = std::fs::read(path)?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_config(seed: bool) -> RuntimeConfig {
        RuntimeConfig {
            profile: PROFILE_ID.into(),
            provider: ProviderConfig {
                provider_id: "fixture".into(),
                model_id: "deterministic-fixture-v1".into(),
                transport: ProviderTransport::Native,
                endpoint: None,
                auth_env: None,
                capabilities: ProviderCapabilities {
                    seed,
                    ..ProviderCapabilities::default()
                },
            },
        }
    }

    #[test]
    fn headless_fixture_runs_without_xai_identity_or_credentials() {
        let root = CompositionRoot::from_config(fixture_config(true)).unwrap();
        let result = root
            .run(InferenceRequest {
                experiment_id: "test-1".into(),
                prompt: "deterministic prompt".into(),
                seed: Some(7),
            })
            .unwrap();
        assert_eq!(result.profile, "NS-LLM");
        assert_eq!(result.provider_identity.provider_id, "fixture");
        assert_eq!(result.receipt.evidence_class, "MODEL_INFERENCE");
        assert_eq!(result.receipt.status, "ok");
        assert_eq!(result.receipt.actor.id, "fixture/deterministic-fixture-v1");
        let serialized = serde_json::to_string(&result).unwrap();
        assert!(!serialized.to_ascii_lowercase().contains("xai_api_key"));
    }

    #[test]
    fn same_request_produces_same_fixture_output_and_receipt() {
        let root = CompositionRoot::from_config(fixture_config(true)).unwrap();
        let request = InferenceRequest {
            experiment_id: "repeat".into(),
            prompt: "same".into(),
            seed: Some(11),
        };
        let first = root.run(request.clone()).unwrap();
        let second = root.run(request).unwrap();
        assert_eq!(first.output, second.output);
        assert_eq!(first.receipt, second.receipt);
    }

    #[test]
    fn seed_capable_provider_fails_closed_without_seed() {
        let root = CompositionRoot::from_config(fixture_config(true)).unwrap();
        let error = root
            .run(InferenceRequest {
                experiment_id: "missing-seed".into(),
                prompt: "prompt".into(),
                seed: None,
            })
            .unwrap_err();
        assert!(matches!(error, RuntimeError::MissingSeed));
    }

    #[test]
    fn wrong_profile_is_rejected() {
        let mut config = fixture_config(false);
        config.profile = "ns-llm".into();
        assert!(CompositionRoot::from_config(config).is_err());
    }

    #[cfg(not(feature = "xai-compat"))]
    #[test]
    fn xai_is_not_required_or_enabled_by_default() {
        let config = RuntimeConfig {
            profile: PROFILE_ID.into(),
            provider: ProviderConfig {
                provider_id: "xai".into(),
                model_id: "grok".into(),
                transport: ProviderTransport::OpenaiCompatible,
                endpoint: None,
                auth_env: None,
                capabilities: ProviderCapabilities::default(),
            },
        };
        assert!(matches!(
            CompositionRoot::from_config(config),
            Err(RuntimeError::UnsupportedProvider(_))
        ));
    }
}

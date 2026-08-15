use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;
use thiserror::Error;

pub const PROFILE_ID: &str = "NS-LLM";
pub const RESULT_SCHEMA: &str = "qsol-harness/headless-result/1";
pub const RECEIPT_SCHEMA: &str = "qsol-harness-receipt/1";
const CANONICAL_REQUEST_SCHEMA: &str = "qsol-harness/canonical-inference-request/1";
const RECEIPT_MATERIAL_SCHEMA: &str = "qsol-harness/receipt-material/1";

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
#[serde(deny_unknown_fields)]
pub struct ProviderCapabilities {
    pub tools: bool,
    pub streaming: bool,
    #[serde(default)]
    pub seed: bool,
    #[serde(default)]
    pub structured_output: bool,
    #[serde(default)]
    pub images: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GenerationConfig {
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub max_tokens: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ProviderConfig {
    pub provider_id: String,
    pub model_id: String,
    pub transport: ProviderTransport,
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub auth_ref: Option<String>,
    pub capabilities: ProviderCapabilities,
    #[serde(default)]
    pub generation: Option<GenerationConfig>,
}

impl ProviderConfig {
    fn validate(&self) -> Result<(), RuntimeError> {
        if self.provider_id.trim().is_empty() {
            return Err(RuntimeError::Config("provider_id must not be empty".into()));
        }
        if self.model_id.trim().is_empty() {
            return Err(RuntimeError::Config("model_id must not be empty".into()));
        }
        if let Some(reference) = &self.auth_ref {
            let Some(name) = reference.strip_prefix("env:") else {
                return Err(RuntimeError::Config(
                    "auth_ref must use env:NAME; raw credentials are forbidden".into(),
                ));
            };
            if name.trim().is_empty() {
                return Err(RuntimeError::Config(
                    "auth_ref environment variable name must not be empty".into(),
                ));
            }
        }
        if self
            .generation
            .as_ref()
            .and_then(|generation| generation.max_tokens)
            == Some(0)
        {
            return Err(RuntimeError::Config(
                "generation.max_tokens must be greater than zero".into(),
            ));
        }
        Ok(())
    }

    fn auth_env_name(&self) -> Option<&str> {
        self.auth_ref
            .as_deref()
            .and_then(|value| value.strip_prefix("env:"))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
pub struct InferenceRequest {
    pub experiment_id: String,
    pub prompt: String,
    #[serde(default)]
    pub seed: Option<u64>,
}

impl InferenceRequest {
    fn validate(&self) -> Result<(), RuntimeError> {
        if self.experiment_id.trim().is_empty() {
            return Err(RuntimeError::Config(
                "experiment_id must not be empty".into(),
            ));
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

impl ProviderIdentity {
    fn binding_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        append_length_prefixed(&mut bytes, self.provider_id.as_bytes());
        append_length_prefixed(&mut bytes, self.model_id.as_bytes());
        append_length_prefixed(&mut bytes, self.transport.to_string().as_bytes());
        bytes
    }

    fn receipt_actor_id(&self) -> String {
        format!("model-sha256:{}", sha256_hex(&self.binding_bytes()))
    }
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

fn append_length_prefixed(target: &mut Vec<u8>, value: &[u8]) {
    target.extend_from_slice(&(value.len() as u64).to_be_bytes());
    target.extend_from_slice(value);
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

#[derive(Serialize)]
struct CanonicalInferenceEnvelope<'a> {
    schema: &'static str,
    profile: &'static str,
    provider_identity: &'a ProviderIdentity,
    request: &'a InferenceRequest,
}

#[derive(Serialize)]
struct ReceiptMaterial<'a> {
    schema: &'static str,
    experiment_id: &'a str,
    provider_identity: &'a ProviderIdentity,
    actor_id: &'a str,
    input_sha256: &'a str,
    output_sha256: &'a str,
}

fn canonical_input_sha256(
    identity: &ProviderIdentity,
    request: &InferenceRequest,
) -> Result<String, RuntimeError> {
    let envelope = CanonicalInferenceEnvelope {
        schema: CANONICAL_REQUEST_SCHEMA,
        profile: PROFILE_ID,
        provider_identity: identity,
        request,
    };
    Ok(sha256_hex(&serde_json::to_vec(&envelope)?))
}

pub trait ModelProvider: Send + Sync {
    fn identity(&self) -> ProviderIdentity;
    fn supports_seed(&self) -> bool;
    fn infer(&self, request: &InferenceRequest) -> Result<String, RuntimeError>;
}

#[cfg(any(test, feature = "ci-fixture"))]
#[derive(Clone, Debug)]
struct FixtureProvider {
    config: ProviderConfig,
}

#[cfg(any(test, feature = "ci-fixture"))]
impl ModelProvider for FixtureProvider {
    fn identity(&self) -> ProviderIdentity {
        ProviderIdentity::from(&self.config)
    }

    fn supports_seed(&self) -> bool {
        self.config.capabilities.seed
    }

    fn infer(&self, request: &InferenceRequest) -> Result<String, RuntimeError> {
        let identity = self.identity();
        let canonical = CanonicalInferenceEnvelope {
            schema: CANONICAL_REQUEST_SCHEMA,
            profile: PROFILE_ID,
            provider_identity: &identity,
            request,
        };
        Ok(format!(
            "fixture:{}:{}",
            self.config.model_id,
            sha256_hex(&serde_json::to_vec(&canonical)?)
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
fn endpoint_is_loopback(url: &reqwest::Url) -> bool {
    let Some(host) = url.host_str() else {
        return false;
    };
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|address| address.is_loopback())
}

#[cfg(feature = "openai-compatible")]
fn validate_endpoint(endpoint: &str, will_attach_credentials: bool) -> Result<(), RuntimeError> {
    let url = reqwest::Url::parse(endpoint)
        .map_err(|error| RuntimeError::Config(format!("invalid provider endpoint: {error}")))?;

    if !url.username().is_empty() || url.password().is_some() {
        return Err(RuntimeError::Config(
            "provider endpoint must not embed credentials".into(),
        ));
    }

    if will_attach_credentials
        && url.scheme() != "https"
        && !(url.scheme() == "http" && endpoint_is_loopback(&url))
    {
        return Err(RuntimeError::Config(
            "bearer credentials require HTTPS; plaintext HTTP is allowed only for loopback endpoints"
                .into(),
        ));
    }

    Ok(())
}

#[cfg(feature = "openai-compatible")]
fn extract_openai_compatible_content(
    value: &serde_json::Value,
    expected_model: &str,
) -> Result<String, RuntimeError> {
    let resolved_model = value
        .get("model")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            RuntimeError::Provider("response omitted top-level model identity".into())
        })?;
    if resolved_model != expected_model {
        return Err(RuntimeError::Provider(format!(
            "response model mismatch: requested {expected_model}, resolved {resolved_model}"
        )));
    }

    value
        .pointer("/choices/0/message/content")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| RuntimeError::Provider("response omitted choices[0].message.content".into()))
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

        let xai_implicit_auth = config.provider_id.eq_ignore_ascii_case("xai");
        validate_endpoint(&endpoint, config.auth_ref.is_some() || xai_implicit_auth)?;
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
        if let Some(name) = self.config.auth_env_name() {
            let token =
                std::env::var(name).map_err(|_| RuntimeError::MissingAuthEnv(name.to_owned()))?;
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
        extract_openai_compatible_content(&value, &self.config.model_id)
    }
}

fn build_provider(config: ProviderConfig) -> Result<Box<dyn ModelProvider>, RuntimeError> {
    config.validate()?;

    if matches!(config.transport, ProviderTransport::Native)
        && config.provider_id.eq_ignore_ascii_case("fixture")
    {
        #[cfg(any(test, feature = "ci-fixture"))]
        {
            return Ok(Box::new(FixtureProvider { config }));
        }
        #[cfg(not(any(test, feature = "ci-fixture")))]
        {
            return Err(RuntimeError::UnsupportedProvider(
                "deterministic fixture is test/CI-only; rebuild with --features ci-fixture for explicit CI execution"
                    .into(),
            ));
        }
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
        let input_sha256 = canonical_input_sha256(&identity, &request)?;
        let output = self.provider.infer(&request)?;
        let output_sha256 = sha256_hex(output.as_bytes());
        let actor_id = identity.receipt_actor_id();
        let receipt_material = ReceiptMaterial {
            schema: RECEIPT_MATERIAL_SCHEMA,
            experiment_id: &request.experiment_id,
            provider_identity: &identity,
            actor_id: &actor_id,
            input_sha256: &input_sha256,
            output_sha256: &output_sha256,
        };
        let receipt_id = sha256_hex(&serde_json::to_vec(&receipt_material)?);
        let receipt = EvidenceReceipt {
            schema_version: RECEIPT_SCHEMA,
            receipt_id,
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

    fn capabilities(seed: bool) -> ProviderCapabilities {
        ProviderCapabilities {
            tools: false,
            streaming: false,
            seed,
            structured_output: false,
            images: false,
        }
    }

    fn fixture_config(seed: bool) -> RuntimeConfig {
        RuntimeConfig {
            profile: PROFILE_ID.into(),
            provider: ProviderConfig {
                provider_id: "fixture".into(),
                model_id: "deterministic-fixture-v1".into(),
                transport: ProviderTransport::Native,
                endpoint: None,
                auth_ref: None,
                capabilities: capabilities(seed),
                generation: None,
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
        assert!(result.receipt.actor.id.starts_with("model-sha256:"));
        let serialized = serde_json::to_string(&result).unwrap();
        assert!(!serialized.to_ascii_lowercase().contains("xai_api_key"));
        assert!(!serialized.to_ascii_lowercase().contains("bearer "));
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
    fn provider_identity_changes_canonical_input_hash() {
        let request = InferenceRequest {
            experiment_id: "identity-bound".into(),
            prompt: "same request".into(),
            seed: Some(3),
        };
        let first = ProviderIdentity {
            provider_id: "provider-a".into(),
            model_id: "model".into(),
            transport: ProviderTransport::OpenaiCompatible,
        };
        let second = ProviderIdentity {
            provider_id: "provider-b".into(),
            model_id: "model".into(),
            transport: ProviderTransport::OpenaiCompatible,
        };
        assert_ne!(
            canonical_input_sha256(&first, &request).unwrap(),
            canonical_input_sha256(&second, &request).unwrap()
        );
    }

    #[test]
    fn actor_identity_encoding_resists_delimiter_collisions() {
        let first = ProviderIdentity {
            provider_id: "a".into(),
            model_id: "b/c".into(),
            transport: ProviderTransport::Native,
        };
        let second = ProviderIdentity {
            provider_id: "a/b".into(),
            model_id: "c".into(),
            transport: ProviderTransport::Native,
        };
        assert_ne!(first.receipt_actor_id(), second.receipt_actor_id());
    }

    #[test]
    fn raw_credentials_are_rejected_in_provider_config() {
        let mut config = fixture_config(false);
        config.provider.auth_ref = Some("secret-token".into());
        assert!(CompositionRoot::from_config(config).is_err());
    }

    #[test]
    fn unknown_provider_fields_are_rejected() {
        let config = r#"{
            "profile":"NS-LLM",
            "provider":{
                "provider_id":"example",
                "model_id":"model",
                "transport":"openai-compatible",
                "api_key":"secret",
                "capabilities":{"tools":false,"streaming":false}
            }
        }"#;
        assert!(serde_json::from_str::<RuntimeConfig>(config).is_err());
    }

    #[test]
    fn capabilities_object_is_required() {
        let config = r#"{
            "profile":"NS-LLM",
            "provider":{
                "provider_id":"example",
                "model_id":"model",
                "transport":"openai-compatible"
            }
        }"#;
        assert!(serde_json::from_str::<RuntimeConfig>(config).is_err());
    }

    #[test]
    fn mandatory_capability_fields_are_required() {
        let config = r#"{
            "profile":"NS-LLM",
            "provider":{
                "provider_id":"example",
                "model_id":"model",
                "transport":"openai-compatible",
                "capabilities":{"seed":true}
            }
        }"#;
        assert!(serde_json::from_str::<RuntimeConfig>(config).is_err());
    }

    #[test]
    fn wrong_profile_is_rejected() {
        let mut config = fixture_config(false);
        config.profile = "ns-llm".into();
        assert!(CompositionRoot::from_config(config).is_err());
    }

    #[cfg(feature = "openai-compatible")]
    #[test]
    fn response_model_mismatch_fails_closed() {
        let response = serde_json::json!({
            "model": "fallback-model",
            "choices": [{"message": {"content": "answer"}}]
        });
        assert!(extract_openai_compatible_content(&response, "requested-model").is_err());
    }

    #[cfg(feature = "openai-compatible")]
    #[test]
    fn authenticated_plaintext_remote_endpoint_is_rejected() {
        assert!(validate_endpoint("http://example.com/v1/chat/completions", true).is_err());
        assert!(validate_endpoint("http://127.0.0.1:11434/v1/chat/completions", true).is_ok());
        assert!(validate_endpoint("https://example.com/v1/chat/completions", true).is_ok());
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
                auth_ref: Some("env:XAI_API_KEY".into()),
                capabilities: capabilities(false),
                generation: None,
            },
        };
        assert!(matches!(
            CompositionRoot::from_config(config),
            Err(RuntimeError::UnsupportedProvider(_))
        ));
    }
}

use serde::Deserialize;

use crate::rewrite_profile::RewriteProfile;
use crate::rewrite_protocol::RewriteCorrectionPolicy;

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub audio: AudioConfig,
    pub transcription: TranscriptionConfig,
    #[serde(default, rename = "whisper")]
    pub(crate) legacy_whisper: LegacyWhisperConfig,
    pub postprocess: PostprocessConfig,
    pub session: SessionConfig,
    pub personalization: PersonalizationConfig,
    pub rewrite: RewriteConfig,
    #[serde(default, rename = "agentic_rewrite")]
    pub(crate) legacy_agentic_rewrite: LegacyAgenticRewriteConfig,
    pub cloud: CloudConfig,
    pub cleanup: CleanupConfig,
    pub inject: InjectConfig,
    pub feedback: FeedbackConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    pub device: String,
    pub sample_rate: u32,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptionBackend {
    #[default]
    WhisperCpp,
    FasterWhisper,
    Nemo,
    Cloud,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptionFallback {
    None,
    #[default]
    ConfiguredLocal,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TranscriptionConfig {
    pub backend: TranscriptionBackend,
    pub fallback: TranscriptionFallback,
    pub local_backend: TranscriptionBackend,
    pub selected_model: String,
    pub model_path: String,
    pub language: String,
    pub use_gpu: bool,
    pub flash_attn: bool,
    pub idle_timeout_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub(crate) struct LegacyWhisperConfig {
    pub(crate) model_path: String,
    pub(crate) language: String,
    pub(crate) use_gpu: bool,
    pub(crate) flash_attn: bool,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct PostprocessConfig {
    pub mode: PostprocessMode,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PostprocessMode {
    #[default]
    Raw,
    #[serde(alias = "advanced_local", alias = "agentic_rewrite")]
    Rewrite,
    LegacyBasic,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RewriteBackend {
    #[default]
    Local,
    Cloud,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RewriteFallback {
    None,
    #[default]
    Local,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct RewriteConfig {
    pub backend: RewriteBackend,
    pub fallback: RewriteFallback,
    pub selected_model: String,
    pub model_path: String,
    pub instructions_path: String,
    pub profile: RewriteProfile,
    pub timeout_ms: u64,
    pub idle_timeout_ms: u64,
    pub max_output_chars: usize,
    pub max_tokens: usize,
    pub policy_path: String,
    pub glossary_path: String,
    pub default_correction_policy: RewriteCorrectionPolicy,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub(crate) struct LegacyAgenticRewriteConfig {
    pub policy_path: String,
    pub glossary_path: String,
    pub default_correction_policy: RewriteCorrectionPolicy,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CloudProvider {
    #[default]
    #[serde(rename = "openai")]
    OpenAi,
    #[serde(rename = "openai_compatible")]
    OpenAiCompatible,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CloudLanguageMode {
    #[default]
    InheritLocal,
    Force,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CloudConfig {
    pub provider: CloudProvider,
    pub base_url: String,
    pub api_key: String,
    pub api_key_env: String,
    pub connect_timeout_ms: u64,
    pub request_timeout_ms: u64,
    pub transcription: CloudTranscriptionConfig,
    pub rewrite: CloudRewriteConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CloudTranscriptionConfig {
    pub model: String,
    pub language_mode: CloudLanguageMode,
    pub language: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CloudRewriteConfig {
    pub model: String,
    pub temperature: f32,
    pub max_output_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct CloudSettingsUpdate<'a> {
    pub provider: CloudProvider,
    pub base_url: &'a str,
    pub api_key: &'a str,
    pub api_key_env: &'a str,
    pub connect_timeout_ms: u64,
    pub request_timeout_ms: u64,
    pub transcription_model: &'a str,
    pub transcription_language_mode: CloudLanguageMode,
    pub transcription_language: &'a str,
    pub rewrite_model: &'a str,
    pub rewrite_temperature: f32,
    pub rewrite_max_output_tokens: usize,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct SessionConfig {
    pub enabled: bool,
    pub max_entries: usize,
    pub max_age_ms: u64,
    pub max_replace_graphemes: usize,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct PersonalizationConfig {
    pub dictionary_path: String,
    pub snippets_path: String,
    pub snippet_trigger: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct CleanupConfig {
    pub enabled: bool,
    pub profile: CleanupProfile,
    pub spoken_formatting: bool,
    pub backtrack: bool,
    pub remove_fillers: bool,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CleanupProfile {
    #[default]
    Basic,
    Aggressive,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct InjectConfig {}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct FeedbackConfig {
    pub enabled: bool,
    pub start_sound: String,
    pub stop_sound: String,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            device: String::new(),
            sample_rate: 16000,
        }
    }
}

impl Default for TranscriptionConfig {
    fn default() -> Self {
        Self {
            backend: TranscriptionBackend::WhisperCpp,
            fallback: TranscriptionFallback::ConfiguredLocal,
            local_backend: TranscriptionBackend::WhisperCpp,
            selected_model: "large-v3-turbo".into(),
            model_path: "~/.local/share/whispers/ggml-large-v3-turbo.bin".into(),
            language: "auto".into(),
            use_gpu: true,
            flash_attn: true,
            idle_timeout_ms: 120000,
        }
    }
}

impl Default for LegacyWhisperConfig {
    fn default() -> Self {
        let default = TranscriptionConfig::default();
        Self {
            model_path: default.model_path,
            language: default.language,
            use_gpu: default.use_gpu,
            flash_attn: default.flash_attn,
        }
    }
}

impl Default for PostprocessConfig {
    fn default() -> Self {
        Self {
            mode: PostprocessMode::Raw,
        }
    }
}

impl PostprocessMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::Rewrite => "rewrite",
            Self::LegacyBasic => "legacy_basic",
        }
    }

    pub fn uses_rewrite(self) -> bool {
        matches!(self, Self::Rewrite)
    }
}

impl TranscriptionBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::WhisperCpp => "whisper_cpp",
            Self::FasterWhisper => "faster_whisper",
            Self::Nemo => "nemo",
            Self::Cloud => "cloud",
        }
    }
}

impl TranscriptionFallback {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::ConfiguredLocal => "configured_local",
        }
    }
}

impl RewriteBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Cloud => "cloud",
        }
    }
}

impl RewriteFallback {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Local => "local",
        }
    }
}

impl CloudProvider {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::OpenAiCompatible => "openai_compatible",
        }
    }
}

impl CloudLanguageMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InheritLocal => "inherit_local",
            Self::Force => "force",
        }
    }
}

impl Default for RewriteConfig {
    fn default() -> Self {
        Self {
            backend: RewriteBackend::Local,
            fallback: RewriteFallback::Local,
            selected_model: "qwen-3.5-4b-q4_k_m".into(),
            model_path: String::new(),
            instructions_path: "~/.local/share/whispers/rewrite-instructions.txt".into(),
            profile: RewriteProfile::Auto,
            timeout_ms: 30000,
            idle_timeout_ms: 120000,
            max_output_chars: 1200,
            max_tokens: 256,
            policy_path: crate::agentic_rewrite::default_policy_path().into(),
            glossary_path: crate::agentic_rewrite::default_glossary_path().into(),
            default_correction_policy: RewriteCorrectionPolicy::Balanced,
        }
    }
}

impl Default for LegacyAgenticRewriteConfig {
    fn default() -> Self {
        Self {
            policy_path: crate::agentic_rewrite::default_policy_path().into(),
            glossary_path: crate::agentic_rewrite::default_glossary_path().into(),
            default_correction_policy: RewriteCorrectionPolicy::Balanced,
        }
    }
}

impl Default for CloudConfig {
    fn default() -> Self {
        Self {
            provider: CloudProvider::OpenAi,
            base_url: String::new(),
            api_key: String::new(),
            api_key_env: "OPENAI_API_KEY".into(),
            connect_timeout_ms: 3000,
            request_timeout_ms: 15000,
            transcription: CloudTranscriptionConfig::default(),
            rewrite: CloudRewriteConfig::default(),
        }
    }
}

impl Default for CloudTranscriptionConfig {
    fn default() -> Self {
        Self {
            model: "gpt-4o-mini-transcribe".into(),
            language_mode: CloudLanguageMode::InheritLocal,
            language: String::new(),
        }
    }
}

impl Default for CloudRewriteConfig {
    fn default() -> Self {
        Self {
            model: "gpt-4.1-mini".into(),
            temperature: 0.1,
            max_output_tokens: 256,
        }
    }
}

impl Default for PersonalizationConfig {
    fn default() -> Self {
        Self {
            dictionary_path: "~/.local/share/whispers/dictionary.toml".into(),
            snippets_path: "~/.local/share/whispers/snippets.toml".into(),
            snippet_trigger: "insert".into(),
        }
    }
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_entries: 3,
            max_age_ms: 8000,
            max_replace_graphemes: 400,
        }
    }
}

impl Default for CleanupConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            profile: CleanupProfile::Basic,
            spoken_formatting: true,
            backtrack: true,
            remove_fillers: true,
        }
    }
}

impl Default for FeedbackConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            start_sound: String::new(),
            stop_sound: String::new(),
        }
    }
}

impl TranscriptionConfig {
    pub fn resolved_local_backend(&self) -> TranscriptionBackend {
        match self.local_backend {
            TranscriptionBackend::WhisperCpp
            | TranscriptionBackend::FasterWhisper
            | TranscriptionBackend::Nemo => self.local_backend,
            TranscriptionBackend::Cloud => TranscriptionBackend::WhisperCpp,
        }
    }
}

use serde::Deserialize;
use std::path::{Path, PathBuf};

use crate::error::{Result, WhsprError};
use crate::rewrite_profile::RewriteProfile;
use crate::rewrite_protocol::RewriteCorrectionPolicy;

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub audio: AudioConfig,
    pub transcription: TranscriptionConfig,
    #[serde(default, rename = "whisper")]
    legacy_whisper: LegacyWhisperConfig,
    pub postprocess: PostprocessConfig,
    pub session: SessionConfig,
    pub personalization: PersonalizationConfig,
    pub rewrite: RewriteConfig,
    pub agentic_rewrite: AgenticRewriteConfig,
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
struct LegacyWhisperConfig {
    model_path: String,
    language: String,
    use_gpu: bool,
    flash_attn: bool,
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
    AdvancedLocal,
    AgenticRewrite,
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
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct AgenticRewriteConfig {
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
            Self::AdvancedLocal => "advanced_local",
            Self::AgenticRewrite => "agentic_rewrite",
            Self::LegacyBasic => "legacy_basic",
        }
    }

    pub fn uses_rewrite(self) -> bool {
        matches!(self, Self::AdvancedLocal | Self::AgenticRewrite)
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
        }
    }
}

impl Default for AgenticRewriteConfig {
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

impl Config {
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let config_path = resolve_config_path(path);

        if !config_path.exists() {
            tracing::info!(
                "no config file found at {}, using defaults",
                config_path.display()
            );
            return Ok(Config::default());
        }

        let contents = std::fs::read_to_string(&config_path).map_err(|e| {
            WhsprError::Config(format!("failed to read {}: {e}", config_path.display()))
        })?;

        let mut config: Config = toml::from_str(&contents).map_err(|e| {
            WhsprError::Config(format!("failed to parse {}: {e}", config_path.display()))
        })?;

        config.apply_legacy_transcription_migration(&contents, &config_path);
        config.apply_legacy_cleanup_migration(&contents, &config_path);
        config.apply_cloud_sanitization();
        Ok(config)
    }

    pub fn resolved_model_path(&self) -> PathBuf {
        PathBuf::from(expand_tilde(&self.transcription.model_path))
    }

    pub fn resolved_rewrite_model_path(&self) -> Option<PathBuf> {
        (!self.rewrite.model_path.trim().is_empty())
            .then(|| PathBuf::from(expand_tilde(&self.rewrite.model_path)))
    }

    pub fn resolved_rewrite_instructions_path(&self) -> Option<PathBuf> {
        (!self.rewrite.instructions_path.trim().is_empty())
            .then(|| PathBuf::from(expand_tilde(&self.rewrite.instructions_path)))
    }

    pub fn resolved_dictionary_path(&self) -> PathBuf {
        PathBuf::from(expand_tilde(&self.personalization.dictionary_path))
    }

    pub fn resolved_snippets_path(&self) -> PathBuf {
        PathBuf::from(expand_tilde(&self.personalization.snippets_path))
    }

    pub fn resolved_agentic_policy_path(&self) -> PathBuf {
        PathBuf::from(expand_tilde(&self.agentic_rewrite.policy_path))
    }

    pub fn resolved_agentic_glossary_path(&self) -> PathBuf {
        PathBuf::from(expand_tilde(&self.agentic_rewrite.glossary_path))
    }

    fn apply_legacy_transcription_migration(&mut self, contents: &str, config_path: &Path) {
        let transcription_present = section_present(contents, "transcription");
        let whisper_present = section_present(contents, "whisper");

        if !transcription_present && whisper_present {
            tracing::warn!(
                "config {} uses deprecated [whisper]; mapping to [transcription]",
                config_path.display()
            );
            self.transcription.backend = TranscriptionBackend::WhisperCpp;
            self.transcription.model_path = self.legacy_whisper.model_path.clone();
            self.transcription.language = self.legacy_whisper.language.clone();
            self.transcription.use_gpu = self.legacy_whisper.use_gpu;
            self.transcription.flash_attn = self.legacy_whisper.flash_attn;
        } else if whisper_present {
            tracing::warn!(
                "config {} contains deprecated [whisper]; [transcription] takes precedence",
                config_path.display()
            );
        }
    }

    fn apply_legacy_cleanup_migration(&mut self, contents: &str, config_path: &Path) {
        let cleanup_present = cleanup_section_present(contents);

        if cleanup_present && self.postprocess.mode == PostprocessMode::Raw {
            if self.cleanup.enabled {
                tracing::warn!(
                    "config {} uses deprecated [cleanup]; mapping to postprocess.mode = \"legacy_basic\"",
                    config_path.display()
                );
                self.postprocess.mode = PostprocessMode::LegacyBasic;
            } else {
                tracing::warn!(
                    "config {} disables deprecated [cleanup]; keeping postprocess.mode = \"raw\"",
                    config_path.display()
                );
            }
        } else if cleanup_present && self.postprocess.mode != PostprocessMode::LegacyBasic {
            tracing::warn!(
                "config {} contains deprecated [cleanup]; [postprocess] takes precedence",
                config_path.display()
            );
        }
    }

    fn apply_cloud_sanitization(&mut self) {
        if self.transcription.local_backend == TranscriptionBackend::Cloud {
            tracing::warn!(
                "transcription.local_backend cannot be cloud; falling back to whisper_cpp"
            );
            self.transcription.local_backend = TranscriptionBackend::WhisperCpp;
        }
        if self.cloud.api_key.trim().is_empty() && looks_like_cloud_api_key(&self.cloud.api_key_env)
        {
            tracing::warn!(
                "cloud.api_key_env looks like a literal API key; treating it as cloud.api_key"
            );
            self.cloud.api_key = self.cloud.api_key_env.trim().to_string();
            self.cloud.api_key_env = "OPENAI_API_KEY".into();
        }
    }
}

pub fn default_config_path() -> PathBuf {
    xdg_dir("config").join("whispers").join("config.toml")
}

pub fn resolve_config_path(path: Option<&Path>) -> PathBuf {
    match path {
        Some(p) => p.to_path_buf(),
        None => default_config_path(),
    }
}

pub fn data_dir() -> PathBuf {
    xdg_dir("data").join("whispers")
}

fn xdg_dir(kind: &str) -> PathBuf {
    match kind {
        "config" => {
            if let Ok(dir) = std::env::var("XDG_CONFIG_HOME") {
                PathBuf::from(dir)
            } else if let Ok(home) = std::env::var("HOME") {
                PathBuf::from(home).join(".config")
            } else {
                tracing::warn!("neither XDG_CONFIG_HOME nor HOME is set, falling back to /tmp");
                PathBuf::from("/tmp")
            }
        }
        "data" => {
            if let Ok(dir) = std::env::var("XDG_DATA_HOME") {
                PathBuf::from(dir)
            } else if let Ok(home) = std::env::var("HOME") {
                PathBuf::from(home).join(".local").join("share")
            } else {
                tracing::warn!("neither XDG_DATA_HOME nor HOME is set, falling back to /tmp");
                PathBuf::from("/tmp")
            }
        }
        _ => {
            tracing::warn!("unknown XDG directory kind '{kind}', falling back to /tmp");
            PathBuf::from("/tmp")
        }
    }
}

pub fn expand_tilde(path: &str) -> String {
    match path.strip_prefix("~/") {
        Some(rest) => {
            if let Ok(home) = std::env::var("HOME") {
                return format!("{home}/{rest}");
            }
            tracing::warn!("HOME is not set, cannot expand tilde in path: {path}");
        }
        None if path == "~" => {
            if let Ok(home) = std::env::var("HOME") {
                return home;
            }
            tracing::warn!("HOME is not set, cannot expand tilde in path: {path}");
        }
        _ => {}
    }
    path.to_string()
}

pub fn write_default_config(path: &Path, model_path: &str) -> Result<()> {
    let contents = format!(
        r#"# whispers configuration
#
# Keybinding is handled by your compositor. Example for Hyprland:
#   bind = SUPER ALT, D, exec, whispers
#
# First invocation starts recording, second invocation stops + transcribes + pastes.

[audio]
# Input device name (empty = system default)
device = ""
# Sample rate in Hz (ASR requires 16000)
sample_rate = 16000

[transcription]
# Active transcription backend ("whisper_cpp", "faster_whisper", "nemo", or "cloud")
backend = "whisper_cpp"
# Cloud fallback behavior ("configured_local" or "none")
fallback = "configured_local"
# Local backend used directly in local mode and as the cloud fallback backend
local_backend = "whisper_cpp"
# Managed ASR model name for the selected backend
selected_model = "large-v3-turbo"
# Path to the local backend-specific model or empty to use the selected managed model
# Manage models with: whispers asr-model list / download / select
model_path = "{model_path}"
# Language code ("en", "fr", "de", etc.) or "auto" for auto-detect
language = "auto"
# Enable GPU acceleration (set false to force CPU)
use_gpu = true
# Enable flash attention when GPU is enabled
flash_attn = true
# How long the hidden ASR worker stays warm without requests (0 = never expire)
idle_timeout_ms = 120000

[postprocess]
# "raw" (default), "advanced_local", "agentic_rewrite", or "legacy_basic" for deprecated cleanup configs
mode = "raw"

[session]
# Enable short-lived session backtracking in rewrite modes
enabled = true
# How many recent dictation entries to keep in the runtime session ledger
max_entries = 3
# How long a recent dictation entry stays eligible for revision
max_age_ms = 8000
# Maximum graphemes that may be deleted when revising the latest entry
max_replace_graphemes = 400

[personalization]
# Dictionary replacements applied in all modes
dictionary_path = "~/.local/share/whispers/dictionary.toml"
# Snippets expanded via an explicit spoken trigger
snippets_path = "~/.local/share/whispers/snippets.toml"
# Spoken trigger phrase used before snippet names
snippet_trigger = "insert"

[rewrite]
# Rewrite backend ("local" or "cloud")
backend = "local"
# Cloud fallback behavior ("local" or "none")
fallback = "local"
# Managed rewrite model name for advanced_local mode
selected_model = "qwen-3.5-4b-q4_k_m"
# Manual GGUF path override (empty = use selected managed model)
# Custom rewrite models should be chat-capable GGUFs with an embedded
# chat template that llama.cpp can apply at runtime.
model_path = ""
# Append-only custom rewrite instructions file (empty = disabled)
instructions_path = "~/.local/share/whispers/rewrite-instructions.txt"
# Rewrite profile selection ("auto", "qwen", "generic", or "llama_compat")
profile = "auto"
# Timeout for local rewrite inference in milliseconds
timeout_ms = 30000
# How long the hidden rewrite worker stays warm without requests
idle_timeout_ms = 120000
# Maximum characters accepted from the rewrite model
max_output_chars = 1200
# Maximum tokens to generate for rewritten output
max_tokens = 256

[agentic_rewrite]
# App-aware rewrite policy rules used by postprocess.mode = "agentic_rewrite"
policy_path = "~/.local/share/whispers/app-rewrite-policy.toml"
# Technical glossary used by postprocess.mode = "agentic_rewrite"
glossary_path = "~/.local/share/whispers/technical-glossary.toml"
# Default correction policy ("conservative", "balanced", or "aggressive")
default_correction_policy = "balanced"

[cloud]
# Cloud provider ("openai" or "openai_compatible")
provider = "openai"
# Custom base URL for openai_compatible providers (empty uses the OpenAI default)
base_url = ""
# Optional API key stored directly in the config (empty = use api_key_env instead)
api_key = ""
# Environment variable holding the API key
api_key_env = "OPENAI_API_KEY"
# Network connect timeout in milliseconds
connect_timeout_ms = 3000
# End-to-end request timeout in milliseconds
request_timeout_ms = 15000

[cloud.transcription]
# Cloud transcription model
model = "gpt-4o-mini-transcribe"
# "inherit_local" uses [transcription].language when it is not "auto"; "force" uses the value below
language_mode = "inherit_local"
# Language code used when language_mode = "force"
language = ""

[cloud.rewrite]
# Cloud rewrite model
model = "gpt-4.1-mini"
# Sampling temperature for cloud rewrite
temperature = 0.1
# Maximum tokens requested from the cloud rewrite model
max_output_tokens = 256

[feedback]
# Play sound feedback on start/stop
enabled = true
# Custom sound file paths (empty = use bundled sounds)
start_sound = ""
stop_sound = ""
"#
    );

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| WhsprError::Config(format!("failed to create config directory: {e}")))?;
    }

    std::fs::write(path, contents)
        .map_err(|e| WhsprError::Config(format!("failed to write config: {e}")))?;

    Ok(())
}

pub fn update_config_transcription_selection(
    config_path: &Path,
    backend: TranscriptionBackend,
    selected_model: &str,
    model_path: &str,
    set_active_backend: bool,
) -> Result<()> {
    let contents = std::fs::read_to_string(config_path)
        .map_err(|e| WhsprError::Config(format!("failed to read config: {e}")))?;

    let mut doc = contents
        .parse::<toml_edit::DocumentMut>()
        .map_err(|e| WhsprError::Config(format!("failed to parse config: {e}")))?;

    ensure_standard_postprocess_tables(&mut doc);
    doc.as_table_mut().remove("whisper");
    if set_active_backend {
        doc["transcription"]["backend"] = toml_edit::value(backend.as_str());
    }
    doc["transcription"]["local_backend"] = toml_edit::value(backend.as_str());
    doc["transcription"]["fallback"] =
        toml_edit::value(TranscriptionFallback::ConfiguredLocal.as_str());
    doc["transcription"]["selected_model"] = toml_edit::value(selected_model);
    doc["transcription"]["model_path"] = toml_edit::value(model_path);
    let idle_timeout_ms = if backend == TranscriptionBackend::Nemo {
        0
    } else {
        120000
    };
    doc["transcription"]["idle_timeout_ms"] = toml_edit::value(idle_timeout_ms);

    std::fs::write(config_path, doc.to_string())
        .map_err(|e| WhsprError::Config(format!("failed to write config: {e}")))?;

    Ok(())
}

pub fn update_config_postprocess_mode(config_path: &Path, mode: PostprocessMode) -> Result<()> {
    let contents = std::fs::read_to_string(config_path)
        .map_err(|e| WhsprError::Config(format!("failed to read config: {e}")))?;

    let mut doc = contents
        .parse::<toml_edit::DocumentMut>()
        .map_err(|e| WhsprError::Config(format!("failed to parse config: {e}")))?;

    ensure_standard_postprocess_tables(&mut doc);
    doc["postprocess"]["mode"] = toml_edit::value(mode.as_str());

    std::fs::write(config_path, doc.to_string())
        .map_err(|e| WhsprError::Config(format!("failed to write config: {e}")))?;

    Ok(())
}

pub fn update_config_rewrite_selection(config_path: &Path, selected_model: &str) -> Result<()> {
    let contents = std::fs::read_to_string(config_path)
        .map_err(|e| WhsprError::Config(format!("failed to read config: {e}")))?;

    let mut doc = contents
        .parse::<toml_edit::DocumentMut>()
        .map_err(|e| WhsprError::Config(format!("failed to parse config: {e}")))?;

    ensure_standard_postprocess_tables(&mut doc);
    let mode = match doc["postprocess"]["mode"].as_str() {
        Some("agentic_rewrite") => PostprocessMode::AgenticRewrite,
        _ => PostprocessMode::AdvancedLocal,
    };
    doc["postprocess"]["mode"] = toml_edit::value(mode.as_str());
    let rewrite_backend = doc["rewrite"]
        .as_table_like()
        .and_then(|table| table.get("backend"))
        .and_then(|item| item.as_str());
    if !matches!(rewrite_backend, Some("cloud")) {
        doc["rewrite"]["backend"] = toml_edit::value(RewriteBackend::Local.as_str());
    }
    doc["rewrite"]["fallback"] = toml_edit::value(RewriteFallback::Local.as_str());
    doc["rewrite"]["selected_model"] = toml_edit::value(selected_model);
    doc["rewrite"]["model_path"] = toml_edit::value("");
    doc["rewrite"]["instructions_path"] =
        toml_edit::value("~/.local/share/whispers/rewrite-instructions.txt");
    doc["rewrite"]["profile"] = toml_edit::value(RewriteProfile::Auto.as_str());
    doc["rewrite"]["timeout_ms"] = toml_edit::value(30000);
    doc["rewrite"]["idle_timeout_ms"] = toml_edit::value(120000);
    doc["rewrite"]["max_output_chars"] = toml_edit::value(1200);
    doc["rewrite"]["max_tokens"] = toml_edit::value(256);
    doc["agentic_rewrite"]["policy_path"] =
        toml_edit::value(crate::agentic_rewrite::default_policy_path());
    doc["agentic_rewrite"]["glossary_path"] =
        toml_edit::value(crate::agentic_rewrite::default_glossary_path());
    doc["agentic_rewrite"]["default_correction_policy"] =
        toml_edit::value(RewriteCorrectionPolicy::Balanced.as_str());

    std::fs::write(config_path, doc.to_string())
        .map_err(|e| WhsprError::Config(format!("failed to write config: {e}")))?;

    Ok(())
}

pub fn update_config_transcription_runtime(
    config_path: &Path,
    backend: TranscriptionBackend,
    fallback: TranscriptionFallback,
) -> Result<()> {
    let contents = std::fs::read_to_string(config_path)
        .map_err(|e| WhsprError::Config(format!("failed to read config: {e}")))?;
    let mut doc = contents
        .parse::<toml_edit::DocumentMut>()
        .map_err(|e| WhsprError::Config(format!("failed to parse config: {e}")))?;

    ensure_standard_postprocess_tables(&mut doc);
    doc["transcription"]["backend"] = toml_edit::value(backend.as_str());
    doc["transcription"]["fallback"] = toml_edit::value(fallback.as_str());

    std::fs::write(config_path, doc.to_string())
        .map_err(|e| WhsprError::Config(format!("failed to write config: {e}")))?;
    Ok(())
}

pub fn update_config_rewrite_runtime(
    config_path: &Path,
    backend: RewriteBackend,
    fallback: RewriteFallback,
) -> Result<()> {
    let contents = std::fs::read_to_string(config_path)
        .map_err(|e| WhsprError::Config(format!("failed to read config: {e}")))?;
    let mut doc = contents
        .parse::<toml_edit::DocumentMut>()
        .map_err(|e| WhsprError::Config(format!("failed to parse config: {e}")))?;

    ensure_standard_postprocess_tables(&mut doc);
    normalize_postprocess_mode(&mut doc);
    doc["rewrite"]["backend"] = toml_edit::value(backend.as_str());
    doc["rewrite"]["fallback"] = toml_edit::value(fallback.as_str());

    std::fs::write(config_path, doc.to_string())
        .map_err(|e| WhsprError::Config(format!("failed to write config: {e}")))?;
    Ok(())
}

pub fn update_config_cloud_settings(
    config_path: &Path,
    settings: &CloudSettingsUpdate<'_>,
) -> Result<()> {
    let contents = std::fs::read_to_string(config_path)
        .map_err(|e| WhsprError::Config(format!("failed to read config: {e}")))?;
    let mut doc = contents
        .parse::<toml_edit::DocumentMut>()
        .map_err(|e| WhsprError::Config(format!("failed to parse config: {e}")))?;

    ensure_standard_postprocess_tables(&mut doc);
    ensure_root_table(&mut doc, "cloud");
    ensure_nested_table(&mut doc, "cloud", "transcription");
    ensure_nested_table(&mut doc, "cloud", "rewrite");
    doc["cloud"]["provider"] = toml_edit::value(settings.provider.as_str());
    doc["cloud"]["base_url"] = toml_edit::value(settings.base_url);
    doc["cloud"]["api_key"] = toml_edit::value(settings.api_key);
    doc["cloud"]["api_key_env"] = toml_edit::value(settings.api_key_env);
    doc["cloud"]["connect_timeout_ms"] = toml_edit::value(settings.connect_timeout_ms as i64);
    doc["cloud"]["request_timeout_ms"] = toml_edit::value(settings.request_timeout_ms as i64);
    doc["cloud"]["transcription"]["model"] = toml_edit::value(settings.transcription_model);
    doc["cloud"]["transcription"]["language_mode"] =
        toml_edit::value(settings.transcription_language_mode.as_str());
    doc["cloud"]["transcription"]["language"] = toml_edit::value(settings.transcription_language);
    doc["cloud"]["rewrite"]["model"] = toml_edit::value(settings.rewrite_model);
    doc["cloud"]["rewrite"]["temperature"] = toml_edit::value(settings.rewrite_temperature as f64);
    doc["cloud"]["rewrite"]["max_output_tokens"] =
        toml_edit::value(settings.rewrite_max_output_tokens as i64);

    std::fs::write(config_path, doc.to_string())
        .map_err(|e| WhsprError::Config(format!("failed to write config: {e}")))?;
    Ok(())
}

fn ensure_standard_postprocess_tables(doc: &mut toml_edit::DocumentMut) {
    ensure_root_table(doc, "transcription");
    ensure_root_table(doc, "postprocess");
    ensure_root_table(doc, "session");
    ensure_root_table(doc, "rewrite");
    ensure_root_table(doc, "agentic_rewrite");
    ensure_root_table(doc, "cloud");
    ensure_nested_table(doc, "cloud", "transcription");
    ensure_nested_table(doc, "cloud", "rewrite");
    ensure_root_table(doc, "personalization");
}

fn normalize_postprocess_mode(doc: &mut toml_edit::DocumentMut) {
    let current = doc["postprocess"]["mode"].as_str().unwrap_or_default();
    if !matches!(
        current,
        "raw" | "advanced_local" | "agentic_rewrite" | "legacy_basic"
    ) {
        doc["postprocess"]["mode"] = toml_edit::value(PostprocessMode::AdvancedLocal.as_str());
    }
}

fn ensure_root_table(doc: &mut toml_edit::DocumentMut, key: &str) {
    let root = doc.as_table_mut();
    let needs_insert = !root.contains_key(key) || !root[key].is_table();
    if needs_insert {
        root.insert(key, toml_edit::Item::Table(toml_edit::Table::new()));
    }
}

fn ensure_nested_table(doc: &mut toml_edit::DocumentMut, parent: &str, child: &str) {
    ensure_root_table(doc, parent);
    let root = doc.as_table_mut();
    let Some(parent_item) = root.get_mut(parent) else {
        return;
    };
    let Some(parent_table) = parent_item.as_table_like_mut() else {
        return;
    };
    let needs_insert = !parent_table.contains_key(child)
        || parent_table
            .get(child)
            .map(|item| !item.is_table())
            .unwrap_or(true);
    if needs_insert {
        parent_table.insert(child, toml_edit::Item::Table(toml_edit::Table::new()));
    }
}

fn cleanup_section_present(contents: &str) -> bool {
    section_present(contents, "cleanup")
}

fn section_present(contents: &str, name: &str) -> bool {
    toml::from_str::<toml::Value>(contents)
        .ok()
        .and_then(|value| value.get(name).cloned())
        .is_some()
}

fn looks_like_cloud_api_key(value: &str) -> bool {
    let trimmed = value.trim();
    trimmed.starts_with("sk-")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::WhsprError;

    #[test]
    fn load_missing_file_uses_defaults() {
        let path = crate::test_support::unique_temp_path("config-missing", "toml");
        let config = Config::load(Some(&path)).expect("missing config should load defaults");
        assert_eq!(config.audio.sample_rate, 16000);
        assert_eq!(config.transcription.language, "auto");
        assert_eq!(
            config.transcription.backend,
            TranscriptionBackend::WhisperCpp
        );
        assert_eq!(config.postprocess.mode, PostprocessMode::Raw);
        assert_eq!(config.personalization.snippet_trigger, "insert");
        assert_eq!(config.rewrite.selected_model, "qwen-3.5-4b-q4_k_m");
    }

    #[test]
    fn load_invalid_toml_returns_parse_error() {
        let path = crate::test_support::unique_temp_path("config-invalid", "toml");
        std::fs::write(&path, "not = [valid = toml").expect("write invalid config");
        let err = Config::load(Some(&path)).expect_err("invalid config should fail");
        match err {
            WhsprError::Config(msg) => {
                assert!(msg.contains("failed to parse"), "unexpected message: {msg}");
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn expand_tilde_uses_home_when_present() {
        let _env_lock = crate::test_support::env_lock();
        let _guard = crate::test_support::EnvVarGuard::capture(&["HOME"]);
        crate::test_support::set_env("HOME", "/tmp/whispers-home");
        assert_eq!(
            expand_tilde("~/models/ggml.bin"),
            "/tmp/whispers-home/models/ggml.bin"
        );
        assert_eq!(expand_tilde("~"), "/tmp/whispers-home");
    }

    #[test]
    fn expand_tilde_without_home_returns_original_path() {
        let _env_lock = crate::test_support::env_lock();
        let _guard = crate::test_support::EnvVarGuard::capture(&["HOME"]);
        crate::test_support::remove_env("HOME");
        assert_eq!(expand_tilde("~/models/ggml.bin"), "~/models/ggml.bin");
        assert_eq!(expand_tilde("~"), "~");
    }

    #[test]
    fn write_default_and_update_model_path_roundtrip() {
        let dir = crate::test_support::unique_temp_dir("config-roundtrip");
        let config_path = dir.join("nested").join("config.toml");

        write_default_config(&config_path, "~/old-model.bin").expect("write config");
        assert!(config_path.exists(), "config file should exist");

        update_config_transcription_selection(
            &config_path,
            TranscriptionBackend::WhisperCpp,
            "large-v3-turbo",
            "~/new-model.bin",
            true,
        )
        .expect("update config");
        let loaded = Config::load(Some(&config_path)).expect("load config");
        assert_eq!(loaded.transcription.model_path, "~/new-model.bin");
        assert_eq!(
            loaded.transcription.backend,
            TranscriptionBackend::WhisperCpp
        );
        assert_eq!(loaded.audio.sample_rate, 16000);
        assert_eq!(loaded.postprocess.mode, PostprocessMode::Raw);
        assert_eq!(
            loaded.personalization.dictionary_path,
            "~/.local/share/whispers/dictionary.toml"
        );
        assert!(loaded.session.enabled);
        assert_eq!(loaded.session.max_entries, 3);
        assert_eq!(loaded.rewrite.timeout_ms, 30000);
        assert!(loaded.feedback.enabled);

        let raw = std::fs::read_to_string(&config_path).expect("read config");
        assert!(raw.contains("[audio]"));
        assert!(raw.contains("[transcription]"));
        assert!(raw.contains("[postprocess]"));
        assert!(raw.contains("[session]"));
        assert!(raw.contains("[rewrite]"));
        assert!(!raw.contains("[whisper]"));
    }

    #[test]
    fn selecting_nemo_model_sets_non_expiring_asr_worker_timeout() {
        let config_path = crate::test_support::unique_temp_path("config-nemo-timeout", "toml");
        write_default_config(&config_path, "~/old-model.bin").expect("write config");

        update_config_transcription_selection(
            &config_path,
            TranscriptionBackend::Nemo,
            "parakeet-tdt_ctc-1.1b",
            "~/.local/share/whispers/nemo/models/parakeet-tdt_ctc-1.1b",
            true,
        )
        .expect("select nemo model");

        let loaded = Config::load(Some(&config_path)).expect("load config");
        assert_eq!(loaded.transcription.backend, TranscriptionBackend::Nemo);
        assert_eq!(loaded.transcription.idle_timeout_ms, 0);
    }

    #[test]
    fn load_legacy_whisper_section_maps_to_transcription() {
        let path = crate::test_support::unique_temp_path("config-whisper-legacy", "toml");
        std::fs::write(
            &path,
            r#"[whisper]
model_path = "~/legacy-model.bin"
language = "en"
use_gpu = false
flash_attn = false
"#,
        )
        .expect("write config");

        let loaded = Config::load(Some(&path)).expect("load config");
        assert_eq!(
            loaded.transcription.backend,
            TranscriptionBackend::WhisperCpp
        );
        assert_eq!(loaded.transcription.model_path, "~/legacy-model.bin");
        assert_eq!(loaded.transcription.language, "en");
        assert!(!loaded.transcription.use_gpu);
        assert!(!loaded.transcription.flash_attn);
    }

    #[test]
    fn load_legacy_cleanup_section_maps_to_legacy_basic() {
        let path = crate::test_support::unique_temp_path("config-cleanup", "toml");
        std::fs::write(
            &path,
            r#"[cleanup]
profile = "aggressive"
spoken_formatting = false
remove_fillers = false
"#,
        )
        .expect("write config");

        let config = Config::load(Some(&path)).expect("load config");
        assert_eq!(config.postprocess.mode, PostprocessMode::LegacyBasic);
        assert_eq!(config.cleanup.profile, CleanupProfile::Aggressive);
        assert!(!config.cleanup.spoken_formatting);
        assert!(config.cleanup.backtrack);
        assert!(!config.cleanup.remove_fillers);
    }

    #[test]
    fn update_rewrite_selection_enables_advanced_mode() {
        let dir = crate::test_support::unique_temp_dir("config-rewrite-select");
        let config_path = dir.join("config.toml");
        write_default_config(&config_path, "~/model.bin").expect("write config");

        update_config_rewrite_selection(&config_path, "qwen-3.5-2b-q4_k_m")
            .expect("select rewrite model");

        let loaded = Config::load(Some(&config_path)).expect("load config");
        assert_eq!(loaded.postprocess.mode, PostprocessMode::AdvancedLocal);
        assert_eq!(loaded.rewrite.selected_model, "qwen-3.5-2b-q4_k_m");
        assert!(loaded.rewrite.model_path.is_empty());
        assert_eq!(
            loaded.rewrite.instructions_path,
            "~/.local/share/whispers/rewrite-instructions.txt"
        );
        assert_eq!(loaded.rewrite.profile, RewriteProfile::Auto);
        assert_eq!(loaded.rewrite.timeout_ms, 30000);
        assert_eq!(loaded.rewrite.idle_timeout_ms, 120000);
    }

    #[test]
    fn update_helpers_upgrade_legacy_configs_without_panicking() {
        let config_path = crate::test_support::unique_temp_path("config-legacy-upgrade", "toml");
        std::fs::write(
            &config_path,
            r#"[audio]
sample_rate = 16000

[whisper]
model_path = "~/.local/share/whispers/ggml-large-v3-turbo.bin"
language = "auto"
"#,
        )
        .expect("write legacy config");

        update_config_transcription_selection(
            &config_path,
            TranscriptionBackend::WhisperCpp,
            "large-v3-turbo",
            "~/.local/share/whispers/ggml-large-v3-turbo.bin",
            true,
        )
        .expect("update transcription selection");
        update_config_rewrite_selection(&config_path, "qwen-3.5-4b-q4_k_m")
            .expect("update rewrite selection");

        let loaded = Config::load(Some(&config_path)).expect("load upgraded config");
        assert_eq!(
            loaded.transcription.backend,
            TranscriptionBackend::WhisperCpp
        );
        assert_eq!(loaded.transcription.selected_model, "large-v3-turbo");
        assert_eq!(loaded.postprocess.mode, PostprocessMode::AdvancedLocal);
        assert_eq!(loaded.rewrite.selected_model, "qwen-3.5-4b-q4_k_m");

        let raw = std::fs::read_to_string(&config_path).expect("read upgraded config");
        assert!(!raw.contains("[whisper]"));
    }

    #[test]
    fn load_cloud_literal_key_from_legacy_api_key_env() {
        let path = crate::test_support::unique_temp_path("config-cloud-literal-key", "toml");
        std::fs::write(
            &path,
            r#"[cloud]
api_key_env = "sk-test-inline"
"#,
        )
        .expect("write config");

        let loaded = Config::load(Some(&path)).expect("load config");
        assert_eq!(loaded.cloud.api_key, "sk-test-inline");
        assert_eq!(loaded.cloud.api_key_env, "OPENAI_API_KEY");
    }
}

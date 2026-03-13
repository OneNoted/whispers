use serde::Deserialize;
use std::path::{Path, PathBuf};

use crate::branding;
use crate::error::{Result, WhsprError};

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub audio: AudioConfig,
    pub whisper: WhisperConfig,
    pub postprocess: PostprocessConfig,
    pub rewrite: RewriteConfig,
    pub inject: InjectConfig,
    pub feedback: FeedbackConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    pub device: String,
    pub sample_rate: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct WhisperConfig {
    pub model_path: String,
    pub language: String,
    pub use_gpu: bool,
    pub flash_attn: bool,
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
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct RewriteConfig {
    pub selected_model: String,
    pub model_path: String,
    pub timeout_ms: u64,
    pub max_output_chars: usize,
    pub max_tokens: usize,
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

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            model_path: managed_data_path_for_config("ggml-large-v3-turbo.bin"),
            language: "auto".into(),
            use_gpu: true,
            flash_attn: true,
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
        }
    }
}

impl Default for RewriteConfig {
    fn default() -> Self {
        Self {
            selected_model: "llama-3.2-3b-q4_k_m".into(),
            model_path: String::new(),
            timeout_ms: 30000,
            max_output_chars: 1200,
            max_tokens: 256,
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

        let config: Config = toml::from_str(&contents).map_err(|e| {
            WhsprError::Config(format!("failed to parse {}: {e}", config_path.display()))
        })?;
        Ok(config)
    }

    pub fn resolved_model_path(&self) -> PathBuf {
        PathBuf::from(expand_tilde(&self.whisper.model_path))
    }

    pub fn resolved_rewrite_model_path(&self) -> Option<PathBuf> {
        (!self.rewrite.model_path.trim().is_empty())
            .then(|| PathBuf::from(expand_tilde(&self.rewrite.model_path)))
    }
}

pub fn default_config_path() -> PathBuf {
    xdg_dir("config")
        .join(branding::APP_NAME)
        .join("config.toml")
}

pub fn resolve_config_path(path: Option<&Path>) -> PathBuf {
    match path {
        Some(p) => p.to_path_buf(),
        None => default_config_path(),
    }
}

pub fn data_dir() -> PathBuf {
    preferred_data_dir()
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

fn preferred_data_dir() -> PathBuf {
    xdg_dir("data").join(branding::APP_NAME)
}

fn managed_data_path_for_config(filename: &str) -> String {
    format!("~/.local/share/{}/{}", branding::APP_NAME, filename)
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
# Sample rate in Hz (whisper requires 16000)
sample_rate = 16000

[whisper]
# Path to ggml whisper model file
# Manage models with: whispers model list / download / select
model_path = "{model_path}"
# Language code ("en", "fr", "de", etc.) or "auto" for auto-detect
language = "auto"
# Enable GPU acceleration (set false to force CPU)
use_gpu = true
# Enable flash attention when GPU is enabled
flash_attn = true

[postprocess]
# "raw" (default) or "advanced_local"
mode = "raw"

[rewrite]
# Managed rewrite model name for advanced_local mode
selected_model = "llama-3.2-3b-q4_k_m"
# Manual GGUF path override (empty = use selected managed model)
# Custom rewrite models should be chat-capable GGUFs with an embedded
# chat template that llama.cpp can apply at runtime.
model_path = ""
# Timeout for local rewrite inference in milliseconds
timeout_ms = 30000
# Maximum characters accepted from the rewrite model
max_output_chars = 1200
# Maximum tokens to generate for rewritten output
max_tokens = 256

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

pub fn update_config_model_path(config_path: &Path, new_model_path: &str) -> Result<()> {
    let contents = std::fs::read_to_string(config_path)
        .map_err(|e| WhsprError::Config(format!("failed to read config: {e}")))?;

    let mut doc = contents
        .parse::<toml_edit::DocumentMut>()
        .map_err(|e| WhsprError::Config(format!("failed to parse config: {e}")))?;

    doc["whisper"]["model_path"] = toml_edit::value(new_model_path);

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
    doc["postprocess"]["mode"] = toml_edit::value(PostprocessMode::AdvancedLocal.as_str());
    doc["rewrite"]["selected_model"] = toml_edit::value(selected_model);
    doc["rewrite"]["model_path"] = toml_edit::value("");
    doc["rewrite"]["timeout_ms"] = toml_edit::value(30000);
    doc["rewrite"]["max_output_chars"] = toml_edit::value(1200);
    doc["rewrite"]["max_tokens"] = toml_edit::value(256);

    std::fs::write(config_path, doc.to_string())
        .map_err(|e| WhsprError::Config(format!("failed to write config: {e}")))?;

    Ok(())
}

fn ensure_standard_postprocess_tables(doc: &mut toml_edit::DocumentMut) {
    if !doc["postprocess"].is_table() {
        doc["postprocess"] = toml_edit::Item::Table(toml_edit::Table::new());
    }

    if !doc["rewrite"].is_table() {
        doc["rewrite"] = toml_edit::Item::Table(toml_edit::Table::new());
    }
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
        assert_eq!(config.whisper.language, "auto");
        assert_eq!(config.postprocess.mode, PostprocessMode::Raw);
        assert_eq!(config.rewrite.selected_model, "llama-3.2-3b-q4_k_m");
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

        update_config_model_path(&config_path, "~/new-model.bin").expect("update config");
        let loaded = Config::load(Some(&config_path)).expect("load config");
        assert_eq!(loaded.whisper.model_path, "~/new-model.bin");
        assert_eq!(loaded.audio.sample_rate, 16000);
        assert_eq!(loaded.postprocess.mode, PostprocessMode::Raw);
        assert_eq!(loaded.rewrite.timeout_ms, 30000);
        assert!(loaded.feedback.enabled);

        let raw = std::fs::read_to_string(&config_path).expect("read config");
        assert!(raw.contains("[audio]"));
        assert!(raw.contains("[whisper]"));
        assert!(raw.contains("[postprocess]"));
        assert!(raw.contains("[rewrite]"));
    }

    #[test]
    fn update_rewrite_selection_enables_advanced_mode() {
        let dir = crate::test_support::unique_temp_dir("config-rewrite-select");
        let config_path = dir.join("config.toml");
        write_default_config(&config_path, "~/model.bin").expect("write config");

        update_config_rewrite_selection(&config_path, "llama-3.2-1b-q4_k_m")
            .expect("select rewrite model");

        let loaded = Config::load(Some(&config_path)).expect("load config");
        assert_eq!(loaded.postprocess.mode, PostprocessMode::AdvancedLocal);
        assert_eq!(loaded.rewrite.selected_model, "llama-3.2-1b-q4_k_m");
        assert!(loaded.rewrite.model_path.is_empty());
        assert_eq!(loaded.rewrite.timeout_ms, 30000);
    }
}

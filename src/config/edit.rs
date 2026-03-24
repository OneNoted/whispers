use std::path::Path;

use crate::error::{Result, WhsprError};
use crate::rewrite_profile::RewriteProfile;

use super::{
    CloudSettingsUpdate, PostprocessMode, RewriteBackend, RewriteFallback, TranscriptionBackend,
    TranscriptionFallback,
};

pub(crate) fn default_config_template(model_path: &str) -> String {
    format!(
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
# "raw" (default), "rewrite", or "legacy_basic" for deprecated cleanup configs
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
# Managed rewrite model name for rewrite mode
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
# App-aware rewrite policy rules used by postprocess.mode = "rewrite"
policy_path = "~/.local/share/whispers/app-rewrite-policy.toml"
# Technical glossary used by postprocess.mode = "rewrite"
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
    )
}

pub fn write_default_config(path: &Path, model_path: &str) -> Result<()> {
    let contents = default_config_template(model_path);

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
    doc["postprocess"]["mode"] = toml_edit::value(PostprocessMode::Rewrite.as_str());
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
    ensure_root_table(doc, "cloud");
    ensure_nested_table(doc, "cloud", "transcription");
    ensure_nested_table(doc, "cloud", "rewrite");
    ensure_root_table(doc, "personalization");
}

fn normalize_postprocess_mode(doc: &mut toml_edit::DocumentMut) {
    let current = doc["postprocess"]["mode"].as_str().unwrap_or_default();
    if !matches!(current, "raw" | "rewrite" | "legacy_basic") {
        doc["postprocess"]["mode"] = toml_edit::value(PostprocessMode::Rewrite.as_str());
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

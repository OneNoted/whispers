use std::fmt::Write;
use std::path::{Path, PathBuf};

use console::style;

use crate::config::{self, CloudProvider, Config, RewriteBackend, TranscriptionBackend};
use crate::error::Result;

pub fn print_status(config_path_override: Option<&Path>) -> Result<()> {
    let config_path = config::resolve_config_path(config_path_override);
    let config_exists = config_path.exists();
    let config = Config::load(Some(&config_path))?;
    print!("{}", render_status(&config_path, config_exists, &config));
    Ok(())
}

fn render_status(config_path: &Path, config_exists: bool, config: &Config) -> String {
    let mut out = String::new();

    let _ = writeln!(out, "{}", crate::ui::header("Whispers status"));
    push_section(&mut out, "Config");
    push_path_field(&mut out, "path", config_path);
    push_field(
        &mut out,
        "source",
        if config_exists {
            "config file"
        } else {
            "defaults (config file missing)"
        },
        ValueStyle::Status,
    );
    push_path_field(&mut out, "data_dir", &config::data_dir());

    push_section(&mut out, "Transcription");
    push_field(
        &mut out,
        "backend",
        config.transcription.backend.as_str(),
        ValueStyle::Backend,
    );
    push_field(
        &mut out,
        "local_backend",
        config.transcription.resolved_local_backend().as_str(),
        ValueStyle::Backend,
    );
    push_field(
        &mut out,
        "fallback",
        config.transcription.fallback.as_str(),
        ValueStyle::Value,
    );
    push_field(
        &mut out,
        "selected_model",
        &config.transcription.selected_model,
        ValueStyle::Value,
    );
    push_field(
        &mut out,
        "model_status",
        transcription_model_status(config),
        ValueStyle::Status,
    );
    push_path_field(&mut out, "model_path", &config.resolved_model_path());
    push_field(
        &mut out,
        "language",
        &config.transcription.language,
        ValueStyle::Value,
    );
    push_field(
        &mut out,
        "use_gpu",
        yes_no(config.transcription.use_gpu),
        ValueStyle::Boolean,
    );
    push_field(
        &mut out,
        "flash_attn",
        yes_no(config.transcription.flash_attn),
        ValueStyle::Boolean,
    );
    push_field(
        &mut out,
        "idle_timeout_ms",
        config.transcription.idle_timeout_ms.to_string(),
        ValueStyle::Value,
    );

    push_section(&mut out, "Postprocess");
    push_field(
        &mut out,
        "mode",
        config.postprocess.mode.as_str(),
        ValueStyle::Value,
    );
    push_field(
        &mut out,
        "rewrite_enabled",
        yes_no(config.postprocess.mode.uses_rewrite()),
        ValueStyle::Boolean,
    );

    push_section(&mut out, "Rewrite");
    push_field(
        &mut out,
        "backend",
        config.rewrite.backend.as_str(),
        if config.rewrite.backend == RewriteBackend::Cloud {
            ValueStyle::Provider
        } else {
            ValueStyle::Backend
        },
    );
    push_field(
        &mut out,
        "fallback",
        config.rewrite.fallback.as_str(),
        ValueStyle::Value,
    );
    push_field(
        &mut out,
        "selected_model",
        &config.rewrite.selected_model,
        ValueStyle::Value,
    );
    push_field(
        &mut out,
        "local_model_status",
        rewrite_model_status(config),
        ValueStyle::Status,
    );
    push_field(
        &mut out,
        "local_model_path",
        optional_path_display(resolve_rewrite_model_path(config).as_deref()),
        ValueStyle::Path,
    );
    push_field(
        &mut out,
        "profile",
        config.rewrite.profile.as_str(),
        ValueStyle::Value,
    );
    push_field(
        &mut out,
        "instructions_status",
        path_presence_status(config.resolved_rewrite_instructions_path().as_deref()),
        ValueStyle::Status,
    );
    push_field(
        &mut out,
        "instructions_path",
        optional_path_display(config.resolved_rewrite_instructions_path().as_deref()),
        ValueStyle::Path,
    );
    push_field(
        &mut out,
        "timeout_ms",
        config.rewrite.timeout_ms.to_string(),
        ValueStyle::Value,
    );
    push_field(
        &mut out,
        "idle_timeout_ms",
        config.rewrite.idle_timeout_ms.to_string(),
        ValueStyle::Value,
    );
    push_field(
        &mut out,
        "max_output_chars",
        config.rewrite.max_output_chars.to_string(),
        ValueStyle::Value,
    );
    push_field(
        &mut out,
        "max_tokens",
        config.rewrite.max_tokens.to_string(),
        ValueStyle::Value,
    );

    push_section(&mut out, "Agentic Rewrite");
    push_field(
        &mut out,
        "enabled",
        yes_no(matches!(
            config.postprocess.mode,
            crate::config::PostprocessMode::AgenticRewrite
        )),
        ValueStyle::Boolean,
    );
    push_field(
        &mut out,
        "default_correction_policy",
        config.agentic_rewrite.default_correction_policy.as_str(),
        ValueStyle::Value,
    );
    push_field(
        &mut out,
        "policy_status",
        path_presence_status(Some(config.resolved_agentic_policy_path().as_path())),
        ValueStyle::Status,
    );
    push_path_field(
        &mut out,
        "policy_path",
        &config.resolved_agentic_policy_path(),
    );
    push_field(
        &mut out,
        "glossary_status",
        path_presence_status(Some(config.resolved_agentic_glossary_path().as_path())),
        ValueStyle::Status,
    );
    push_path_field(
        &mut out,
        "glossary_path",
        &config.resolved_agentic_glossary_path(),
    );

    push_section(&mut out, "Cloud");
    push_field(
        &mut out,
        "transcription_active",
        yes_no(config.transcription.backend == TranscriptionBackend::Cloud),
        ValueStyle::Boolean,
    );
    push_field(
        &mut out,
        "rewrite_active",
        yes_no(
            config.postprocess.mode.uses_rewrite()
                && config.rewrite.backend == RewriteBackend::Cloud,
        ),
        ValueStyle::Boolean,
    );
    push_field(
        &mut out,
        "provider",
        config.cloud.provider.as_str(),
        ValueStyle::Provider,
    );
    push_field(
        &mut out,
        "base_url",
        cloud_base_url_display(config),
        ValueStyle::Path,
    );
    push_field(
        &mut out,
        "api_key",
        cloud_api_key_status(config),
        ValueStyle::Status,
    );
    push_field(
        &mut out,
        "transcription_model",
        &config.cloud.transcription.model,
        ValueStyle::Value,
    );
    push_field(
        &mut out,
        "language_mode",
        config.cloud.transcription.language_mode.as_str(),
        ValueStyle::Value,
    );
    push_field(
        &mut out,
        "forced_language",
        if config.cloud.transcription.language.trim().is_empty() {
            "(none)"
        } else {
            config.cloud.transcription.language.as_str()
        },
        ValueStyle::Value,
    );
    push_field(
        &mut out,
        "rewrite_model",
        &config.cloud.rewrite.model,
        ValueStyle::Value,
    );
    push_field(
        &mut out,
        "rewrite_temperature",
        format!("{:.2}", config.cloud.rewrite.temperature),
        ValueStyle::Value,
    );
    push_field(
        &mut out,
        "rewrite_max_output_tokens",
        config.cloud.rewrite.max_output_tokens.to_string(),
        ValueStyle::Value,
    );
    push_field(
        &mut out,
        "connect_timeout_ms",
        config.cloud.connect_timeout_ms.to_string(),
        ValueStyle::Value,
    );
    push_field(
        &mut out,
        "request_timeout_ms",
        config.cloud.request_timeout_ms.to_string(),
        ValueStyle::Value,
    );

    push_section(&mut out, "Personalization");
    push_field(
        &mut out,
        "dictionary_status",
        path_presence_status(Some(config.resolved_dictionary_path().as_path())),
        ValueStyle::Status,
    );
    push_path_field(
        &mut out,
        "dictionary_path",
        &config.resolved_dictionary_path(),
    );
    push_field(
        &mut out,
        "snippets_status",
        path_presence_status(Some(config.resolved_snippets_path().as_path())),
        ValueStyle::Status,
    );
    push_path_field(&mut out, "snippets_path", &config.resolved_snippets_path());
    push_field(
        &mut out,
        "snippet_trigger",
        &config.personalization.snippet_trigger,
        ValueStyle::Value,
    );

    push_section(&mut out, "Voice");
    push_field(
        &mut out,
        "live_inject",
        yes_no(config.voice.live_inject),
        ValueStyle::Boolean,
    );
    push_field(
        &mut out,
        "live_rewrite",
        yes_no(config.voice.live_rewrite),
        ValueStyle::Boolean,
    );
    push_field(
        &mut out,
        "partial_interval_ms",
        config.voice.partial_interval_ms.to_string(),
        ValueStyle::Value,
    );
    push_field(
        &mut out,
        "rewrite_interval_ms",
        config.voice.rewrite_interval_ms.to_string(),
        ValueStyle::Value,
    );
    push_field(
        &mut out,
        "context_window_ms",
        config.voice.context_window_ms.to_string(),
        ValueStyle::Value,
    );
    push_field(
        &mut out,
        "min_chunk_ms",
        config.voice.min_chunk_ms.to_string(),
        ValueStyle::Value,
    );
    push_field(
        &mut out,
        "freeze_on_focus_change",
        yes_no(config.voice.freeze_on_focus_change),
        ValueStyle::Boolean,
    );

    out
}

fn push_section(out: &mut String, name: &str) {
    let _ = writeln!(out, "\n{}", crate::ui::section(name));
}

fn push_field(out: &mut String, label: &str, value: impl AsRef<str>, style: ValueStyle) {
    let _ = writeln!(
        out,
        "  {}: {}",
        crate::ui::summary_key(label),
        style_value(style, value.as_ref())
    );
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn push_path_field(out: &mut String, label: &str, path: &Path) {
    push_field(out, label, path.display().to_string(), ValueStyle::Path);
}

fn path_presence_status(path: Option<&Path>) -> &'static str {
    match path {
        Some(path) if path.exists() => "present",
        Some(_) => "missing",
        None => "not configured",
    }
}

fn optional_path_display(path: Option<&Path>) -> String {
    path.map(|path| path.display().to_string())
        .unwrap_or_else(|| "(not configured)".into())
}

#[derive(Clone, Copy)]
enum ValueStyle {
    Value,
    Boolean,
    Status,
    Backend,
    Provider,
    Path,
}

fn style_value(style_kind: ValueStyle, value: &str) -> String {
    match style_kind {
        ValueStyle::Value => crate::ui::value(value),
        ValueStyle::Boolean => bool_token(value),
        ValueStyle::Status => status_value_token(value),
        ValueStyle::Backend => crate::ui::backend_token(value),
        ValueStyle::Provider => crate::ui::provider_token(value),
        ValueStyle::Path => crate::ui::subtle(value),
    }
}

fn bool_token(value: &str) -> String {
    match value.trim() {
        "yes" => style(value).bold().green().to_string(),
        "no" => style(value).bold().red().to_string(),
        _ => crate::ui::value(value),
    }
}

fn status_value_token(value: &str) -> String {
    match status_category(value) {
        StatusCategory::Good => style(value).bold().green().to_string(),
        StatusCategory::Warn => style(value).bold().yellow().to_string(),
        StatusCategory::Bad => style(value).bold().red().to_string(),
        StatusCategory::Info => crate::ui::provider_token(value),
        StatusCategory::Neutral => crate::ui::value(value),
        StatusCategory::Subtle => crate::ui::subtle(value),
    }
}

fn status_category(value: &str) -> StatusCategory {
    match value.trim() {
        "ready" | "present" | "set" | "config file" | "config value set" => StatusCategory::Good,
        "cloud" => StatusCategory::Info,
        "missing" | "env:OPENAI_API_KEY (missing)" => StatusCategory::Bad,
        value if value.contains("(missing)") => StatusCategory::Warn,
        "not configured" | "defaults (config file missing)" => StatusCategory::Warn,
        _ if value.starts_with("env:") && value.ends_with("(set)") => StatusCategory::Good,
        _ if value.starts_with("https://") || value.starts_with("http://") => {
            StatusCategory::Subtle
        }
        _ => StatusCategory::Neutral,
    }
}

#[derive(Clone, Copy)]
enum StatusCategory {
    Good,
    Warn,
    Bad,
    Info,
    Neutral,
    Subtle,
}

fn transcription_model_status(config: &Config) -> &'static str {
    let model_path = config.resolved_model_path();
    match config.transcription.resolved_local_backend() {
        TranscriptionBackend::WhisperCpp => {
            if model_path.exists() {
                "ready"
            } else {
                "missing"
            }
        }
        TranscriptionBackend::FasterWhisper => {
            if crate::faster_whisper::model_dir_is_ready(&model_path) {
                "ready"
            } else {
                "missing"
            }
        }
        TranscriptionBackend::Nemo => {
            if crate::nemo_asr::model_dir_is_ready(&model_path) {
                "ready"
            } else {
                "missing"
            }
        }
        TranscriptionBackend::Cloud => "cloud",
    }
}

fn resolve_rewrite_model_path(config: &Config) -> Option<PathBuf> {
    if let Some(path) = config.resolved_rewrite_model_path() {
        return Some(path);
    }
    crate::rewrite_model::selected_model_path(&config.rewrite.selected_model)
}

fn rewrite_model_status(config: &Config) -> &'static str {
    let Some(model_path) = resolve_rewrite_model_path(config) else {
        return "not configured";
    };
    if model_path.exists() {
        "ready"
    } else {
        "missing"
    }
}

fn cloud_base_url_display(config: &Config) -> String {
    let configured = config.cloud.base_url.trim();
    if !configured.is_empty() {
        return configured.to_string();
    }

    match config.cloud.provider {
        CloudProvider::OpenAi => "https://api.openai.com/v1 (default)".into(),
        CloudProvider::OpenAiCompatible => "(missing for openai_compatible)".into(),
    }
}

fn cloud_api_key_status(config: &Config) -> String {
    if !config.cloud.api_key.trim().is_empty() {
        return "config value set".into();
    }

    let env_name = config.cloud.api_key_env.trim();
    if env_name.is_empty() {
        return "missing".into();
    }

    let env_status = if std::env::var_os(env_name).is_some() {
        "set"
    } else {
        "missing"
    };
    format!("env:{env_name} ({env_status})")
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config::{PostprocessMode, RewriteBackend, TranscriptionBackend};

    #[test]
    fn render_status_reports_selected_runtime_settings() {
        let _env_lock = crate::test_support::env_lock();
        let temp = crate::test_support::unique_temp_dir("status-render");
        let config_path = temp.join("config.toml");

        let asr_model = temp.join("ggml-large-v3-turbo.bin");
        std::fs::write(&asr_model, "bin").unwrap();

        let rewrite_model = temp.join("rewrite.gguf");
        std::fs::write(&rewrite_model, "gguf").unwrap();

        let instructions_path = temp.join("rewrite-instructions.txt");
        std::fs::write(&instructions_path, "Keep terms exact.").unwrap();

        let policy_path = temp.join("app-rewrite-policy.toml");
        std::fs::write(&policy_path, "").unwrap();

        let glossary_path = temp.join("technical-glossary.toml");
        std::fs::write(&glossary_path, "").unwrap();

        let dictionary_path = temp.join("dictionary.toml");
        std::fs::write(&dictionary_path, "").unwrap();

        let snippets_path = temp.join("snippets.toml");
        std::fs::write(&snippets_path, "").unwrap();

        crate::test_support::set_env("WHISPERS_STATUS_TEST_KEY", "secret");

        let mut config = Config::default();
        config.transcription.backend = TranscriptionBackend::WhisperCpp;
        config.transcription.local_backend = TranscriptionBackend::WhisperCpp;
        config.transcription.model_path = asr_model.display().to_string();
        config.transcription.selected_model = "large-v3-turbo".into();
        config.postprocess.mode = PostprocessMode::AgenticRewrite;
        config.rewrite.backend = RewriteBackend::Local;
        config.rewrite.model_path = rewrite_model.display().to_string();
        config.rewrite.instructions_path = instructions_path.display().to_string();
        config.agentic_rewrite.policy_path = policy_path.display().to_string();
        config.agentic_rewrite.glossary_path = glossary_path.display().to_string();
        config.personalization.dictionary_path = dictionary_path.display().to_string();
        config.personalization.snippets_path = snippets_path.display().to_string();
        config.cloud.api_key = String::new();
        config.cloud.api_key_env = "WHISPERS_STATUS_TEST_KEY".into();
        config.voice.live_inject = true;
        config.voice.live_rewrite = true;

        let rendered = render_status(&config_path, true, &config);
        assert!(rendered.contains("Whispers status"));
        assert!(rendered.contains(&crate::ui::section("Config")));
        assert!(rendered.contains(&format!(
            "{}: {}",
            crate::ui::summary_key("source"),
            style_value(ValueStyle::Status, "config file")
        )));
        assert!(rendered.contains(&format!(
            "{}: {}",
            crate::ui::summary_key("backend"),
            style_value(ValueStyle::Backend, "whisper_cpp")
        )));
        assert!(rendered.contains(&format!(
            "{}: {}",
            crate::ui::summary_key("model_status"),
            style_value(ValueStyle::Status, "ready")
        )));
        assert!(rendered.contains(&format!(
            "{}: {}",
            crate::ui::summary_key("mode"),
            style_value(ValueStyle::Value, "agentic_rewrite")
        )));
        assert!(rendered.contains(&format!(
            "{}: {}",
            crate::ui::summary_key("default_correction_policy"),
            style_value(ValueStyle::Value, "balanced")
        )));
        assert!(rendered.contains(&format!(
            "{}: {}",
            crate::ui::summary_key("api_key"),
            style_value(ValueStyle::Status, "env:WHISPERS_STATUS_TEST_KEY (set)")
        )));
        assert!(rendered.contains(&format!(
            "{}: {}",
            crate::ui::summary_key("live_inject"),
            style_value(ValueStyle::Boolean, "yes")
        )));
        assert!(rendered.contains(&format!(
            "{}: {}",
            crate::ui::summary_key("live_rewrite"),
            style_value(ValueStyle::Boolean, "yes")
        )));
    }

    #[test]
    fn render_status_marks_missing_optional_files() {
        let config = Config::default();
        let rendered = render_status(Path::new("/tmp/whispers-status.toml"), false, &config);
        assert!(rendered.contains(&format!(
            "{}: {}",
            crate::ui::summary_key("source"),
            style_value(ValueStyle::Status, "defaults (config file missing)")
        )));
        assert!(rendered.contains(&format!(
            "{}: {}",
            crate::ui::summary_key("policy_status"),
            style_value(ValueStyle::Status, "missing")
        )));
        assert!(rendered.contains(&format!(
            "{}: {}",
            crate::ui::summary_key("glossary_status"),
            style_value(ValueStyle::Status, "missing")
        )));
    }
}

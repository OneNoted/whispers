use std::path::Path;

use crate::error::{Result, WhsprError};

use super::{Config, PostprocessMode, TranscriptionBackend, resolve_config_path};

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
        config.apply_legacy_agentic_rewrite_migration(&contents, &config_path);
        config.apply_cloud_sanitization();
        Ok(config)
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

    fn apply_legacy_agentic_rewrite_migration(&mut self, contents: &str, config_path: &Path) {
        let legacy_present = section_present(contents, "agentic_rewrite");
        if !legacy_present {
            return;
        }

        let rewrite_has_policy = table_key_present(contents, "rewrite", "policy_path");
        let rewrite_has_glossary = table_key_present(contents, "rewrite", "glossary_path");
        let rewrite_has_default_policy =
            table_key_present(contents, "rewrite", "default_correction_policy");

        if !rewrite_has_policy {
            self.rewrite.policy_path = self.legacy_agentic_rewrite.policy_path.clone();
        }
        if !rewrite_has_glossary {
            self.rewrite.glossary_path = self.legacy_agentic_rewrite.glossary_path.clone();
        }
        if !rewrite_has_default_policy {
            self.rewrite.default_correction_policy =
                self.legacy_agentic_rewrite.default_correction_policy;
        }

        if rewrite_has_policy || rewrite_has_glossary || rewrite_has_default_policy {
            tracing::warn!(
                "config {} contains deprecated [agentic_rewrite]; [rewrite] takes precedence",
                config_path.display()
            );
        } else {
            tracing::warn!(
                "config {} uses deprecated [agentic_rewrite]; mapping fields to [rewrite]",
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

fn cleanup_section_present(contents: &str) -> bool {
    section_present(contents, "cleanup")
}

fn section_present(contents: &str, name: &str) -> bool {
    toml::from_str::<toml::Value>(contents)
        .ok()
        .and_then(|value| value.get(name).cloned())
        .is_some()
}

fn table_key_present(contents: &str, table: &str, key: &str) -> bool {
    toml::from_str::<toml::Value>(contents)
        .ok()
        .and_then(|value| value.get(table).cloned())
        .and_then(|value| value.get(key).cloned())
        .is_some()
}

fn looks_like_cloud_api_key(value: &str) -> bool {
    let trimmed = value.trim();
    trimmed.starts_with("sk-")
}

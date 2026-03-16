use std::path::Path;

use crate::asr_model;
use crate::config::{
    self, CloudLanguageMode, CloudSettingsUpdate, PostprocessMode, RewriteBackend, RewriteFallback,
    TranscriptionBackend, TranscriptionFallback,
};
use crate::error::Result;
use crate::ui::SetupUi;

use super::{CloudSetup, SetupSelections};

pub(super) fn apply_setup_config(
    ui: &SetupUi,
    config_path: &Path,
    config_path_override: Option<&Path>,
    selections: &SetupSelections,
) -> Result<()> {
    ensure_config_exists(ui, config_path)?;
    asr_model::select_model(selections.asr_model.name, config_path_override)?;

    if let Some(rewrite_model) = selections.rewrite_model {
        config::update_config_rewrite_selection(config_path, rewrite_model)?;
    }

    apply_postprocess_selection(ui, config_path, selections)?;
    apply_runtime_backend_selection(config_path, selections.asr_model.backend, &selections.cloud)?;
    apply_cloud_settings(ui, config_path, &selections.cloud)?;
    Ok(())
}

fn ensure_config_exists(ui: &SetupUi, config_path: &Path) -> Result<()> {
    let default_model_path = crate::model::model_path_for_config("ggml-large-v3-turbo.bin");
    if !config_path.exists() {
        tracing::info!("writing new config at {}", config_path.display());
        config::write_default_config(config_path, &default_model_path)?;
        return Ok(());
    }

    tracing::info!("updating existing config at {}", config_path.display());
    ui.print_info("Updating existing config.");
    Ok(())
}

fn apply_postprocess_selection(
    ui: &SetupUi,
    config_path: &Path,
    selections: &SetupSelections,
) -> Result<()> {
    if selections.postprocess_mode == PostprocessMode::Raw {
        config::update_config_postprocess_mode(config_path, PostprocessMode::Raw)?;
        ui.print_info("Rewrite cleanup: disabled (raw mode).");
    } else {
        config::update_config_postprocess_mode(config_path, selections.postprocess_mode)?;
    }
    Ok(())
}

fn apply_cloud_settings(ui: &SetupUi, config_path: &Path, cloud: &CloudSetup) -> Result<()> {
    if !cloud.enabled() {
        return Ok(());
    }

    config::update_config_cloud_settings(
        config_path,
        &CloudSettingsUpdate {
            provider: cloud.provider,
            base_url: &cloud.base_url,
            api_key: &cloud.api_key,
            api_key_env: &cloud.api_key_env,
            connect_timeout_ms: 3000,
            request_timeout_ms: 15000,
            transcription_model: "gpt-4o-mini-transcribe",
            transcription_language_mode: CloudLanguageMode::InheritLocal,
            transcription_language: "",
            rewrite_model: "gpt-4.1-mini",
            rewrite_temperature: 0.1,
            rewrite_max_output_tokens: 256,
        },
    )?;

    if cloud.api_key.is_empty() {
        ui.print_ok(format!(
            "Cloud provider: {} (using API key env {}).",
            crate::ui::provider_token(cloud.provider.as_str()),
            crate::ui::value(&cloud.api_key_env)
        ));
    } else {
        ui.print_ok(format!(
            "Cloud provider: {} (using a locally stored API key).",
            crate::ui::provider_token(cloud.provider.as_str())
        ));
    }

    if cloud.api_key.is_empty() && std::env::var(&cloud.api_key_env).is_err() {
        ui.print_warn(format!(
            "{} is not set in the current environment yet.",
            crate::ui::value(&cloud.api_key_env)
        ));
    }

    Ok(())
}

pub(super) fn apply_runtime_backend_selection(
    config_path: &Path,
    selected_asr_backend: TranscriptionBackend,
    cloud: &CloudSetup,
) -> Result<()> {
    let transcription_backend = if cloud.asr_enabled {
        TranscriptionBackend::Cloud
    } else {
        selected_asr_backend
    };
    let transcription_fallback = if cloud.asr_enabled {
        cloud.asr_fallback
    } else {
        TranscriptionFallback::ConfiguredLocal
    };
    config::update_config_transcription_runtime(
        config_path,
        transcription_backend,
        transcription_fallback,
    )?;

    let rewrite_backend = if cloud.rewrite_enabled {
        RewriteBackend::Cloud
    } else {
        RewriteBackend::Local
    };
    let rewrite_fallback = if cloud.rewrite_enabled {
        if crate::rewrite::local_rewrite_available() {
            cloud.rewrite_fallback
        } else {
            RewriteFallback::None
        }
    } else {
        RewriteFallback::Local
    };
    config::update_config_rewrite_runtime(config_path, rewrite_backend, rewrite_fallback)?;
    Ok(())
}

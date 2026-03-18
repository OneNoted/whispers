use std::path::Path;

use crate::config::{self, TranscriptionBackend};
use crate::error::Result;
use crate::ui::SetupUi;

use super::SetupSelections;

pub(super) async fn download_asr_model(
    ui: &SetupUi,
    asr_model: &'static crate::asr_model::AsrModelInfo,
) -> Result<()> {
    ui.blank();
    tracing::info!("setup selected ASR model: {}", asr_model.name);
    crate::asr_model::download_model(asr_model.name).await?;
    ui.blank();
    Ok(())
}

pub(super) async fn download_rewrite_model(
    ui: &SetupUi,
    rewrite_model: &'static str,
) -> Result<()> {
    ui.blank();
    tracing::info!("setup selected rewrite model: {}", rewrite_model);
    crate::rewrite_model::download_model(rewrite_model).await?;
    ui.blank();
    Ok(())
}

pub(super) fn maybe_create_agentic_starter_files(
    ui: &SetupUi,
    config_path: &Path,
    selections: &SetupSelections,
) -> Result<()> {
    if !selections.postprocess_mode.uses_rewrite() {
        return Ok(());
    }

    let config = config::Config::load(Some(config_path))?;
    let created = crate::agentic_rewrite::ensure_starter_files(&config)?;
    for path in created {
        ui.print_info(format!("Created rewrite starter file: {}", path));
    }
    Ok(())
}

pub(super) fn cleanup_stale_asr_workers(ui: &SetupUi, config_path: &Path) -> Result<()> {
    match config::Config::load(Some(config_path))
        .and_then(|config| crate::asr::cleanup::cleanup_stale_transcribers(&config))
    {
        Ok(()) => Ok(()),
        Err(err) => {
            ui.print_warn(format!(
                "Failed to retire stale ASR workers after setup: {err}"
            ));
            Ok(())
        }
    }
}

pub(super) fn maybe_prewarm_experimental_nemo(
    ui: &SetupUi,
    config_path: &Path,
    selections: &SetupSelections,
) -> Result<()> {
    if selections.asr_model.backend != TranscriptionBackend::Nemo || selections.cloud.asr_enabled {
        return Ok(());
    }

    let spinner =
        crate::ui::spinner("Starting background warm-up for the experimental NeMo backend...");
    match config::Config::load(Some(config_path)).and_then(|config| asr_model_prewarm(&config)) {
        Ok(()) => {
            spinner.finish_and_clear();
            ui.print_info("Background warm-up started for the experimental NeMo backend.");
        }
        Err(err) => {
            spinner.finish_and_clear();
            ui.print_warn(format!(
                "Failed to prewarm NeMo ASR backend after setup: {err}"
            ));
        }
    }

    Ok(())
}

fn asr_model_prewarm(config: &config::Config) -> Result<()> {
    let prepared = crate::asr::prepare::prepare_transcriber(config)?;
    crate::asr::prepare::prewarm_transcriber(&prepared, "setup");
    Ok(())
}

mod apply;
mod report;
mod select;
mod side_effects;

#[cfg(test)]
mod tests;

use std::path::Path;

use crate::asr_model::{self, ASR_MODELS, AsrModelInfo};
use crate::config::{
    CloudProvider, PostprocessMode, RewriteFallback, TranscriptionFallback, resolve_config_path,
};
use crate::error::Result;
use crate::ui::SetupUi;

struct SetupSelections {
    asr_model: &'static AsrModelInfo,
    rewrite_model: Option<&'static str>,
    postprocess_mode: PostprocessMode,
    cloud: CloudSetup,
}

struct CloudSetup {
    provider: CloudProvider,
    base_url: String,
    api_key: String,
    api_key_env: String,
    asr_enabled: bool,
    asr_fallback: TranscriptionFallback,
    rewrite_enabled: bool,
    rewrite_fallback: RewriteFallback,
}

impl Default for CloudSetup {
    fn default() -> Self {
        Self {
            provider: CloudProvider::OpenAi,
            base_url: String::new(),
            api_key: String::new(),
            api_key_env: "OPENAI_API_KEY".to_string(),
            asr_enabled: false,
            asr_fallback: TranscriptionFallback::ConfiguredLocal,
            rewrite_enabled: false,
            rewrite_fallback: RewriteFallback::Local,
        }
    }
}

impl CloudSetup {
    fn enabled(&self) -> bool {
        self.asr_enabled || self.rewrite_enabled
    }
}

pub async fn run_setup(config_path_override: Option<&Path>) -> Result<()> {
    let ui = SetupUi::new();
    ui.print_header("whispers setup");
    ui.blank();
    report::print_setup_intro(&ui);

    let available_asr_models: Vec<_> = ASR_MODELS
        .iter()
        .filter(|model| asr_model::is_model_available(model.name))
        .collect();

    let asr_model = select::choose_asr_model(&ui, &available_asr_models)?;
    side_effects::download_asr_model(&ui, asr_model).await?;

    let mut rewrite_model = None;
    let mut postprocess_mode = PostprocessMode::Raw;
    if crate::rewrite::local_rewrite_available()
        && ui.confirm("Enable smarter local rewrite cleanup?", false)?
    {
        let selected_rewrite_model =
            select::choose_rewrite_model(&ui, "Choose a local rewrite model", 1)?;
        side_effects::download_rewrite_model(&ui, selected_rewrite_model).await?;
        rewrite_model = Some(selected_rewrite_model);
        postprocess_mode = select::choose_rewrite_mode(&ui)?;
    } else if !crate::rewrite::local_rewrite_available() {
        ui.print_info(
            "This build does not include local rewrite support. You can still enable cloud rewrite.",
        );
        ui.blank();
    }

    let rewrite_model_before_cloud = rewrite_model;
    let cloud = if ui.confirm("Add optional cloud ASR or rewrite?", false)? {
        select::configure_cloud(&ui, &mut rewrite_model)?
    } else {
        CloudSetup::default()
    };
    if rewrite_model_before_cloud.is_none() {
        if let Some(selected_rewrite_model) = rewrite_model {
            side_effects::download_rewrite_model(&ui, selected_rewrite_model).await?;
        }
    }
    if cloud.rewrite_enabled && postprocess_mode == PostprocessMode::Raw {
        postprocess_mode = select::choose_rewrite_mode(&ui)?;
    }

    let selections = SetupSelections {
        asr_model,
        rewrite_model,
        postprocess_mode,
        cloud,
    };

    let config_path = resolve_config_path(config_path_override);
    apply::apply_setup_config(&ui, &config_path, config_path_override, &selections)?;
    side_effects::maybe_create_agentic_starter_files(&ui, &config_path, &selections)?;
    side_effects::cleanup_stale_asr_workers(&ui, &config_path)?;

    if let Some(rewrite_model) = selections.rewrite_model {
        ui.print_ok(format!(
            "Local rewrite model: {}",
            crate::ui::value(rewrite_model)
        ));
    }

    side_effects::maybe_prewarm_experimental_nemo(&ui, &config_path, &selections)?;

    ui.print_ok("Config saved.");
    ui.blank();
    report::print_setup_summary(&ui, &selections);
    ui.blank();
    report::print_setup_complete(&ui);

    Ok(())
}

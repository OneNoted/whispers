use crate::asr_model::{self, AsrModelInfo};
use crate::config::{CloudProvider, PostprocessMode, RewriteFallback, TranscriptionFallback};
use crate::error::Result;
use crate::rewrite_model::{self, REWRITE_MODELS};
use crate::ui::SetupUi;

use super::CloudSetup;

pub(super) fn choose_asr_model(
    ui: &SetupUi,
    available_asr_models: &[&'static AsrModelInfo],
) -> Result<&'static AsrModelInfo> {
    loop {
        let items: Vec<String> = available_asr_models
            .iter()
            .map(|model| asr_model::setup_label(model))
            .collect();
        let selection = ui.select("Choose an ASR model", &items, 0)?;
        let chosen = available_asr_models[selection];

        if let Some(warning) = asr_model::experimental_warning(chosen) {
            ui.blank();
            tracing::debug!(
                "experimental setup warning for {}: {}",
                chosen.name,
                warning
            );
            ui.print_experimental_notice(chosen.name, asr_model::experimental_notice_facts(chosen));
            if !ui.danger_confirm(
                crate::ui::danger_text("Continue with this experimental ASR backend?"),
                false,
            )? {
                ui.blank();
                continue;
            }
        }

        return Ok(chosen);
    }
}

pub(super) fn choose_rewrite_model(
    ui: &SetupUi,
    prompt: &str,
    default_index: usize,
) -> Result<&'static str> {
    let items: Vec<String> = REWRITE_MODELS
        .iter()
        .map(rewrite_model::setup_label)
        .collect();
    let selection = ui.select(prompt, &items, default_index)?;
    let chosen = &REWRITE_MODELS[selection];
    Ok(chosen.name)
}

pub(super) fn choose_rewrite_mode(ui: &SetupUi) -> Result<PostprocessMode> {
    let items = ["rewrite: unified rewrite with app-aware policy and glossary support"];
    let _selection = ui.select("Choose the rewrite mode", &items, 0)?;
    Ok(PostprocessMode::Rewrite)
}

pub(super) fn configure_cloud(
    ui: &SetupUi,
    rewrite_model: &mut Option<&'static str>,
) -> Result<CloudSetup> {
    let mut cloud = CloudSetup::default();

    let provider_items = vec![
        format!(
            "{}  {}",
            crate::ui::provider_token("OpenAI"),
            crate::ui::description_token("Official OpenAI hosted API")
        ),
        format!(
            "{}  {}",
            crate::ui::provider_token("OpenAI-compatible endpoint"),
            crate::ui::description_token("Third-party endpoint that speaks the OpenAI API")
        ),
    ];
    let provider_selection = ui.select("Choose a cloud provider", &provider_items, 0)?;
    if provider_selection == 1 {
        cloud.provider = CloudProvider::OpenAiCompatible;
        cloud.base_url = ui.input_string("Base URL for the OpenAI-compatible API", None)?;
    }

    let cloud_key_input = ui.input_string(
        "Cloud API key or environment variable name",
        Some("OPENAI_API_KEY"),
    )?;
    if looks_like_cloud_api_key(&cloud_key_input) {
        cloud.api_key = cloud_key_input.trim().to_string();
        cloud.api_key_env = "OPENAI_API_KEY".into();
    } else {
        cloud.api_key_env = cloud_key_input.trim().to_string();
    }

    let cloud_mode_items = [
        "Cloud rewrite only",
        "Cloud ASR only",
        "Cloud ASR + rewrite",
    ];
    let cloud_mode = ui.select("Choose the cloud mode", &cloud_mode_items, 0)?;
    cloud.rewrite_enabled = matches!(cloud_mode, 0 | 2);
    cloud.asr_enabled = matches!(cloud_mode, 1 | 2);

    if cloud.asr_enabled {
        let fallback_items = ["Use configured local fallback", "Fail if cloud ASR fails"];
        let selection = ui.select("If cloud ASR fails", &fallback_items, 0)?;
        cloud.asr_fallback = if selection == 0 {
            TranscriptionFallback::ConfiguredLocal
        } else {
            TranscriptionFallback::None
        };
    }

    if cloud.rewrite_enabled {
        if crate::rewrite::local_rewrite_available() {
            let fallback_items = [
                "Use local rewrite fallback",
                "Fail back to deterministic text",
            ];
            let selection = ui.select("If cloud rewrite fails", &fallback_items, 0)?;
            cloud.rewrite_fallback = if selection == 0 {
                RewriteFallback::Local
            } else {
                RewriteFallback::None
            };
        } else {
            cloud.rewrite_fallback = RewriteFallback::None;
            ui.print_info(
                "Local rewrite fallback is unavailable in this build; cloud rewrite will fall back to deterministic text.",
            );
        }
    }

    if cloud.rewrite_enabled
        && cloud.rewrite_fallback == RewriteFallback::Local
        && rewrite_model.is_none()
    {
        ui.blank();
        ui.print_info("Cloud rewrite fallback uses a local rewrite model.");
        *rewrite_model = Some(choose_rewrite_model(
            ui,
            "Choose a local rewrite fallback model to download",
            1,
        )?);
    }

    Ok(cloud)
}

fn looks_like_cloud_api_key(value: &str) -> bool {
    value.trim().starts_with("sk-")
}

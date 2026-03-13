use std::path::Path;

use crate::asr_model::{self, ASR_MODELS, AsrModelInfo};
use crate::config::{
    self, CloudLanguageMode, CloudProvider, CloudSettingsUpdate, PostprocessMode, RewriteBackend,
    RewriteFallback, TranscriptionBackend, TranscriptionFallback, resolve_config_path,
};
use crate::error::Result;
use crate::rewrite_model::{self, REWRITE_MODELS};
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
    print_setup_intro(&ui);

    let available_asr_models: Vec<_> = ASR_MODELS
        .iter()
        .filter(|model| asr_model::is_model_available(model.name))
        .collect();

    let asr_model = choose_asr_model(&ui, &available_asr_models)?;
    ui.blank();
    tracing::info!("setup selected ASR model: {}", asr_model.name);
    asr_model::download_model(asr_model.name).await?;
    ui.blank();

    let mut rewrite_model = None;
    let mut postprocess_mode = PostprocessMode::Raw;
    if ui.confirm("Enable smarter local rewrite cleanup?", false)? {
        rewrite_model = Some(choose_rewrite_model(&ui, "Choose a local rewrite model", 1).await?);
        postprocess_mode = choose_rewrite_mode(&ui)?;
    }

    let cloud = if ui.confirm("Add optional cloud ASR or rewrite?", false)? {
        configure_cloud(&ui, &mut rewrite_model).await?
    } else {
        CloudSetup::default()
    };
    if cloud.rewrite_enabled && postprocess_mode == PostprocessMode::Raw {
        postprocess_mode = choose_rewrite_mode(&ui)?;
    }

    let selections = SetupSelections {
        asr_model,
        rewrite_model,
        postprocess_mode,
        cloud,
    };

    let config_path = resolve_config_path(config_path_override);
    ensure_config_exists(&ui, &config_path)?;
    asr_model::select_model(selections.asr_model.name, config_path_override)?;

    if let Some(rewrite_model) = selections.rewrite_model {
        config::update_config_rewrite_selection(&config_path, rewrite_model)?;
    }

    apply_postprocess_selection(&ui, &config_path, &selections)?;
    apply_runtime_backend_selection(
        &config_path,
        selections.asr_model.backend,
        &selections.cloud,
    )?;
    apply_cloud_settings(&ui, &config_path, &selections.cloud)?;
    maybe_create_agentic_starter_files(&ui, &config_path, &selections)?;
    cleanup_stale_asr_workers(&ui, &config_path)?;

    if let Some(rewrite_model) = selections.rewrite_model {
        ui.print_ok(format!(
            "Local rewrite model: {}",
            crate::ui::value(rewrite_model)
        ));
    }

    maybe_prewarm_experimental_nemo(&ui, &config_path, &selections)?;

    ui.print_ok("Config saved.");
    ui.blank();
    print_setup_summary(&ui, &selections);
    ui.blank();
    print_setup_complete(&ui);

    Ok(())
}

fn print_setup_intro(ui: &SetupUi) {
    ui.print_subtle(
        "Recommended models are listed first. Experimental backends are available, but not the default recommendation.",
    );
    ui.blank();
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

fn maybe_prewarm_experimental_nemo(
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

fn cleanup_stale_asr_workers(ui: &SetupUi, config_path: &Path) -> Result<()> {
    match config::Config::load(Some(config_path))
        .and_then(|config| crate::asr::cleanup_stale_transcribers(&config))
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

fn asr_model_prewarm(config: &config::Config) -> Result<()> {
    let prepared = crate::asr::prepare_transcriber(config)?;
    crate::asr::prewarm_transcriber(&prepared, "setup");
    Ok(())
}

fn choose_asr_model(
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

async fn choose_rewrite_model(
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
    ui.blank();
    tracing::info!("setup selected rewrite model: {}", chosen.name);
    rewrite_model::download_model(chosen.name).await?;
    ui.blank();
    Ok(chosen.name)
}

fn choose_rewrite_mode(ui: &SetupUi) -> Result<PostprocessMode> {
    let items = [
        "advanced_local: smart rewrite cleanup with current bounded-candidate behavior",
        "agentic_rewrite: app-aware rewrite with policy and technical glossary support",
    ];
    let selection = ui.select("Choose the rewrite mode", &items, 1)?;
    Ok(if selection == 0 {
        PostprocessMode::AdvancedLocal
    } else {
        PostprocessMode::AgenticRewrite
    })
}

async fn configure_cloud(
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
    }

    if cloud.rewrite_enabled
        && cloud.rewrite_fallback == RewriteFallback::Local
        && rewrite_model.is_none()
    {
        ui.blank();
        ui.print_info("Cloud rewrite fallback uses a local rewrite model.");
        *rewrite_model = Some(
            choose_rewrite_model(ui, "Choose a local rewrite fallback model to download", 1)
                .await?,
        );
    }

    Ok(cloud)
}

fn looks_like_cloud_api_key(value: &str) -> bool {
    value.trim().starts_with("sk-")
}

fn print_setup_summary(ui: &SetupUi, selections: &SetupSelections) {
    ui.print_section("Setup summary");
    println!(
        "  {}: {} ({}, {}, {})",
        crate::ui::summary_key("ASR"),
        crate::ui::value(selections.asr_model.name),
        crate::ui::backend_token(selections.asr_model.backend.as_str()),
        crate::ui::scope_token(selections.asr_model.language_scope.as_str()),
        crate::ui::tier_token(selections.asr_model.support_tier.as_str())
    );

    if let Some(note) = selections.asr_model.setup_note {
        println!("  {}: {}", crate::ui::summary_key("ASR note"), note);
    }

    if selections.cloud.asr_enabled {
        println!(
            "  {}: enabled via {} (fallback: {})",
            crate::ui::summary_key("Cloud ASR"),
            crate::ui::provider_token(selections.cloud.provider.as_str()),
            selections.cloud.asr_fallback.as_str()
        );
    } else {
        println!("  {}: disabled", crate::ui::summary_key("Cloud ASR"));
    }

    match (selections.cloud.rewrite_enabled, selections.rewrite_model) {
        (true, Some(model)) => println!(
            "  {}: cloud with local fallback ({}, mode: {})",
            crate::ui::summary_key("Rewrite"),
            crate::ui::value(model),
            selections.postprocess_mode.as_str()
        ),
        (true, None) => println!(
            "  {}: cloud only (fallback: {}, mode: {})",
            crate::ui::summary_key("Rewrite"),
            selections.cloud.rewrite_fallback.as_str(),
            selections.postprocess_mode.as_str()
        ),
        (false, Some(model)) => println!(
            "  {}: local ({}, mode: {})",
            crate::ui::summary_key("Rewrite"),
            crate::ui::value(model),
            selections.postprocess_mode.as_str()
        ),
        (false, None) => println!(
            "  {}: disabled (raw mode)",
            crate::ui::summary_key("Rewrite")
        ),
    }

    if selections.asr_model.backend == TranscriptionBackend::Nemo && !selections.cloud.asr_enabled {
        println!(
            "  {}: first use may be slower than steady-state while the worker warms.",
            crate::ui::summary_key("NeMo note")
        );
    }
}

fn print_setup_complete(ui: &SetupUi) {
    ui.print_header("Setup complete");
    println!("You can now use whispers.");
    ui.print_section("Example keybind");
    ui.print_subtle("Bind it to a key in your compositor, e.g. for Hyprland:");
    println!("  bind = SUPER ALT, D, exec, whispers");
}

fn apply_runtime_backend_selection(
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
        cloud.rewrite_fallback
    } else {
        RewriteFallback::Local
    };
    config::update_config_rewrite_runtime(config_path, rewrite_backend, rewrite_fallback)?;
    Ok(())
}

fn maybe_create_agentic_starter_files(
    ui: &SetupUi,
    config_path: &Path,
    selections: &SetupSelections,
) -> Result<()> {
    if selections.postprocess_mode != PostprocessMode::AgenticRewrite {
        return Ok(());
    }

    let config = config::Config::load(Some(config_path))?;
    let created = crate::agentic_rewrite::ensure_starter_files(&config)?;
    for path in created {
        ui.print_info(format!("Created agentic rewrite starter file: {}", path));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn runtime_selection_resets_cloud_asr_when_disabled() {
        let config_path = crate::test_support::unique_temp_path("setup-runtime-reset", "toml");
        config::write_default_config(&config_path, "~/model.bin").expect("write config");
        config::update_config_transcription_runtime(
            &config_path,
            TranscriptionBackend::Cloud,
            TranscriptionFallback::None,
        )
        .expect("set cloud runtime");

        let cloud = CloudSetup::default();
        apply_runtime_backend_selection(&config_path, TranscriptionBackend::WhisperCpp, &cloud)
            .expect("reset runtime");

        let config = Config::load(Some(&config_path)).expect("load config");
        assert_eq!(
            config.transcription.backend,
            TranscriptionBackend::WhisperCpp
        );
        assert_eq!(
            config.transcription.fallback,
            TranscriptionFallback::ConfiguredLocal
        );
    }
}

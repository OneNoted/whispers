use std::path::Path;

use crate::asr_model::{self, ASR_MODELS, AsrModelInfo};
use crate::cli::CompletionShell;
use crate::config::{
    self, CloudLanguageMode, CloudProvider, CloudSettingsUpdate, PostprocessMode, RewriteBackend,
    RewriteFallback, TranscriptionBackend, TranscriptionFallback, VoiceConfig, resolve_config_path,
};
use crate::error::Result;
use crate::rewrite_model::{self, REWRITE_MODELS};
use crate::ui::SetupUi;

struct SetupSelections {
    asr_model: &'static AsrModelInfo,
    rewrite_model: Option<&'static str>,
    postprocess_mode: PostprocessMode,
    cloud: CloudSetup,
    voice: VoiceSetup,
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct VoiceSetup {
    enabled: bool,
    live_inject: bool,
    live_rewrite: bool,
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
    let voice = configure_voice(&ui, postprocess_mode)?;
    let completion_shells = choose_completion_shells(&ui)?;

    let selections = SetupSelections {
        asr_model,
        rewrite_model,
        postprocess_mode,
        cloud,
        voice,
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
    apply_voice_selection(&ui, &config_path, &selections.voice)?;
    maybe_create_agentic_starter_files(&ui, &config_path, &selections)?;
    cleanup_stale_asr_workers(&ui, &config_path)?;
    maybe_install_shell_completions(&ui, &completion_shells)?;

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
    print_setup_complete(&ui, &selections);

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

fn apply_voice_selection(ui: &SetupUi, config_path: &Path, voice: &VoiceSetup) -> Result<()> {
    let mut voice_config = VoiceConfig::default();
    voice_config.live_inject = voice.enabled && voice.live_inject;
    voice_config.live_rewrite = voice.enabled && voice.live_rewrite;
    config::update_config_voice_settings(config_path, &voice_config)?;

    if !voice.enabled {
        ui.print_info("Live voice mode: disabled.");
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

fn configure_voice(ui: &SetupUi, postprocess_mode: PostprocessMode) -> Result<VoiceSetup> {
    if !ui.confirm("Enable experimental live voice mode?", false)? {
        return Ok(VoiceSetup::default());
    }

    let items = [
        "Preview-only: live transcript OSD while recording, final insert on stop",
        "Live inject: update the target app while recording (freezes on focus change)",
    ];
    let selection = ui.select("Choose the live voice behavior", &items, 0)?;
    let live_inject = selection == 1;
    let live_rewrite = if postprocess_mode.uses_rewrite() {
        ui.confirm(
            "Show live rewrite preview in the OSD while recording?",
            false,
        )?
    } else {
        ui.print_info("Live rewrite preview needs a rewrite-enabled postprocess mode.");
        false
    };

    Ok(VoiceSetup {
        enabled: true,
        live_inject,
        live_rewrite,
    })
}

fn choose_completion_shells(ui: &SetupUi) -> Result<Vec<CompletionShell>> {
    let detected_shells = crate::completions::detect_installed_shells();
    let current_shell = crate::completions::detect_shell();

    if detected_shells.is_empty() {
        ui.print_info("Could not find any supported shells on PATH automatically.");
        if !ui.confirm("Install shell completions anyway?", false)? {
            return Ok(Vec::new());
        }
        return Ok(vec![choose_shell_manually(ui)?]);
    }

    let mut detected_names = detected_shells
        .iter()
        .map(|shell| shell.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    if let Some(shell) = current_shell {
        detected_names.push_str(&format!(" (current shell hint: {})", shell.as_str()));
    }
    ui.print_info(format!("Detected supported shells: {detected_names}."));

    if !ui.confirm("Install shell completions?", true)? {
        return Ok(Vec::new());
    }

    if detected_shells.len() == 1 {
        return Ok(detected_shells);
    }

    let mut items = vec!["all detected shells".to_string()];
    items.extend(
        detected_shells
            .iter()
            .map(|shell| shell_choice_label(*shell, current_shell)),
    );
    let default = current_shell
        .and_then(|shell| {
            detected_shells
                .iter()
                .position(|candidate| *candidate == shell)
        })
        .map_or(0, |index| index + 1);
    let selection = ui.select("Choose shells for completion install", &items, default)?;
    if selection == 0 {
        return Ok(detected_shells);
    }

    Ok(vec![detected_shells[selection - 1]])
}

fn choose_shell_manually(ui: &SetupUi) -> Result<CompletionShell> {
    let items = ["bash", "zsh", "fish", "nushell"];
    let selection = ui.select("Choose a shell for completions", &items, 0)?;
    Ok(match selection {
        0 => CompletionShell::Bash,
        1 => CompletionShell::Zsh,
        2 => CompletionShell::Fish,
        _ => CompletionShell::Nushell,
    })
}

fn shell_choice_label(shell: CompletionShell, current_shell: Option<CompletionShell>) -> String {
    if current_shell == Some(shell) {
        format!("{} (current shell)", shell.as_str())
    } else {
        shell.as_str().to_string()
    }
}

fn maybe_install_shell_completions(ui: &SetupUi, shells: &[CompletionShell]) -> Result<()> {
    if shells.is_empty() {
        return Ok(());
    }

    for shell in shells {
        match crate::completions::install_completions(*shell) {
            Ok(path) => {
                ui.print_ok(format!(
                    "Installed {} completions at {}.",
                    crate::ui::value(shell.as_str()),
                    crate::ui::value(path.display().to_string())
                ));
                if let Some(note) = crate::completions::install_note(*shell, &path) {
                    ui.print_info(note);
                }
            }
            Err(err) => {
                ui.print_warn(format!(
                    "Failed to install {} completions automatically: {err}",
                    shell.as_str()
                ));
                ui.print_info(format!(
                    "You can still run {} later.",
                    crate::ui::value(format!("whispers completions {}", shell.as_str()))
                ));
            }
        }
    }

    Ok(())
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

    match (
        selections.voice.enabled,
        selections.voice.live_inject,
        selections.voice.live_rewrite,
    ) {
        (false, _, _) => println!("  {}: disabled", crate::ui::summary_key("Voice")),
        (true, false, false) => println!(
            "  {}: preview-only via {}",
            crate::ui::summary_key("Voice"),
            crate::ui::value("whispers voice")
        ),
        (true, true, false) => println!(
            "  {}: live inject via {}",
            crate::ui::summary_key("Voice"),
            crate::ui::value("whispers voice")
        ),
        (true, false, true) => println!(
            "  {}: preview-only via {} with live rewrite preview",
            crate::ui::summary_key("Voice"),
            crate::ui::value("whispers voice")
        ),
        (true, true, true) => println!(
            "  {}: live inject via {} with live rewrite preview",
            crate::ui::summary_key("Voice"),
            crate::ui::value("whispers voice")
        ),
    }
}

fn print_setup_complete(ui: &SetupUi, selections: &SetupSelections) {
    ui.print_header("Setup complete");
    println!("You can now use whispers.");
    ui.print_section("Example keybind");
    ui.print_subtle("Bind it to a key in your compositor, e.g. for Hyprland:");
    println!("  bind = SUPER ALT, D, exec, whispers");
    if selections.voice.enabled {
        println!("  bind = SUPER ALT, V, exec, whispers voice");
        ui.print_subtle("Voice mode is separate so you can keep the existing dictation flow.");
    }
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

    #[test]
    fn apply_voice_selection_persists_voice_defaults_and_toggles() {
        let config_path = crate::test_support::unique_temp_path("setup-voice-selection", "toml");
        config::write_default_config(&config_path, "~/model.bin").expect("write config");

        let ui = SetupUi::new();
        let voice = VoiceSetup {
            enabled: true,
            live_inject: true,
            live_rewrite: true,
        };
        apply_voice_selection(&ui, &config_path, &voice).expect("apply voice selection");

        let config = Config::load(Some(&config_path)).expect("load config");
        assert!(config.voice.live_inject);
        assert!(config.voice.live_rewrite);
        assert_eq!(
            config.voice.partial_interval_ms,
            VoiceConfig::default().partial_interval_ms
        );
        assert_eq!(
            config.voice.freeze_on_focus_change,
            VoiceConfig::default().freeze_on_focus_change
        );
    }

    #[test]
    fn maybe_install_shell_completions_writes_fish_script() {
        let _env_lock = crate::test_support::env_lock();
        let _guard = crate::test_support::EnvVarGuard::capture(&[
            "HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
        ]);
        let root = crate::test_support::unique_temp_dir("setup-shell-completions");
        crate::test_support::set_env("HOME", &root.to_string_lossy());
        crate::test_support::remove_env("XDG_CONFIG_HOME");
        crate::test_support::remove_env("XDG_DATA_HOME");

        let ui = SetupUi::new();
        maybe_install_shell_completions(&ui, &[CompletionShell::Fish])
            .expect("install fish completions");

        let path = root.join(".config/fish/completions/whispers.fish");
        let contents = std::fs::read_to_string(path).expect("read fish completions");
        assert!(contents.contains("complete -c whispers"));
    }

    #[test]
    fn shell_choice_label_marks_current_shell() {
        assert_eq!(
            shell_choice_label(CompletionShell::Fish, Some(CompletionShell::Fish)),
            "fish (current shell)"
        );
        assert_eq!(
            shell_choice_label(CompletionShell::Bash, Some(CompletionShell::Fish)),
            "bash"
        );
    }
}

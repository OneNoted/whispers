use std::path::PathBuf;
use std::process::Command;

use std::path::Path;

use crate::config::{self, TranscriptionBackend};
use crate::error::Result;
use crate::ui::SetupUi;

use super::SetupSelections;

const MODULES_LOAD_PATH: &str = "/etc/modules-load.d/whispers-uinput.conf";
const UDEV_RULE_PATH: &str = "/etc/udev/rules.d/70-whispers-uinput.rules";
const UDEV_RULE_CONTENT: &str = "KERNEL==\"uinput\", SUBSYSTEM==\"misc\", GROUP=\"input\", MODE=\"0660\", OPTIONS+=\"static_node=uinput\"\n";
const MODULES_LOAD_CONTENT: &str = "uinput\n";

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct InjectionSetupOutcome {
    pub changed_groups: bool,
}

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

pub(super) fn maybe_setup_injection_access(ui: &SetupUi) -> Result<InjectionSetupOutcome> {
    let readiness = crate::inject::InjectionReadinessReport::collect();
    if readiness.is_ready() || !readiness.has_uinput_issue() {
        return Ok(InjectionSetupOutcome::default());
    }

    ui.blank();
    ui.print_section("System setup");
    ui.print_warn("`/dev/uinput` is not ready, so paste injection will fail.");
    for line in readiness.issue_lines() {
        println!("  - {line}");
    }

    if !ui.confirm("Set up `/dev/uinput` access now? This uses sudo.", true)? {
        return Ok(InjectionSetupOutcome::default());
    }

    let mut outcome = InjectionSetupOutcome::default();
    if let Err(err) = ensure_uinput_module_loaded(ui) {
        ui.print_warn(format!("Failed to load the `uinput` module: {err}"));
    }
    if let Err(err) = install_root_file(ui, MODULES_LOAD_PATH, MODULES_LOAD_CONTENT) {
        ui.print_warn(format!("Failed to persist the `uinput` module load: {err}"));
    }
    if let Err(err) = install_root_file(ui, UDEV_RULE_PATH, UDEV_RULE_CONTENT) {
        ui.print_warn(format!(
            "Failed to install the `/dev/uinput` udev rule: {err}"
        ));
    }
    if add_user_to_group(ui, "input")? {
        outcome.changed_groups = true;
    }
    if let Err(err) = reload_udev(ui) {
        ui.print_warn(format!(
            "Failed to reload `udev` after updating `/dev/uinput`: {err}"
        ));
    }

    if outcome.changed_groups {
        ui.print_info("Group membership changed. Log out and back in before testing dictation.");
    }

    Ok(outcome)
}

fn asr_model_prewarm(config: &config::Config) -> Result<()> {
    let prepared = crate::asr::prepare::prepare_transcriber(config)?;
    crate::asr::prepare::prewarm_transcriber(&prepared, "setup");
    Ok(())
}

fn ensure_uinput_module_loaded(ui: &SetupUi) -> Result<()> {
    ui.print_info("Loading the `uinput` kernel module...");
    run_sudo(&["modprobe", "uinput"])
}

fn reload_udev(ui: &SetupUi) -> Result<()> {
    ui.print_info("Reloading `udev` rules for `/dev/uinput`...");
    run_sudo(&["udevadm", "control", "--reload"])?;
    run_sudo(&[
        "udevadm",
        "trigger",
        "--subsystem-match=misc",
        "--sysname-match=uinput",
    ])
}

fn add_user_to_group(ui: &SetupUi, group: &str) -> Result<bool> {
    let username = current_username()?;
    if current_user_in_group(&username, group)? {
        ui.print_info(format!(
            "User is already configured for the `{group}` group."
        ));
        return Ok(false);
    }

    ui.print_info(format!("Adding `{username}` to the `{group}` group..."));
    run_sudo(&["usermod", "-aG", group, &username])?;
    Ok(true)
}

fn current_user_in_group(username: &str, group: &str) -> Result<bool> {
    let output = Command::new("id")
        .args(["-nG", username])
        .output()
        .map_err(|err| {
            crate::error::WhsprError::Config(format!("failed to inspect groups: {err}"))
        })?;
    if !output.status.success() {
        return Err(crate::error::WhsprError::Config(format!(
            "`id -nG {username}` exited with {}",
            output.status
        )));
    }
    let groups = String::from_utf8_lossy(&output.stdout);
    Ok(groups.split_whitespace().any(|entry| entry == group))
}

fn current_username() -> Result<String> {
    if let Some(name) = std::env::var_os("SUDO_USER").or_else(|| std::env::var_os("USER")) {
        let username = name.to_string_lossy().trim().to_string();
        if !username.is_empty() {
            return Ok(username);
        }
    }

    let output = Command::new("id").arg("-un").output().map_err(|err| {
        crate::error::WhsprError::Config(format!("failed to determine username: {err}"))
    })?;
    if !output.status.success() {
        return Err(crate::error::WhsprError::Config(format!(
            "`id -un` exited with {}",
            output.status
        )));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn install_root_file(ui: &SetupUi, target: &str, contents: &str) -> Result<()> {
    let temp_path = temp_file_path(target);
    std::fs::write(&temp_path, contents)?;
    let temp_path_str = temp_path.to_string_lossy().to_string();
    let result = run_sudo(&["install", "-Dm644", &temp_path_str, target]);
    let _ = std::fs::remove_file(&temp_path);
    result?;
    ui.print_info(format!("Installed `{target}`."));
    Ok(())
}

fn temp_file_path(target: &str) -> PathBuf {
    let basename = Path::new(target)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("whispers-temp");
    std::env::temp_dir().join(format!("whispers-{}-{basename}", std::process::id()))
}

fn run_sudo(args: &[&str]) -> Result<()> {
    let status = Command::new("sudo").args(args).status().map_err(|err| {
        crate::error::WhsprError::Config(format!("failed to run sudo {:?}: {err}", args))
    })?;
    if !status.success() {
        return Err(crate::error::WhsprError::Config(format!(
            "`sudo {}` exited with {status}",
            args.join(" ")
        )));
    }
    Ok(())
}

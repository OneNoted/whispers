use std::ffi::CStr;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use crate::config::{self, TranscriptionBackend};
use crate::error::Result;
use crate::ui::SetupUi;

use super::SetupSelections;

const UINPUT_GROUP: &str = "uinput";
const MODULES_LOAD_PATH: &str = "/etc/modules-load.d/whispers-uinput.conf";
const UDEV_RULE_PATH: &str = "/etc/udev/rules.d/70-whispers-uinput.rules";
const UDEV_RULE_CONTENT: &str = "KERNEL==\"uinput\", SUBSYSTEM==\"misc\", GROUP=\"uinput\", MODE=\"0660\", OPTIONS+=\"static_node=uinput\"\n";
const MODULES_LOAD_CONTENT: &str = "uinput\n";
pub(super) const UDEV_TRIGGER_ARGS: &[&str] = &[
    "udevadm",
    "trigger",
    "--subsystem-match=misc",
    "--sysname-match=uinput",
    "--settle",
];

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct InjectionSetupOutcome {
    pub changed_groups: bool,
    pub group_membership_ready: bool,
    pub udev_reload_succeeded: bool,
}

impl InjectionSetupOutcome {
    pub(super) fn setup_group_change_message(self) -> Option<&'static str> {
        if !self.changed_groups {
            None
        } else if self.udev_reload_succeeded {
            Some("Group membership changed. Log out and back in before testing dictation.")
        } else {
            Some(
                "Group membership changed. Log out and back in after finishing the remaining paste injection steps.",
            )
        }
    }

    pub(super) fn report_group_change_message(self) -> Option<&'static str> {
        if !self.changed_groups {
            None
        } else if self.udev_reload_succeeded {
            Some(
                "If you were just added to the `uinput` group, log out and back in before testing.",
            )
        } else {
            Some(
                "If you were just added to the `uinput` group, log out and back in after finishing the remaining paste injection steps.",
            )
        }
    }

    pub(super) fn can_finish_with_relogin_only(self, only_requires_relogin: bool) -> bool {
        self.group_membership_ready && self.udev_reload_succeeded && only_requires_relogin
    }
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
    if let Err(err) = ensure_group_exists(ui, UINPUT_GROUP) {
        ui.print_warn(format!(
            "Failed to ensure the `{UINPUT_GROUP}` group exists: {err}"
        ));
    }
    if let Some(warning) = record_group_membership_change_result(
        &mut outcome,
        UINPUT_GROUP,
        add_user_to_group(ui, UINPUT_GROUP),
    ) {
        ui.print_warn(warning);
    }
    match reload_udev(ui) {
        Ok(()) => outcome.udev_reload_succeeded = true,
        Err(err) => ui.print_warn(format!(
            "Failed to reload `udev` after updating `/dev/uinput`: {err}"
        )),
    }

    if let Some(message) = outcome.setup_group_change_message() {
        ui.print_info(message);
    }

    Ok(outcome)
}

pub(super) fn record_group_membership_change_result(
    outcome: &mut InjectionSetupOutcome,
    group: &str,
    result: Result<bool>,
) -> Option<String> {
    match result {
        Ok(true) => {
            outcome.changed_groups = true;
            outcome.group_membership_ready = true;
            None
        }
        Ok(false) => {
            outcome.group_membership_ready = true;
            None
        }
        Err(err) => Some(format!(
            "Failed to add the current user to the `{group}` group: {err}"
        )),
    }
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
    run_sudo(UDEV_TRIGGER_ARGS)
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

fn ensure_group_exists(ui: &SetupUi, group: &str) -> Result<()> {
    if group_exists(group)? {
        ui.print_info(format!("Group `{group}` already exists."));
        return Ok(());
    }

    ui.print_info(format!(
        "Creating dedicated `{group}` group for `/dev/uinput`..."
    ));
    run_sudo(&["groupadd", "--system", group])
}

fn group_exists(group: &str) -> Result<bool> {
    let group_file = std::fs::read_to_string("/etc/group").map_err(|err| {
        crate::error::WhsprError::Config(format!("failed to read /etc/group: {err}"))
    })?;

    Ok(group_file
        .lines()
        .filter_map(|line| line.split(':').next())
        .any(|entry| entry == group))
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
    let uid = unsafe { libc::geteuid() };
    if uid == 0 {
        return Err(crate::error::WhsprError::Config(
            "run `whispers setup` as your normal user, not as root".into(),
        ));
    }

    let buffer_len = match unsafe { libc::sysconf(libc::_SC_GETPW_R_SIZE_MAX) } {
        value if value > 0 => value as usize,
        _ => 1024,
    };
    let mut buffer = vec![0u8; buffer_len];
    let mut passwd = std::mem::MaybeUninit::<libc::passwd>::uninit();
    let mut result = std::ptr::null_mut();

    let status = unsafe {
        libc::getpwuid_r(
            uid,
            passwd.as_mut_ptr(),
            buffer.as_mut_ptr().cast(),
            buffer.len(),
            &mut result,
        )
    };
    if status != 0 || result.is_null() {
        return Err(crate::error::WhsprError::Config(format!(
            "failed to resolve current username for uid {uid}"
        )));
    }

    let passwd = unsafe { passwd.assume_init() };
    let name = unsafe { CStr::from_ptr(passwd.pw_name) };
    Ok(name.to_string_lossy().into_owned())
}

fn install_root_file(ui: &SetupUi, target: &str, contents: &str) -> Result<()> {
    let Some(parent) = Path::new(target).parent().and_then(|path| path.to_str()) else {
        return Err(crate::error::WhsprError::Config(format!(
            "failed to determine parent directory for `{target}`"
        )));
    };
    run_sudo(&["mkdir", "-p", parent])?;
    run_sudo_with_input(&["tee", target], contents)?;
    run_sudo(&["chmod", "0644", target])?;
    ui.print_info(format!("Installed `{target}`."));
    Ok(())
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

fn run_sudo_with_input(args: &[&str], input: &str) -> Result<()> {
    let mut child = Command::new("sudo")
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .spawn()
        .map_err(|err| {
            crate::error::WhsprError::Config(format!("failed to run sudo {:?}: {err}", args))
        })?;

    let mut stdin = child.stdin.take().ok_or_else(|| {
        crate::error::WhsprError::Config(format!("failed to open stdin for sudo {:?}", args))
    })?;
    stdin.write_all(input.as_bytes()).map_err(|err| {
        crate::error::WhsprError::Config(format!(
            "failed to write stdin for sudo {:?}: {err}",
            args
        ))
    })?;
    drop(stdin);

    let status = child.wait().map_err(|err| {
        crate::error::WhsprError::Config(format!("failed to wait for sudo {:?}: {err}", args))
    })?;
    if !status.success() {
        return Err(crate::error::WhsprError::Config(format!(
            "`sudo {}` exited with {status}",
            args.join(" ")
        )));
    }
    Ok(())
}

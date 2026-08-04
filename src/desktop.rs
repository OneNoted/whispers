use std::fs;
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::{OnceLock, mpsc};
use std::thread;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::{Error, Result, paths};

const NOTIFICATION_ID: &str = "99172";
const OSD_QML: &str = include_str!("../ui/shell.qml");

enum OsdCommand {
    Ensure,
    Show {
        stage: &'static str,
        detail: String,
        select_output: bool,
    },
    Hide,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Focus {
    pub app_id: String,
    pub title: String,
    pub identity: String,
}

pub fn capture_focus() -> Option<Focus> {
    capture_niri().or_else(capture_hyprland)
}

pub fn prewarm_osd() {
    send_osd(OsdCommand::Ensure);
}

pub fn show_listening() {
    show_osd("listening", "", true);
}

pub fn show_preview(text: &str) {
    show_osd("listening", text, false);
}

pub fn show_transcribing() {
    show_osd("transcribing", "", false);
}

pub fn show_enhancing() {
    show_osd("enhancing", "", false);
}

pub fn hide_osd() {
    send_osd(OsdCommand::Hide);
}

pub fn show_message(text: &str) {
    show_osd("info", text, false);
}

pub fn show_error(text: &str) {
    show_osd("error", text, false);
}

pub fn play_cue(name: &str) {
    let candidates = [
        format!("/usr/share/whispers/sounds/{name}.wav"),
        format!("{}/sounds/{name}.wav", env!("CARGO_MANIFEST_DIR")),
    ];
    if let Some(path) = candidates
        .iter()
        .find(|path| std::path::Path::new(path).is_file())
    {
        let _ = Command::new("pw-play")
            .arg(path)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

pub fn deliver(text: &str, target: Option<&str>) -> Result<()> {
    let result = deliver_inner(text, target);
    if result.is_err() {
        show_error("Delivery failed");
    }
    result
}

fn deliver_inner(text: &str, target: Option<&str>) -> Result<()> {
    show_osd("delivering", "", false);
    let current = capture_focus().map(|focus| focus.identity);
    if !focus_is_unchanged(target, current.as_deref()) {
        copy_to_clipboard(text)?;
        show_osd("copied", "Focus changed", false);
        return Ok(());
    }
    if type_with_wtype(text).is_ok() {
        show_osd("done", "Text inserted", false);
        return Ok(());
    }
    copy_to_clipboard(text)?;
    let current = capture_focus();
    if !focus_is_unchanged(
        target,
        current.as_ref().map(|focus| focus.identity.as_str()),
    ) {
        show_osd("copied", "Focus changed", false);
        return Ok(());
    }
    let terminal = current.is_some_and(|focus| is_terminal(&focus.app_id));
    if paste_with_wtype(terminal).is_ok() {
        show_osd("done", "Text inserted", false);
    } else {
        show_osd("copied", "Automatic paste failed", false);
    }
    Ok(())
}

fn focus_is_unchanged(target: Option<&str>, current: Option<&str>) -> bool {
    matches!((target, current), (Some(target), Some(current)) if target == current)
}

fn show_osd(stage: &'static str, detail: &str, select_output: bool) {
    send_osd(OsdCommand::Show {
        stage,
        detail: detail.to_string(),
        select_output,
    });
}

fn send_osd(command: OsdCommand) {
    static OSD: OnceLock<mpsc::Sender<OsdCommand>> = OnceLock::new();
    let sender = OSD.get_or_init(|| {
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || osd_worker(&receiver));
        sender
    });
    let _ = sender.send(command);
}

fn osd_worker(receiver: &mpsc::Receiver<OsdCommand>) {
    let mut available = false;
    let mut output = String::new();
    while let Ok(command) = receiver.recv() {
        match command {
            OsdCommand::Ensure => available = ensure_osd(),
            OsdCommand::Show {
                stage,
                detail,
                select_output,
            } => {
                if select_output {
                    output = capture_output().unwrap_or_default();
                }
                if (!available && !ensure_osd()) || !call_osd("display", &[stage, &detail, &output])
                {
                    available = false;
                    notify(&format!("{}{}", stage_label(stage), detail_suffix(&detail)));
                } else {
                    available = true;
                }
            }
            OsdCommand::Hide => {
                if available {
                    let _ = call_osd("hide", &[]);
                }
            }
        }
    }
}

fn ensure_osd() -> bool {
    let Ok(config) = install_osd_config() else {
        return false;
    };
    if osd_is_running(&config) {
        return true;
    }
    if !Command::new("quickshell")
        .arg("--path")
        .arg(&config)
        .args(["--daemonize", "--no-duplicate"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
    {
        return false;
    }
    for _ in 0..20 {
        if osd_is_running(&config) {
            return true;
        }
        thread::sleep(Duration::from_millis(25));
    }
    false
}

fn install_osd_config() -> Result<std::path::PathBuf> {
    let config = paths::osd_config()?;
    if fs::read_to_string(&config).is_ok_and(|contents| contents == OSD_QML) {
        return Ok(config);
    }
    let parent = config
        .parent()
        .ok_or_else(|| Error::message("OSD configuration has no parent"))?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(".shell.qml.{}.tmp", std::process::id()));
    fs::write(&temporary, OSD_QML)?;
    fs::rename(temporary, &config)?;
    Ok(config)
}

fn osd_is_running(config: &std::path::Path) -> bool {
    Command::new("qs")
        .arg("ipc")
        .args(["--path", &config.to_string_lossy(), "show"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn call_osd(function: &str, arguments: &[&str]) -> bool {
    let Ok(config) = paths::osd_config() else {
        return false;
    };
    Command::new("qs")
        .arg("ipc")
        .arg("--path")
        .arg(config)
        .args(["call", "whispers", function])
        .args(arguments)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn capture_output() -> Option<String> {
    let niri = Command::new("niri")
        .args(["msg", "--json", "focused-output"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| serde_json::from_slice::<serde_json::Value>(&output.stdout).ok())
        .and_then(|value| value.get("name")?.as_str().map(str::to_string));
    niri.or_else(|| {
        Command::new("hyprctl")
            .args(["-j", "activeworkspace"])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| serde_json::from_slice::<serde_json::Value>(&output.stdout).ok())
            .and_then(|value| value.get("monitor")?.as_str().map(str::to_string))
    })
}

fn notify(body: &str) {
    let _ = Command::new("notify-send")
        .args([
            "--transient",
            "--replace-id",
            NOTIFICATION_ID,
            "whispers",
            body,
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

fn stage_label(stage: &str) -> &'static str {
    match stage {
        "listening" => "Listening",
        "transcribing" => "Transcribing",
        "enhancing" => "Enhancing",
        "delivering" => "Typing",
        "copied" => "Copied",
        "error" => "Couldn't finish",
        "info" => "Whispers",
        _ => "Done",
    }
}

fn detail_suffix(detail: &str) -> String {
    if detail.is_empty() {
        String::new()
    } else {
        format!(": {detail}")
    }
}

fn type_with_wtype(text: &str) -> Result<()> {
    let mut child = Command::new("wtype")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| Error::message(format!("could not start wtype: {error}")))?;
    child
        .stdin
        .take()
        .ok_or_else(|| Error::message("wtype stdin is unavailable"))?
        .write_all(text.as_bytes())?;
    let status = child.wait()?;
    if status.success() {
        Ok(())
    } else {
        Err(Error::message(
            "wtype could not type into the active surface",
        ))
    }
}

fn copy_to_clipboard(text: &str) -> Result<()> {
    let mut child = Command::new("wl-copy")
        .args(["--type", "text/plain;charset=utf-8"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| Error::message(format!("could not start wl-copy: {error}")))?;
    child
        .stdin
        .take()
        .ok_or_else(|| Error::message("wl-copy stdin is unavailable"))?
        .write_all(text.as_bytes())?;
    if child.wait()?.success() {
        Ok(())
    } else {
        Err(Error::message("wl-copy failed"))
    }
}

fn paste_with_wtype(terminal: bool) -> Result<()> {
    let modifier = "ctrl";
    let mut command = Command::new("wtype");
    command.args(["-M", modifier]);
    if terminal {
        command.args(["-M", "shift", "-k", "v", "-m", "shift"]);
    } else {
        command.args(["-k", "v"]);
    }
    let status = command.args(["-m", modifier]).status()?;
    if status.success() {
        Ok(())
    } else {
        Err(Error::message("paste keystroke failed"))
    }
}

fn is_terminal(app_id: &str) -> bool {
    let app_id = app_id.to_ascii_lowercase();
    [
        "foot",
        "kitty",
        "alacritty",
        "wezterm",
        "ghostty",
        "terminal",
    ]
    .iter()
    .any(|candidate| app_id.contains(candidate))
}

fn capture_niri() -> Option<Focus> {
    let output = Command::new("niri")
        .args(["msg", "--json", "focused-window"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    let identity = value.get("id")?.to_string();
    Some(Focus {
        app_id: value.get("app_id")?.as_str()?.into(),
        title: value
            .get("title")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .into(),
        identity,
    })
}

fn capture_hyprland() -> Option<Focus> {
    let output = Command::new("hyprctl")
        .args(["-j", "activewindow"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    Some(Focus {
        app_id: value.get("class")?.as_str()?.into(),
        title: value
            .get("title")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .into(),
        identity: value.get("address")?.as_str()?.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_terminal_surfaces() {
        assert!(is_terminal("com.mitchellh.ghostty"));
        assert!(is_terminal("foot"));
        assert!(!is_terminal("libreoffice-writer"));
    }

    #[test]
    fn focus_must_be_known_at_both_boundaries() {
        assert!(focus_is_unchanged(Some("window-1"), Some("window-1")));
        assert!(!focus_is_unchanged(Some("window-1"), Some("window-2")));
        assert!(!focus_is_unchanged(Some("window-1"), None));
        assert!(!focus_is_unchanged(None, Some("window-1")));
        assert!(!focus_is_unchanged(None, None));
    }
}

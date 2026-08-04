use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::{OnceLock, mpsc};
use std::thread;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

const NOTIFICATION_ID: &str = "99172";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Focus {
    pub app_id: String,
    pub title: String,
    pub identity: String,
}

pub fn capture_focus() -> Option<Focus> {
    capture_niri().or_else(capture_hyprland)
}

pub fn show_listening() {
    notify("Listening…");
}

pub fn show_preview(text: &str) {
    notify(text);
}

pub fn show_message(text: &str) {
    notify(text);
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
    let current = capture_focus().map(|focus| focus.identity);
    if !focus_is_unchanged(target, current.as_deref()) {
        copy_to_clipboard(text)?;
        show_message("Dictation copied because focus could not be verified");
        return Ok(());
    }
    if type_with_wtype(text).is_ok() {
        return Ok(());
    }
    copy_to_clipboard(text)?;
    let current = capture_focus();
    if !focus_is_unchanged(
        target,
        current.as_ref().map(|focus| focus.identity.as_str()),
    ) {
        show_message("Dictation copied because focus changed");
        return Ok(());
    }
    let terminal = current.is_some_and(|focus| is_terminal(&focus.app_id));
    let result = paste_with_wtype(terminal);
    if result.is_err() {
        show_message("Dictation is on the clipboard; automatic paste failed");
    }
    result
}

fn focus_is_unchanged(target: Option<&str>, current: Option<&str>) -> bool {
    matches!((target, current), (Some(target), Some(current)) if target == current)
}

fn notify(body: &str) {
    static NOTIFICATIONS: OnceLock<mpsc::SyncSender<String>> = OnceLock::new();
    let sender = NOTIFICATIONS.get_or_init(|| {
        let (sender, receiver) = mpsc::sync_channel::<String>(1);
        thread::spawn(move || {
            while let Ok(body) = receiver.recv() {
                let _ = Command::new("notify-send")
                    .args([
                        "--transient",
                        "--replace-id",
                        NOTIFICATION_ID,
                        "whispers",
                        &body,
                    ])
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status();
            }
        });
        sender
    });
    let _ = sender.try_send(body.to_string());
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

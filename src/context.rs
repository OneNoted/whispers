use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TypingContext {
    pub focus_fingerprint: String,
    pub app_id: Option<String>,
    pub window_title: Option<String>,
    pub surface_kind: SurfaceKind,
    pub browser_domain: Option<String>,
    pub captured_at_ms: u64,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceKind {
    Browser,
    Terminal,
    Editor,
    #[default]
    GenericText,
    Unknown,
}

impl Default for TypingContext {
    fn default() -> Self {
        Self::unknown()
    }
}

impl TypingContext {
    pub fn unknown() -> Self {
        Self {
            focus_fingerprint: String::new(),
            app_id: None,
            window_title: None,
            surface_kind: SurfaceKind::Unknown,
            browser_domain: None,
            captured_at_ms: now_ms(),
        }
    }

    pub fn is_known_focus(&self) -> bool {
        !self.focus_fingerprint.is_empty()
    }
}

pub fn capture_typing_context() -> TypingContext {
    let captured_at_ms = now_ms();

    if let Some(context) = capture_niri_context(captured_at_ms) {
        return context;
    }

    if let Some(context) = capture_hyprland_context(captured_at_ms) {
        return context;
    }

    TypingContext::unknown()
}

fn capture_niri_context(captured_at_ms: u64) -> Option<TypingContext> {
    let desktop = std::env::var("XDG_CURRENT_DESKTOP")
        .ok()
        .map(|value| value.to_ascii_lowercase())
        .unwrap_or_default();
    if !desktop.contains("niri") {
        return None;
    }

    let output = Command::new("niri")
        .args(["msg", "-j", "focused-window"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let raw = String::from_utf8(output.stdout).ok()?;
    parse_niri_focused_window_json(&raw, captured_at_ms)
}

fn capture_hyprland_context(captured_at_ms: u64) -> Option<TypingContext> {
    std::env::var_os("HYPRLAND_INSTANCE_SIGNATURE")?;

    let output = Command::new("hyprctl")
        .args(["activewindow", "-j"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let raw = String::from_utf8(output.stdout).ok()?;
    parse_hyprland_activewindow_json(&raw, captured_at_ms)
}

fn parse_niri_focused_window_json(raw: &str, captured_at_ms: u64) -> Option<TypingContext> {
    let value: Value = serde_json::from_str(raw).ok()?;
    let window_id = value.get("id").and_then(Value::as_i64);
    let app_id = json_string(&value, "app_id").filter(|value| !value.is_empty());
    let window_title = json_string(&value, "title").filter(|value| !value.is_empty());

    if window_id.is_none() && app_id.is_none() && window_title.is_none() {
        return None;
    }

    let focus_fingerprint = if let Some(id) = window_id {
        format!("niri:{id}")
    } else {
        format!(
            "niri:{}:{}",
            app_id.as_deref().unwrap_or(""),
            window_title.as_deref().unwrap_or("")
        )
    };

    let surface_kind = classify_surface_kind(app_id.as_deref(), window_title.as_deref());
    let browser_domain = infer_browser_domain(surface_kind, window_title.as_deref());

    Some(TypingContext {
        focus_fingerprint,
        app_id,
        window_title,
        surface_kind,
        browser_domain,
        captured_at_ms,
    })
}

fn parse_hyprland_activewindow_json(raw: &str, captured_at_ms: u64) -> Option<TypingContext> {
    let value: Value = serde_json::from_str(raw).ok()?;
    let address = json_string(&value, "address").unwrap_or_default();
    let app_id = json_string(&value, "class")
        .or_else(|| json_string(&value, "initialClass"))
        .filter(|value| !value.is_empty());
    let window_title = json_string(&value, "title")
        .or_else(|| json_string(&value, "initialTitle"))
        .filter(|value| !value.is_empty());

    if address.is_empty() && app_id.is_none() && window_title.is_none() {
        return None;
    }

    let focus_fingerprint = if !address.is_empty() {
        format!("hyprland:{address}")
    } else {
        format!(
            "hyprland:{}:{}",
            app_id.as_deref().unwrap_or(""),
            window_title.as_deref().unwrap_or("")
        )
    };

    let surface_kind = classify_surface_kind(app_id.as_deref(), window_title.as_deref());
    let browser_domain = infer_browser_domain(surface_kind, window_title.as_deref());

    Some(TypingContext {
        focus_fingerprint,
        app_id,
        window_title,
        surface_kind,
        browser_domain,
        captured_at_ms,
    })
}

fn classify_surface_kind(app_id: Option<&str>, window_title: Option<&str>) -> SurfaceKind {
    let haystack = format!(
        "{} {}",
        app_id.unwrap_or_default().to_ascii_lowercase(),
        window_title.unwrap_or_default().to_ascii_lowercase()
    );

    if haystack.trim().is_empty() {
        return SurfaceKind::Unknown;
    }

    if [
        "firefox", "chromium", "chrome", "brave", "vivaldi", "zen", "browser",
    ]
    .iter()
    .any(|needle| haystack.contains(needle))
    {
        return SurfaceKind::Browser;
    }

    if [
        "kitty",
        "alacritty",
        "foot",
        "wezterm",
        "ghostty",
        "terminal",
        "konsole",
        "gnome-terminal",
        "xterm",
    ]
    .iter()
    .any(|needle| haystack.contains(needle))
    {
        return SurfaceKind::Terminal;
    }

    if [
        "code",
        "codium",
        "zed",
        "emacs",
        "nvim",
        "neovim",
        "jetbrains",
        "cursor",
        "sublime",
        "helix",
    ]
    .iter()
    .any(|needle| haystack.contains(needle))
    {
        return SurfaceKind::Editor;
    }

    SurfaceKind::GenericText
}

fn json_string(value: &Value, key: &str) -> Option<String> {
    value.get(key)?.as_str().map(str::to_string)
}

fn infer_browser_domain(surface_kind: SurfaceKind, window_title: Option<&str>) -> Option<String> {
    if surface_kind != SurfaceKind::Browser {
        return None;
    }

    let title = window_title?;
    extract_browser_domain_from_title(title)
}

fn extract_browser_domain_from_title(title: &str) -> Option<String> {
    let normalized = title
        .replace(" — ", "\n")
        .replace(" – ", "\n")
        .replace(" - ", "\n")
        .replace(" | ", "\n")
        .replace(" · ", "\n");

    for segment in normalized.lines() {
        let segment = segment.trim();
        if segment.is_empty() {
            continue;
        }

        if let Some(domain) = extract_domain_candidate(segment) {
            return Some(domain);
        }

        for token in segment.split_whitespace() {
            if let Some(domain) = extract_domain_candidate(token) {
                return Some(domain);
            }
        }
    }

    None
}

fn extract_domain_candidate(candidate: &str) -> Option<String> {
    let trimmed = candidate.trim_matches(|ch: char| {
        ch.is_whitespace() || matches!(ch, '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}' | ',')
    });
    if trimmed.is_empty() {
        return None;
    }

    let without_scheme = trimmed.split("://").nth(1).unwrap_or(trimmed);
    let host = without_scheme
        .split('/')
        .next()
        .unwrap_or(without_scheme)
        .split(':')
        .next()
        .unwrap_or(without_scheme)
        .trim_end_matches('.');

    looks_like_domain(host).then(|| host.to_ascii_lowercase())
}

fn looks_like_domain(host: &str) -> bool {
    let mut labels = host.split('.');
    let Some(first) = labels.next() else {
        return false;
    };
    if first.is_empty() || !is_domain_label(first) {
        return false;
    }

    let rest = labels.collect::<Vec<_>>();
    if rest.is_empty() {
        return false;
    }

    rest.iter().all(|label| is_domain_label(label))
        && rest
            .last()
            .is_some_and(|label| label.chars().any(|ch| ch.is_ascii_alphabetic()))
}

fn is_domain_label(label: &str) -> bool {
    !label.is_empty()
        && label
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-')
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_surface_kind_detects_terminal() {
        let kind = classify_surface_kind(Some("kitty"), Some("notes@host"));
        assert_eq!(kind, SurfaceKind::Terminal);
    }

    #[test]
    fn classify_surface_kind_detects_browser() {
        let kind = classify_surface_kind(Some("firefox"), Some("Example Domain"));
        assert_eq!(kind, SurfaceKind::Browser);
    }

    #[test]
    fn parse_niri_focused_window_json_uses_id_for_focus_fingerprint() {
        let raw = r#"{
            "id": 7,
            "title": "codex ~/P/whispers",
            "app_id": "kitty"
        }"#;

        let context = parse_niri_focused_window_json(raw, 42).expect("context");
        assert_eq!(context.focus_fingerprint, "niri:7");
        assert_eq!(context.app_id.as_deref(), Some("kitty"));
        assert_eq!(context.window_title.as_deref(), Some("codex ~/P/whispers"));
        assert_eq!(context.surface_kind, SurfaceKind::Terminal);
        assert_eq!(context.captured_at_ms, 42);
    }

    #[test]
    fn parse_hyprland_activewindow_json_uses_address_for_focus_fingerprint() {
        let raw = r#"{
            "address": "0x1234",
            "class": "kitty",
            "title": "shell"
        }"#;

        let context = parse_hyprland_activewindow_json(raw, 42).expect("context");
        assert_eq!(context.focus_fingerprint, "hyprland:0x1234");
        assert_eq!(context.app_id.as_deref(), Some("kitty"));
        assert_eq!(context.window_title.as_deref(), Some("shell"));
        assert_eq!(context.surface_kind, SurfaceKind::Terminal);
        assert_eq!(context.captured_at_ms, 42);
    }

    #[test]
    fn parse_hyprland_activewindow_json_returns_none_for_empty_payload() {
        let raw = r#"{"mapped": true}"#;
        assert!(parse_hyprland_activewindow_json(raw, 1).is_none());
    }

    #[test]
    fn parse_niri_focused_window_json_extracts_browser_domain_from_title() {
        let raw = r#"{
            "id": 11,
            "title": "docs.rs - serde_json",
            "app_id": "firefox"
        }"#;

        let context = parse_niri_focused_window_json(raw, 42).expect("context");
        assert_eq!(context.surface_kind, SurfaceKind::Browser);
        assert_eq!(context.browser_domain.as_deref(), Some("docs.rs"));
    }

    #[test]
    fn parse_hyprland_activewindow_json_extracts_browser_domain_from_title() {
        let raw = r#"{
            "address": "0x5678",
            "class": "firefox",
            "title": "https://news.ycombinator.com/item?id=1"
        }"#;

        let context = parse_hyprland_activewindow_json(raw, 42).expect("context");
        assert_eq!(context.surface_kind, SurfaceKind::Browser);
        assert_eq!(
            context.browser_domain.as_deref(),
            Some("news.ycombinator.com")
        );
    }
}

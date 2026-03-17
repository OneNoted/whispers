use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use unicode_segmentation::UnicodeSegmentation;

use crate::config::SessionConfig;
use crate::context::TypingContext;
use crate::error::{Result, WhsprError};

use super::{EligibleSessionEntry, SessionEntry, SessionRewriteSummary};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct SessionFile {
    next_id: u64,
    entries: Vec<SessionEntry>,
}

pub fn load_recent_entry(
    config: &SessionConfig,
    context: &TypingContext,
) -> Result<Option<EligibleSessionEntry>> {
    if !config.enabled || !context.is_known_focus() {
        return Ok(None);
    }

    let mut state = load_session_file()?;
    prune_state(&mut state, config);
    persist_session_file(&state)?;

    let Some(entry) = state.entries.last().cloned() else {
        return Ok(None);
    };

    if entry.focus_fingerprint != context.focus_fingerprint {
        return Ok(None);
    }

    if entry.grapheme_len == 0 || entry.grapheme_len > config.max_replace_graphemes {
        return Ok(None);
    }

    Ok(Some(EligibleSessionEntry {
        delete_graphemes: entry.grapheme_len,
        entry,
    }))
}

pub fn record_append(
    config: &SessionConfig,
    context: &TypingContext,
    final_text: &str,
    rewrite_summary: SessionRewriteSummary,
) -> Result<()> {
    if !config.enabled || final_text.trim().is_empty() || !context.is_known_focus() {
        return Ok(());
    }

    let mut state = load_session_file()?;
    prune_state(&mut state, config);
    let entry = SessionEntry {
        id: state.next_id,
        final_text: final_text.to_string(),
        grapheme_len: grapheme_count(final_text),
        injected_at_ms: now_ms(),
        focus_fingerprint: context.focus_fingerprint.clone(),
        surface_kind: context.surface_kind,
        app_id: context.app_id.clone(),
        window_title: context.window_title.clone(),
        rewrite_summary,
    };
    state.next_id = state.next_id.saturating_add(1);
    state.entries.push(entry);
    trim_state(&mut state, config);
    persist_session_file(&state)
}

pub fn record_replace(
    config: &SessionConfig,
    context: &TypingContext,
    replaced_entry_id: u64,
    final_text: &str,
    rewrite_summary: SessionRewriteSummary,
) -> Result<()> {
    if !config.enabled || final_text.trim().is_empty() || !context.is_known_focus() {
        return Ok(());
    }

    let mut state = load_session_file()?;
    prune_state(&mut state, config);
    if let Some(entry) = state
        .entries
        .iter_mut()
        .find(|entry| entry.id == replaced_entry_id)
    {
        entry.final_text = final_text.to_string();
        entry.grapheme_len = grapheme_count(final_text);
        entry.injected_at_ms = now_ms();
        entry.focus_fingerprint = context.focus_fingerprint.clone();
        entry.surface_kind = context.surface_kind;
        entry.app_id = context.app_id.clone();
        entry.window_title = context.window_title.clone();
        entry.rewrite_summary = rewrite_summary;
    } else {
        return record_append(config, context, final_text, rewrite_summary);
    }
    trim_state(&mut state, config);
    persist_session_file(&state)
}

fn load_session_file() -> Result<SessionFile> {
    let path = session_file_path();
    if !path.exists() {
        return Ok(SessionFile::default());
    }

    let contents = std::fs::read_to_string(&path).map_err(|e| {
        WhsprError::Config(format!(
            "failed to read session state {}: {e}",
            path.display()
        ))
    })?;
    serde_json::from_str(&contents).map_err(|e| {
        WhsprError::Config(format!(
            "failed to parse session state {}: {e}",
            path.display()
        ))
    })
}

fn persist_session_file(state: &SessionFile) -> Result<()> {
    let path = session_file_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            WhsprError::Config(format!(
                "failed to create session runtime dir {}: {e}",
                parent.display()
            ))
        })?;
    }
    let encoded = serde_json::to_vec(state)
        .map_err(|e| WhsprError::Config(format!("failed to encode session state: {e}")))?;
    std::fs::write(&path, encoded).map_err(|e| {
        WhsprError::Config(format!(
            "failed to write session state {}: {e}",
            path.display()
        ))
    })
}

fn prune_state(state: &mut SessionFile, config: &SessionConfig) {
    let cutoff = now_ms().saturating_sub(config.max_age_ms);
    state.entries.retain(|entry| entry.injected_at_ms >= cutoff);
    trim_state(state, config);
    if state.next_id == 0 {
        state.next_id = 1;
    }
}

fn trim_state(state: &mut SessionFile, config: &SessionConfig) {
    if state.entries.len() > config.max_entries {
        let remove_count = state.entries.len() - config.max_entries;
        state.entries.drain(0..remove_count);
    }
}

fn session_file_path() -> PathBuf {
    let runtime_dir = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(runtime_dir)
        .join("whispers")
        .join("session.json")
}

fn grapheme_count(text: &str) -> usize {
    UnicodeSegmentation::graphemes(text, true).count()
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::{load_recent_entry, record_append, record_replace};
    use crate::config::SessionConfig;
    use crate::context::{SurfaceKind, TypingContext};
    use crate::session::SessionRewriteSummary;
    use crate::test_support::{EnvVarGuard, env_lock, set_env, unique_temp_dir};

    fn config() -> SessionConfig {
        SessionConfig {
            enabled: true,
            max_entries: 3,
            max_age_ms: 8_000,
            max_replace_graphemes: 400,
        }
    }

    fn context() -> TypingContext {
        TypingContext {
            focus_fingerprint: "hyprland:0x123".into(),
            app_id: Some("kitty".into()),
            window_title: Some("shell".into()),
            surface_kind: SurfaceKind::Terminal,
            browser_domain: None,
            captured_at_ms: 10,
        }
    }

    fn with_runtime_dir<T>(f: impl FnOnce() -> T) -> T {
        let _env_lock = env_lock();
        let _guard = EnvVarGuard::capture(&["XDG_RUNTIME_DIR"]);
        let runtime_dir = unique_temp_dir("session-runtime");
        let runtime_dir = runtime_dir
            .to_str()
            .expect("temp runtime dir should be valid UTF-8");
        set_env("XDG_RUNTIME_DIR", runtime_dir);
        f()
    }

    #[test]
    fn record_append_then_load_recent_entry() {
        with_runtime_dir(|| {
            record_append(
                &config(),
                &context(),
                "Hello there",
                SessionRewriteSummary {
                    had_edit_cues: false,
                    rewrite_used: false,
                    recommended_candidate: None,
                },
            )
            .expect("record");

            let entry = load_recent_entry(&config(), &context())
                .expect("load")
                .expect("entry");
            assert_eq!(entry.entry.final_text, "Hello there");
            assert_eq!(entry.delete_graphemes, 11);
        });
    }

    #[test]
    fn load_recent_entry_requires_matching_focus() {
        with_runtime_dir(|| {
            record_append(
                &config(),
                &context(),
                "Hello there",
                SessionRewriteSummary {
                    had_edit_cues: false,
                    rewrite_used: false,
                    recommended_candidate: None,
                },
            )
            .expect("record");

            let mut other = context();
            other.focus_fingerprint = "hyprland:0x999".into();
            assert!(
                load_recent_entry(&config(), &other)
                    .expect("load")
                    .is_none()
            );
        });
    }

    #[test]
    fn record_replace_updates_existing_entry() {
        with_runtime_dir(|| {
            let rewrite_summary = SessionRewriteSummary {
                had_edit_cues: false,
                rewrite_used: false,
                recommended_candidate: None,
            };
            record_append(
                &config(),
                &context(),
                "Hello there",
                rewrite_summary.clone(),
            )
            .expect("record");
            let entry = load_recent_entry(&config(), &context())
                .expect("load")
                .expect("entry");

            record_replace(&config(), &context(), entry.entry.id, "Hi", rewrite_summary)
                .expect("replace");

            let replaced = load_recent_entry(&config(), &context())
                .expect("load")
                .expect("entry");
            assert_eq!(replaced.entry.final_text, "Hi");
            assert_eq!(replaced.delete_graphemes, 2);
        });
    }
}

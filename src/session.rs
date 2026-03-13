use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use unicode_segmentation::UnicodeSegmentation;

use crate::cleanup;
use crate::config::SessionConfig;
use crate::context::{SurfaceKind, TypingContext};
use crate::error::{Result, WhsprError};
use crate::rewrite_protocol::{
    RewriteSessionBacktrackCandidate, RewriteSessionBacktrackCandidateKind, RewriteSessionEntry,
    RewriteSurfaceKind, RewriteTranscript, RewriteTypingContext,
};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct SessionFile {
    next_id: u64,
    entries: Vec<SessionEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionEntry {
    pub id: u64,
    pub final_text: String,
    pub grapheme_len: usize,
    pub injected_at_ms: u64,
    pub focus_fingerprint: String,
    pub surface_kind: SurfaceKind,
    pub app_id: Option<String>,
    pub window_title: Option<String>,
    pub rewrite_summary: SessionRewriteSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionRewriteSummary {
    pub had_edit_cues: bool,
    pub rewrite_used: bool,
    pub recommended_candidate: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EligibleSessionEntry {
    pub entry: SessionEntry,
    pub delete_graphemes: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionBacktrackPlan {
    pub recent_entries: Vec<RewriteSessionEntry>,
    pub candidates: Vec<RewriteSessionBacktrackCandidate>,
    pub recommended: Option<RewriteSessionBacktrackCandidate>,
    pub deterministic_replacement_text: Option<String>,
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

pub fn build_backtrack_plan(
    transcript: &RewriteTranscript,
    recent_entry: Option<&EligibleSessionEntry>,
) -> SessionBacktrackPlan {
    let Some(recent_entry) = recent_entry else {
        return SessionBacktrackPlan::default();
    };
    if !should_offer_session_backtrack(transcript) {
        return SessionBacktrackPlan::default();
    }

    let append_text = preferred_current_text(transcript);
    if append_text.is_empty() {
        return SessionBacktrackPlan::default();
    }

    let append_candidate = RewriteSessionBacktrackCandidate {
        kind: RewriteSessionBacktrackCandidateKind::AppendCurrent,
        entry_id: None,
        delete_graphemes: 0,
        text: append_text.clone(),
    };
    let replace_candidate = RewriteSessionBacktrackCandidate {
        kind: RewriteSessionBacktrackCandidateKind::ReplaceLastEntry,
        entry_id: Some(recent_entry.entry.id),
        delete_graphemes: recent_entry.delete_graphemes,
        text: append_text,
    };

    SessionBacktrackPlan {
        recent_entries: vec![to_rewrite_session_entry(&recent_entry.entry)],
        candidates: vec![replace_candidate.clone(), append_candidate],
        recommended: Some(replace_candidate),
        deterministic_replacement_text: preferred_current_text_for_exact_followup(transcript),
    }
}

pub fn to_rewrite_typing_context(context: &TypingContext) -> Option<RewriteTypingContext> {
    context.is_known_focus().then(|| RewriteTypingContext {
        focus_fingerprint: context.focus_fingerprint.clone(),
        app_id: context.app_id.clone(),
        window_title: context.window_title.clone(),
        surface_kind: map_surface_kind(context.surface_kind),
        browser_domain: context.browser_domain.clone(),
        captured_at_ms: context.captured_at_ms,
    })
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

fn should_offer_session_backtrack(transcript: &RewriteTranscript) -> bool {
    if cleanup::explicit_followup_replacement(&transcript.raw_text).is_some() {
        return true;
    }

    if transcript.correction_aware_text.trim() == transcript.raw_text.trim() {
        return false;
    }

    let raw_prefix = normalize_prefix(&transcript.raw_text);
    if ![
        "scratch that",
        "actually scratch that",
        "never mind",
        "nevermind",
        "actually never mind",
        "actually nevermind",
        "oh wait never mind",
        "oh wait nevermind",
        "forget that",
        "wait no",
        "actually wait no",
        "i meant",
        "actually i meant",
        "i mean",
        "actually i mean",
    ]
    .iter()
    .any(|cue| raw_prefix.starts_with(cue))
    {
        return false;
    }

    transcript.edit_hypotheses.iter().any(|hypothesis| {
        hypothesis.strength == crate::rewrite_protocol::RewriteEditSignalStrength::Strong
            && matches!(
                hypothesis.match_source,
                crate::rewrite_protocol::RewriteEditHypothesisMatchSource::Exact
                    | crate::rewrite_protocol::RewriteEditHypothesisMatchSource::Alias
            )
    })
}

fn preferred_current_text(transcript: &RewriteTranscript) -> String {
    transcript
        .recommended_candidate
        .as_ref()
        .map(|candidate| candidate.text.trim())
        .filter(|text: &&str| !text.is_empty())
        .or_else(|| {
            Some(transcript.correction_aware_text.trim()).filter(|text: &&str| !text.is_empty())
        })
        .or_else(|| Some(transcript.raw_text.trim()).filter(|text: &&str| !text.is_empty()))
        .unwrap_or_default()
        .to_string()
}

fn preferred_current_text_for_exact_followup(transcript: &RewriteTranscript) -> Option<String> {
    if let Some(text) = cleanup::explicit_followup_replacement(&transcript.raw_text) {
        return Some(text);
    }

    if !has_strong_explicit_followup_cue(transcript) {
        return None;
    }

    let raw_prefix = normalize_prefix(&transcript.raw_text);
    if ![
        "scratch that",
        "actually scratch that",
        "never mind",
        "nevermind",
        "actually never mind",
        "actually nevermind",
        "oh wait never mind",
        "oh wait nevermind",
        "forget that",
    ]
    .iter()
    .any(|cue| raw_prefix.starts_with(cue))
    {
        return None;
    }

    let preferred = preferred_current_text(transcript);
    (!preferred.is_empty()).then_some(preferred)
}

fn has_strong_explicit_followup_cue(transcript: &RewriteTranscript) -> bool {
    transcript.edit_hypotheses.iter().any(|hypothesis| {
        hypothesis.strength == crate::rewrite_protocol::RewriteEditSignalStrength::Strong
            && matches!(
                hypothesis.match_source,
                crate::rewrite_protocol::RewriteEditHypothesisMatchSource::Exact
                    | crate::rewrite_protocol::RewriteEditHypothesisMatchSource::Alias
            )
            && matches!(
                hypothesis.cue_family.as_str(),
                "scratch_that" | "never_mind"
            )
    })
}

fn normalize_prefix(text: &str) -> String {
    text.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch.is_ascii_whitespace() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .take(4)
        .collect::<Vec<_>>()
        .join(" ")
}

fn to_rewrite_session_entry(entry: &SessionEntry) -> RewriteSessionEntry {
    RewriteSessionEntry {
        id: entry.id,
        final_text: entry.final_text.clone(),
        grapheme_len: entry.grapheme_len,
        focus_fingerprint: entry.focus_fingerprint.clone(),
        surface_kind: map_surface_kind(entry.surface_kind),
        app_id: entry.app_id.clone(),
        window_title: entry.window_title.clone(),
    }
}

fn map_surface_kind(kind: SurfaceKind) -> RewriteSurfaceKind {
    match kind {
        SurfaceKind::Browser => RewriteSurfaceKind::Browser,
        SurfaceKind::Terminal => RewriteSurfaceKind::Terminal,
        SurfaceKind::Editor => RewriteSurfaceKind::Editor,
        SurfaceKind::GenericText => RewriteSurfaceKind::GenericText,
        SurfaceKind::Unknown => RewriteSurfaceKind::Unknown,
    }
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

    #[test]
    fn build_backtrack_plan_prefers_replacing_recent_entry_for_follow_up_correction() {
        let transcript = RewriteTranscript {
            raw_text: "scratch that hi".into(),
            correction_aware_text: "Hi".into(),
            aggressive_correction_text: None,
            detected_language: Some("en".into()),
            typing_context: None,
            recent_session_entries: Vec::new(),
            session_backtrack_candidates: Vec::new(),
            recommended_session_candidate: None,
            segments: Vec::new(),
            edit_intents: Vec::new(),
            edit_signals: Vec::new(),
            edit_hypotheses: vec![crate::rewrite_protocol::RewriteEditHypothesis {
                cue_family: "scratch_that".into(),
                matched_text: "scratch that".into(),
                match_source: crate::rewrite_protocol::RewriteEditHypothesisMatchSource::Exact,
                kind: crate::rewrite_protocol::RewriteEditSignalKind::Cancel,
                scope_hint: crate::rewrite_protocol::RewriteEditSignalScope::Sentence,
                replacement_scope: crate::rewrite_protocol::RewriteReplacementScope::Sentence,
                tail_shape: crate::rewrite_protocol::RewriteTailShape::Phrase,
                strength: crate::rewrite_protocol::RewriteEditSignalStrength::Strong,
            }],
            rewrite_candidates: Vec::new(),
            recommended_candidate: Some(crate::rewrite_protocol::RewriteCandidate {
                kind: crate::rewrite_protocol::RewriteCandidateKind::SentenceReplacement,
                text: "Hi".into(),
            }),
            policy_context: crate::rewrite_protocol::RewritePolicyContext::default(),
        };

        let recent = EligibleSessionEntry {
            entry: SessionEntry {
                id: 7,
                final_text: "Hello there".into(),
                grapheme_len: 11,
                injected_at_ms: 1,
                focus_fingerprint: "hyprland:0x123".into(),
                surface_kind: SurfaceKind::GenericText,
                app_id: Some("firefox".into()),
                window_title: Some("Example".into()),
                rewrite_summary: SessionRewriteSummary {
                    had_edit_cues: false,
                    rewrite_used: true,
                    recommended_candidate: Some("Hello there".into()),
                },
            },
            delete_graphemes: 11,
        };

        let plan = build_backtrack_plan(&transcript, Some(&recent));
        assert_eq!(plan.recent_entries.len(), 1);
        assert_eq!(plan.candidates.len(), 2);
        assert_eq!(
            plan.recommended.as_ref().map(|candidate| candidate.kind),
            Some(RewriteSessionBacktrackCandidateKind::ReplaceLastEntry)
        );
        assert_eq!(
            plan.recommended
                .as_ref()
                .and_then(|candidate| candidate.entry_id),
            Some(7)
        );
        assert_eq!(plan.deterministic_replacement_text.as_deref(), Some("Hi"));
    }

    #[test]
    fn build_backtrack_plan_uses_raw_followup_fallback_without_hypotheses() {
        let transcript = RewriteTranscript {
            raw_text: "scratch that hi".into(),
            correction_aware_text: "scratch that hi".into(),
            aggressive_correction_text: None,
            detected_language: None,
            typing_context: None,
            recent_session_entries: Vec::new(),
            session_backtrack_candidates: Vec::new(),
            recommended_session_candidate: None,
            segments: Vec::new(),
            edit_intents: Vec::new(),
            edit_signals: Vec::new(),
            edit_hypotheses: Vec::new(),
            rewrite_candidates: Vec::new(),
            recommended_candidate: None,
            policy_context: crate::rewrite_protocol::RewritePolicyContext::default(),
        };

        let recent = EligibleSessionEntry {
            entry: SessionEntry {
                id: 7,
                final_text: "Hello there".into(),
                grapheme_len: 11,
                injected_at_ms: 1,
                focus_fingerprint: "hyprland:0x123".into(),
                surface_kind: SurfaceKind::GenericText,
                app_id: Some("firefox".into()),
                window_title: Some("Example".into()),
                rewrite_summary: SessionRewriteSummary {
                    had_edit_cues: false,
                    rewrite_used: true,
                    recommended_candidate: Some("Hello there".into()),
                },
            },
            delete_graphemes: 11,
        };

        let plan = build_backtrack_plan(&transcript, Some(&recent));
        assert_eq!(
            plan.recommended.as_ref().map(|candidate| candidate.kind),
            Some(RewriteSessionBacktrackCandidateKind::ReplaceLastEntry)
        );
        assert_eq!(plan.deterministic_replacement_text.as_deref(), Some("Hi"));
    }
}

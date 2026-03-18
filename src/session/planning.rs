use crate::cleanup;
use crate::context::{SurfaceKind, TypingContext};
use crate::rewrite_protocol::{
    RewriteSessionBacktrackCandidate, RewriteSessionBacktrackCandidateKind, RewriteSessionEntry,
    RewriteSurfaceKind, RewriteTranscript, RewriteTypingContext,
};

use super::{EligibleSessionEntry, SessionBacktrackPlan, SessionEntry};

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

#[cfg(test)]
mod tests {
    use super::build_backtrack_plan;
    use crate::context::SurfaceKind;
    use crate::rewrite_protocol::{
        RewriteCandidate, RewriteCandidateKind, RewriteEditHypothesis,
        RewriteEditHypothesisMatchSource, RewriteEditSignalKind, RewriteEditSignalStrength,
        RewritePolicyContext, RewriteReplacementScope, RewriteSessionBacktrackCandidateKind,
        RewriteTailShape, RewriteTranscript,
    };
    use crate::session::{EligibleSessionEntry, SessionEntry, SessionRewriteSummary};

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
            edit_hypotheses: vec![RewriteEditHypothesis {
                cue_family: "scratch_that".into(),
                matched_text: "scratch that".into(),
                match_source: RewriteEditHypothesisMatchSource::Exact,
                kind: RewriteEditSignalKind::Cancel,
                scope_hint: crate::rewrite_protocol::RewriteEditSignalScope::Sentence,
                replacement_scope: RewriteReplacementScope::Sentence,
                tail_shape: RewriteTailShape::Phrase,
                strength: RewriteEditSignalStrength::Strong,
            }],
            rewrite_candidates: Vec::new(),
            recommended_candidate: Some(RewriteCandidate {
                kind: RewriteCandidateKind::SentenceReplacement,
                text: "Hi".into(),
            }),
            edit_context: Default::default(),
            policy_context: RewritePolicyContext::default(),
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
            edit_context: Default::default(),
            policy_context: RewritePolicyContext::default(),
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
    }
}

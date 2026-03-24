use std::path::PathBuf;

use crate::agentic_rewrite;
use crate::cleanup;
use crate::config::{Config, PostprocessMode};
use crate::context::TypingContext;
use crate::personalization::{self, PersonalizationRules};
use crate::rewrite_model;
use crate::rewrite_protocol::{
    RewriteCandidateKind, RewriteSessionBacktrackCandidateKind, RewriteTranscript,
};
use crate::session::{self, EligibleSessionEntry};
use crate::structured_text;
use crate::transcribe::Transcript;

use super::finalize::FinalizedOperation;

#[derive(Debug, Clone, Default)]
pub struct RuntimeTextResources {
    pub(crate) rules: PersonalizationRules,
    pub(crate) runtime_policy: agentic_rewrite::RuntimePolicyResources,
}

pub(crate) struct RewritePlan {
    pub rules: PersonalizationRules,
    pub fallback_text: String,
    pub rewrite_transcript: RewriteTranscript,
    pub custom_instructions: Option<String>,
    pub local_model_path: Option<PathBuf>,
    pub operation: FinalizedOperation,
    pub had_edit_cues: bool,
    pub recommended_candidate: Option<String>,
}

pub fn raw_text(transcript: &Transcript) -> String {
    transcript.raw_text.trim().to_string()
}

pub(crate) fn resolve_rewrite_model_path(config: &Config) -> Option<PathBuf> {
    if let Some(path) = config.resolved_rewrite_model_path() {
        return Some(path);
    }

    rewrite_model::selected_model_path(&config.rewrite.selected_model)
}

pub(crate) fn load_runtime_rules_with_status(config: &Config) -> (PersonalizationRules, bool) {
    match personalization::load_rules(config) {
        Ok(rules) => (rules, false),
        Err(err) => {
            tracing::warn!("failed to load personalization rules: {err}");
            (PersonalizationRules::default(), true)
        }
    }
}

pub fn load_runtime_text_resources(config: &Config) -> RuntimeTextResources {
    load_runtime_text_resources_with_status(config).0
}

pub fn load_runtime_text_resources_with_status(config: &Config) -> (RuntimeTextResources, bool) {
    let (rules, rules_degraded) = load_runtime_rules_with_status(config);
    let (runtime_policy, policy_degraded) =
        agentic_rewrite::load_runtime_resources_with_status(config);

    (
        RuntimeTextResources {
            rules,
            runtime_policy,
        },
        rules_degraded || policy_degraded,
    )
}

pub(crate) fn build_rewrite_plan(
    config: &Config,
    resources: &RuntimeTextResources,
    transcript: &Transcript,
    typing_context: Option<&TypingContext>,
    recent_session: Option<&EligibleSessionEntry>,
) -> RewritePlan {
    let rules = resources.rules.clone();
    let local_model_path = resolve_rewrite_model_path(config);
    let mut rewrite_transcript = personalization::build_rewrite_transcript(transcript, &rules);
    rewrite_transcript.typing_context = typing_context.and_then(session::to_rewrite_typing_context);
    agentic_rewrite::apply_runtime_policy_with_resources(
        config,
        &mut rewrite_transcript,
        &resources.runtime_policy,
    );
    let session_plan = session::build_backtrack_plan(&rewrite_transcript, recent_session);
    let mut fallback_text = base_text(config, transcript);
    if let Some(candidate) = rewrite_transcript
        .rewrite_candidates
        .iter()
        .find(|candidate| candidate.kind == RewriteCandidateKind::StructuredLiteral)
    {
        let candidate_text = candidate.text.as_str();
        let prefer_structured_fallback = [
            Some(transcript.raw_text.as_str()),
            Some(rewrite_transcript.correction_aware_text.as_str()),
            rewrite_transcript.aggressive_correction_text.as_deref(),
        ]
        .into_iter()
        .flatten()
        .any(|text| structured_text::output_matches_candidate(text, candidate_text));
        if prefer_structured_fallback {
            fallback_text = candidate.text.clone();
        }
    }
    if session_plan.recommended.as_ref().is_some_and(|candidate| {
        matches!(
            candidate.kind,
            RewriteSessionBacktrackCandidateKind::ReplaceLastEntry
        )
    }) {
        if let Some(explicit_followup_text) =
            cleanup::explicit_followup_replacement(&rewrite_transcript.raw_text)
        {
            fallback_text = explicit_followup_text;
        }
    }
    rewrite_transcript.edit_context.has_recent_same_focus_entry = recent_session.is_some();
    rewrite_transcript
        .edit_context
        .recommended_session_action_is_replace =
        session_plan.recommended.as_ref().is_some_and(|candidate| {
            matches!(
                candidate.kind,
                RewriteSessionBacktrackCandidateKind::ReplaceLastEntry
            )
        });
    rewrite_transcript.recent_session_entries = session_plan.recent_entries.clone();
    rewrite_transcript.session_backtrack_candidates = session_plan.candidates.clone();
    rewrite_transcript.recommended_session_candidate = session_plan.recommended.clone();
    tracing::debug!(
        mode = config.postprocess.mode.as_str(),
        edit_hypotheses = rewrite_transcript.edit_hypotheses.len(),
        rewrite_candidates = rewrite_transcript.rewrite_candidates.len(),
        session_backtrack_candidates = rewrite_transcript.session_backtrack_candidates.len(),
        recommended_candidate = rewrite_transcript
            .recommended_candidate
            .as_ref()
            .map(|candidate| candidate.text.as_str())
            .unwrap_or(""),
        "prepared rewrite request"
    );

    RewritePlan {
        custom_instructions: personalization::custom_instructions(&rules).map(str::to_string),
        local_model_path,
        operation: recommended_operation(&rewrite_transcript),
        had_edit_cues: !rewrite_transcript.edit_signals.is_empty()
            || !rewrite_transcript.edit_hypotheses.is_empty(),
        recommended_candidate: rewrite_transcript
            .recommended_session_candidate
            .as_ref()
            .map(|candidate| candidate.text.clone())
            .or_else(|| {
                rewrite_transcript
                    .recommended_candidate
                    .as_ref()
                    .map(|candidate| candidate.text.clone())
            }),
        rules,
        fallback_text,
        rewrite_transcript,
    }
}

fn base_text(config: &Config, transcript: &Transcript) -> String {
    match config.postprocess.mode {
        PostprocessMode::LegacyBasic => cleanup::clean_transcript(transcript, &config.cleanup),
        PostprocessMode::Rewrite => cleanup::correction_aware_text(transcript),
        PostprocessMode::Raw => raw_text(transcript),
    }
}

fn recommended_operation(rewrite_transcript: &RewriteTranscript) -> FinalizedOperation {
    rewrite_transcript
        .recommended_session_candidate
        .as_ref()
        .and_then(|candidate| {
            matches!(
                candidate.kind,
                RewriteSessionBacktrackCandidateKind::ReplaceLastEntry
            )
            .then_some(FinalizedOperation::ReplaceLastEntry {
                entry_id: candidate.entry_id?,
                delete_graphemes: candidate.delete_graphemes,
            })
        })
        .unwrap_or(FinalizedOperation::Append)
}

#[cfg(test)]
mod tests {
    use super::{build_rewrite_plan, load_runtime_text_resources};
    use crate::config::{Config, PostprocessMode};
    use crate::context::SurfaceKind;
    use crate::postprocess::finalize::FinalizedOperation;
    use crate::session::{EligibleSessionEntry, SessionEntry, SessionRewriteSummary};
    use crate::transcribe::Transcript;

    #[test]
    fn build_rewrite_plan_uses_explicit_followup_replacement_for_replace_fallback() {
        let mut config = Config::default();
        config.postprocess.mode = PostprocessMode::Rewrite;

        let transcript = Transcript {
            raw_text: "srajvat, hi".into(),
            detected_language: Some("en".into()),
            segments: Vec::new(),
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

        let plan = build_rewrite_plan(
            &config,
            &load_runtime_text_resources(&config),
            &transcript,
            None,
            Some(&recent),
        );
        assert_eq!(plan.fallback_text, "Hi");
        assert_eq!(plan.recommended_candidate.as_deref(), Some("Hi"));
        assert_eq!(
            plan.operation,
            FinalizedOperation::ReplaceLastEntry {
                entry_id: 7,
                delete_graphemes: 11,
            }
        );
    }

    #[test]
    fn build_rewrite_plan_prefers_structured_literal_for_fallback() {
        let mut config = Config::default();
        config.postprocess.mode = PostprocessMode::Rewrite;

        let transcript = Transcript {
            raw_text: "portfolio. Notes. Supply is the URL".into(),
            detected_language: Some("en".into()),
            segments: Vec::new(),
        };

        let plan = build_rewrite_plan(
            &config,
            &load_runtime_text_resources(&config),
            &transcript,
            None,
            None,
        );
        assert_eq!(plan.fallback_text, "portfolio.notes.supply");
    }

    #[test]
    fn build_rewrite_plan_keeps_full_fallback_when_structured_text_is_embedded() {
        let mut config = Config::default();
        config.postprocess.mode = PostprocessMode::Rewrite;

        let transcript = Transcript {
            raw_text: "Check portfolio. Notes. Supply tomorrow".into(),
            detected_language: Some("en".into()),
            segments: Vec::new(),
        };

        let plan = build_rewrite_plan(
            &config,
            &load_runtime_text_resources(&config),
            &transcript,
            None,
            None,
        );
        assert_eq!(
            plan.fallback_text,
            "Check portfolio. Notes. Supply tomorrow"
        );
    }

    #[test]
    fn build_rewrite_plan_keeps_possessive_structured_literal_fallback() {
        let mut config = Config::default();
        config.postprocess.mode = PostprocessMode::Rewrite;

        let transcript = Transcript {
            raw_text: "example.com's".into(),
            detected_language: Some("en".into()),
            segments: Vec::new(),
        };

        let plan = build_rewrite_plan(
            &config,
            &load_runtime_text_resources(&config),
            &transcript,
            None,
            None,
        );
        assert_eq!(plan.fallback_text, crate::cleanup::correction_aware_text(&transcript));
    }
}

use std::path::PathBuf;

use crate::agentic_rewrite;
use crate::cleanup;
use crate::config::{Config, PostprocessMode};
use crate::context::TypingContext;
use crate::personalization::{self, PersonalizationRules};
use crate::rewrite_model;
use crate::rewrite_protocol::{RewriteSessionBacktrackCandidateKind, RewriteTranscript};
use crate::session::{self, EligibleSessionEntry};
use crate::transcribe::Transcript;

use super::finalize::FinalizedOperation;

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

pub(crate) fn load_runtime_rules(config: &Config) -> PersonalizationRules {
    match personalization::load_rules(config) {
        Ok(rules) => rules,
        Err(err) => {
            tracing::warn!("failed to load personalization rules: {err}");
            PersonalizationRules::default()
        }
    }
}

pub(crate) fn build_rewrite_plan(
    config: &Config,
    transcript: &Transcript,
    typing_context: Option<&TypingContext>,
    recent_session: Option<&EligibleSessionEntry>,
) -> RewritePlan {
    let rules = load_runtime_rules(config);
    let fallback_text = base_text(config, transcript);
    let local_model_path = resolve_rewrite_model_path(config);
    let mut rewrite_transcript = personalization::build_rewrite_transcript(transcript, &rules);
    rewrite_transcript.typing_context = typing_context.and_then(session::to_rewrite_typing_context);
    agentic_rewrite::apply_runtime_policy(config, &mut rewrite_transcript);
    let session_plan = session::build_backtrack_plan(&rewrite_transcript, recent_session);
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

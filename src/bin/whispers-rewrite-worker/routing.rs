use crate::rewrite_protocol::{
    RewriteEditHypothesisMatchSource, RewriteEditSignalStrength, RewriteTranscript,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RewriteRoute {
    Fast,
    ResolvedCorrection,
    SessionCandidateAdjudication,
    CandidateAdjudication,
}

pub(super) fn rewrite_route(transcript: &RewriteTranscript) -> RewriteRoute {
    if has_session_backtrack_candidate(transcript) {
        RewriteRoute::SessionCandidateAdjudication
    } else if requires_candidate_adjudication(transcript) {
        RewriteRoute::CandidateAdjudication
    } else if transcript.correction_aware_text.trim() != transcript.raw_text.trim() {
        RewriteRoute::ResolvedCorrection
    } else {
        RewriteRoute::Fast
    }
}

pub(super) fn requires_candidate_adjudication(transcript: &RewriteTranscript) -> bool {
    !transcript.edit_signals.is_empty() || !transcript.edit_hypotheses.is_empty()
}

pub(super) fn has_strong_explicit_edit_cue(transcript: &RewriteTranscript) -> bool {
    transcript.edit_hypotheses.iter().any(|hypothesis| {
        hypothesis.strength == RewriteEditSignalStrength::Strong
            && matches!(
                hypothesis.match_source,
                RewriteEditHypothesisMatchSource::Exact | RewriteEditHypothesisMatchSource::Alias
            )
    })
}

pub(super) fn has_session_backtrack_candidate(transcript: &RewriteTranscript) -> bool {
    transcript.recommended_session_candidate.is_some()
        || !transcript.session_backtrack_candidates.is_empty()
}

pub(super) fn has_policy_context(transcript: &RewriteTranscript) -> bool {
    transcript.policy_context.is_active()
}

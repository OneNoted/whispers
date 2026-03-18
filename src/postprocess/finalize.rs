use std::time::Duration;
use std::time::Instant;

use crate::agentic_rewrite;
use crate::config::{Config, PostprocessMode, RewriteBackend, RewriteFallback};
use crate::context::TypingContext;
use crate::personalization::{self, PersonalizationRules};
use crate::rewrite_protocol::{RewriteCorrectionPolicy, RewriteTranscript};
use crate::rewrite_worker::RewriteService;
use crate::session::{EligibleSessionEntry, SessionRewriteSummary};
use crate::transcribe::Transcript;

use super::{execution, planning};

const FEEDBACK_DRAIN_DELAY: Duration = Duration::from_millis(150);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinalizedOperation {
    Append,
    ReplaceLastEntry {
        entry_id: u64,
        delete_graphemes: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinalizedTranscript {
    pub text: String,
    pub operation: FinalizedOperation,
    pub rewrite_summary: SessionRewriteSummary,
}

pub async fn finalize_transcript(
    config: &Config,
    transcript: Transcript,
    rewrite_service: Option<&RewriteService>,
    typing_context: Option<&TypingContext>,
    recent_session: Option<&EligibleSessionEntry>,
) -> FinalizedTranscript {
    let started = Instant::now();
    let finalized = match config.postprocess.mode {
        PostprocessMode::Raw => {
            let rules = planning::load_runtime_rules(config);
            finalize_plain_text(
                planning::raw_text(&transcript),
                SessionRewriteSummary {
                    had_edit_cues: false,
                    rewrite_used: false,
                    recommended_candidate: None,
                },
                &rules,
            )
        }
        PostprocessMode::LegacyBasic => {
            let rules = planning::load_runtime_rules(config);
            finalize_plain_text(
                crate::cleanup::clean_transcript(&transcript, &config.cleanup),
                SessionRewriteSummary {
                    had_edit_cues: false,
                    rewrite_used: false,
                    recommended_candidate: None,
                },
                &rules,
            )
        }
        PostprocessMode::Rewrite => {
            finalize_rewrite_plan_or_fallback(
                config,
                rewrite_service,
                planning::build_rewrite_plan(config, &transcript, typing_context, recent_session),
            )
            .await
        }
    };
    tracing::info!(
        elapsed_ms = started.elapsed().as_millis(),
        mode = config.postprocess.mode.as_str(),
        rewrite_used = finalized.rewrite_summary.rewrite_used,
        output_chars = finalized.text.len(),
        "finalize_transcript finished"
    );
    finalized
}

pub async fn wait_for_feedback_drain() {
    tokio::time::sleep(FEEDBACK_DRAIN_DELAY).await;
}

async fn finalize_rewrite_plan_or_fallback(
    config: &Config,
    rewrite_service: Option<&RewriteService>,
    plan: planning::RewritePlan,
) -> FinalizedTranscript {
    let local_rewrite_available = crate::rewrite::local_rewrite_available();
    let local_backend_requested = config.rewrite.backend == RewriteBackend::Local;

    if local_backend_requested && !local_rewrite_available {
        tracing::warn!(
            "local rewrite backend requested, but this build does not include local rewrite support; using fallback"
        );
        return finalize_unavailable_rewrite_fallback(plan);
    }

    let local_rewrite_required = local_backend_requested
        || (config.rewrite.fallback == RewriteFallback::Local && local_rewrite_available);
    if local_rewrite_required && plan.local_model_path.is_none() {
        tracing::warn!(
            "rewrite backend requires a local model but none is configured; using fallback"
        );
        return finalize_unavailable_rewrite_fallback(plan);
    }

    let rewrite_result = execution::execute_rewrite(config, rewrite_service, &plan).await;
    finalize_rewrite_attempt(config, plan, rewrite_result)
}

fn finalize_rewrite_attempt(
    config: &Config,
    plan: planning::RewritePlan,
    rewrite_result: crate::error::Result<String>,
) -> FinalizedTranscript {
    let (base, rewrite_used) = match rewrite_result {
        Ok(text) if rewrite_output_accepted(config, &plan.rewrite_transcript, &text) => {
            tracing::debug!(
                output_len = text.len(),
                mode = config.postprocess.mode.as_str(),
                "rewrite applied successfully"
            );
            (text, true)
        }
        Ok(text) if text.trim().is_empty() => {
            tracing::warn!("rewrite model returned empty text; using fallback");
            (plan.fallback_text, false)
        }
        Ok(text) => {
            tracing::warn!(
                mode = config.postprocess.mode.as_str(),
                output_len = text.len(),
                "rewrite output failed acceptance guard; using fallback"
            );
            (plan.fallback_text, false)
        }
        Err(err) => {
            tracing::warn!("rewrite failed: {err}; using fallback");
            (plan.fallback_text, false)
        }
    };

    finalize_plain_text(
        base,
        SessionRewriteSummary {
            had_edit_cues: plan.had_edit_cues,
            rewrite_used,
            recommended_candidate: plan.recommended_candidate,
        },
        &plan.rules,
    )
    .with_operation(plan.operation)
}

fn finalize_plain_text(
    text: String,
    rewrite_summary: SessionRewriteSummary,
    rules: &PersonalizationRules,
) -> FinalizedTranscript {
    FinalizedTranscript {
        text: personalization::finalize_text(&text, rules),
        operation: FinalizedOperation::Append,
        rewrite_summary,
    }
}

fn finalize_unavailable_rewrite_fallback(plan: planning::RewritePlan) -> FinalizedTranscript {
    finalize_plain_text(
        plan.fallback_text,
        SessionRewriteSummary {
            had_edit_cues: plan.had_edit_cues,
            rewrite_used: false,
            recommended_candidate: plan.recommended_candidate,
        },
        &plan.rules,
    )
    .with_operation(plan.operation)
}

impl FinalizedTranscript {
    fn with_operation(mut self, operation: FinalizedOperation) -> Self {
        self.operation = operation;
        self
    }
}

fn rewrite_output_accepted(
    _config: &Config,
    rewrite_transcript: &RewriteTranscript,
    text: &str,
) -> bool {
    if text.trim().is_empty() {
        return false;
    }

    match rewrite_transcript.policy_context.correction_policy {
        RewriteCorrectionPolicy::Conservative => {
            agentic_rewrite::conservative_output_allowed(rewrite_transcript, text)
        }
        RewriteCorrectionPolicy::Balanced | RewriteCorrectionPolicy::Aggressive => true,
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::rewrite_protocol::{
        RewriteCandidate, RewriteCandidateKind, RewritePolicyContext, RewriteTranscript,
    };

    fn plan_config(mode: PostprocessMode, backend: RewriteBackend) -> Config {
        let mut config = Config::default();
        config.postprocess.mode = mode;
        config.rewrite.backend = backend;
        config
    }

    fn rewrite_plan() -> planning::RewritePlan {
        planning::RewritePlan {
            rules: PersonalizationRules::default(),
            fallback_text: "fallback text".into(),
            rewrite_transcript: RewriteTranscript {
                raw_text: "raw text".into(),
                correction_aware_text: "fallback text".into(),
                aggressive_correction_text: None,
                detected_language: Some("en".into()),
                typing_context: None,
                recent_session_entries: Vec::new(),
                session_backtrack_candidates: Vec::new(),
                recommended_session_candidate: None,
                segments: Vec::new(),
                edit_intents: Vec::new(),
                edit_signals: Vec::new(),
                edit_hypotheses: Vec::new(),
                rewrite_candidates: vec![RewriteCandidate {
                    kind: RewriteCandidateKind::ConservativeCorrection,
                    text: "allowed rewrite".into(),
                }],
                recommended_candidate: None,
                edit_context: Default::default(),
                policy_context: RewritePolicyContext::default(),
            },
            custom_instructions: None,
            local_model_path: Some(PathBuf::from("/tmp/model.gguf")),
            operation: FinalizedOperation::Append,
            had_edit_cues: false,
            recommended_candidate: Some("allowed rewrite".into()),
        }
    }

    #[tokio::test]
    #[cfg(not(feature = "local-rewrite"))]
    async fn local_rewrite_unavailable_build_falls_back_to_plain_text() {
        let config = plan_config(PostprocessMode::Rewrite, RewriteBackend::Local);
        let mut plan = rewrite_plan();
        plan.operation = FinalizedOperation::ReplaceLastEntry {
            entry_id: 11,
            delete_graphemes: 6,
        };

        let finalized = finalize_rewrite_plan_or_fallback(&config, None, plan).await;

        assert_eq!(finalized.text, "fallback text");
        assert!(!finalized.rewrite_summary.rewrite_used);
        assert_eq!(
            finalized.operation,
            FinalizedOperation::ReplaceLastEntry {
                entry_id: 11,
                delete_graphemes: 6,
            }
        );
    }

    #[tokio::test]
    #[cfg(feature = "local-rewrite")]
    async fn missing_local_model_falls_back_to_plain_text() {
        let config = plan_config(PostprocessMode::Rewrite, RewriteBackend::Local);
        let mut plan = rewrite_plan();
        plan.local_model_path = None;

        let finalized = finalize_rewrite_plan_or_fallback(&config, None, plan).await;

        assert_eq!(finalized.text, "fallback text");
        assert!(!finalized.rewrite_summary.rewrite_used);
    }

    #[test]
    fn conservative_agentic_rejection_falls_back_to_precomputed_text() {
        let mut config = plan_config(PostprocessMode::Rewrite, RewriteBackend::Cloud);
        config.rewrite.fallback = RewriteFallback::None;
        let mut plan = rewrite_plan();
        plan.rewrite_transcript.policy_context.correction_policy =
            RewriteCorrectionPolicy::Conservative;

        let finalized = finalize_rewrite_attempt(&config, plan, Ok("rejected rewrite".into()));

        assert_eq!(finalized.text, "fallback text");
        assert!(!finalized.rewrite_summary.rewrite_used);
    }
}

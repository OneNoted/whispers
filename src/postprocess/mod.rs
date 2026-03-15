mod execution;
mod planning;

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

pub use execution::{prepare_rewrite_service, prewarm_rewrite_service};
pub use planning::raw_text;

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
        PostprocessMode::AdvancedLocal | PostprocessMode::AgenticRewrite => {
            rewrite_plan_or_fallback(
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

async fn rewrite_plan_or_fallback(
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
        return finalize_plain_text(
            plan.fallback_text,
            SessionRewriteSummary {
                had_edit_cues: false,
                rewrite_used: false,
                recommended_candidate: None,
            },
            &plan.rules,
        );
    }

    let local_rewrite_required = local_backend_requested
        || (config.rewrite.fallback == RewriteFallback::Local && local_rewrite_available);
    if local_rewrite_required && plan.local_model_path.is_none() {
        tracing::warn!(
            "rewrite backend requires a local model but none is configured; using fallback"
        );
        return finalize_plain_text(
            plan.fallback_text,
            SessionRewriteSummary {
                had_edit_cues: false,
                rewrite_used: false,
                recommended_candidate: None,
            },
            &plan.rules,
        );
    }

    if let Some(text) = plan.deterministic_replacement_text.clone() {
        tracing::debug!(
            output_len = text.len(),
            mode = config.postprocess.mode.as_str(),
            "using deterministic session replacement"
        );
        return finalize_plain_text(
            text,
            SessionRewriteSummary {
                had_edit_cues: plan.had_edit_cues,
                rewrite_used: false,
                recommended_candidate: plan.recommended_candidate.clone(),
            },
            &plan.rules,
        )
        .with_operation(plan.operation.clone());
    }

    let rewrite_result = execution::execute_rewrite(config, rewrite_service, &plan).await;

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

impl FinalizedTranscript {
    fn with_operation(mut self, operation: FinalizedOperation) -> Self {
        self.operation = operation;
        self
    }
}

fn rewrite_output_accepted(
    config: &Config,
    rewrite_transcript: &RewriteTranscript,
    text: &str,
) -> bool {
    if text.trim().is_empty() {
        return false;
    }

    if config.postprocess.mode != PostprocessMode::AgenticRewrite {
        return true;
    }

    match rewrite_transcript.policy_context.correction_policy {
        RewriteCorrectionPolicy::Conservative => {
            agentic_rewrite::conservative_output_allowed(rewrite_transcript, text)
        }
        RewriteCorrectionPolicy::Balanced | RewriteCorrectionPolicy::Aggressive => true,
    }
}

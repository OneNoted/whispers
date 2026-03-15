mod planning;

use std::path::Path;
use std::time::Duration;
use std::time::Instant;

use crate::agentic_rewrite;
use crate::cloud;
use crate::config::{Config, PostprocessMode, RewriteBackend, RewriteFallback};
use crate::context::TypingContext;
use crate::personalization::{self, PersonalizationRules};
use crate::rewrite_protocol::{RewriteCorrectionPolicy, RewriteTranscript};
use crate::rewrite_worker::{self, RewriteService};
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

pub fn prepare_rewrite_service(config: &Config) -> Option<RewriteService> {
    if !config.postprocess.mode.uses_rewrite() {
        return None;
    }

    if config.rewrite.backend != RewriteBackend::Local
        && config.rewrite.fallback != RewriteFallback::Local
    {
        return None;
    }

    if !crate::rewrite::local_rewrite_available() {
        return None;
    }

    let model_path = planning::resolve_rewrite_model_path(config)?;
    Some(rewrite_worker::RewriteService::new(
        &config.rewrite,
        &model_path,
    ))
}

pub fn prewarm_rewrite_service(service: &RewriteService, phase: &str) {
    match service.prewarm() {
        Ok(()) => tracing::info!("prewarming rewrite worker via {}", phase,),
        Err(err) => tracing::warn!("failed to prewarm rewrite worker: {err}"),
    }
}

async fn rewrite_plan_or_fallback(
    config: &Config,
    rewrite_service: Option<&RewriteService>,
    plan: planning::RewritePlan,
) -> FinalizedTranscript {
    let planning::RewritePlan {
        rules,
        fallback_text,
        rewrite_transcript,
        custom_instructions,
        local_model_path,
        operation,
        had_edit_cues,
        recommended_candidate,
        deterministic_replacement_text,
    } = plan;
    let local_rewrite_available = crate::rewrite::local_rewrite_available();
    let local_backend_requested = config.rewrite.backend == RewriteBackend::Local;

    if local_backend_requested && !local_rewrite_available {
        tracing::warn!(
            "local rewrite backend requested, but this build does not include local rewrite support; using fallback"
        );
        return finalize_plain_text(
            fallback_text,
            SessionRewriteSummary {
                had_edit_cues: false,
                rewrite_used: false,
                recommended_candidate: None,
            },
            &rules,
        );
    }

    let local_rewrite_required = local_backend_requested
        || (config.rewrite.fallback == RewriteFallback::Local && local_rewrite_available);
    if local_rewrite_required && local_model_path.is_none() {
        tracing::warn!(
            "rewrite backend requires a local model but none is configured; using fallback"
        );
        return finalize_plain_text(
            fallback_text,
            SessionRewriteSummary {
                had_edit_cues: false,
                rewrite_used: false,
                recommended_candidate: None,
            },
            &rules,
        );
    }

    if let Some(text) = deterministic_replacement_text {
        tracing::debug!(
            output_len = text.len(),
            mode = config.postprocess.mode.as_str(),
            "using deterministic session replacement"
        );
        return finalize_plain_text(
            text,
            SessionRewriteSummary {
                had_edit_cues,
                rewrite_used: false,
                recommended_candidate,
            },
            &rules,
        )
        .with_operation(operation);
    }

    let rewrite_result = match config.rewrite.backend {
        RewriteBackend::Local => {
            local_rewrite_result(
                config,
                rewrite_service,
                local_model_path
                    .as_ref()
                    .expect("local rewrite requires resolved model path"),
                &rewrite_transcript,
                custom_instructions.as_deref(),
            )
            .await
        }
        RewriteBackend::Cloud => {
            let cloud_service = cloud::CloudService::new(config);
            let cloud_result = match cloud_service {
                Ok(service) => {
                    service
                        .rewrite_transcript(
                            config,
                            &rewrite_transcript,
                            custom_instructions.as_deref(),
                        )
                        .await
                }
                Err(err) => Err(err),
            };
            match cloud_result {
                Ok(text) => Ok(text),
                Err(err)
                    if config.rewrite.fallback == RewriteFallback::Local
                        && local_rewrite_available =>
                {
                    tracing::warn!("cloud rewrite failed: {err}; falling back to local rewrite");
                    local_rewrite_result(
                        config,
                        rewrite_service,
                        local_model_path
                            .as_ref()
                            .expect("local rewrite fallback requires resolved model path"),
                        &rewrite_transcript,
                        custom_instructions.as_deref(),
                    )
                    .await
                }
                Err(err) => Err(err),
            }
        }
    };

    let (base, rewrite_used) = match rewrite_result {
        Ok(text) if rewrite_output_accepted(config, &rewrite_transcript, &text) => {
            tracing::debug!(
                output_len = text.len(),
                mode = config.postprocess.mode.as_str(),
                "rewrite applied successfully"
            );
            (text, true)
        }
        Ok(text) if text.trim().is_empty() => {
            tracing::warn!("rewrite model returned empty text; using fallback");
            (fallback_text, false)
        }
        Ok(text) => {
            tracing::warn!(
                mode = config.postprocess.mode.as_str(),
                output_len = text.len(),
                "rewrite output failed acceptance guard; using fallback"
            );
            (fallback_text, false)
        }
        Err(err) => {
            tracing::warn!("rewrite failed: {err}; using fallback");
            (fallback_text, false)
        }
    };

    finalize_plain_text(
        base,
        SessionRewriteSummary {
            had_edit_cues,
            rewrite_used,
            recommended_candidate,
        },
        &rules,
    )
    .with_operation(operation)
}

async fn local_rewrite_result(
    config: &Config,
    rewrite_service: Option<&RewriteService>,
    model_path: &Path,
    rewrite_transcript: &crate::rewrite_protocol::RewriteTranscript,
    custom_instructions: Option<&str>,
) -> crate::error::Result<String> {
    if !crate::rewrite::local_rewrite_available() {
        return Err(crate::error::WhsprError::Rewrite(
            "local rewrite is unavailable in this build; rebuild with --features local-rewrite"
                .into(),
        ));
    }

    if let Some(service) = rewrite_service {
        rewrite_worker::rewrite_with_service(
            service,
            &config.rewrite,
            rewrite_transcript,
            custom_instructions,
        )
        .await
    } else {
        rewrite_worker::rewrite_transcript(
            &config.rewrite,
            model_path,
            rewrite_transcript,
            custom_instructions,
        )
        .await
    }
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

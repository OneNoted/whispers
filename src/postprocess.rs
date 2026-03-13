use std::path::{Path, PathBuf};
use std::time::Duration;
use std::time::Instant;

use crate::cleanup;
use crate::cloud;
use crate::config::{Config, PostprocessMode, RewriteBackend, RewriteFallback};
use crate::context::TypingContext;
use crate::personalization::{self, PersonalizationRules};
use crate::rewrite_model;
use crate::rewrite_protocol::RewriteSessionBacktrackCandidateKind;
use crate::rewrite_worker::{self, RewriteService};
use crate::session::{self, EligibleSessionEntry, SessionRewriteSummary};
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

pub fn raw_text(transcript: &Transcript) -> String {
    transcript.raw_text.trim().to_string()
}

fn base_text(config: &Config, transcript: &Transcript) -> String {
    match config.postprocess.mode {
        PostprocessMode::LegacyBasic => cleanup::clean_transcript(transcript, &config.cleanup),
        PostprocessMode::AdvancedLocal => cleanup::correction_aware_text(transcript),
        PostprocessMode::Raw => raw_text(transcript),
    }
}

pub fn resolve_rewrite_model_path(config: &Config) -> Option<PathBuf> {
    if let Some(path) = config.resolved_rewrite_model_path() {
        return Some(path);
    }

    rewrite_model::selected_model_path(&config.rewrite.selected_model)
}

pub async fn finalize_transcript(
    config: &Config,
    transcript: Transcript,
    rewrite_service: Option<&RewriteService>,
    typing_context: Option<&TypingContext>,
    recent_session: Option<&EligibleSessionEntry>,
) -> FinalizedTranscript {
    let started = Instant::now();
    let rules = load_runtime_rules(config);
    let finalized = match config.postprocess.mode {
        PostprocessMode::Raw => finalize_plain_text(
            raw_text(&transcript),
            SessionRewriteSummary {
                had_edit_cues: false,
                rewrite_used: false,
                recommended_candidate: None,
            },
            &rules,
        ),
        PostprocessMode::LegacyBasic => finalize_plain_text(
            cleanup::clean_transcript(&transcript, &config.cleanup),
            SessionRewriteSummary {
                had_edit_cues: false,
                rewrite_used: false,
                recommended_candidate: None,
            },
            &rules,
        ),
        PostprocessMode::AdvancedLocal => {
            rewrite_transcript_or_fallback(
                config,
                &transcript,
                rewrite_service,
                &rules,
                typing_context,
                recent_session,
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
    if config.postprocess.mode != PostprocessMode::AdvancedLocal {
        return None;
    }

    if config.rewrite.backend != RewriteBackend::Local
        && config.rewrite.fallback != RewriteFallback::Local
    {
        return None;
    }

    let model_path = resolve_rewrite_model_path(config)?;
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

async fn rewrite_transcript_or_fallback(
    config: &Config,
    transcript: &Transcript,
    rewrite_service: Option<&RewriteService>,
    rules: &PersonalizationRules,
    typing_context: Option<&TypingContext>,
    recent_session: Option<&EligibleSessionEntry>,
) -> FinalizedTranscript {
    let fallback = base_text(config, transcript);
    let local_model_path = resolve_rewrite_model_path(config);
    let local_rewrite_required = config.rewrite.backend == RewriteBackend::Local
        || config.rewrite.fallback == RewriteFallback::Local;
    if local_rewrite_required && local_model_path.is_none() {
        tracing::warn!(
            "rewrite backend requires a local model but none is configured; using fallback"
        );
        return finalize_plain_text(
            fallback,
            SessionRewriteSummary {
                had_edit_cues: false,
                rewrite_used: false,
                recommended_candidate: None,
            },
            rules,
        );
    }
    let mut rewrite_transcript = personalization::build_rewrite_transcript(transcript, rules);
    rewrite_transcript.typing_context = typing_context.and_then(session::to_rewrite_typing_context);
    let session_plan = session::build_backtrack_plan(&rewrite_transcript, recent_session);
    rewrite_transcript.recent_session_entries = session_plan.recent_entries.clone();
    rewrite_transcript.session_backtrack_candidates = session_plan.candidates.clone();
    rewrite_transcript.recommended_session_candidate = session_plan.recommended.clone();
    tracing::debug!(
        edit_hypotheses = rewrite_transcript.edit_hypotheses.len(),
        rewrite_candidates = rewrite_transcript.rewrite_candidates.len(),
        session_backtrack_candidates = rewrite_transcript.session_backtrack_candidates.len(),
        recommended_candidate = rewrite_transcript
            .recommended_candidate
            .as_ref()
            .map(|candidate| candidate.text.as_str())
            .unwrap_or(""),
        "advanced_local prepared rewrite request"
    );
    let custom_instructions = personalization::custom_instructions(rules);
    let deterministic_session_replacement = session_plan.deterministic_replacement_text.clone();

    if let Some(text) = deterministic_session_replacement {
        tracing::debug!(
            output_len = text.len(),
            "advanced_local using deterministic session replacement"
        );
        let operation = rewrite_transcript
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
            .unwrap_or(FinalizedOperation::Append);
        return finalize_plain_text(
            text,
            SessionRewriteSummary {
                had_edit_cues: !rewrite_transcript.edit_signals.is_empty()
                    || !rewrite_transcript.edit_hypotheses.is_empty(),
                rewrite_used: false,
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
            },
            rules,
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
                custom_instructions,
            )
            .await
        }
        RewriteBackend::Cloud => {
            let cloud_service = cloud::CloudService::new(config);
            let cloud_result = match cloud_service {
                Ok(service) => {
                    service
                        .rewrite_transcript(config, &rewrite_transcript, custom_instructions)
                        .await
                }
                Err(err) => Err(err),
            };
            match cloud_result {
                Ok(text) => Ok(text),
                Err(err) if config.rewrite.fallback == RewriteFallback::Local => {
                    tracing::warn!("cloud rewrite failed: {err}; falling back to local rewrite");
                    local_rewrite_result(
                        config,
                        rewrite_service,
                        local_model_path
                            .as_ref()
                            .expect("local rewrite fallback requires resolved model path"),
                        &rewrite_transcript,
                        custom_instructions,
                    )
                    .await
                }
                Err(err) => Err(err),
            }
        }
    };

    let had_edit_cues = !rewrite_transcript.edit_signals.is_empty()
        || !rewrite_transcript.edit_hypotheses.is_empty();
    let recommended_candidate = rewrite_transcript
        .recommended_session_candidate
        .as_ref()
        .map(|candidate| candidate.text.clone())
        .or_else(|| {
            rewrite_transcript
                .recommended_candidate
                .as_ref()
                .map(|candidate| candidate.text.clone())
        });

    let (base, rewrite_used) = match rewrite_result {
        Ok(text) if !text.trim().is_empty() => {
            tracing::debug!(
                output_len = text.len(),
                "advanced_local rewrite applied successfully"
            );
            (text, true)
        }
        Ok(_) => {
            tracing::warn!("rewrite model returned empty text; using fallback");
            (fallback, false)
        }
        Err(err) => {
            tracing::warn!("rewrite failed: {err}; using fallback");
            (fallback, false)
        }
    };
    let operation = rewrite_transcript
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
        .unwrap_or(FinalizedOperation::Append);

    finalize_plain_text(
        base,
        SessionRewriteSummary {
            had_edit_cues,
            rewrite_used,
            recommended_candidate,
        },
        rules,
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

fn load_runtime_rules(config: &Config) -> PersonalizationRules {
    match personalization::load_rules(config) {
        Ok(rules) => rules,
        Err(err) => {
            tracing::warn!("failed to load personalization rules: {err}");
            PersonalizationRules::default()
        }
    }
}

impl FinalizedTranscript {
    fn with_operation(mut self, operation: FinalizedOperation) -> Self {
        self.operation = operation;
        self
    }
}

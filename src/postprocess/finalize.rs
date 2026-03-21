use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Duration;
use std::time::Instant;

use crate::agentic_rewrite;
use crate::config::{Config, PostprocessMode, RewriteBackend, RewriteFallback};
use crate::context::TypingContext;
use crate::error::WhsprError;
use crate::personalization::{self, PersonalizationRules};
use crate::rewrite_protocol::{RewriteCandidateKind, RewriteCorrectionPolicy, RewriteTranscript};
use crate::rewrite_worker::RewriteService;
use crate::session::{EligibleSessionEntry, SessionRewriteSummary};
use crate::structured_text;
use crate::transcribe::Transcript;

use super::{execution, planning};

const FEEDBACK_DRAIN_DELAY: Duration = Duration::from_millis(150);
const REWRITE_PLAN_TIMEOUT: Duration = Duration::from_secs(5);

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
            match build_rewrite_plan_with_timeout(
                config,
                &transcript,
                typing_context,
                recent_session,
            )
            .await
            {
                Ok(plan) => finalize_rewrite_plan_or_fallback(config, rewrite_service, plan).await,
                Err(err) => finalize_rewrite_planning_failure(&transcript, &err),
            }
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

async fn build_rewrite_plan_with_timeout(
    config: &Config,
    transcript: &Transcript,
    typing_context: Option<&TypingContext>,
    recent_session: Option<&EligibleSessionEntry>,
) -> crate::error::Result<planning::RewritePlan> {
    let config = config.clone();
    let transcript = transcript.clone();
    let typing_context = typing_context.cloned();
    let recent_session = recent_session.cloned();

    run_rewrite_thread_with_timeout(REWRITE_PLAN_TIMEOUT, "rewrite planning", move || {
        planning::build_rewrite_plan(
            &config,
            &transcript,
            typing_context.as_ref(),
            recent_session.as_ref(),
        )
    })
    .await
}

async fn run_rewrite_thread_with_timeout<T, F>(
    timeout: Duration,
    phase: &'static str,
    task: F,
) -> crate::error::Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    let (tx, rx) = tokio::sync::oneshot::channel();
    std::thread::Builder::new()
        .name(format!("whispers-{phase}"))
        .spawn(move || {
            let result = catch_unwind(AssertUnwindSafe(task))
                .map_err(|_| format!("{phase} thread panicked"));
            let _ = tx.send(result);
        })
        .map_err(|err| WhsprError::Rewrite(format!("failed to spawn {phase} thread: {err}")))?;

    let result = tokio::time::timeout(timeout, rx)
        .await
        .map_err(|_| {
            WhsprError::Rewrite(format!("{phase} timed out after {}ms", timeout.as_millis()))
        })?
        .map_err(|_| WhsprError::Rewrite(format!("{phase} thread exited without a result")))?;

    result.map_err(WhsprError::Rewrite)
}

fn finalize_rewrite_planning_failure(
    transcript: &Transcript,
    err: &crate::error::WhsprError,
) -> FinalizedTranscript {
    tracing::warn!("rewrite planning failed: {err}; using raw transcript fallback");
    FinalizedTranscript {
        text: planning::raw_text(transcript),
        operation: FinalizedOperation::Append,
        rewrite_summary: SessionRewriteSummary {
            had_edit_cues: false,
            rewrite_used: false,
            recommended_candidate: None,
        },
    }
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
            let text =
                canonicalize_structured_output(&plan.rewrite_transcript, &text).unwrap_or(text);
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

    if let Some(candidate) = strict_structured_literal_candidate(rewrite_transcript) {
        return structured_text::output_matches_candidate(text, candidate);
    }
    if structured_literal_candidate(rewrite_transcript)
        .is_some_and(|candidate| structured_text::output_matches_candidate(text, candidate))
    {
        return false;
    }

    match rewrite_transcript.policy_context.correction_policy {
        RewriteCorrectionPolicy::Conservative => {
            agentic_rewrite::conservative_output_allowed(rewrite_transcript, text)
        }
        RewriteCorrectionPolicy::Balanced | RewriteCorrectionPolicy::Aggressive => true,
    }
}

fn structured_literal_candidate(rewrite_transcript: &RewriteTranscript) -> Option<&str> {
    rewrite_transcript
        .rewrite_candidates
        .iter()
        .find(|candidate| candidate.kind == RewriteCandidateKind::StructuredLiteral)
        .map(|candidate| candidate.text.as_str())
}

fn strict_structured_literal_candidate(rewrite_transcript: &RewriteTranscript) -> Option<&str> {
    let candidate = structured_literal_candidate(rewrite_transcript)?;
    structured_literal_source_matches_candidate(rewrite_transcript, candidate).then_some(candidate)
}

fn structured_literal_source_matches_candidate(
    rewrite_transcript: &RewriteTranscript,
    candidate: &str,
) -> bool {
    [
        Some(rewrite_transcript.raw_text.as_str()),
        Some(rewrite_transcript.correction_aware_text.as_str()),
        rewrite_transcript.aggressive_correction_text.as_deref(),
    ]
    .into_iter()
    .flatten()
    .any(|text| structured_text::output_matches_candidate(text, candidate))
}

fn canonicalize_structured_output(
    rewrite_transcript: &RewriteTranscript,
    text: &str,
) -> Option<String> {
    let candidate = strict_structured_literal_candidate(rewrite_transcript)?;
    structured_text::output_matches_candidate(text, candidate).then(|| candidate.to_string())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

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

    #[tokio::test]
    async fn rewrite_thread_timeout_returns_value_when_task_finishes() {
        let result = run_rewrite_thread_with_timeout(Duration::from_millis(50), "test task", || 7)
            .await
            .expect("task should finish");

        assert_eq!(result, 7);
    }

    #[tokio::test]
    async fn rewrite_thread_timeout_returns_error_when_task_hangs() {
        let completed = Arc::new(AtomicBool::new(false));
        let completed_in_thread = Arc::clone(&completed);

        let err =
            run_rewrite_thread_with_timeout(Duration::from_millis(10), "test task", move || {
                std::thread::sleep(Duration::from_millis(50));
                completed_in_thread.store(true, Ordering::Relaxed);
            })
            .await
            .expect_err("task should time out");

        match err {
            WhsprError::Rewrite(message) => {
                assert!(message.contains("test task timed out after 10ms"));
            }
            other => panic!("unexpected error: {other:?}"),
        }

        assert!(!completed.load(Ordering::Relaxed));
    }

    #[test]
    fn rewrite_planning_failure_falls_back_to_trimmed_raw_text() {
        let transcript = Transcript {
            raw_text: "  raw transcript  ".into(),
            detected_language: Some("en".into()),
            segments: Vec::new(),
        };

        let finalized = finalize_rewrite_planning_failure(
            &transcript,
            &WhsprError::Rewrite("rewrite planning timed out after 5000ms".into()),
        );

        assert_eq!(finalized.text, "raw transcript");
        assert!(!finalized.rewrite_summary.rewrite_used);
        assert_eq!(finalized.operation, FinalizedOperation::Append);
    }

    #[test]
    fn structured_literal_meta_wrapper_is_canonicalized() {
        let config = plan_config(PostprocessMode::Rewrite, RewriteBackend::Cloud);
        let mut plan = rewrite_plan();
        plan.rewrite_transcript.raw_text = "portfolio. Notes. Supply".into();
        plan.rewrite_transcript.correction_aware_text = "portfolio. Notes. Supply".into();
        plan.fallback_text = "portfolio.notes.supply".into();
        plan.rewrite_transcript.rewrite_candidates = vec![RewriteCandidate {
            kind: RewriteCandidateKind::StructuredLiteral,
            text: "portfolio.notes.supply".into(),
        }];

        let finalized = finalize_rewrite_attempt(
            &config,
            plan,
            Ok("portfolio. Notes. Supply is the URL".into()),
        );

        assert_eq!(finalized.text, "portfolio.notes.supply");
        assert!(finalized.rewrite_summary.rewrite_used);
    }

    #[test]
    fn structured_literal_url_with_query_and_fragment_meta_wrapper_is_canonicalized() {
        let config = plan_config(PostprocessMode::Rewrite, RewriteBackend::Cloud);
        let mut plan = rewrite_plan();
        plan.rewrite_transcript.raw_text = "https://example.com/?q=test&lang=en#frag".into();
        plan.rewrite_transcript.correction_aware_text =
            "https://example.com/?q=test&lang=en#frag".into();
        plan.fallback_text = "https://example.com/?q=test&lang=en#frag".into();
        plan.rewrite_transcript.rewrite_candidates = vec![RewriteCandidate {
            kind: RewriteCandidateKind::StructuredLiteral,
            text: "https://example.com/?q=test&lang=en#frag".into(),
        }];

        let finalized = finalize_rewrite_attempt(
            &config,
            plan,
            Ok("https://example.com/?q=test&lang=en#frag is the URL".into()),
        );

        assert_eq!(finalized.text, "https://example.com/?q=test&lang=en#frag");
        assert!(finalized.rewrite_summary.rewrite_used);
    }

    #[test]
    fn structured_literal_output_is_canonicalized_when_accepted() {
        let config = plan_config(PostprocessMode::Rewrite, RewriteBackend::Cloud);
        let mut plan = rewrite_plan();
        plan.rewrite_transcript.raw_text = "portfolio. Notes. Supply".into();
        plan.rewrite_transcript.correction_aware_text = "portfolio. Notes. Supply".into();
        plan.rewrite_transcript.rewrite_candidates = vec![RewriteCandidate {
            kind: RewriteCandidateKind::StructuredLiteral,
            text: "portfolio.notes.supply".into(),
        }];

        let finalized =
            finalize_rewrite_attempt(&config, plan, Ok("portfolio . notes . supply".into()));

        assert_eq!(finalized.text, "portfolio.notes.supply");
        assert!(finalized.rewrite_summary.rewrite_used);
    }

    #[test]
    fn structured_literal_embedded_sentence_rejects_lossy_candidate_only_output() {
        let config = plan_config(PostprocessMode::Rewrite, RewriteBackend::Cloud);
        let mut plan = rewrite_plan();
        plan.fallback_text = "Check portfolio. Notes. Supply tomorrow".into();
        plan.rewrite_transcript.rewrite_candidates = vec![RewriteCandidate {
            kind: RewriteCandidateKind::StructuredLiteral,
            text: "portfolio.notes.supply".into(),
        }];

        let finalized =
            finalize_rewrite_attempt(&config, plan, Ok("portfolio.notes.supply".into()));

        assert_eq!(finalized.text, "Check portfolio. Notes. Supply tomorrow");
        assert!(!finalized.rewrite_summary.rewrite_used);
    }

    #[test]
    fn structured_literal_embedded_sentence_accepts_full_sentence_rewrite() {
        let config = plan_config(PostprocessMode::Rewrite, RewriteBackend::Cloud);
        let mut plan = rewrite_plan();
        plan.fallback_text = "Check portfolio. Notes. Supply tomorrow".into();
        plan.rewrite_transcript.rewrite_candidates = vec![RewriteCandidate {
            kind: RewriteCandidateKind::StructuredLiteral,
            text: "portfolio.notes.supply".into(),
        }];

        let finalized = finalize_rewrite_attempt(
            &config,
            plan,
            Ok("Check portfolio.notes.supply tomorrow.".into()),
        );

        assert_eq!(finalized.text, "Check portfolio.notes.supply tomorrow.");
        assert!(finalized.rewrite_summary.rewrite_used);
    }
}

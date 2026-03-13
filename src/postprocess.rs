use std::path::PathBuf;
use std::time::Duration;

use crate::config::{Config, PostprocessMode};
use crate::rewrite_model;
use crate::rewrite_worker::{self, RewriteWorker};
use crate::transcribe::Transcript;

const FEEDBACK_DRAIN_DELAY: Duration = Duration::from_millis(150);

pub fn raw_text(transcript: &Transcript) -> String {
    transcript.raw_text.trim().to_string()
}

pub fn fallback_text(config: &Config, transcript: &Transcript) -> String {
    let _ = config;
    raw_text(transcript)
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
    preloaded_worker: Option<&mut RewriteWorker>,
) -> String {
    match config.postprocess.mode {
        PostprocessMode::Raw => raw_text(&transcript),
        PostprocessMode::AdvancedLocal => {
            rewrite_transcript_or_fallback(config, transcript, preloaded_worker).await
        }
    }
}

pub async fn wait_for_feedback_drain() {
    tokio::time::sleep(FEEDBACK_DRAIN_DELAY).await;
}

pub fn preload_rewrite_worker(config: &Config, phase: &str) -> Option<RewriteWorker> {
    if config.postprocess.mode != PostprocessMode::AdvancedLocal {
        return None;
    }

    let model_path = resolve_rewrite_model_path(config)?;
    match rewrite_worker::RewriteWorker::spawn(&config.rewrite, &model_path) {
        Ok(worker) => {
            tracing::info!(
                "preloading rewrite worker from {} {phase}",
                model_path.display()
            );
            Some(worker)
        }
        Err(err) => {
            tracing::warn!("failed to preload rewrite worker: {err}");
            None
        }
    }
}

async fn rewrite_transcript_or_fallback(
    config: &Config,
    transcript: Transcript,
    preloaded_worker: Option<&mut RewriteWorker>,
) -> String {
    let fallback = fallback_text(config, &transcript);
    let Some(model_path) = resolve_rewrite_model_path(config) else {
        tracing::warn!(
            "advanced_local selected but no rewrite model is configured; using fallback"
        );
        return fallback;
    };

    let rewrite_result = if let Some(worker) = preloaded_worker {
        rewrite_worker::rewrite_with_worker(worker, &config.rewrite, &transcript).await
    } else {
        rewrite_worker::rewrite_transcript(&config.rewrite, &model_path, &transcript).await
    };

    match rewrite_result {
        Ok(text) if !text.trim().is_empty() => text,
        Ok(_) => {
            tracing::warn!("rewrite model returned empty text; using fallback");
            fallback
        }
        Err(err) => {
            tracing::warn!("rewrite failed: {err}; using fallback");
            fallback
        }
    }
}

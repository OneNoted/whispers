use std::time::Instant;

use crate::config::Config;
use crate::error::Result;
use crate::postprocess::finalize;

mod osd;
mod runtime;

use runtime::DictationRuntime;

pub async fn run(config: Config) -> Result<()> {
    let activation_started = Instant::now();
    // Register signals before startup work to minimize early-signal races.
    let mut sigusr1 =
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::user_defined1())?;
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    let mut runtime = DictationRuntime::new(config);
    let recording = runtime.start_recording()?;
    runtime.prepare_services()?;

    tokio::select! {
        _ = sigusr1.recv() => {
            tracing::info!("toggle signal received, stopping recording");
        }
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("interrupted, cancelling");
            runtime.cancel_recording(recording)?;
            return Ok(());
        }
        _ = sigterm.recv() => {
            tracing::info!("terminated, cancelling");
            runtime.cancel_recording(recording)?;
            return Ok(());
        }
    }

    let captured = runtime.finish_recording(recording)?;
    let transcribed = runtime.transcribe_recording(captured).await?;

    if transcribed.is_empty() {
        tracing::warn!("transcription returned empty text");
        finalize::wait_for_feedback_drain().await;
        return Ok(());
    }

    let finalized = runtime.finalize_recording(transcribed).await;
    if finalized.is_empty() {
        tracing::warn!("post-processing produced empty text");
        // When the RMS/duration gates skip transcription, the process would
        // exit almost immediately after play_stop().  PipeWire may still be
        // draining the stop sound's last buffer; exiting while it's "warm"
        // causes an audible click as the OS closes our audio file descriptors.
        // With speech, transcription takes seconds — providing natural drain time.
        finalize::wait_for_feedback_drain().await;
        return Ok(());
    }

    runtime.inject_finalized(finalized).await?;

    tracing::info!("done");
    tracing::info!(
        total_elapsed_ms = activation_started.elapsed().as_millis(),
        "dictation pipeline finished"
    );
    Ok(())
}

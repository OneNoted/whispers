use std::process::Child;
use std::time::Instant;

#[cfg(feature = "osd")]
use std::process::Command;

use crate::asr;
use crate::audio::AudioRecorder;
use crate::config::Config;
use crate::context;
use crate::error::Result;
use crate::feedback::FeedbackPlayer;
use crate::inject::TextInjector;
use crate::postprocess::{execution, finalize};
use crate::session;

pub async fn run(config: Config) -> Result<()> {
    let activation_started = Instant::now();
    // Register signals before startup work to minimize early-signal races.
    let mut sigusr1 =
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::user_defined1())?;
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;

    let feedback = FeedbackPlayer::new(
        config.feedback.enabled,
        &config.feedback.start_sound,
        &config.feedback.stop_sound,
    );

    // Play start sound first (blocking), then start recording so the sound
    // doesn't leak into the mic.
    feedback.play_start();
    let recording_context = context::capture_typing_context();
    let session_enabled = config.postprocess.mode.uses_rewrite();
    let recent_session = if session_enabled {
        session::load_recent_entry(&config.session, &recording_context)?
    } else {
        None
    };
    let mut recorder = AudioRecorder::new(&config.audio);
    recorder.start()?;
    let mut osd = spawn_osd();
    tracing::info!("recording... (run whispers again to stop)");

    let transcriber = asr::prepare_transcriber(&config)?;
    let rewrite_service = execution::prepare_rewrite_service(&config);
    asr::prewarm_transcriber(&transcriber, "recording");
    if let Some(service) = rewrite_service.as_ref() {
        execution::prewarm_rewrite_service(service, "recording");
    }

    tokio::select! {
        _ = sigusr1.recv() => {
            tracing::info!("toggle signal received, stopping recording");
        }
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("interrupted, cancelling");
            kill_osd(&mut osd);
            recorder.stop()?;
            return Ok(());
        }
        _ = sigterm.recv() => {
            tracing::info!("terminated, cancelling");
            kill_osd(&mut osd);
            recorder.stop()?;
            return Ok(());
        }
    }

    // Stop recording before playing feedback so the stop sound doesn't
    // leak into the mic.
    kill_osd(&mut osd);
    let audio = recorder.stop()?;
    feedback.play_stop();
    let sample_rate = config.audio.sample_rate;
    let audio_duration_ms = ((audio.len() as f64 / sample_rate as f64) * 1000.0).round() as u64;

    tracing::info!(
        samples = audio.len(),
        sample_rate,
        audio_duration_ms,
        "transcribing captured audio"
    );

    let transcribe_started = Instant::now();
    let transcript = asr::transcribe_audio(&config, transcriber, audio, sample_rate).await?;
    tracing::info!(
        elapsed_ms = transcribe_started.elapsed().as_millis(),
        transcript_chars = transcript.raw_text.len(),
        "transcription stage finished"
    );

    if transcript.is_empty() {
        tracing::warn!("transcription returned empty text");
        finalize::wait_for_feedback_drain().await;
        return Ok(());
    }

    let injection_context = context::capture_typing_context();
    let recent_session = recent_session.filter(|entry| {
        let same_focus = entry.entry.focus_fingerprint == injection_context.focus_fingerprint;
        if !same_focus {
            tracing::debug!(
                previous_focus = entry.entry.focus_fingerprint,
                current_focus = injection_context.focus_fingerprint,
                "session backtrack blocked because focus changed before injection"
            );
        }
        same_focus
    });
    let finalize_started = Instant::now();
    let finalized = finalize::finalize_transcript(
        &config,
        transcript,
        rewrite_service.as_ref(),
        Some(&injection_context),
        recent_session.as_ref(),
    )
    .await;
    tracing::info!(
        elapsed_ms = finalize_started.elapsed().as_millis(),
        output_chars = finalized.text.len(),
        operation = match finalized.operation {
            finalize::FinalizedOperation::Append => "append",
            finalize::FinalizedOperation::ReplaceLastEntry { .. } => "replace_last_entry",
        },
        rewrite_used = finalized.rewrite_summary.rewrite_used,
        "post-processing stage finished"
    );

    if finalized.text.is_empty() {
        tracing::warn!("post-processing produced empty text");
        // When the RMS/duration gates skip transcription, the process would
        // exit almost immediately after play_stop().  PipeWire may still be
        // draining the stop sound's last buffer; exiting while it's "warm"
        // causes an audible click as the OS closes our audio file descriptors.
        // With speech, transcription takes seconds — providing natural drain time.
        finalize::wait_for_feedback_drain().await;
        return Ok(());
    }

    // Inject text
    tracing::info!("injecting text: {:?}", finalized.text);
    let injector = TextInjector::new();
    match finalized.operation {
        finalize::FinalizedOperation::Append => {
            injector.inject(&finalized.text).await?;
            if session_enabled {
                session::record_append(
                    &config.session,
                    &injection_context,
                    &finalized.text,
                    finalized.rewrite_summary,
                )?;
            }
        }
        finalize::FinalizedOperation::ReplaceLastEntry {
            entry_id,
            delete_graphemes,
        } => {
            injector
                .replace_recent_text(delete_graphemes, &finalized.text)
                .await?;
            if session_enabled {
                session::record_replace(
                    &config.session,
                    &injection_context,
                    entry_id,
                    &finalized.text,
                    finalized.rewrite_summary,
                )?;
            }
        }
    }

    tracing::info!("done");
    tracing::info!(
        total_elapsed_ms = activation_started.elapsed().as_millis(),
        "dictation pipeline finished"
    );
    Ok(())
}

#[cfg(feature = "osd")]
fn spawn_osd() -> Option<Child> {
    // Look for whispers-osd next to our own binary first, then fall back to PATH
    let osd_path = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|dir| dir.join("whispers-osd")))
        .filter(|p| p.exists())
        .unwrap_or_else(|| "whispers-osd".into());

    match Command::new(&osd_path).spawn() {
        Ok(child) => {
            tracing::debug!("spawned whispers-osd (pid {})", child.id());
            Some(child)
        }
        Err(e) => {
            tracing::warn!(
                "failed to spawn whispers-osd from {}: {e}",
                osd_path.display()
            );
            None
        }
    }
}

#[cfg(not(feature = "osd"))]
fn spawn_osd() -> Option<Child> {
    None
}

fn kill_osd(child: &mut Option<Child>) {
    if let Some(mut c) = child.take() {
        let pid = c.id() as libc::pid_t;
        unsafe {
            libc::kill(pid, libc::SIGTERM);
        }
        let _ = c.wait();
        tracing::debug!("whispers-osd (pid {pid}) terminated");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kill_osd_none_is_noop() {
        let mut child: Option<Child> = None;
        kill_osd(&mut child);
        assert!(child.is_none());
    }
}

use std::process::Child;

#[cfg(feature = "osd")]
use std::process::Command;

use crate::audio::AudioRecorder;
use crate::config::Config;
use crate::error::{Result, WhsprError};
use crate::feedback::FeedbackPlayer;
use crate::inject::TextInjector;
use crate::postprocess;
use crate::transcribe::{TranscriptionBackend, WhisperLocal};

pub async fn run(config: Config) -> Result<()> {
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
    let mut recorder = AudioRecorder::new(&config.audio);
    recorder.start()?;
    let mut osd = spawn_osd();
    tracing::info!("recording... (run whispers again to stop)");

    // Preload whisper model in background while recording
    let whisper_config = config.whisper.clone();
    let model_path = config.resolved_model_path();
    let model_handle =
        tokio::task::spawn_blocking(move || WhisperLocal::new(&whisper_config, &model_path));
    let mut preloaded_rewrite_worker =
        postprocess::preload_rewrite_worker(&config, "while recording");

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

    tracing::info!("transcribing {} samples...", audio.len());

    // Await preloaded model (instant if it finished during recording)
    let backend = model_handle
        .await
        .map_err(|e| WhsprError::Transcription(format!("model loading task failed: {e}")))??;

    let transcript = tokio::task::spawn_blocking(move || backend.transcribe(&audio, sample_rate))
        .await
        .map_err(|e| WhsprError::Transcription(format!("task panicked: {e}")))??;

    if transcript.is_empty() {
        tracing::warn!("transcription returned empty text");
        postprocess::wait_for_feedback_drain().await;
        return Ok(());
    }

    let text =
        postprocess::finalize_transcript(&config, transcript, preloaded_rewrite_worker.as_mut())
            .await;

    if text.is_empty() {
        tracing::warn!("post-processing produced empty text");
        // When the RMS/duration gates skip transcription, the process would
        // exit almost immediately after play_stop().  PipeWire may still be
        // draining the stop sound's last buffer; exiting while it's "warm"
        // causes an audible click as the OS closes our audio file descriptors.
        // With speech, transcription takes seconds — providing natural drain time.
        postprocess::wait_for_feedback_drain().await;
        return Ok(());
    }

    // Inject text
    tracing::info!("injecting text: {text:?}");
    let injector = TextInjector::new();
    injector.inject(&text).await?;

    tracing::info!("done");
    Ok(())
}

#[cfg(feature = "osd")]
fn spawn_osd() -> Option<Child> {
    let osd_path = crate::branding::resolve_sidecar_executable(&[crate::branding::OSD_BINARY]);

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

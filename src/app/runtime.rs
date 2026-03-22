use std::process::Child;
use std::time::Instant;

use crate::asr;
use crate::audio::AudioRecorder;
use crate::config::Config;
use crate::context::{self, TypingContext};
use crate::error::Result;
use crate::feedback::FeedbackPlayer;
use crate::inject::TextInjector;
use crate::postprocess::{execution, finalize, planning};
use crate::rewrite_worker::RewriteService;
use crate::runtime_diagnostics::{
    DictationRuntimeDiagnostics, DictationStage, DictationStageMetadata,
};
use crate::session::{self, EligibleSessionEntry};
use crate::transcribe::Transcript;

pub(super) struct DictationRuntime {
    config: Config,
    diagnostics: DictationRuntimeDiagnostics,
    feedback: FeedbackPlayer,
    session_enabled: bool,
    runtime_text_resources: planning::RuntimeTextResources,
    runtime_text_resources_degraded_reason: Option<String>,
    transcriber: Option<asr::prepare::PreparedTranscriber>,
    rewrite_service: Option<RewriteService>,
}

pub(super) struct ActiveRecording {
    recorder: AudioRecorder,
    osd: Option<Child>,
    recent_session: Option<EligibleSessionEntry>,
}

pub(super) struct CapturedRecording {
    audio: Vec<f32>,
    sample_rate: u32,
    recent_session: Option<EligibleSessionEntry>,
}

pub(super) struct TranscribedRecording {
    transcript: Transcript,
    recent_session: Option<EligibleSessionEntry>,
}

pub(super) struct ReadyInjection {
    finalized: finalize::FinalizedTranscript,
    injection_context: TypingContext,
}

impl DictationRuntime {
    pub(super) fn new(config: Config, diagnostics: DictationRuntimeDiagnostics) -> Self {
        let feedback = FeedbackPlayer::new(
            config.feedback.enabled,
            &config.feedback.start_sound,
            &config.feedback.stop_sound,
        );
        let session_enabled = config.postprocess.mode.uses_rewrite();
        let (runtime_text_resources, runtime_text_resources_degraded_reason) =
            load_runtime_text_resources_or_default(&config);

        Self {
            config,
            diagnostics,
            feedback,
            session_enabled,
            runtime_text_resources,
            runtime_text_resources_degraded_reason,
            transcriber: None,
            rewrite_service: None,
        }
    }

    pub(super) fn start_recording(&self) -> Result<ActiveRecording> {
        // Play start sound first (blocking), then start recording so the sound
        // doesn't leak into the mic.
        self.feedback.play_start();
        let recording_context = context::capture_typing_context();
        let recent_session = if self.session_enabled {
            load_recent_session_entry_guarded(&self.config.session, &recording_context)
        } else {
            None
        };

        let mut recorder = AudioRecorder::new(&self.config.audio);
        recorder.start()?;
        let osd = super::osd::spawn_osd();
        self.diagnostics
            .enter_stage(DictationStage::Recording, DictationStageMetadata::default());
        tracing::info!("recording... (run whispers again to stop)");

        Ok(ActiveRecording {
            recorder,
            osd,
            recent_session,
        })
    }

    pub(super) fn prepare_services(&mut self) -> Result<()> {
        self.diagnostics.enter_stage(
            DictationStage::AsrPrepare,
            DictationStageMetadata::default(),
        );
        let transcriber = asr::prepare::prepare_transcriber(&self.config)?;
        let rewrite_service = execution::prepare_rewrite_service(&self.config);
        asr::prepare::prewarm_transcriber(&transcriber, "recording");
        if let Some(service) = rewrite_service.as_ref() {
            execution::prewarm_rewrite_service(service, "recording");
        }

        self.transcriber = Some(transcriber);
        self.rewrite_service = rewrite_service;
        Ok(())
    }

    pub(super) fn cancel_recording(&self, mut recording: ActiveRecording) -> Result<()> {
        super::osd::kill_osd(&mut recording.osd);
        recording.recorder.stop()?;
        self.diagnostics.clear_with_stage(DictationStage::Cancelled);
        Ok(())
    }

    pub(super) fn finish_recording(
        &self,
        mut recording: ActiveRecording,
    ) -> Result<CapturedRecording> {
        // Stop recording before playing feedback so the stop sound doesn't
        // leak into the mic.
        super::osd::kill_osd(&mut recording.osd);
        let audio = recording.recorder.stop()?;
        self.feedback.play_stop();
        let sample_rate = self.config.audio.sample_rate;
        let audio_duration_ms = ((audio.len() as f64 / sample_rate as f64) * 1000.0).round() as u64;

        tracing::info!(
            samples = audio.len(),
            sample_rate,
            audio_duration_ms,
            "transcribing captured audio"
        );
        self.diagnostics.enter_stage(
            DictationStage::RecordingStopped,
            DictationStageMetadata {
                audio_samples: Some(audio.len()),
                sample_rate: Some(sample_rate),
                audio_duration_ms: Some(audio_duration_ms),
                ..DictationStageMetadata::default()
            },
        );

        Ok(CapturedRecording {
            audio,
            sample_rate,
            recent_session: recording.recent_session,
        })
    }

    pub(super) async fn transcribe_recording(
        &mut self,
        recording: CapturedRecording,
    ) -> Result<TranscribedRecording> {
        let audio_samples = recording.audio.len();
        let sample_rate = recording.sample_rate;
        let audio_duration_ms =
            ((audio_samples as f64 / sample_rate as f64) * 1000.0).round() as u64;
        self.diagnostics.enter_stage(
            DictationStage::AsrTranscribe,
            DictationStageMetadata {
                audio_samples: Some(audio_samples),
                sample_rate: Some(sample_rate),
                audio_duration_ms: Some(audio_duration_ms),
                ..DictationStageMetadata::default()
            },
        );
        let transcriber = self
            .transcriber
            .take()
            .expect("transcriber prepared before transcription");
        let transcribe_started = Instant::now();
        let transcript = asr::execute::transcribe_audio(
            &self.config,
            transcriber,
            recording.audio,
            recording.sample_rate,
        )
        .await?;

        tracing::info!(
            elapsed_ms = transcribe_started.elapsed().as_millis(),
            transcript_chars = transcript.raw_text.len(),
            "transcription stage finished"
        );

        Ok(TranscribedRecording {
            transcript,
            recent_session: recording.recent_session,
        })
    }

    pub(super) async fn finalize_recording(
        &self,
        recording: TranscribedRecording,
    ) -> ReadyInjection {
        let injection_context = context::capture_typing_context();
        let recent_session = recording.recent_session.filter(|entry| {
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
        self.diagnostics.enter_stage(
            DictationStage::Postprocess,
            DictationStageMetadata {
                substage: Some("planning".into()),
                detail: Some("build_rewrite_plan".into()),
                transcript_chars: Some(recording.transcript.raw_text.len()),
                degraded_reason: self.runtime_text_resources_degraded_reason.clone(),
                ..DictationStageMetadata::default()
            },
        );
        let finalized = finalize::finalize_transcript(
            &self.config,
            recording.transcript,
            self.rewrite_service.as_ref(),
            Some(&self.runtime_text_resources),
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

        ReadyInjection {
            finalized,
            injection_context,
        }
    }

    pub(super) async fn inject_finalized(&self, ready: ReadyInjection) -> Result<()> {
        let ReadyInjection {
            finalized,
            injection_context,
        } = ready;
        let finalize::FinalizedTranscript {
            text,
            operation,
            rewrite_summary,
            degraded_reason,
        } = finalized;

        tracing::info!("injecting text: {:?}", text);
        let stage_metadata = DictationStageMetadata {
            substage: Some("inject".to_string()),
            detail: Some(match operation {
                finalize::FinalizedOperation::Append => "clipboard_paste".to_string(),
                finalize::FinalizedOperation::ReplaceLastEntry { .. } => {
                    "replace_recent_text".to_string()
                }
            }),
            output_chars: Some(text.len()),
            operation: Some(match operation {
                finalize::FinalizedOperation::Append => "append".to_string(),
                finalize::FinalizedOperation::ReplaceLastEntry { .. } => {
                    "replace_last_entry".to_string()
                }
            }),
            rewrite_used: Some(rewrite_summary.rewrite_used),
            degraded_reason: degraded_reason.clone(),
            ..DictationStageMetadata::default()
        };
        self.diagnostics
            .enter_stage(DictationStage::Inject, stage_metadata.clone());
        let injector = TextInjector::new();
        match operation {
            finalize::FinalizedOperation::Append => {
                injector.inject(&text).await?;
                if self.session_enabled {
                    let mut session_metadata = stage_metadata.clone();
                    session_metadata.substage = Some("session_write".into());
                    session_metadata.detail = Some("record_append".into());
                    self.diagnostics
                        .enter_stage(DictationStage::SessionWrite, session_metadata);
                    record_session_append_guarded(
                        &self.config.session,
                        &injection_context,
                        &text,
                        rewrite_summary,
                    );
                }
            }
            finalize::FinalizedOperation::ReplaceLastEntry {
                entry_id,
                delete_graphemes,
            } => {
                injector
                    .replace_recent_text(delete_graphemes, &text)
                    .await?;
                if self.session_enabled {
                    let mut session_metadata = stage_metadata;
                    session_metadata.substage = Some("session_write".into());
                    session_metadata.detail = Some("record_replace".into());
                    self.diagnostics
                        .enter_stage(DictationStage::SessionWrite, session_metadata);
                    record_session_replace_guarded(
                        &self.config.session,
                        &injection_context,
                        entry_id,
                        &text,
                        rewrite_summary,
                    );
                }
            }
        }

        Ok(())
    }
}

fn load_runtime_text_resources_or_default(
    config: &Config,
) -> (planning::RuntimeTextResources, Option<String>) {
    let (resources, degraded) = planning::load_runtime_text_resources_with_status(config);
    if degraded {
        (resources, Some("runtime_text_resources_unavailable".into()))
    } else {
        (resources, None)
    }
}

fn load_recent_session_entry_guarded(
    config: &crate::config::SessionConfig,
    context: &TypingContext,
) -> Option<EligibleSessionEntry> {
    match session::load_recent_entry(config, context) {
        Ok(entry) => entry,
        Err(err) => {
            tracing::warn!("failed to load recent session entry: {err}; continuing without it");
            None
        }
    }
}

fn record_session_append_guarded(
    config: &crate::config::SessionConfig,
    context: &TypingContext,
    text: &str,
    rewrite_summary: crate::session::SessionRewriteSummary,
) {
    match session::record_append(config, context, text, rewrite_summary) {
        Ok(()) => {}
        Err(err) => {
            tracing::warn!("failed to persist session append: {err}; continuing");
        }
    }
}

fn record_session_replace_guarded(
    config: &crate::config::SessionConfig,
    context: &TypingContext,
    entry_id: u64,
    text: &str,
    rewrite_summary: crate::session::SessionRewriteSummary,
) {
    match session::record_replace(config, context, entry_id, text, rewrite_summary) {
        Ok(()) => {}
        Err(err) => {
            tracing::warn!("failed to persist session replacement: {err}; continuing");
        }
    }
}

impl TranscribedRecording {
    pub(super) fn is_empty(&self) -> bool {
        self.transcript.is_empty()
    }
}

impl ReadyInjection {
    pub(super) fn is_empty(&self) -> bool {
        self.finalized.text.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;

    use super::*;
    use crate::context::SurfaceKind;
    use crate::session::SessionRewriteSummary;
    use crate::test_support::{EnvVarGuard, env_lock, set_env, unique_temp_dir};

    fn typing_context() -> TypingContext {
        TypingContext {
            focus_fingerprint: "niri:7".into(),
            app_id: Some("kitty".into()),
            window_title: Some("shell".into()),
            surface_kind: SurfaceKind::Terminal,
            browser_domain: None,
            captured_at_ms: 42,
        }
    }

    fn with_runtime_dir<T>(f: impl FnOnce(&Path) -> T) -> T {
        let _env_lock = env_lock();
        let _guard = EnvVarGuard::capture(&["XDG_RUNTIME_DIR"]);
        let runtime_dir = unique_temp_dir("runtime-session-timeout");
        set_env(
            "XDG_RUNTIME_DIR",
            runtime_dir.to_str().expect("runtime dir should be utf-8"),
        );
        f(&runtime_dir)
    }

    fn mkfifo(path: &Path) {
        let c_path = CString::new(path.as_os_str().as_bytes()).expect("fifo path");
        let result = unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) };
        assert_eq!(
            result,
            0,
            "mkfifo failed: {}",
            std::io::Error::last_os_error()
        );
    }

    #[test]
    fn load_recent_session_entry_skips_blocking_fifo() {
        with_runtime_dir(|runtime_dir| {
            let session_dir = runtime_dir.join("whispers");
            std::fs::create_dir_all(&session_dir).expect("session dir");
            mkfifo(&session_dir.join("session.json"));

            let recent = load_recent_session_entry_guarded(
                &crate::config::SessionConfig::default(),
                &typing_context(),
            );

            assert!(recent.is_none());
        });
    }

    #[test]
    fn record_session_append_skips_blocking_fifo() {
        with_runtime_dir(|runtime_dir| {
            let session_dir = runtime_dir.join("whispers");
            std::fs::create_dir_all(&session_dir).expect("session dir");
            mkfifo(&session_dir.join("session.json"));

            record_session_append_guarded(
                &crate::config::SessionConfig::default(),
                &typing_context(),
                "hello",
                SessionRewriteSummary {
                    had_edit_cues: false,
                    rewrite_used: false,
                    recommended_candidate: None,
                },
            );
        });
    }

    #[test]
    fn load_runtime_text_resources_falls_back_to_defaults_for_blocking_fifo() {
        let mut config = Config::default();
        config.postprocess.mode = crate::config::PostprocessMode::Rewrite;
        with_runtime_dir(|runtime_dir| {
            let dictionary_path = runtime_dir.join("blocking-dictionary.toml");
            mkfifo(&dictionary_path);
            config.personalization.dictionary_path = dictionary_path
                .to_str()
                .expect("dictionary path")
                .to_string();

            let (_, degraded_reason) = load_runtime_text_resources_or_default(&config);

            assert_eq!(
                degraded_reason.as_deref(),
                Some("runtime_text_resources_unavailable")
            );
        });
    }
}

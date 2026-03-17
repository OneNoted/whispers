use std::process::Child;
use std::time::Instant;

use crate::asr;
use crate::audio::AudioRecorder;
use crate::config::Config;
use crate::context::{self, TypingContext};
use crate::error::Result;
use crate::feedback::FeedbackPlayer;
use crate::inject::TextInjector;
use crate::postprocess::{execution, finalize};
use crate::rewrite_worker::RewriteService;
use crate::session::{self, EligibleSessionEntry};
use crate::transcribe::Transcript;

pub(super) struct DictationRuntime {
    config: Config,
    feedback: FeedbackPlayer,
    session_enabled: bool,
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
    pub(super) fn new(config: Config) -> Self {
        let feedback = FeedbackPlayer::new(
            config.feedback.enabled,
            &config.feedback.start_sound,
            &config.feedback.stop_sound,
        );
        let session_enabled = config.postprocess.mode.uses_rewrite();

        Self {
            config,
            feedback,
            session_enabled,
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
            session::load_recent_entry(&self.config.session, &recording_context)?
        } else {
            None
        };

        let mut recorder = AudioRecorder::new(&self.config.audio);
        recorder.start()?;
        let osd = super::osd::spawn_osd();
        tracing::info!("recording... (run whispers again to stop)");

        Ok(ActiveRecording {
            recorder,
            osd,
            recent_session,
        })
    }

    pub(super) fn prepare_services(&mut self) -> Result<()> {
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
        let finalized = finalize::finalize_transcript(
            &self.config,
            recording.transcript,
            self.rewrite_service.as_ref(),
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
        } = finalized;

        tracing::info!("injecting text: {:?}", text);
        let injector = TextInjector::new();
        match operation {
            finalize::FinalizedOperation::Append => {
                injector.inject(&text).await?;
                if self.session_enabled {
                    session::record_append(
                        &self.config.session,
                        &injection_context,
                        &text,
                        rewrite_summary,
                    )?;
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
                    session::record_replace(
                        &self.config.session,
                        &injection_context,
                        entry_id,
                        &text,
                        rewrite_summary,
                    )?;
                }
            }
        }

        Ok(())
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

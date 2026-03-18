use std::time::{Duration, Instant};

use tokio::time::MissedTickBehavior;
use unicode_segmentation::UnicodeSegmentation;

use crate::asr::{self, LiveTranscriber};
use crate::audio::{self, AudioRecorder};
use crate::config::Config;
use crate::context::{self, TypingContext};
use crate::error::Result;
use crate::feedback::FeedbackPlayer;
use crate::inject::{InjectionPolicy, TextInjector};
use crate::osd::{OsdHandle, OsdMode};
use crate::osd_protocol::{VoiceOsdStatus, VoiceOsdUpdate};
use crate::postprocess::{self, FinalizedOperation, FinalizedTranscript};
use crate::rewrite_worker::RewriteService;
use crate::session::{self, EligibleSessionEntry};
use crate::transcribe::Transcript;

const UNSTABLE_TAIL_MS: u32 = 3500;
const LIVE_MIN_TRANSCRIBE_DELTA_MS: u64 = 220;
const LIVE_SILENCE_WINDOW_MS: u64 = 450;
const LIVE_SILENCE_RMS_THRESHOLD: f32 = 0.0022;
const LIVE_SILENCE_SETTLE_MS: u64 = 220;

pub async fn run(config: Config) -> Result<()> {
    let activation_started = Instant::now();
    let mut sigusr1 =
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::user_defined1())?;
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;

    let feedback = FeedbackPlayer::new(
        config.feedback.enabled,
        &config.feedback.start_sound,
        &config.feedback.stop_sound,
    );

    feedback.play_start();
    let recording_context = context::capture_typing_context();
    let session_enabled = config.postprocess.mode.uses_rewrite();
    let recent_session = if session_enabled {
        session::load_recent_entry(&config.session, &recording_context)?
    } else {
        None
    };
    let replaceable_prefix_graphemes = recent_session
        .as_ref()
        .map(|entry| entry.delete_graphemes)
        .unwrap_or(0);

    let mut recorder = AudioRecorder::new(&config.audio);
    recorder.start()?;

    let mut osd = OsdHandle::spawn(OsdMode::Voice);
    let mut accumulator = VoiceTranscriptAccumulator::default();
    let mut live_preview_pacing = LivePreviewPacing::default();
    let mut rewrite_preview = None::<String>;
    let mut live_injection = LiveInjectionState::new(
        config.voice.live_inject,
        config.voice.freeze_on_focus_change,
        &recording_context,
        replaceable_prefix_graphemes,
    );
    osd.send_voice_update(&build_osd_update(
        VoiceOsdStatus::Listening,
        &accumulator,
        rewrite_preview.as_deref(),
        &live_injection,
        config.voice.live_inject,
    ));

    tracing::info!("voice recording... (run whispers voice again to stop)");

    let transcriber = asr::prepare_live_transcriber(&config).await?;
    let rewrite_service = postprocess::prepare_rewrite_service(&config);
    asr::prewarm_live_transcriber(&transcriber, "voice recording");
    if let Some(service) = rewrite_service.as_ref() {
        postprocess::prewarm_rewrite_service(service, "voice recording");
    }

    let partial_interval_ms = config.voice.partial_interval_ms.max(50);
    let mut partial_tick = tokio::time::interval(Duration::from_millis(partial_interval_ms));
    partial_tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
    partial_tick.tick().await;
    let rewrite_interval = Duration::from_millis(config.voice.rewrite_interval_ms.max(1));
    let mut last_rewrite_at = Instant::now() - rewrite_interval;

    loop {
        tokio::select! {
            _ = sigusr1.recv() => {
                tracing::info!("toggle signal received, stopping voice recording");
                break;
            }
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("interrupted, cancelling voice recording");
                osd.kill();
                recorder.stop()?;
                return Ok(());
            }
            _ = sigterm.recv() => {
                tracing::info!("terminated, cancelling voice recording");
                osd.kill();
                recorder.stop()?;
                return Ok(());
            }
            _ = partial_tick.tick() => {
                if let Err(err) = process_partial_tick(
                    &config,
                    &recorder,
                    &transcriber,
                    rewrite_service.as_ref(),
                    recent_session.as_ref(),
                    &mut accumulator,
                    &mut live_preview_pacing,
                    &mut rewrite_preview,
                    &mut last_rewrite_at,
                    &mut live_injection,
                    &mut osd,
                ).await {
                    tracing::warn!("live partial update failed: {err}");
                    osd.send_voice_update(&build_osd_update(
                        if live_injection.is_frozen() {
                            VoiceOsdStatus::Frozen
                        } else {
                            VoiceOsdStatus::Listening
                        },
                        &accumulator,
                        rewrite_preview.as_deref(),
                        &live_injection,
                        config.voice.live_inject,
                    ));
                }
            }
        }
    }

    let audio = recorder.stop()?;
    feedback.play_stop();
    let sample_rate = config.audio.sample_rate;
    let audio_duration_ms = audio_duration_ms(audio.len(), sample_rate);
    osd.send_voice_update(&build_osd_update(
        VoiceOsdStatus::Finalizing,
        &accumulator,
        rewrite_preview.as_deref(),
        &live_injection,
        config.voice.live_inject,
    ));

    tracing::info!(
        samples = audio.len(),
        sample_rate,
        audio_duration_ms,
        "transcribing final voice-mode audio"
    );

    let transcript = asr::transcribe_live_audio(&config, &transcriber, audio, sample_rate).await?;
    if transcript.is_empty() {
        tracing::warn!("final voice-mode transcription returned empty text");
        postprocess::wait_for_feedback_drain().await;
        osd.kill();
        return Ok(());
    }

    let injection_context = context::capture_typing_context();
    let recent_session = recent_session.filter(|entry| {
        let same_focus = entry.entry.focus_fingerprint == injection_context.focus_fingerprint;
        if !same_focus {
            tracing::debug!(
                previous_focus = entry.entry.focus_fingerprint,
                current_focus = injection_context.focus_fingerprint,
                "session backtrack blocked because focus changed before final voice injection"
            );
        }
        same_focus
    });

    let finalized = postprocess::finalize_transcript(
        &config,
        transcript,
        rewrite_service.as_ref(),
        Some(&injection_context),
        recent_session.as_ref(),
    )
    .await;
    if finalized.text.is_empty() {
        tracing::warn!("final voice-mode post-processing produced empty text");
        postprocess::wait_for_feedback_drain().await;
        osd.kill();
        return Ok(());
    }

    let injector = TextInjector::new();
    let injection_applied = if config.voice.live_inject {
        apply_final_live_output(
            &injector,
            &mut live_injection,
            &injection_context,
            &finalized,
        )
        .await?
    } else {
        inject_final_output(&injector, &injection_context, &finalized).await?;
        true
    };

    if injection_applied && session_enabled {
        record_final_session(&config, &injection_context, &finalized)?;
    } else if !injection_applied {
        tracing::warn!("skipping session recording because final live injection was not applied");
    }

    tracing::info!(
        total_elapsed_ms = activation_started.elapsed().as_millis(),
        final_chars = finalized.text.len(),
        "voice dictation pipeline finished"
    );
    osd.kill();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn process_partial_tick(
    config: &Config,
    recorder: &AudioRecorder,
    transcriber: &LiveTranscriber,
    rewrite_service: Option<&RewriteService>,
    recent_session: Option<&EligibleSessionEntry>,
    accumulator: &mut VoiceTranscriptAccumulator,
    live_preview_pacing: &mut LivePreviewPacing,
    rewrite_preview: &mut Option<String>,
    last_rewrite_at: &mut Instant,
    live_injection: &mut LiveInjectionState,
    osd: &mut OsdHandle,
) -> Result<()> {
    let snapshot = recorder.snapshot()?;
    let total_audio_ms = audio_duration_ms(snapshot.len(), config.audio.sample_rate);
    if total_audio_ms < config.voice.min_chunk_ms {
        return Ok(());
    }
    let new_audio_ms = audio_duration_ms(
        snapshot
            .len()
            .saturating_sub(live_preview_pacing.last_processed_samples),
        config.audio.sample_rate,
    );
    let recent_rms = recent_audio_rms(&snapshot, config.audio.sample_rate, LIVE_SILENCE_WINDOW_MS);
    if recent_rms >= LIVE_SILENCE_RMS_THRESHOLD {
        live_preview_pacing.last_voice_activity_at = Some(Instant::now());
    } else {
        let settled_silence = live_preview_pacing
            .last_voice_activity_at
            .map(|at| at.elapsed() >= Duration::from_millis(LIVE_SILENCE_SETTLE_MS))
            .unwrap_or(true);
        if settled_silence {
            tracing::trace!(
                total_audio_ms,
                new_audio_ms,
                recent_rms,
                "skipping live partial update during settled silence"
            );
            live_preview_pacing.last_processed_samples = snapshot.len();
            osd.send_voice_update(&build_osd_update(
                if live_injection.is_frozen() {
                    VoiceOsdStatus::Frozen
                } else {
                    VoiceOsdStatus::Listening
                },
                accumulator,
                rewrite_preview.as_deref(),
                live_injection,
                config.voice.live_inject,
            ));
            return Ok(());
        }
    }
    if new_audio_ms < LIVE_MIN_TRANSCRIBE_DELTA_MS {
        return Ok(());
    }

    osd.send_voice_update(&build_osd_update(
        VoiceOsdStatus::Transcribing,
        accumulator,
        rewrite_preview.as_deref(),
        live_injection,
        config.voice.live_inject,
    ));

    let (mut chunk, chunk_start_ms) = clip_audio_tail(
        &snapshot,
        config.audio.sample_rate,
        config.voice.context_window_ms,
    );
    audio::preprocess_live_audio(&mut chunk, config.audio.sample_rate);
    let mut transcript =
        asr::transcribe_live_audio(config, transcriber, chunk, config.audio.sample_rate).await?;
    if transcript.is_empty() {
        tracing::debug!(
            total_audio_ms,
            new_audio_ms,
            recent_rms,
            "live partial transcription returned empty text; preserving previous preview"
        );
        osd.send_voice_update(&build_osd_update(
            if live_injection.is_frozen() {
                VoiceOsdStatus::Frozen
            } else {
                VoiceOsdStatus::Listening
            },
            accumulator,
            rewrite_preview.as_deref(),
            live_injection,
            config.voice.live_inject,
        ));
        return Ok(());
    }
    offset_transcript_segments(&mut transcript, chunk_start_ms);
    live_preview_pacing.last_processed_samples = snapshot.len();

    accumulator.update(&transcript, total_audio_ms as u32);
    let live_preview_text = accumulator.full_text();
    let current_context = (config.voice.live_rewrite || config.voice.live_inject)
        .then(context::capture_typing_context);

    if config.voice.live_rewrite
        && !live_preview_text.is_empty()
        && last_rewrite_at.elapsed()
            >= Duration::from_millis(config.voice.rewrite_interval_ms.max(1))
    {
        osd.send_voice_update(&build_osd_update(
            VoiceOsdStatus::Rewriting,
            accumulator,
            rewrite_preview.as_deref(),
            live_injection,
            config.voice.live_inject,
        ));
        let preview_transcript =
            build_live_rewrite_transcript(accumulator, &live_preview_text, &transcript);
        let live_recent_session = recent_session.filter(|entry| {
            current_context
                .as_ref()
                .map(|context| entry.entry.focus_fingerprint == context.focus_fingerprint)
                .unwrap_or(false)
        });
        let finalized = postprocess::finalize_transcript(
            config,
            preview_transcript,
            rewrite_service,
            current_context.as_ref(),
            live_recent_session,
        )
        .await;
        *rewrite_preview = match finalized.text.trim() {
            "" => None,
            text if text == live_preview_text => None,
            text => Some(text.to_string()),
        };
        tracing::debug!(
            live_preview_chars = live_preview_text.len(),
            rewrite_preview_chars = rewrite_preview.as_ref().map(|text| text.len()).unwrap_or(0),
            "updated live rewrite preview"
        );
        *last_rewrite_at = Instant::now();
    }

    if config.voice.live_inject && !live_preview_text.is_empty() {
        let current_context = current_context
            .as_ref()
            .cloned()
            .unwrap_or_else(context::capture_typing_context);
        if let Some(command) = live_injection.plan_update(&live_preview_text, &current_context) {
            tracing::trace!(
                delete_graphemes = command.delete_graphemes,
                insert_chars = command.text.len(),
                desired_chars = live_preview_text.len(),
                "applying live injection command"
            );
            if let Err(err) =
                apply_injection_command(&TextInjector::new(), &command, &current_context).await
            {
                live_injection.freeze();
                return Err(err);
            }
        }
    }

    osd.send_voice_update(&build_osd_update(
        if live_injection.is_frozen() {
            VoiceOsdStatus::Frozen
        } else {
            VoiceOsdStatus::Listening
        },
        accumulator,
        rewrite_preview.as_deref(),
        live_injection,
        config.voice.live_inject,
    ));
    Ok(())
}

async fn inject_final_output(
    injector: &TextInjector,
    injection_context: &TypingContext,
    finalized: &FinalizedTranscript,
) -> Result<()> {
    match finalized.operation {
        FinalizedOperation::Append => injector.inject(&finalized.text, injection_context).await,
        FinalizedOperation::ReplaceLastEntry {
            delete_graphemes, ..
        } => {
            injector
                .replace_recent_text(delete_graphemes, &finalized.text, injection_context)
                .await
        }
    }
}

async fn apply_final_live_output(
    injector: &TextInjector,
    live_injection: &mut LiveInjectionState,
    injection_context: &TypingContext,
    finalized: &FinalizedTranscript,
) -> Result<bool> {
    match live_injection.plan_finalize(&finalized.operation, &finalized.text, injection_context) {
        FinalizeInjectionDecision::Apply(command) => {
            apply_injection_command(injector, &command, injection_context).await?;
            Ok(true)
        }
        FinalizeInjectionDecision::Noop => Ok(true),
        FinalizeInjectionDecision::Blocked => Ok(false),
    }
}

fn record_final_session(
    config: &Config,
    injection_context: &TypingContext,
    finalized: &FinalizedTranscript,
) -> Result<()> {
    match finalized.operation {
        FinalizedOperation::Append => session::record_append(
            &config.session,
            injection_context,
            &finalized.text,
            finalized.rewrite_summary.clone(),
        ),
        FinalizedOperation::ReplaceLastEntry { entry_id, .. } => session::record_replace(
            &config.session,
            injection_context,
            entry_id,
            &finalized.text,
            finalized.rewrite_summary.clone(),
        ),
    }
}

async fn apply_injection_command(
    injector: &TextInjector,
    command: &InjectionCommand,
    current_context: &TypingContext,
) -> Result<()> {
    injector
        .replace_recent_text(command.delete_graphemes, &command.text, current_context)
        .await
}

fn build_osd_update(
    status: VoiceOsdStatus,
    accumulator: &VoiceTranscriptAccumulator,
    rewrite_preview: Option<&str>,
    live_injection: &LiveInjectionState,
    live_inject_enabled: bool,
) -> VoiceOsdUpdate {
    VoiceOsdUpdate {
        status,
        stable_text: accumulator.stable_text.clone(),
        unstable_text: accumulator.unstable_text.clone(),
        rewrite_preview: rewrite_preview.map(str::to_string),
        live_inject: live_inject_enabled,
        frozen: live_injection.is_frozen(),
    }
}

fn build_live_rewrite_transcript(
    accumulator: &VoiceTranscriptAccumulator,
    live_preview_text: &str,
    transcript: &Transcript,
) -> Transcript {
    let mut preview_transcript = transcript.clone();
    preview_transcript.raw_text = live_preview_text.to_string();

    if preview_transcript.segments.is_empty() && !live_preview_text.trim().is_empty() {
        preview_transcript
            .segments
            .push(crate::transcribe::TranscriptSegment {
                text: accumulator.full_text(),
                start_ms: 0,
                end_ms: accumulator.committed_until_ms.max(1),
            });
    }

    preview_transcript
}

fn clip_audio_tail(samples: &[f32], sample_rate: u32, context_window_ms: u64) -> (Vec<f32>, u32) {
    if sample_rate == 0 || samples.is_empty() {
        return (Vec::new(), 0);
    }

    let context_samples = ((context_window_ms as u128 * sample_rate as u128) / 1000) as usize;
    let start = if context_samples == 0 {
        0
    } else {
        samples.len().saturating_sub(context_samples)
    };
    (
        samples[start..].to_vec(),
        audio_duration_ms(start, sample_rate) as u32,
    )
}

fn offset_transcript_segments(transcript: &mut Transcript, offset_ms: u32) {
    for segment in &mut transcript.segments {
        segment.start_ms = segment.start_ms.saturating_add(offset_ms);
        segment.end_ms = segment.end_ms.saturating_add(offset_ms);
    }
}

fn audio_duration_ms(samples: usize, sample_rate: u32) -> u64 {
    if sample_rate == 0 {
        return 0;
    }
    ((samples as f64 / sample_rate as f64) * 1000.0).round() as u64
}

fn audio_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let energy: f32 = samples.iter().map(|sample| sample * sample).sum();
    (energy / samples.len() as f32).sqrt()
}

fn recent_audio_rms(samples: &[f32], sample_rate: u32, window_ms: u64) -> f32 {
    let (tail, _) = clip_audio_tail(samples, sample_rate, window_ms);
    audio_rms(&tail)
}

fn grapheme_count(text: &str) -> usize {
    UnicodeSegmentation::graphemes(text, true).count()
}

fn shared_grapheme_prefix(current: &str, desired: &str) -> (usize, usize) {
    let mut current_end = 0;
    let mut desired_end = 0;

    for ((current_idx, current_grapheme), (desired_idx, desired_grapheme)) in current
        .grapheme_indices(true)
        .zip(desired.grapheme_indices(true))
    {
        if current_grapheme != desired_grapheme {
            break;
        }
        current_end = current_idx + current_grapheme.len();
        desired_end = desired_idx + desired_grapheme.len();
    }

    (current_end, desired_end)
}

fn build_suffix_rewrite_command(
    current_text: &str,
    desired_text: &str,
) -> Option<InjectionCommand> {
    if current_text == desired_text {
        return None;
    }

    let (current_prefix_end, desired_prefix_end) =
        shared_grapheme_prefix(current_text, desired_text);
    let delete_graphemes = grapheme_count(&current_text[current_prefix_end..]);
    let text = desired_text[desired_prefix_end..].to_string();

    if delete_graphemes == 0 && text.is_empty() {
        return None;
    }

    Some(InjectionCommand {
        delete_graphemes,
        text,
    })
}

fn append_segment_text(output: &mut String, text: &str) {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return;
    }
    if !output.is_empty() {
        output.push(' ');
    }
    output.push_str(trimmed);
}

fn join_segment_text<I>(segments: I) -> String
where
    I: IntoIterator,
    I::Item: AsRef<str>,
{
    let mut joined = String::new();
    for segment in segments {
        append_segment_text(&mut joined, segment.as_ref());
    }
    joined
}

#[derive(Debug, Clone, Default)]
struct VoiceTranscriptAccumulator {
    stable_text: String,
    unstable_text: String,
    committed_until_ms: u32,
}

#[derive(Debug, Clone, Default)]
struct LivePreviewPacing {
    last_processed_samples: usize,
    last_voice_activity_at: Option<Instant>,
}

impl VoiceTranscriptAccumulator {
    fn update(&mut self, transcript: &Transcript, total_audio_ms: u32) {
        let stable_boundary_ms = total_audio_ms.saturating_sub(UNSTABLE_TAIL_MS);
        let committed_until_ms = self.committed_until_ms;
        for segment in transcript.segments.iter().filter(|segment| {
            segment.end_ms <= stable_boundary_ms && segment.end_ms > committed_until_ms
        }) {
            append_segment_text(&mut self.stable_text, &segment.text);
            self.committed_until_ms = self.committed_until_ms.max(segment.end_ms);
        }

        if transcript.segments.is_empty() {
            self.unstable_text = transcript.raw_text.trim().to_string();
            return;
        }

        self.unstable_text = join_segment_text(
            transcript
                .segments
                .iter()
                .filter(|segment| segment.end_ms > self.committed_until_ms)
                .map(|segment| segment.text.as_str()),
        );
    }

    fn full_text(&self) -> String {
        join_segment_text([self.stable_text.as_str(), self.unstable_text.as_str()])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct InjectionCommand {
    delete_graphemes: usize,
    text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum FinalizeInjectionDecision {
    Apply(InjectionCommand),
    Noop,
    Blocked,
}

#[derive(Debug, Clone)]
struct LiveInjectionState {
    enabled: bool,
    freeze_on_focus_change: bool,
    original_focus: String,
    replaceable_prefix_graphemes: usize,
    current_text: String,
    pending_correction: Option<PendingCorrection>,
    frozen: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PendingCorrection {
    desired_text: String,
    confirmations: usize,
}

impl LiveInjectionState {
    fn new(
        enabled: bool,
        freeze_on_focus_change: bool,
        recording_context: &TypingContext,
        replaceable_prefix_graphemes: usize,
    ) -> Self {
        Self {
            enabled,
            freeze_on_focus_change,
            original_focus: recording_context.focus_fingerprint.clone(),
            replaceable_prefix_graphemes,
            current_text: String::new(),
            pending_correction: None,
            frozen: false,
        }
    }

    fn is_frozen(&self) -> bool {
        self.frozen
    }

    fn freeze(&mut self) {
        self.frozen = true;
    }

    fn plan_update(
        &mut self,
        desired_text: &str,
        current_context: &TypingContext,
    ) -> Option<InjectionCommand> {
        if !self.enabled || self.frozen {
            return None;
        }
        if self.should_freeze(current_context) {
            self.frozen = true;
            return None;
        }
        let policy = InjectionPolicy::for_context(current_context);
        let Some(command) = build_suffix_rewrite_command(&self.current_text, desired_text) else {
            self.pending_correction = None;
            return None;
        };
        if command.delete_graphemes > 0 {
            if !policy.allows_live_destructive_correction(command.delete_graphemes) {
                tracing::trace!(
                    surface_policy = policy.label(),
                    delete_graphemes = command.delete_graphemes,
                    "deferring live destructive correction until final reconciliation"
                );
                self.pending_correction = None;
                return None;
            }
            let required_confirmations = policy.destructive_correction_confirmations();
            let should_apply = match self.pending_correction.as_mut() {
                Some(pending) if pending.desired_text == desired_text => {
                    pending.confirmations = pending.confirmations.saturating_add(1);
                    pending.confirmations >= required_confirmations
                }
                _ => {
                    self.pending_correction = Some(PendingCorrection {
                        desired_text: desired_text.to_string(),
                        confirmations: 1,
                    });
                    required_confirmations <= 1
                }
            };
            if !should_apply {
                return None;
            }
        }
        self.pending_correction = None;
        self.current_text = desired_text.to_string();
        Some(command)
    }

    fn plan_finalize(
        &mut self,
        operation: &FinalizedOperation,
        final_text: &str,
        current_context: &TypingContext,
    ) -> FinalizeInjectionDecision {
        if !self.enabled {
            return FinalizeInjectionDecision::Blocked;
        }
        if self.frozen && self.should_freeze(current_context) {
            return FinalizeInjectionDecision::Blocked;
        }

        let extra_delete = match operation {
            FinalizedOperation::Append => 0,
            FinalizedOperation::ReplaceLastEntry {
                delete_graphemes, ..
            } => {
                if *delete_graphemes > self.replaceable_prefix_graphemes {
                    return FinalizeInjectionDecision::Blocked;
                }
                *delete_graphemes
            }
        };
        let delete_graphemes = extra_delete + grapheme_count(&self.current_text);
        if extra_delete == 0 {
            if let Some(command) = build_suffix_rewrite_command(&self.current_text, final_text) {
                self.current_text = final_text.to_string();
                self.frozen = false;
                self.pending_correction = None;
                return FinalizeInjectionDecision::Apply(command);
            }
            return FinalizeInjectionDecision::Noop;
        }

        let command = InjectionCommand {
            delete_graphemes,
            text: final_text.to_string(),
        };
        self.current_text = final_text.to_string();
        self.pending_correction = None;
        self.frozen = false;
        FinalizeInjectionDecision::Apply(command)
    }

    fn should_freeze(&self, current_context: &TypingContext) -> bool {
        self.freeze_on_focus_change
            && !self.original_focus.is_empty()
            && !current_context.focus_fingerprint.is_empty()
            && current_context.focus_fingerprint != self.original_focus
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::SurfaceKind;
    use crate::transcribe::TranscriptSegment;

    fn transcript(segments: &[(&str, u32, u32)]) -> Transcript {
        Transcript {
            raw_text: join_segment_text(segments.iter().map(|(text, _, _)| *text)),
            detected_language: Some("en".into()),
            segments: segments
                .iter()
                .map(|(text, start_ms, end_ms)| TranscriptSegment {
                    text: (*text).to_string(),
                    start_ms: *start_ms,
                    end_ms: *end_ms,
                })
                .collect(),
        }
    }

    fn context_with_surface(focus: &str, surface_kind: SurfaceKind) -> TypingContext {
        TypingContext {
            focus_fingerprint: focus.into(),
            app_id: Some("app".into()),
            window_title: Some("window".into()),
            surface_kind,
            browser_domain: None,
            captured_at_ms: 0,
        }
    }

    fn context(focus: &str) -> TypingContext {
        context_with_surface(focus, SurfaceKind::GenericText)
    }

    #[test]
    fn accumulator_commits_stable_prefix_and_keeps_tail_mutable() {
        let mut accumulator = VoiceTranscriptAccumulator::default();
        accumulator.update(
            &transcript(&[
                ("hello", 0, 900),
                ("world", 900, 1900),
                ("again", 1900, 3300),
            ]),
            UNSTABLE_TAIL_MS + 2000,
        );

        assert_eq!(accumulator.stable_text, "hello world");
        assert_eq!(accumulator.unstable_text, "again");
        assert_eq!(accumulator.full_text(), "hello world again");
    }

    #[test]
    fn accumulator_handles_short_tail_regressions() {
        let mut accumulator = VoiceTranscriptAccumulator::default();
        accumulator.update(
            &transcript(&[("hello", 0, 900), ("world", 900, 2100)]),
            3000,
        );
        assert_eq!(accumulator.full_text(), "hello world");

        accumulator.update(&transcript(&[("hello", 0, 900)]), 3000);
        assert_eq!(accumulator.stable_text, "");
        assert_eq!(accumulator.unstable_text, "hello");
    }

    #[test]
    fn accumulator_allows_earlier_correction_inside_mutable_suffix() {
        let mut accumulator = VoiceTranscriptAccumulator::default();
        accumulator.update(
            &transcript(&[("ship", 1000, 1800), ("it", 1800, 2400)]),
            3000,
        );
        assert_eq!(accumulator.full_text(), "ship it");

        accumulator.update(
            &transcript(&[("shift", 1000, 1800), ("it", 1800, 2400)]),
            3000,
        );
        assert_eq!(accumulator.full_text(), "shift it");
    }

    #[test]
    fn live_rewrite_transcript_uses_accumulated_preview_text() {
        let mut accumulator = VoiceTranscriptAccumulator::default();
        accumulator.update(
            &transcript(&[
                ("i'm", 0, 400),
                ("using", 400, 800),
                ("hyperland", 800, 1200),
            ]),
            UNSTABLE_TAIL_MS + 800,
        );
        assert_eq!(accumulator.stable_text, "i'm using");
        assert_eq!(accumulator.unstable_text, "hyperland");

        let live_preview_text = accumulator.full_text();
        let preview = build_live_rewrite_transcript(
            &accumulator,
            &live_preview_text,
            &transcript(&[("hyperland", 800, 1200)]),
        );

        assert_eq!(preview.raw_text, "i'm using hyperland");
        assert_eq!(preview.segments.len(), 1);
        assert_eq!(preview.segments[0].text, "hyperland");
    }

    #[test]
    fn final_reconciliation_overrides_last_partial_text() {
        let mut state = LiveInjectionState::new(true, true, &context("focus-a"), 0);
        let _ = state.plan_update("draft text", &context("focus-a"));

        let decision = state.plan_finalize(
            &FinalizedOperation::Append,
            "final text",
            &context("focus-a"),
        );
        assert_eq!(
            decision,
            FinalizeInjectionDecision::Apply(InjectionCommand {
                delete_graphemes: 10,
                text: "final text".into(),
            })
        );
    }

    #[test]
    fn preview_only_mode_never_plans_live_mutation() {
        let mut state = LiveInjectionState::new(false, true, &context("focus-a"), 0);
        assert_eq!(state.plan_update("hello", &context("focus-a")), None);
    }

    #[test]
    fn live_inject_only_rewrites_owned_text() {
        let mut state = LiveInjectionState::new(true, true, &context("focus-a"), 0);
        let first = state
            .plan_update("hello", &context("focus-a"))
            .expect("first update");
        assert_eq!(
            first,
            InjectionCommand {
                delete_graphemes: 0,
                text: "hello".into()
            }
        );

        let second = state
            .plan_update("hello world", &context("focus-a"))
            .expect("second update");
        assert_eq!(
            second,
            InjectionCommand {
                delete_graphemes: 0,
                text: " world".into()
            }
        );
    }

    #[test]
    fn live_inject_only_rewrites_changed_suffix_when_correcting() {
        let mut state = LiveInjectionState::new(true, true, &context("focus-a"), 0);
        let _ = state.plan_update("ship it", &context("focus-a"));

        assert_eq!(state.plan_update("shift it", &context("focus-a")), None);

        let correction = state
            .plan_update("shift it", &context("focus-a"))
            .expect("confirmed correction update");
        assert_eq!(
            correction,
            InjectionCommand {
                delete_graphemes: 4,
                text: "ft it".into()
            }
        );
    }

    #[test]
    fn destructive_correction_confirmation_resets_when_target_changes() {
        let mut state = LiveInjectionState::new(true, true, &context("focus-a"), 0);
        let _ = state.plan_update("hyperland", &context("focus-a"));

        assert_eq!(state.plan_update("hyprland", &context("focus-a")), None);
        assert_eq!(state.plan_update("highprland", &context("focus-a")), None);
        assert_eq!(state.plan_update("hyprland", &context("focus-a")), None);

        let correction = state
            .plan_update("hyprland", &context("focus-a"))
            .expect("correction update");
        assert_eq!(
            correction,
            InjectionCommand {
                delete_graphemes: 6,
                text: "rland".into()
            }
        );
    }

    #[test]
    fn destructive_correction_confirmation_resets_when_correction_disappears() {
        let mut state = LiveInjectionState::new(true, true, &context("focus-a"), 0);
        let _ = state.plan_update("ship it", &context("focus-a"));

        assert_eq!(state.plan_update("shift it", &context("focus-a")), None);
        assert_eq!(state.plan_update("ship it", &context("focus-a")), None);
        assert_eq!(state.plan_update("shift it", &context("focus-a")), None);
        assert!(state.plan_update("shift it", &context("focus-a")).is_some());
    }

    #[test]
    fn final_append_only_appends_delta_when_live_text_is_already_correct() {
        let mut state = LiveInjectionState::new(true, true, &context("focus-a"), 0);
        let _ = state.plan_update("hello", &context("focus-a"));

        let decision = state.plan_finalize(
            &FinalizedOperation::Append,
            "hello world",
            &context("focus-a"),
        );
        assert_eq!(
            decision,
            FinalizeInjectionDecision::Apply(InjectionCommand {
                delete_graphemes: 0,
                text: " world".into(),
            })
        );
    }

    #[test]
    fn focus_change_freezes_live_injection_immediately() {
        let mut state = LiveInjectionState::new(true, true, &context("focus-a"), 0);
        assert_eq!(state.plan_update("hello", &context("focus-b")), None);
        assert!(state.is_frozen());
    }

    #[test]
    fn browser_surface_requires_extra_confirmation_for_destructive_live_updates() {
        let browser = context_with_surface("focus-a", SurfaceKind::Browser);
        let mut state = LiveInjectionState::new(true, true, &browser, 0);
        let _ = state.plan_update("hyperland", &browser);

        assert_eq!(state.plan_update("hyprland", &browser), None);
        assert_eq!(state.plan_update("hyprland", &browser), None);
        let correction = state
            .plan_update("hyprland", &browser)
            .expect("third confirmation should apply");
        assert_eq!(
            correction,
            InjectionCommand {
                delete_graphemes: 6,
                text: "rland".into()
            }
        );
    }

    #[test]
    fn unknown_surface_keeps_live_mode_append_only_but_allows_final_reconciliation() {
        let unknown = context_with_surface("focus-a", SurfaceKind::Unknown);
        let mut state = LiveInjectionState::new(true, true, &unknown, 0);
        let _ = state
            .plan_update("hello", &unknown)
            .expect("initial append");

        assert_eq!(state.plan_update("help", &unknown), None);
        assert_eq!(state.plan_update("help", &unknown), None);

        let decision = state.plan_finalize(&FinalizedOperation::Append, "help", &unknown);
        assert_eq!(
            decision,
            FinalizeInjectionDecision::Apply(InjectionCommand {
                delete_graphemes: 2,
                text: "p".into(),
            })
        );
    }

    #[test]
    fn final_reconciliation_respects_owned_delete_bounds() {
        let mut state = LiveInjectionState::new(true, true, &context("focus-a"), 3);
        let _ = state.plan_update("hello", &context("focus-a"));

        let blocked = state.plan_finalize(
            &FinalizedOperation::ReplaceLastEntry {
                entry_id: 7,
                delete_graphemes: 4,
            },
            "replacement",
            &context("focus-a"),
        );
        assert_eq!(blocked, FinalizeInjectionDecision::Blocked);

        let allowed = state.plan_finalize(
            &FinalizedOperation::ReplaceLastEntry {
                entry_id: 7,
                delete_graphemes: 3,
            },
            "replacement",
            &context("focus-a"),
        );
        assert_eq!(
            allowed,
            FinalizeInjectionDecision::Apply(InjectionCommand {
                delete_graphemes: 8,
                text: "replacement".into(),
            })
        );
    }
}

use std::path::Path;

use serde::{Deserialize, Serialize};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, get_lang_str,
};

use crate::config::TranscriptionConfig;
use crate::error::{Result, WhsprError};

pub trait TranscriptionBackend: Send + Sync {
    fn transcribe(&self, audio: &[f32], sample_rate: u32) -> Result<Transcript>;
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Transcript {
    pub raw_text: String,
    pub detected_language: Option<String>,
    pub segments: Vec<TranscriptSegment>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TranscriptSegment {
    pub text: String,
    pub start_ms: u32,
    pub end_ms: u32,
}

impl Transcript {
    fn empty(detected_language: Option<String>) -> Self {
        Self {
            raw_text: String::new(),
            detected_language,
            segments: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.raw_text.trim().is_empty()
    }

    fn append_chunk(&mut self, mut chunk: Transcript, offset_ms: u32) {
        if self.detected_language.is_none() {
            self.detected_language = chunk.detected_language.take();
        }

        let chunk_text = chunk.raw_text.trim();
        if !chunk_text.is_empty() {
            if !self.raw_text.is_empty() {
                self.raw_text.push(' ');
            }
            self.raw_text.push_str(chunk_text);
        }

        for mut segment in chunk.segments {
            segment.start_ms = segment.start_ms.saturating_add(offset_ms);
            segment.end_ms = segment.end_ms.saturating_add(offset_ms);
            self.segments.push(segment);
        }
    }
}

pub struct WhisperLocal {
    ctx: WhisperContext,
    language: String,
}

impl WhisperLocal {
    pub fn new(config: &TranscriptionConfig, model_path: &Path) -> Result<Self> {
        if !model_path.exists() {
            return Err(WhsprError::Transcription(format!(
                "model file not found: {}",
                model_path.display()
            )));
        }

        tracing::info!("loading whisper model from {}", model_path.display());

        let mut ctx_params = WhisperContextParameters::default();
        ctx_params.use_gpu(config.use_gpu);
        ctx_params.flash_attn(config.use_gpu && config.flash_attn);
        tracing::info!(
            "whisper acceleration: use_gpu={}, flash_attn={}",
            config.use_gpu,
            config.use_gpu && config.flash_attn
        );

        let model_path_str = model_path.to_str().ok_or_else(|| {
            WhsprError::Transcription(format!(
                "model path contains invalid UTF-8: {}",
                model_path.display()
            ))
        })?;

        let ctx = WhisperContext::new_with_params(model_path_str, ctx_params)
            .map_err(|e| {
                if config.use_gpu {
                    WhsprError::Transcription(format!(
                        "failed to load whisper model with GPU enabled: {e}. Set [transcription].use_gpu = false to force CPU."
                    ))
                } else {
                    WhsprError::Transcription(format!("failed to load whisper model: {e}"))
                }
            })?;

        tracing::info!("whisper model loaded successfully");

        Ok(Self {
            ctx,
            language: config.language.clone(),
        })
    }
}

const CHUNK_DURATION_SECS: f64 = 30.0;
const OVERLAP_SECS: f64 = 1.0;
const WHISPER_GREEDY_BEST_OF: i32 = 3;

/// Minimum RMS energy to consider audio as containing speech (~-40 dBFS).
const MIN_RMS_THRESHOLD: f32 = 0.01;
/// Minimum duration in seconds for meaningful speech input.
const MIN_DURATION_SECS: f64 = 0.3;

impl TranscriptionBackend for WhisperLocal {
    fn transcribe(&self, audio: &[f32], sample_rate: u32) -> Result<Transcript> {
        if audio.is_empty() || sample_rate == 0 {
            tracing::info!("empty audio or zero sample rate, skipping");
            return Ok(Transcript::empty(configured_language_hint(&self.language)));
        }

        // Audio diagnostics
        let duration_secs = audio.len() as f64 / sample_rate as f64;
        let rms = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt();
        tracing::info!(
            "audio: {:.1}s, {} samples, RMS={:.4}",
            duration_secs,
            audio.len(),
            rms
        );

        // Gate: skip silent or too-short audio to avoid Whisper hallucinations
        if duration_secs < MIN_DURATION_SECS {
            tracing::info!(
                "audio too short ({:.2}s < {:.1}s), skipping",
                duration_secs,
                MIN_DURATION_SECS
            );
            return Ok(Transcript::empty(configured_language_hint(&self.language)));
        }
        if rms < MIN_RMS_THRESHOLD {
            tracing::info!(
                "audio too quiet (RMS {:.4} < {}), skipping",
                rms,
                MIN_RMS_THRESHOLD
            );
            return Ok(Transcript::empty(configured_language_hint(&self.language)));
        }

        let chunk_size = (CHUNK_DURATION_SECS * sample_rate as f64) as usize;
        let overlap = (OVERLAP_SECS * sample_rate as f64) as usize;

        if audio.len() <= chunk_size {
            // Short audio: process directly
            self.transcribe_chunk(audio)
        } else {
            // Long audio: split into overlapping chunks
            let mut transcript = Transcript::empty(configured_language_hint(&self.language));
            let mut offset = 0;

            while offset < audio.len() {
                let end = (offset + chunk_size).min(audio.len());
                let chunk = &audio[offset..end];
                tracing::info!(
                    "processing chunk: {:.1}s - {:.1}s",
                    offset as f64 / sample_rate as f64,
                    end as f64 / sample_rate as f64
                );

                let chunk_transcript = self.transcribe_chunk(chunk)?;
                let offset_ms = samples_to_ms(offset, sample_rate);
                transcript.append_chunk(chunk_transcript, offset_ms);

                if end == audio.len() {
                    break;
                }
                offset = end - overlap;
            }

            tracing::info!("transcription result: {:?}", transcript.raw_text);
            Ok(transcript)
        }
    }
}

impl WhisperLocal {
    fn transcribe_chunk(&self, audio: &[f32]) -> Result<Transcript> {
        let mut params = FullParams::new(SamplingStrategy::Greedy {
            best_of: WHISPER_GREEDY_BEST_OF,
        });

        if self.language == "auto" {
            params.set_language(None);
        } else {
            params.set_language(Some(&self.language));
        }
        params.set_translate(false);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);
        params.set_n_threads(n_threads);

        let mut state = self.ctx.create_state().map_err(|e| {
            WhsprError::Transcription(format!("failed to create whisper state: {e}"))
        })?;

        state
            .full(params, audio)
            .map_err(|e| WhsprError::Transcription(format!("transcription failed: {e}")))?;

        let detected_language = if self.language == "auto" {
            get_lang_str(state.full_lang_id_from_state()).map(ToOwned::to_owned)
        } else {
            Some(self.language.clone())
        };

        let mut raw_text = String::new();
        let mut segments = Vec::new();
        let num_segments = state.full_n_segments();
        for i in 0..num_segments {
            let Some(segment) = state.get_segment(i) else {
                continue;
            };

            let text = read_segment_text(i, &segment);
            let trimmed = text.trim();
            if trimmed.is_empty() {
                continue;
            }

            raw_text.push_str(&text);
            segments.push(TranscriptSegment {
                text: trimmed.to_string(),
                start_ms: centiseconds_to_ms(segment.start_timestamp()),
                end_ms: centiseconds_to_ms(segment.end_timestamp()),
            });
        }

        let raw_text = raw_text.trim().to_string();
        if !raw_text.is_empty() {
            tracing::debug!("chunk transcription: {raw_text:?}");
        }

        Ok(Transcript {
            raw_text,
            detected_language,
            segments,
        })
    }
}

fn read_segment_text(i: i32, segment: &whisper_rs::WhisperSegment<'_>) -> String {
    match segment.to_str() {
        Ok(s) => s.to_string(),
        Err(_) => match segment.to_str_lossy() {
            Ok(lossy) => {
                tracing::warn!("segment {i} contains invalid UTF-8, using lossy conversion");
                lossy.to_string()
            }
            Err(_) => String::new(),
        },
    }
}

fn configured_language_hint(language: &str) -> Option<String> {
    (language != "auto").then(|| language.to_string())
}

fn centiseconds_to_ms(value: i64) -> u32 {
    value.saturating_mul(10).clamp(0, u32::MAX as i64) as u32
}

fn samples_to_ms(samples: usize, sample_rate: u32) -> u32 {
    if sample_rate == 0 {
        return 0;
    }

    ((samples as u64).saturating_mul(1000) / sample_rate as u64).min(u32::MAX as u64) as u32
}

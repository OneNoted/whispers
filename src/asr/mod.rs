pub mod cleanup;
pub mod validation;

use crate::cloud::CloudService;
use crate::config::{Config, TranscriptionBackend, TranscriptionConfig, TranscriptionFallback};
use crate::error::{Result, WhsprError};
use crate::faster_whisper::{self, FasterWhisperService};
use crate::model;
use crate::nemo_asr::{self, NemoAsrService};
use crate::transcribe::{Transcript, TranscriptionBackend as _, WhisperLocal};

pub enum PreparedTranscriber {
    Whisper(tokio::task::JoinHandle<Result<WhisperLocal>>),
    Faster(FasterWhisperService),
    Nemo(NemoAsrService),
    Cloud(CloudService),
}

pub fn prepare_transcriber(config: &Config) -> Result<PreparedTranscriber> {
    cleanup::cleanup_stale_transcribers(config)?;

    match config.transcription.backend {
        TranscriptionBackend::WhisperCpp => {
            let whisper_config = config.transcription.clone();
            let model_path = config.resolved_model_path();
            Ok(PreparedTranscriber::Whisper(tokio::task::spawn_blocking(
                move || WhisperLocal::new(&whisper_config, &model_path),
            )))
        }
        TranscriptionBackend::FasterWhisper => {
            faster_whisper::prepare_service(&config.transcription)
                .map(PreparedTranscriber::Faster)
                .ok_or_else(|| {
                    WhsprError::Transcription(
                        "faster-whisper backend selected but no model path could be resolved"
                            .into(),
                    )
                })
        }
        TranscriptionBackend::Nemo => nemo_asr::prepare_service(&config.transcription)
            .map(PreparedTranscriber::Nemo)
            .ok_or_else(|| {
                WhsprError::Transcription(
                    "nemo backend selected but no model reference could be resolved".into(),
                )
            }),
        TranscriptionBackend::Cloud => Ok(PreparedTranscriber::Cloud(CloudService::new(config)?)),
    }
}

pub fn prewarm_transcriber(prepared: &PreparedTranscriber, phase: &str) {
    match prepared {
        PreparedTranscriber::Faster(service) => match service.prewarm() {
            Ok(()) => tracing::info!("prewarming faster-whisper worker via {}", phase),
            Err(err) => tracing::warn!("failed to prewarm faster-whisper worker: {err}"),
        },
        PreparedTranscriber::Nemo(service) => match service.prewarm() {
            Ok(()) => tracing::info!("prewarming NeMo ASR worker via {}", phase),
            Err(err) => tracing::warn!("failed to prewarm NeMo ASR worker: {err}"),
        },
        _ => {}
    }
}

pub async fn transcribe_audio(
    config: &Config,
    prepared: PreparedTranscriber,
    audio: Vec<f32>,
    sample_rate: u32,
) -> Result<Transcript> {
    match prepared {
        PreparedTranscriber::Whisper(handle) => {
            let backend = handle.await.map_err(|e| {
                WhsprError::Transcription(format!("model loading task failed: {e}"))
            })??;
            tokio::task::spawn_blocking(move || backend.transcribe(&audio, sample_rate))
                .await
                .map_err(|e| WhsprError::Transcription(format!("transcription task failed: {e}")))?
        }
        PreparedTranscriber::Faster(service) => match service.transcribe(&audio, sample_rate).await
        {
            Ok(transcript) => Ok(transcript),
            Err(err) => {
                tracing::warn!("faster-whisper transcription failed: {err}");
                fallback_whisper_cpp_transcribe(config, audio, sample_rate).await
            }
        },
        PreparedTranscriber::Nemo(service) => match service.transcribe(&audio, sample_rate).await {
            Ok(transcript) => Ok(transcript),
            Err(err) => {
                tracing::warn!("NeMo ASR transcription failed: {err}");
                fallback_whisper_cpp_transcribe(config, audio, sample_rate).await
            }
        },
        PreparedTranscriber::Cloud(service) => {
            match service.transcribe_audio(config, &audio, sample_rate).await {
                Ok(transcript) => Ok(transcript),
                Err(err) => {
                    tracing::warn!("cloud transcription failed: {err}");
                    fallback_local_transcribe(config, audio, sample_rate).await
                }
            }
        }
    }
}

async fn fallback_local_transcribe(
    config: &Config,
    audio: Vec<f32>,
    sample_rate: u32,
) -> Result<Transcript> {
    if config.transcription.backend == TranscriptionBackend::Cloud
        && config.transcription.fallback == TranscriptionFallback::None
    {
        return Err(WhsprError::Transcription(
            "cloud transcription failed and [transcription].fallback = \"none\"".into(),
        ));
    }

    let mut local_config = config.transcription.clone();
    local_config.backend = config.transcription.resolved_local_backend();
    let model_path = config.resolved_model_path();
    tracing::warn!(
        "falling back to local ASR backend '{}' using {}",
        local_config.backend.as_str(),
        model_path.display()
    );
    let prepared = match local_config.backend {
        TranscriptionBackend::WhisperCpp => {
            let whisper_config = local_config.clone();
            Ok(PreparedTranscriber::Whisper(tokio::task::spawn_blocking(
                move || WhisperLocal::new(&whisper_config, &model_path),
            )))
        }
        TranscriptionBackend::FasterWhisper => faster_whisper::prepare_service(&local_config)
            .map(PreparedTranscriber::Faster)
            .ok_or_else(|| {
                WhsprError::Transcription(
                    "faster-whisper fallback selected but no model path could be resolved".into(),
                )
            }),
        TranscriptionBackend::Nemo => nemo_asr::prepare_service(&local_config)
            .map(PreparedTranscriber::Nemo)
            .ok_or_else(|| {
                WhsprError::Transcription(
                    "nemo fallback selected but no model reference could be resolved".into(),
                )
            }),
        TranscriptionBackend::Cloud => Err(WhsprError::Transcription(
            "cloud backend cannot be prepared as a local transcriber".into(),
        )),
    }?;
    match prepared {
        PreparedTranscriber::Whisper(handle) => {
            let backend = handle.await.map_err(|e| {
                WhsprError::Transcription(format!("fallback model loading task failed: {e}"))
            })??;
            tokio::task::spawn_blocking(move || backend.transcribe(&audio, sample_rate))
                .await
                .map_err(|e| {
                    WhsprError::Transcription(format!("fallback transcription task failed: {e}"))
                })?
        }
        PreparedTranscriber::Faster(service) => service.transcribe(&audio, sample_rate).await,
        PreparedTranscriber::Nemo(service) => service.transcribe(&audio, sample_rate).await,
        PreparedTranscriber::Cloud(_) => Err(WhsprError::Transcription(
            "cloud fallback resolved to cloud backend".into(),
        )),
    }
}

async fn fallback_whisper_cpp_transcribe(
    config: &Config,
    audio: Vec<f32>,
    sample_rate: u32,
) -> Result<Transcript> {
    let Some(model_path) = fallback_whisper_model_path() else {
        return Err(WhsprError::Transcription(
            "faster-whisper failed and no local large-v3-turbo fallback model is available".into(),
        ));
    };
    tracing::warn!("falling back to whisper_cpp using {}", model_path.display());
    let whisper_config = whisper_fallback_config(&config.transcription);
    let backend =
        tokio::task::spawn_blocking(move || WhisperLocal::new(&whisper_config, &model_path))
            .await
            .map_err(|e| {
                WhsprError::Transcription(format!("fallback model loading task failed: {e}"))
            })??;
    tokio::task::spawn_blocking(move || backend.transcribe(&audio, sample_rate))
        .await
        .map_err(|e| {
            WhsprError::Transcription(format!("fallback transcription task failed: {e}"))
        })?
}

fn whisper_fallback_config(config: &TranscriptionConfig) -> TranscriptionConfig {
    let mut fallback = config.clone();
    fallback.backend = TranscriptionBackend::WhisperCpp;
    fallback.local_backend = TranscriptionBackend::WhisperCpp;
    fallback.selected_model = "large-v3-turbo".into();
    fallback.model_path = model::model_path_for_config("ggml-large-v3-turbo.bin");
    fallback
}

fn fallback_whisper_model_path() -> Option<std::path::PathBuf> {
    let path = model::selected_model_local_path("large-v3-turbo")?;
    path.exists().then_some(path)
}

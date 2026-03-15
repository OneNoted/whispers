use crate::cloud::CloudService;
use crate::config::{Config, TranscriptionBackend, TranscriptionConfig};
use crate::error::{Result, WhsprError};
use crate::faster_whisper::{self, FasterWhisperService};
use crate::nemo_asr::{self, NemoAsrService};
use crate::transcribe::WhisperLocal;
use std::path::Path;

pub enum PreparedTranscriber {
    Whisper(tokio::task::JoinHandle<Result<WhisperLocal>>),
    Faster(FasterWhisperService),
    Nemo(NemoAsrService),
    Cloud(CloudService),
}

pub fn prepare_transcriber(config: &Config) -> Result<PreparedTranscriber> {
    super::cleanup::cleanup_stale_transcribers(config)?;

    if config.transcription.backend == TranscriptionBackend::Cloud {
        return Ok(PreparedTranscriber::Cloud(CloudService::new(config)?));
    }

    prepare_local_transcriber(&config.transcription, &config.resolved_model_path())
}

pub(crate) fn prepare_local_transcriber(
    transcription: &TranscriptionConfig,
    model_path: &Path,
) -> Result<PreparedTranscriber> {
    match transcription.backend {
        TranscriptionBackend::WhisperCpp => {
            let whisper_config = transcription.clone();
            let model_path = model_path.to_path_buf();
            Ok(PreparedTranscriber::Whisper(tokio::task::spawn_blocking(
                move || WhisperLocal::new(&whisper_config, &model_path),
            )))
        }
        TranscriptionBackend::FasterWhisper => faster_whisper::prepare_service(transcription)
            .map(PreparedTranscriber::Faster)
            .ok_or_else(|| {
                WhsprError::Transcription(
                    "faster-whisper backend selected but no model path could be resolved".into(),
                )
            }),
        TranscriptionBackend::Nemo => nemo_asr::prepare_service(transcription)
            .map(PreparedTranscriber::Nemo)
            .ok_or_else(|| {
                WhsprError::Transcription(
                    "nemo backend selected but no model reference could be resolved".into(),
                )
            }),
        TranscriptionBackend::Cloud => Err(WhsprError::Transcription(
            "cloud backend cannot be prepared as a local transcriber".into(),
        )),
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

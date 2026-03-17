use super::prepare::{self, PreparedTranscriber};
use crate::config::{Config, TranscriptionBackend, TranscriptionConfig, TranscriptionFallback};
use crate::error::{Result, WhsprError};
use crate::model;
use crate::transcribe::{Transcript, TranscriptionBackend as _};

pub async fn transcribe_audio(
    config: &Config,
    prepared: PreparedTranscriber,
    audio: Vec<f32>,
    sample_rate: u32,
) -> Result<Transcript> {
    match prepared {
        prepared @ PreparedTranscriber::Whisper(_) => {
            transcribe_with_prepared(prepared, &audio, sample_rate, "").await
        }
        prepared @ PreparedTranscriber::Faster(_) => {
            match transcribe_with_prepared(prepared, &audio, sample_rate, "").await {
                Ok(transcript) => Ok(transcript),
                Err(err) => {
                    tracing::warn!("faster-whisper transcription failed: {err}");
                    fallback_whisper_cpp_transcribe(config, audio, sample_rate).await
                }
            }
        }
        prepared @ PreparedTranscriber::Nemo(_) => {
            match transcribe_with_prepared(prepared, &audio, sample_rate, "").await {
                Ok(transcript) => Ok(transcript),
                Err(err) => {
                    tracing::warn!("NeMo ASR transcription failed: {err}");
                    fallback_whisper_cpp_transcribe(config, audio, sample_rate).await
                }
            }
        }
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

async fn transcribe_with_prepared(
    prepared: PreparedTranscriber,
    audio: &[f32],
    sample_rate: u32,
    task_label: &str,
) -> Result<Transcript> {
    match prepared {
        PreparedTranscriber::Whisper(handle) => {
            let audio = audio.to_vec();
            let backend = handle
                .await
                .map_err(|e| transcription_task_error(task_label, "model loading", &e))??;
            tokio::task::spawn_blocking(move || backend.transcribe(&audio, sample_rate))
                .await
                .map_err(|e| transcription_task_error(task_label, "transcription", &e))?
        }
        PreparedTranscriber::Faster(service) => service.transcribe(audio, sample_rate).await,
        PreparedTranscriber::Nemo(service) => service.transcribe(audio, sample_rate).await,
        PreparedTranscriber::Cloud(_) => Err(WhsprError::Transcription(
            "cloud transcriber cannot be executed without the caller-owned config".into(),
        )),
    }
}

async fn fallback_local_transcribe(
    config: &Config,
    audio: Vec<f32>,
    sample_rate: u32,
) -> Result<Transcript> {
    let (local_config, model_path) = local_fallback_config(config)?;
    tracing::warn!(
        "falling back to local ASR backend '{}' using {}",
        local_config.backend.as_str(),
        model_path.display()
    );
    let prepared = prepare::prepare_local_transcriber(&local_config, &model_path)?;
    transcribe_with_prepared(prepared, &audio, sample_rate, "fallback").await
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
    let prepared = prepare::prepare_local_transcriber(&whisper_config, &model_path)?;
    transcribe_with_prepared(prepared, &audio, sample_rate, "fallback").await
}

fn local_fallback_config(config: &Config) -> Result<(TranscriptionConfig, std::path::PathBuf)> {
    if config.transcription.backend == TranscriptionBackend::Cloud
        && config.transcription.fallback == TranscriptionFallback::None
    {
        return Err(WhsprError::Transcription(
            "cloud transcription failed and [transcription].fallback = \"none\"".into(),
        ));
    }

    let mut local_config = config.transcription.clone();
    local_config.backend = config.transcription.resolved_local_backend();
    Ok((local_config, config.resolved_model_path()))
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

fn transcription_task_error(
    task_label: &str,
    phase: &str,
    error: &tokio::task::JoinError,
) -> WhsprError {
    let prefix = if task_label.is_empty() {
        String::new()
    } else {
        format!("{task_label} ")
    };
    WhsprError::Transcription(format!("{prefix}{phase} task failed: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn cloud_fallback_none_returns_existing_error() {
        let mut config = Config::default();
        config.transcription.backend = TranscriptionBackend::Cloud;
        config.transcription.fallback = TranscriptionFallback::None;

        let err = fallback_local_transcribe(&config, vec![0.0; 16], 16_000)
            .await
            .expect_err("fallback should fail");
        match err {
            WhsprError::Transcription(message) => {
                assert_eq!(
                    message,
                    "cloud transcription failed and [transcription].fallback = \"none\""
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn local_fallback_config_resolves_configured_local_backend() {
        let mut config = Config::default();
        config.transcription.backend = TranscriptionBackend::Cloud;
        config.transcription.local_backend = TranscriptionBackend::Nemo;

        let (local_config, _) = local_fallback_config(&config).expect("fallback config");
        assert_eq!(local_config.backend, TranscriptionBackend::Nemo);
    }

    #[test]
    fn whisper_fallback_config_pins_whisper_cpp_large_v3_turbo() {
        let fallback = whisper_fallback_config(&TranscriptionConfig::default());
        assert_eq!(fallback.backend, TranscriptionBackend::WhisperCpp);
        assert_eq!(fallback.local_backend, TranscriptionBackend::WhisperCpp);
        assert_eq!(fallback.selected_model, "large-v3-turbo");
        assert_eq!(
            fallback.model_path,
            model::model_path_for_config("ggml-large-v3-turbo.bin")
        );
    }
}

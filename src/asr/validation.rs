use crate::config::{Config, TranscriptionBackend};
use crate::error::{Result, WhsprError};

pub fn validate_transcription_config(config: &Config) -> Result<()> {
    if config.transcription.backend == TranscriptionBackend::Cloud {
        crate::cloud::validate_config(config)?;
    }

    if config.transcription.resolved_local_backend() == TranscriptionBackend::FasterWhisper
        && !config.transcription.language.eq_ignore_ascii_case("en")
        && !config.transcription.language.eq_ignore_ascii_case("auto")
    {
        return Err(WhsprError::Config(
            "faster-whisper managed models are currently English-focused; set [transcription].language = \"en\" or \"auto\"".into(),
        ));
    }

    if config.transcription.resolved_local_backend() == TranscriptionBackend::FasterWhisper
        && config.transcription.language.eq_ignore_ascii_case("auto")
    {
        tracing::warn!(
            "faster-whisper backend is configured with language = \"auto\"; English dictation is recommended"
        );
    }

    if config.transcription.resolved_local_backend() == TranscriptionBackend::Nemo
        && !config.transcription.language.eq_ignore_ascii_case("en")
        && !config.transcription.language.eq_ignore_ascii_case("auto")
    {
        return Err(WhsprError::Config(
            "NeMo experimental ASR models are currently English-only; set [transcription].language = \"en\" or \"auto\"".into(),
        ));
    }

    Ok(())
}

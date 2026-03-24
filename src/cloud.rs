use std::time::Duration;
use std::time::Instant;

use flacenc::bitsink::ByteSink;
use flacenc::component::BitRepr;
use flacenc::error::Verify;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use reqwest::multipart::{Form, Part};
use serde::Deserialize;

use crate::config::{
    CloudConfig, CloudLanguageMode, CloudProvider, Config, RewriteBackend, TranscriptionBackend,
};
use crate::error::{Result, WhsprError};
use crate::personalization;
use crate::rewrite::{
    RewritePrompt, build_prompt as build_rewrite_prompt, resolved_profile_for_cloud,
    sanitize_rewrite_output,
};
use crate::rewrite_protocol::RewriteTranscript;
use crate::transcribe::{Transcript, TranscriptSegment};

#[derive(Debug, Clone)]
struct EncodedAudioUpload {
    bytes: Vec<u8>,
    file_name: &'static str,
    mime_type: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WavLengths {
    data_len: usize,
    data_len_u32: u32,
    riff_len_u32: u32,
}

#[derive(Clone)]
pub struct CloudService {
    client: reqwest::Client,
    base_url: String,
}

trait ProviderAdapter {
    fn base_url(&self, config: &CloudConfig) -> Result<String>;
}

struct OpenAiAdapter;
struct OpenAiCompatibleAdapter;

impl ProviderAdapter for OpenAiAdapter {
    fn base_url(&self, config: &CloudConfig) -> Result<String> {
        Ok(normalize_base_url(if config.base_url.trim().is_empty() {
            "https://api.openai.com/v1"
        } else {
            config.base_url.trim()
        }))
    }
}

impl ProviderAdapter for OpenAiCompatibleAdapter {
    fn base_url(&self, config: &CloudConfig) -> Result<String> {
        if config.base_url.trim().is_empty() {
            return Err(WhsprError::Config(
                "cloud.provider = \"openai_compatible\" requires [cloud].base_url".into(),
            ));
        }
        Ok(normalize_base_url(config.base_url.trim()))
    }
}

impl CloudService {
    pub fn new(config: &Config) -> Result<Self> {
        validate_config(config)?;
        let adapter = adapter(config.cloud.provider);
        let base_url = adapter.base_url(&config.cloud)?;
        let api_key = api_key(config)?;
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {api_key}"))
                .map_err(|e| WhsprError::Config(format!("invalid cloud API key value: {e}")))?,
        );

        let client = reqwest::Client::builder()
            .default_headers(headers)
            .connect_timeout(Duration::from_millis(config.cloud.connect_timeout_ms))
            .timeout(Duration::from_millis(config.cloud.request_timeout_ms))
            .build()
            .map_err(|e| WhsprError::Config(format!("failed to build cloud HTTP client: {e}")))?;

        Ok(Self { client, base_url })
    }

    pub async fn check(&self) -> Result<()> {
        let started = Instant::now();
        let url = format!("{}/models", self.base_url);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| WhsprError::Config(format!("cloud connectivity check failed: {e}")))?;
        if !response.status().is_success() {
            return Err(WhsprError::Config(format!(
                "cloud connectivity check failed with HTTP {}",
                response.status()
            )));
        }
        tracing::info!(
            elapsed_ms = started.elapsed().as_millis(),
            "cloud connectivity check finished"
        );
        Ok(())
    }

    pub async fn transcribe_audio(
        &self,
        config: &Config,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<Transcript> {
        let started = Instant::now();
        let upload = encode_preferred_audio_upload(audio, sample_rate)?;
        let upload_bytes = upload.bytes.len();
        let form = build_transcription_form(config, upload)?;
        let url = format!("{}/audio/transcriptions", self.base_url);
        let request_started = Instant::now();
        let response = self
            .client
            .post(url)
            .multipart(form)
            .send()
            .await
            .map_err(|e| WhsprError::Transcription(format!("cloud ASR request failed: {e}")))?;
        if !response.status().is_success() {
            return Err(WhsprError::Transcription(format!(
                "cloud ASR request failed with HTTP {}",
                response.status()
            )));
        }
        tracing::info!(
            elapsed_ms = request_started.elapsed().as_millis(),
            upload_bytes,
            audio_duration_ms = ((audio.len() as f64 / sample_rate as f64) * 1000.0).round() as u64,
            "cloud ASR HTTP request finished"
        );

        let bytes = response.bytes().await.map_err(|e| {
            WhsprError::Transcription(format!("failed to read cloud ASR response: {e}"))
        })?;
        let transcript = parse_transcription_response(
            &bytes,
            audio.len(),
            sample_rate,
            configured_language(config),
        )?;
        tracing::info!(
            elapsed_ms = started.elapsed().as_millis(),
            transcript_chars = transcript.raw_text.len(),
            segments = transcript.segments.len(),
            "cloud ASR stage finished"
        );
        Ok(transcript)
    }

    pub async fn rewrite_transcript(
        &self,
        config: &Config,
        transcript: &RewriteTranscript,
        custom_instructions: Option<&str>,
    ) -> Result<String> {
        let started = Instant::now();
        let profile = resolved_profile_for_cloud(config.rewrite.profile);
        let prompt = build_rewrite_prompt(transcript, profile, custom_instructions)
            .map_err(|e| WhsprError::Rewrite(format!("failed to build rewrite prompt: {e}")))?;
        let url = format!("{}/chat/completions", self.base_url);
        let request = ChatCompletionsRequest::from_prompt(config, &prompt);
        let request_started = Instant::now();
        let response = self
            .client
            .post(url)
            .json(&request)
            .send()
            .await
            .map_err(|e| WhsprError::Rewrite(format!("cloud rewrite request failed: {e}")))?;
        if !response.status().is_success() {
            return Err(WhsprError::Rewrite(format!(
                "cloud rewrite request failed with HTTP {}",
                response.status()
            )));
        }
        tracing::info!(
            elapsed_ms = request_started.elapsed().as_millis(),
            raw_chars = transcript.raw_text.len(),
            correction_chars = transcript.correction_aware_text.len(),
            "cloud rewrite HTTP request finished"
        );

        let response: ChatCompletionsResponse = response.json().await.map_err(|e| {
            WhsprError::Rewrite(format!("failed to decode cloud rewrite response: {e}"))
        })?;
        let content = response
            .choices
            .into_iter()
            .find_map(|choice| choice.message.content.into_text())
            .ok_or_else(|| {
                WhsprError::Rewrite("cloud rewrite response did not include text".into())
            })?;
        let sanitized = sanitize_rewrite_output(&content);
        if sanitized.is_empty() {
            return Err(WhsprError::Rewrite(
                "cloud rewrite response sanitized to empty text".into(),
            ));
        }
        tracing::info!(
            elapsed_ms = started.elapsed().as_millis(),
            output_chars = sanitized.len(),
            "cloud rewrite stage finished"
        );
        Ok(sanitized)
    }
}

pub fn validate_config(config: &Config) -> Result<()> {
    let uses_cloud_asr = config.transcription.backend == TranscriptionBackend::Cloud;
    let uses_cloud_rewrite =
        config.postprocess.mode.uses_rewrite() && config.rewrite.backend == RewriteBackend::Cloud;
    if !uses_cloud_asr && !uses_cloud_rewrite {
        return Ok(());
    }

    let adapter = adapter(config.cloud.provider);
    let _ = adapter.base_url(&config.cloud)?;
    if config.cloud.api_key.trim().is_empty() && config.cloud.api_key_env.trim().is_empty() {
        return Err(WhsprError::Config(
            "either [cloud].api_key or [cloud].api_key_env must be set".into(),
        ));
    }
    let _ = api_key(config)?;

    if uses_cloud_asr && config.cloud.transcription.model.trim().is_empty() {
        return Err(WhsprError::Config(
            "[cloud.transcription].model must not be empty when transcription.backend = \"cloud\""
                .into(),
        ));
    }
    if uses_cloud_rewrite && config.cloud.rewrite.model.trim().is_empty() {
        return Err(WhsprError::Config(
            "[cloud.rewrite].model must not be empty when rewrite.backend = \"cloud\"".into(),
        ));
    }
    if uses_cloud_asr && config.transcription.local_backend == TranscriptionBackend::Cloud {
        return Err(WhsprError::Config(
            "[transcription].local_backend must be a local backend".into(),
        ));
    }
    Ok(())
}

fn adapter(provider: CloudProvider) -> Box<dyn ProviderAdapter + Send + Sync> {
    match provider {
        CloudProvider::OpenAi => Box::new(OpenAiAdapter),
        CloudProvider::OpenAiCompatible => Box::new(OpenAiCompatibleAdapter),
    }
}

fn api_key(config: &Config) -> Result<String> {
    if !config.cloud.api_key.trim().is_empty() {
        return Ok(config.cloud.api_key.trim().to_string());
    }
    std::env::var(&config.cloud.api_key_env).map_err(|_| {
        WhsprError::Config(format!(
            "cloud API key env var {} is not set",
            config.cloud.api_key_env
        ))
    })
}

fn normalize_base_url(base_url: &str) -> String {
    base_url.trim_end_matches('/').to_string()
}

fn build_transcription_form(config: &Config, upload: EncodedAudioUpload) -> Result<Form> {
    let mut form = Form::new().text("model", config.cloud.transcription.model.clone());
    let language = match config.cloud.transcription.language_mode {
        CloudLanguageMode::InheritLocal => {
            let local = config.transcription.language.trim();
            (!local.is_empty() && !local.eq_ignore_ascii_case("auto")).then(|| local.to_string())
        }
        CloudLanguageMode::Force => {
            let value = config.cloud.transcription.language.trim();
            (!value.is_empty()).then(|| value.to_string())
        }
    };
    if let Some(language) = language {
        form = form.text("language", language);
    }
    if let Some(prompt) = load_transcription_prompt(config) {
        form = form.text("prompt", prompt);
    }

    let part = Part::bytes(upload.bytes)
        .file_name(upload.file_name)
        .mime_str(upload.mime_type)
        .map_err(|e| {
            WhsprError::Transcription(format!(
                "failed to prepare {} upload: {e}",
                upload.file_name
            ))
        })?;
    Ok(form.part("file", part))
}

fn load_transcription_prompt(config: &Config) -> Option<String> {
    let rules = personalization::load_rules(config).ok()?;
    personalization::transcription_prompt(&rules)
}

fn encode_preferred_audio_upload(audio: &[f32], sample_rate: u32) -> Result<EncodedAudioUpload> {
    match encode_flac_pcm16(audio, sample_rate) {
        Ok(bytes) => Ok(EncodedAudioUpload {
            bytes,
            file_name: "dictation.flac",
            mime_type: "audio/flac",
        }),
        Err(err) => {
            tracing::warn!("FLAC encoding failed, falling back to WAV upload: {err}");
            Ok(EncodedAudioUpload {
                bytes: encode_wav_pcm16(audio, sample_rate)?,
                file_name: "dictation.wav",
                mime_type: "audio/wav",
            })
        }
    }
}

fn encode_flac_pcm16(audio: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let pcm: Vec<i32> = audio
        .iter()
        .map(|sample| ((*sample).clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16 as i32)
        .collect();
    let config = flacenc::config::Encoder::default()
        .into_verified()
        .map_err(|(_, err)| {
            WhsprError::Audio(format!("failed to verify FLAC encoder config: {err}"))
        })?;
    let source = flacenc::source::MemSource::from_samples(&pcm, 1, 16, sample_rate as usize);
    let flac_stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
        .map_err(|e| WhsprError::Audio(format!("FLAC encoding failed: {e}")))?;
    let mut sink = ByteSink::new();
    flac_stream
        .write(&mut sink)
        .map_err(|e| WhsprError::Audio(format!("failed to serialize FLAC stream: {e}")))?;
    Ok(sink.as_slice().to_vec())
}

fn wav_lengths(sample_count: usize, bytes_per_sample: usize) -> Result<WavLengths> {
    let size_error = || WhsprError::Audio("audio buffer too large to encode as WAV".into());
    let data_len = sample_count
        .checked_mul(bytes_per_sample)
        .ok_or_else(size_error)?;
    let data_len_u32 = u32::try_from(data_len).map_err(|_| size_error())?;
    let riff_len_u32 = 36u32.checked_add(data_len_u32).ok_or_else(size_error)?;
    Ok(WavLengths {
        data_len,
        data_len_u32,
        riff_len_u32,
    })
}

fn encode_wav_pcm16(audio: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let channels: u16 = 1;
    let bits_per_sample: u16 = 16;
    let bytes_per_sample = (bits_per_sample / 8) as usize;
    let lengths = wav_lengths(audio.len(), bytes_per_sample)?;
    let data_len = lengths.data_len;
    let mut bytes = Vec::with_capacity(44 + data_len);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&lengths.riff_len_u32.to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&channels.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    let byte_rate = sample_rate * u32::from(channels) * u32::from(bits_per_sample) / 8;
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    let block_align = channels * bits_per_sample / 8;
    bytes.extend_from_slice(&block_align.to_le_bytes());
    bytes.extend_from_slice(&bits_per_sample.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&lengths.data_len_u32.to_le_bytes());
    for sample in audio {
        let pcm = (*sample).clamp(-1.0, 1.0);
        let pcm = (pcm * i16::MAX as f32).round() as i16;
        bytes.extend_from_slice(&pcm.to_le_bytes());
    }
    Ok(bytes)
}

fn parse_transcription_response(
    bytes: &[u8],
    sample_count: usize,
    sample_rate: u32,
    configured_language: Option<String>,
) -> Result<Transcript> {
    if let Ok(response) = serde_json::from_slice::<TranscriptionResponse>(bytes) {
        let duration_ms = samples_to_ms(sample_count, sample_rate);
        let text = response.text.trim().to_string();
        let detected_language = response.language.or(configured_language);
        let segments = response
            .segments
            .unwrap_or_default()
            .into_iter()
            .filter_map(|segment| {
                let text = segment.text.as_deref()?.trim().to_string();
                if text.is_empty() {
                    return None;
                }
                Some(TranscriptSegment {
                    text,
                    start_ms: segment.start_ms(duration_ms),
                    end_ms: segment.end_ms(duration_ms),
                })
            })
            .collect::<Vec<_>>();
        let segments = if text.is_empty() {
            Vec::new()
        } else if segments.is_empty() {
            vec![TranscriptSegment {
                text: text.clone(),
                start_ms: 0,
                end_ms: duration_ms,
            }]
        } else {
            segments
        };
        return Ok(Transcript {
            raw_text: text,
            detected_language,
            segments,
        });
    }

    let text = String::from_utf8(bytes.to_vec()).map_err(|e| {
        WhsprError::Transcription(format!(
            "cloud ASR response was neither JSON nor UTF-8 text: {e}"
        ))
    })?;
    let text = text.trim().to_string();
    let duration_ms = samples_to_ms(sample_count, sample_rate);
    Ok(Transcript {
        raw_text: text.clone(),
        detected_language: configured_language,
        segments: if text.is_empty() {
            Vec::new()
        } else {
            vec![TranscriptSegment {
                text,
                start_ms: 0,
                end_ms: duration_ms,
            }]
        },
    })
}

fn configured_language(config: &Config) -> Option<String> {
    let language = config.transcription.language.trim();
    (!language.is_empty() && !language.eq_ignore_ascii_case("auto")).then(|| language.to_string())
}

fn samples_to_ms(sample_count: usize, sample_rate: u32) -> u32 {
    if sample_rate == 0 {
        return 0;
    }
    ((sample_count as f64 / sample_rate as f64) * 1000.0).round() as u32
}

#[derive(serde::Serialize)]
struct ChatCompletionsRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
    max_tokens: usize,
}

impl ChatCompletionsRequest {
    fn from_prompt(config: &Config, prompt: &RewritePrompt) -> Self {
        Self {
            model: config.cloud.rewrite.model.clone(),
            messages: vec![
                ChatMessage {
                    role: "system",
                    content: prompt.system.clone(),
                },
                ChatMessage {
                    role: "user",
                    content: prompt.user.clone(),
                },
            ],
            temperature: config.cloud.rewrite.temperature,
            max_tokens: config
                .cloud
                .rewrite
                .max_output_tokens
                .min(config.rewrite.max_tokens),
        }
    }
}

#[derive(serde::Serialize)]
struct ChatMessage {
    role: &'static str,
    content: String,
}

#[derive(Deserialize)]
struct ChatCompletionsResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(Deserialize)]
struct ChatChoiceMessage {
    content: ChatMessageContent,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ChatMessageContent {
    Text(String),
    Parts(Vec<ChatMessagePart>),
}

impl ChatMessageContent {
    fn into_text(self) -> Option<String> {
        match self {
            Self::Text(text) => Some(text),
            Self::Parts(parts) => {
                let text = parts
                    .into_iter()
                    .filter_map(|part| part.text)
                    .collect::<Vec<_>>()
                    .join("");
                (!text.is_empty()).then_some(text)
            }
        }
    }
}

#[derive(Deserialize)]
struct ChatMessagePart {
    #[serde(default)]
    text: Option<String>,
}

#[derive(Deserialize)]
struct TranscriptionResponse {
    text: String,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    segments: Option<Vec<TranscriptionSegmentResponse>>,
}

#[derive(Deserialize)]
struct TranscriptionSegmentResponse {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    start: Option<f64>,
    #[serde(default)]
    end: Option<f64>,
    #[serde(default)]
    start_ms: Option<u32>,
    #[serde(default)]
    end_ms: Option<u32>,
}

impl TranscriptionSegmentResponse {
    fn start_ms(&self, duration_ms: u32) -> u32 {
        self.start_ms
            .or_else(|| self.start.map(|secs| (secs * 1000.0).round() as u32))
            .unwrap_or(0)
            .min(duration_ms)
    }

    fn end_ms(&self, duration_ms: u32) -> u32 {
        self.end_ms
            .or_else(|| self.end.map(|secs| (secs * 1000.0).round() as u32))
            .unwrap_or(duration_ms)
            .min(duration_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        CloudProvider, Config, RewriteBackend, TranscriptionBackend, TranscriptionFallback,
    };
    use httpmock::prelude::*;
    use reqwest::header::CONTENT_TYPE;

    #[test]
    fn openai_compatible_requires_base_url() {
        let mut config = Config::default();
        config.transcription.backend = TranscriptionBackend::Cloud;
        config.cloud.provider = CloudProvider::OpenAiCompatible;
        config.cloud.api_key = "test-key".into();
        let err = validate_config(&config).expect_err("missing base url should fail");
        assert!(err.to_string().contains("base_url"));
    }

    #[test]
    fn direct_api_key_is_accepted_without_env_var() {
        let _env_lock = crate::test_support::env_lock();
        let _guard = crate::test_support::EnvVarGuard::capture(&["OPENAI_API_KEY"]);
        crate::test_support::remove_env("OPENAI_API_KEY");

        let mut config = Config::default();
        config.transcription.backend = TranscriptionBackend::Cloud;
        config.cloud.api_key = "sk-inline".into();
        validate_config(&config).expect("inline api key should validate");
    }

    #[test]
    fn encode_wav_pcm16_writes_riff_header() {
        let bytes = encode_wav_pcm16(&[0.0, 0.5, -0.5], 16_000).expect("wav bytes");
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
    }

    #[test]
    fn wav_lengths_reports_expected_sizes() {
        let lengths = wav_lengths(3, 2).expect("wav lengths");
        assert_eq!(
            lengths,
            WavLengths {
                data_len: 6,
                data_len_u32: 6,
                riff_len_u32: 42,
            }
        );
    }

    #[test]
    fn wav_lengths_accepts_max_reachable_pcm16_riff_payload() {
        let sample_count = (u32::MAX as usize - 36) / 2;
        let lengths = wav_lengths(sample_count, 2).expect("max wav lengths");
        assert_eq!(lengths.data_len, sample_count * 2);
        assert_eq!(lengths.data_len_u32 as usize, sample_count * 2);
        assert_eq!(lengths.riff_len_u32 as usize, sample_count * 2 + 36);
    }

    #[test]
    fn wav_lengths_rejects_riff_size_overflow() {
        let sample_count = (u32::MAX as usize - 35) / 2;
        let err = wav_lengths(sample_count, 2).expect_err("riff overflow should fail");
        assert!(err.to_string().contains("too large to encode as WAV"));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn wav_lengths_rejects_data_size_overflow() {
        let sample_count = (u32::MAX as usize / 2) + 1;
        let err = wav_lengths(sample_count, 2).expect_err("data overflow should fail");
        assert!(err.to_string().contains("too large to encode as WAV"));
    }

    #[test]
    fn encode_flac_pcm16_writes_flac_header() {
        let bytes = encode_flac_pcm16(&[0.0, 0.5, -0.5, 0.25], 16_000).expect("flac bytes");
        assert_eq!(&bytes[..4], b"fLaC");
    }

    #[test]
    fn parse_transcription_response_synthesizes_segment_when_missing() {
        let transcript = parse_transcription_response(
            br#"{"text":"Hello there","language":"en"}"#,
            16_000,
            16_000,
            None,
        )
        .expect("parse transcript");
        assert_eq!(transcript.raw_text, "Hello there");
        assert_eq!(transcript.segments.len(), 1);
        assert_eq!(transcript.segments[0].text, "Hello there");
        assert_eq!(transcript.detected_language.as_deref(), Some("en"));
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn cloud_check_hits_models_endpoint() {
        let server = MockServer::start_async().await;
        let _mock = server
            .mock_async(|when, then| {
                when.method(GET).path("/v1/models");
                then.status(200)
                    .header(CONTENT_TYPE.as_str(), "application/json")
                    .body(r#"{"data":[]}"#);
            })
            .await;

        let _env_lock = crate::test_support::env_lock();
        let _guard = crate::test_support::EnvVarGuard::capture(&["OPENAI_API_KEY"]);
        crate::test_support::set_env("OPENAI_API_KEY", "test-key");

        let mut config = Config::default();
        config.transcription.backend = TranscriptionBackend::Cloud;
        config.cloud.base_url = format!("{}/v1", server.base_url());
        let service = CloudService::new(&config).expect("service");
        service.check().await.expect("cloud check");
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn cloud_rewrite_uses_chat_completions_endpoint() {
        let server = MockServer::start_async().await;
        let _mock = server
            .mock_async(|when, then| {
                when.method(POST).path("/v1/chat/completions");
                then.status(200)
                    .header(CONTENT_TYPE.as_str(), "application/json")
                    .body(r#"{"choices":[{"message":{"content":"Hi there."}}]}"#);
            })
            .await;

        let _env_lock = crate::test_support::env_lock();
        let _guard = crate::test_support::EnvVarGuard::capture(&["OPENAI_API_KEY"]);
        crate::test_support::set_env("OPENAI_API_KEY", "test-key");

        let mut config = Config::default();
        config.postprocess.mode = crate::config::PostprocessMode::Rewrite;
        config.rewrite.backend = RewriteBackend::Cloud;
        config.cloud.base_url = format!("{}/v1", server.base_url());
        let service = CloudService::new(&config).expect("service");
        let text = service
            .rewrite_transcript(
                &config,
                &RewriteTranscript {
                    raw_text: "Hi there".into(),
                    correction_aware_text: "Hi there".into(),
                    aggressive_correction_text: None,
                    detected_language: Some("en".into()),
                    typing_context: None,
                    recent_session_entries: Vec::new(),
                    session_backtrack_candidates: Vec::new(),
                    recommended_session_candidate: None,
                    segments: Vec::new(),
                    edit_intents: Vec::new(),
                    edit_signals: Vec::new(),
                    edit_hypotheses: Vec::new(),
                    rewrite_candidates: Vec::new(),
                    recommended_candidate: None,
                    edit_context: Default::default(),
                    policy_context: crate::rewrite_protocol::RewritePolicyContext::default(),
                },
                None,
            )
            .await
            .expect("rewrite");
        assert_eq!(text, "Hi there.");
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn cloud_asr_maps_json_response() {
        let server = MockServer::start_async().await;
        let _mock = server
            .mock_async(|when, then| {
                when.method(POST).path("/v1/audio/transcriptions");
                then.status(200)
                    .header(CONTENT_TYPE.as_str(), "application/json")
                    .body(
                        r#"{"text":"Hello there","language":"en","segments":[{"text":"Hello there","start":0.0,"end":1.0}]}"#,
                    );
            })
            .await;

        let _env_lock = crate::test_support::env_lock();
        let _guard = crate::test_support::EnvVarGuard::capture(&["OPENAI_API_KEY"]);
        crate::test_support::set_env("OPENAI_API_KEY", "test-key");

        let mut config = Config::default();
        config.transcription.backend = TranscriptionBackend::Cloud;
        config.transcription.fallback = TranscriptionFallback::None;
        config.cloud.base_url = format!("{}/v1", server.base_url());
        let service = CloudService::new(&config).expect("service");
        let transcript = service
            .transcribe_audio(&config, &[0.0; 16_000], 16_000)
            .await
            .expect("transcript");
        assert_eq!(transcript.raw_text, "Hello there");
        assert_eq!(transcript.detected_language.as_deref(), Some("en"));
        assert_eq!(transcript.segments.len(), 1);
    }
}

use whispers::config::Config;
use whispers::config::EnhancementConfig;
use whispers::engine::EngineClient;
use whispers::enhance::Enhancer;
use whispers::paths;

#[test]
#[ignore = "requires a managed llama.cpp runtime and local GGUF model"]
fn local_enhancement_preserves_protected_text() {
    let model = std::env::var("WHISPERS_TEST_ENHANCE_MODEL").unwrap_or_default();
    assert!(!model.is_empty(), "WHISPERS_TEST_ENHANCE_MODEL is required");
    let enhancer = Enhancer::new(EnhancementConfig {
        enabled: true,
        command: "managed".into(),
        model,
        timeout_ms: 30_000,
    });
    let output = enhancer
        .enhance("please use this/that in serde_json version 3.2")
        .unwrap_or_default();
    assert!(output.contains("this/that"), "{output}");
    assert!(output.contains("serde_json"), "{output}");
    assert!(output.contains("3.2"), "{output}");
}

#[test]
#[ignore = "requires the managed sherpa-onnx runtime and Nemotron model"]
fn streaming_runtime_accepts_per_stream_language() {
    let client = EngineClient::spawn("sherpa-onnx");
    let error = client
        .as_ref()
        .err()
        .map_or_else(String::new, ToString::to_string);
    assert!(client.is_ok(), "{error}");
    let Some(mut client) = client.ok() else {
        return;
    };
    assert!(client.start(1, "en").is_ok());
    assert!(client.push_audio(&vec![0.0; 16_000]).is_ok());
    assert!(client.finish().is_ok());
}

#[test]
#[ignore = "requires the managed sherpa-onnx runtime and both ASR models"]
fn final_runtime_retranscribes_the_complete_recording() {
    let config = Config::default();
    let wav = paths::recognition_model_dir(&config.recognition.final_model)
        .unwrap_or_default()
        .join("test_wavs/0.wav");
    let bytes = std::fs::read(&wav).unwrap_or_default();
    let samples = pcm16_wav_samples(&bytes);
    assert!(!samples.is_empty(), "could not read {}", wav.display());

    let mut client = EngineClient::spawn("sherpa-onnx").unwrap_or_else(|error| {
        panic!("could not start managed ASR: {error}");
    });
    assert!(client.start(1, "en").is_ok());
    for chunk in samples.chunks(4_096) {
        assert!(client.push_audio(chunk).is_ok());
    }
    let transcript = client.finish().unwrap_or_default();
    assert!(
        transcript.contains("certainly very like the old portrait"),
        "{transcript}"
    );
}

fn pcm16_wav_samples(wav: &[u8]) -> Vec<f32> {
    let Some(data) = wav.windows(4).position(|chunk| chunk == b"data") else {
        return Vec::new();
    };
    let Some(length) = wav
        .get(data + 4..data + 8)
        .and_then(|bytes| <[u8; 4]>::try_from(bytes).ok())
        .map(u32::from_le_bytes)
        .and_then(|length| usize::try_from(length).ok())
    else {
        return Vec::new();
    };
    wav.get(data + 8..data + 8 + length)
        .unwrap_or_default()
        .chunks_exact(2)
        .map(|bytes| f32::from(i16::from_le_bytes([bytes[0], bytes[1]])) / 32_768.0)
        .collect()
}

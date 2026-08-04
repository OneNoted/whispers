use whispers::config::EnhancementConfig;
use whispers::engine::EngineClient;
use whispers::enhance::Enhancer;

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

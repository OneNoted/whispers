use whispers::config::EnhancementConfig;
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

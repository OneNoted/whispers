use crate::error::WhsprError;

use super::*;

#[test]
fn load_missing_file_uses_defaults() {
    let path = crate::test_support::unique_temp_path("config-missing", "toml");
    let config = Config::load(Some(&path)).expect("missing config should load defaults");
    assert_eq!(config.audio.sample_rate, 16000);
    assert_eq!(config.transcription.language, "auto");
    assert_eq!(
        config.transcription.backend,
        TranscriptionBackend::WhisperCpp
    );
    assert_eq!(config.postprocess.mode, PostprocessMode::Raw);
    assert_eq!(config.personalization.snippet_trigger, "insert");
    assert_eq!(config.rewrite.selected_model, "qwen-3.5-4b-q4_k_m");
}

#[test]
fn load_invalid_toml_returns_parse_error() {
    let path = crate::test_support::unique_temp_path("config-invalid", "toml");
    std::fs::write(&path, "not = [valid = toml").expect("write invalid config");
    let err = Config::load(Some(&path)).expect_err("invalid config should fail");
    match err {
        WhsprError::Config(msg) => {
            assert!(msg.contains("failed to parse"), "unexpected message: {msg}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn expand_tilde_uses_home_when_present() {
    let _env_lock = crate::test_support::env_lock();
    let _guard = crate::test_support::EnvVarGuard::capture(&["HOME"]);
    crate::test_support::set_env("HOME", "/tmp/whispers-home");
    assert_eq!(
        expand_tilde("~/models/ggml.bin"),
        "/tmp/whispers-home/models/ggml.bin"
    );
    assert_eq!(expand_tilde("~"), "/tmp/whispers-home");
}

#[test]
fn expand_tilde_without_home_returns_original_path() {
    let _env_lock = crate::test_support::env_lock();
    let _guard = crate::test_support::EnvVarGuard::capture(&["HOME"]);
    crate::test_support::remove_env("HOME");
    assert_eq!(expand_tilde("~/models/ggml.bin"), "~/models/ggml.bin");
    assert_eq!(expand_tilde("~"), "~");
}

#[test]
fn write_default_and_update_model_path_roundtrip() {
    let dir = crate::test_support::unique_temp_dir("config-roundtrip");
    let config_path = dir.join("nested").join("config.toml");

    write_default_config(&config_path, "~/old-model.bin").expect("write config");
    assert!(config_path.exists(), "config file should exist");

    update_config_transcription_selection(
        &config_path,
        TranscriptionBackend::WhisperCpp,
        "large-v3-turbo",
        "~/new-model.bin",
        true,
    )
    .expect("update config");
    let loaded = Config::load(Some(&config_path)).expect("load config");
    assert_eq!(loaded.transcription.model_path, "~/new-model.bin");
    assert_eq!(
        loaded.transcription.backend,
        TranscriptionBackend::WhisperCpp
    );
    assert_eq!(loaded.audio.sample_rate, 16000);
    assert_eq!(loaded.postprocess.mode, PostprocessMode::Raw);
    assert_eq!(
        loaded.personalization.dictionary_path,
        "~/.local/share/whispers/dictionary.toml"
    );
    assert!(loaded.session.enabled);
    assert_eq!(loaded.session.max_entries, 3);
    assert_eq!(loaded.rewrite.timeout_ms, 30000);
    assert!(loaded.feedback.enabled);

    let raw = std::fs::read_to_string(&config_path).expect("read config");
    assert!(raw.contains("[audio]"));
    assert!(raw.contains("[transcription]"));
    assert!(raw.contains("[postprocess]"));
    assert!(raw.contains("[session]"));
    assert!(raw.contains("[rewrite]"));
    assert!(!raw.contains("[whisper]"));
}

#[test]
fn selecting_nemo_model_sets_non_expiring_asr_worker_timeout() {
    let config_path = crate::test_support::unique_temp_path("config-nemo-timeout", "toml");
    write_default_config(&config_path, "~/old-model.bin").expect("write config");

    update_config_transcription_selection(
        &config_path,
        TranscriptionBackend::Nemo,
        "parakeet-tdt_ctc-1.1b",
        "~/.local/share/whispers/nemo/models/parakeet-tdt_ctc-1.1b",
        true,
    )
    .expect("select nemo model");

    let loaded = Config::load(Some(&config_path)).expect("load config");
    assert_eq!(loaded.transcription.backend, TranscriptionBackend::Nemo);
    assert_eq!(loaded.transcription.idle_timeout_ms, 0);
}

#[test]
fn load_legacy_whisper_section_maps_to_transcription() {
    let path = crate::test_support::unique_temp_path("config-whisper-legacy", "toml");
    std::fs::write(
        &path,
        r#"[whisper]
model_path = "~/legacy-model.bin"
language = "en"
use_gpu = false
flash_attn = false
"#,
    )
    .expect("write config");

    let loaded = Config::load(Some(&path)).expect("load config");
    assert_eq!(
        loaded.transcription.backend,
        TranscriptionBackend::WhisperCpp
    );
    assert_eq!(loaded.transcription.model_path, "~/legacy-model.bin");
    assert_eq!(loaded.transcription.language, "en");
    assert!(!loaded.transcription.use_gpu);
    assert!(!loaded.transcription.flash_attn);
}

#[test]
fn load_legacy_cleanup_section_maps_to_legacy_basic() {
    let path = crate::test_support::unique_temp_path("config-cleanup", "toml");
    std::fs::write(
        &path,
        r#"[cleanup]
profile = "aggressive"
spoken_formatting = false
remove_fillers = false
"#,
    )
    .expect("write config");

    let config = Config::load(Some(&path)).expect("load config");
    assert_eq!(config.postprocess.mode, PostprocessMode::LegacyBasic);
    assert_eq!(config.cleanup.profile, CleanupProfile::Aggressive);
    assert!(!config.cleanup.spoken_formatting);
    assert!(config.cleanup.backtrack);
    assert!(!config.cleanup.remove_fillers);
}

#[test]
fn update_rewrite_selection_enables_advanced_mode() {
    let dir = crate::test_support::unique_temp_dir("config-rewrite-select");
    let config_path = dir.join("config.toml");
    write_default_config(&config_path, "~/model.bin").expect("write config");

    update_config_rewrite_selection(&config_path, "qwen-3.5-2b-q4_k_m")
        .expect("select rewrite model");

    let loaded = Config::load(Some(&config_path)).expect("load config");
    assert_eq!(loaded.postprocess.mode, PostprocessMode::AdvancedLocal);
    assert_eq!(loaded.rewrite.selected_model, "qwen-3.5-2b-q4_k_m");
    assert!(loaded.rewrite.model_path.is_empty());
    assert_eq!(
        loaded.rewrite.instructions_path,
        "~/.local/share/whispers/rewrite-instructions.txt"
    );
    assert_eq!(
        loaded.rewrite.profile,
        crate::rewrite_profile::RewriteProfile::Auto
    );
    assert_eq!(loaded.rewrite.timeout_ms, 30000);
    assert_eq!(loaded.rewrite.idle_timeout_ms, 120000);
}

#[test]
fn update_helpers_upgrade_legacy_configs_without_panicking() {
    let config_path = crate::test_support::unique_temp_path("config-legacy-upgrade", "toml");
    std::fs::write(
        &config_path,
        r#"[audio]
sample_rate = 16000

[whisper]
model_path = "~/.local/share/whispers/ggml-large-v3-turbo.bin"
language = "auto"
"#,
    )
    .expect("write legacy config");

    update_config_transcription_selection(
        &config_path,
        TranscriptionBackend::WhisperCpp,
        "large-v3-turbo",
        "~/.local/share/whispers/ggml-large-v3-turbo.bin",
        true,
    )
    .expect("update transcription selection");
    update_config_rewrite_selection(&config_path, "qwen-3.5-4b-q4_k_m")
        .expect("update rewrite selection");

    let loaded = Config::load(Some(&config_path)).expect("load upgraded config");
    assert_eq!(
        loaded.transcription.backend,
        TranscriptionBackend::WhisperCpp
    );
    assert_eq!(loaded.transcription.selected_model, "large-v3-turbo");
    assert_eq!(loaded.postprocess.mode, PostprocessMode::AdvancedLocal);
    assert_eq!(loaded.rewrite.selected_model, "qwen-3.5-4b-q4_k_m");

    let raw = std::fs::read_to_string(&config_path).expect("read upgraded config");
    assert!(!raw.contains("[whisper]"));
}

#[test]
fn load_cloud_literal_key_from_legacy_api_key_env() {
    let path = crate::test_support::unique_temp_path("config-cloud-literal-key", "toml");
    std::fs::write(
        &path,
        r#"[cloud]
api_key_env = "sk-test-inline"
"#,
    )
    .expect("write config");

    let loaded = Config::load(Some(&path)).expect("load config");
    assert_eq!(loaded.cloud.api_key, "sk-test-inline");
    assert_eq!(loaded.cloud.api_key_env, "OPENAI_API_KEY");
}

#[test]
fn default_config_template_matches_example_file() {
    let example_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config.example.toml");
    let example = std::fs::read_to_string(&example_path).expect("read config example");
    let expected =
        super::edit::default_config_template("~/.local/share/whispers/ggml-large-v3-turbo.bin");
    assert_eq!(example, expected);
}

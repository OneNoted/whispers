use crate::config::Config;

use super::{CloudSetup, apply};
use crate::config::{self, TranscriptionBackend, TranscriptionFallback};

#[cfg(not(feature = "local-rewrite"))]
use crate::config::{RewriteBackend, RewriteFallback};

#[test]
fn runtime_selection_resets_cloud_asr_when_disabled() {
    let config_path = crate::test_support::unique_temp_path("setup-runtime-reset", "toml");
    config::write_default_config(&config_path, "~/model.bin").expect("write config");
    config::update_config_transcription_runtime(
        &config_path,
        TranscriptionBackend::Cloud,
        TranscriptionFallback::None,
    )
    .expect("set cloud runtime");

    let cloud = CloudSetup::default();
    apply::apply_runtime_backend_selection(&config_path, TranscriptionBackend::WhisperCpp, &cloud)
        .expect("reset runtime");

    let config = Config::load(Some(&config_path)).expect("load config");
    assert_eq!(
        config.transcription.backend,
        TranscriptionBackend::WhisperCpp
    );
    assert_eq!(
        config.transcription.fallback,
        TranscriptionFallback::ConfiguredLocal
    );
}

#[cfg(not(feature = "local-rewrite"))]
#[test]
fn runtime_selection_disables_local_rewrite_fallback_when_build_lacks_local_rewrite() {
    let config_path = crate::test_support::unique_temp_path("setup-rewrite-fallback-reset", "toml");
    config::write_default_config(&config_path, "~/model.bin").expect("write config");

    let cloud = CloudSetup {
        rewrite_enabled: true,
        rewrite_fallback: RewriteFallback::Local,
        ..CloudSetup::default()
    };
    apply::apply_runtime_backend_selection(&config_path, TranscriptionBackend::WhisperCpp, &cloud)
        .expect("apply runtime");

    let config = Config::load(Some(&config_path)).expect("load config");
    assert_eq!(config.rewrite.backend, RewriteBackend::Cloud);
    assert_eq!(config.rewrite.fallback, RewriteFallback::None);
}

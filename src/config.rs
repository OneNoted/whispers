mod edit;
mod load;
mod paths;
mod schema;

#[cfg(test)]
mod tests;

pub use edit::{
    update_config_cloud_settings, update_config_postprocess_mode, update_config_rewrite_runtime,
    update_config_rewrite_selection, update_config_transcription_runtime,
    update_config_transcription_selection, write_default_config,
};
pub use paths::{data_dir, default_config_path, expand_tilde, resolve_config_path};
pub use schema::{
    AgenticRewriteConfig, AudioConfig, CleanupConfig, CleanupProfile, CloudConfig,
    CloudLanguageMode, CloudProvider, CloudRewriteConfig, CloudSettingsUpdate,
    CloudTranscriptionConfig, Config, FeedbackConfig, InjectConfig, PersonalizationConfig,
    PostprocessConfig, PostprocessMode, RewriteBackend, RewriteConfig, RewriteFallback,
    SessionConfig, TranscriptionBackend, TranscriptionConfig, TranscriptionFallback,
};

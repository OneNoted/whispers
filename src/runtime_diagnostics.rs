use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::config::{Config, RewriteBackend, TranscriptionBackend};

const STATUS_FILE_NAME: &str = "main-status.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DictationStage {
    Starting,
    Recording,
    AsrPrepare,
    RecordingStopped,
    AsrTranscribe,
    Postprocess,
    Inject,
    SessionWrite,
    Done,
    Cancelled,
}

impl DictationStage {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Starting => "starting",
            Self::Recording => "recording",
            Self::AsrPrepare => "asr_prepare",
            Self::RecordingStopped => "recording_stopped",
            Self::AsrTranscribe => "asr_transcribe",
            Self::Postprocess => "postprocess",
            Self::Inject => "inject",
            Self::SessionWrite => "session_write",
            Self::Done => "done",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct DictationStageMetadata {
    pub audio_samples: Option<usize>,
    pub sample_rate: Option<u32>,
    pub audio_duration_ms: Option<u64>,
    pub transcript_chars: Option<usize>,
    pub output_chars: Option<usize>,
    pub operation: Option<String>,
    pub rewrite_used: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct MainStatusSnapshot {
    pub pid: u32,
    pub started_at_ms: u64,
    pub stage: String,
    pub stage_started_at_ms: u64,
    pub transcription_backend: String,
    pub asr_model: String,
    pub rewrite_backend: String,
    pub rewrite_model: String,
    pub metadata: DictationStageMetadata,
}

#[derive(Debug)]
struct State {
    pid: u32,
    started_at_ms: u64,
    stage: DictationStage,
    stage_started_at_ms: u64,
    transcription_backend: String,
    asr_model: String,
    rewrite_backend: String,
    rewrite_model: String,
    metadata: DictationStageMetadata,
}

#[derive(Clone, Debug)]
pub(crate) struct DictationRuntimeDiagnostics {
    status_path: PathBuf,
    state: Arc<Mutex<State>>,
}

impl DictationRuntimeDiagnostics {
    pub(crate) fn new(config: &Config) -> Self {
        let now = now_ms();
        let diagnostics = Self {
            status_path: status_file_path(),
            state: Arc::new(Mutex::new(State {
                pid: std::process::id(),
                started_at_ms: now,
                stage: DictationStage::Starting,
                stage_started_at_ms: now,
                transcription_backend: config.transcription.backend.as_str().to_string(),
                asr_model: transcription_model_label(config),
                rewrite_backend: config.rewrite.backend.as_str().to_string(),
                rewrite_model: rewrite_model_label(config),
                metadata: DictationStageMetadata::default(),
            })),
        };
        diagnostics.persist_snapshot();
        tracing::info!(
            stage = DictationStage::Starting.as_str(),
            "dictation stage entered"
        );
        diagnostics
    }

    pub(crate) fn enter_stage(&self, stage: DictationStage, metadata: DictationStageMetadata) {
        self.transition(stage, metadata, false);
    }

    pub(crate) fn clear_with_stage(&self, stage: DictationStage) {
        self.transition(stage, DictationStageMetadata::default(), true);
    }

    pub(crate) fn snapshot(&self) -> Option<MainStatusSnapshot> {
        let state = self.state.lock().ok()?;
        Some(snapshot_from_state(&state))
    }

    fn transition(
        &self,
        stage: DictationStage,
        metadata: DictationStageMetadata,
        remove_after_write: bool,
    ) {
        let now = now_ms();
        let snapshot = match self.state.lock() {
            Ok(mut state) => {
                let previous_stage = state.stage;
                let previous_elapsed_ms = now.saturating_sub(state.stage_started_at_ms);
                tracing::info!(
                    stage = previous_stage.as_str(),
                    elapsed_ms = previous_elapsed_ms,
                    "dictation stage finished"
                );

                state.stage = stage;
                state.stage_started_at_ms = now;
                state.metadata = metadata;
                let snapshot = snapshot_from_state(&state);
                tracing::info!(stage = stage.as_str(), "dictation stage entered");
                snapshot
            }
            Err(_) => {
                tracing::warn!("dictation diagnostics lock poisoned; skipping stage update");
                return;
            }
        };

        self.persist_snapshot_value(&snapshot);

        if remove_after_write {
            self.remove_status_file();
        }
    }

    fn persist_snapshot(&self) {
        let Some(snapshot) = self.snapshot() else {
            tracing::warn!("failed to snapshot dictation diagnostics state");
            return;
        };
        self.persist_snapshot_value(&snapshot);
    }

    fn persist_snapshot_value(&self, snapshot: &MainStatusSnapshot) {
        if let Some(parent) = self.status_path.parent()
            && let Err(err) = std::fs::create_dir_all(parent)
        {
            tracing::warn!(
                "failed to create dictation runtime directory {}: {err}",
                parent.display()
            );
            return;
        }

        let encoded = match serde_json::to_vec_pretty(snapshot) {
            Ok(encoded) => encoded,
            Err(err) => {
                tracing::warn!("failed to encode dictation runtime status: {err}");
                return;
            }
        };

        if let Err(err) = std::fs::write(&self.status_path, encoded) {
            tracing::warn!(
                "failed to write dictation runtime status {}: {err}",
                self.status_path.display()
            );
        }
    }

    fn remove_status_file(&self) {
        if let Err(err) = std::fs::remove_file(&self.status_path)
            && err.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(
                "failed to remove dictation runtime status {}: {err}",
                self.status_path.display()
            );
        }
    }
}

impl Drop for DictationRuntimeDiagnostics {
    fn drop(&mut self) {
        if Arc::strong_count(&self.state) == 1 {
            self.remove_status_file();
        }
    }
}

fn snapshot_from_state(state: &State) -> MainStatusSnapshot {
    MainStatusSnapshot {
        pid: state.pid,
        started_at_ms: state.started_at_ms,
        stage: state.stage.as_str().to_string(),
        stage_started_at_ms: state.stage_started_at_ms,
        transcription_backend: state.transcription_backend.clone(),
        asr_model: state.asr_model.clone(),
        rewrite_backend: state.rewrite_backend.clone(),
        rewrite_model: state.rewrite_model.clone(),
        metadata: state.metadata.clone(),
    }
}

fn transcription_model_label(config: &Config) -> String {
    match config.transcription.backend {
        TranscriptionBackend::Cloud => config.cloud.transcription.model.clone(),
        _ => {
            if !config.transcription.model_path.trim().is_empty() {
                config.resolved_model_path().display().to_string()
            } else {
                config.transcription.selected_model.clone()
            }
        }
    }
}

fn rewrite_model_label(config: &Config) -> String {
    match config.rewrite.backend {
        RewriteBackend::Cloud => config.cloud.rewrite.model.clone(),
        RewriteBackend::Local => config
            .resolved_rewrite_model_path()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| config.rewrite.selected_model.clone()),
    }
}

fn status_file_path() -> PathBuf {
    let runtime_dir = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(runtime_dir)
        .join("whispers")
        .join(STATUS_FILE_NAME)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::test_support::{EnvVarGuard, env_lock, set_env, unique_temp_dir};

    fn with_runtime_dir<T>(f: impl FnOnce() -> T) -> T {
        let _env_lock = env_lock();
        let _guard = EnvVarGuard::capture(&["XDG_RUNTIME_DIR"]);
        let runtime_dir = unique_temp_dir("runtime-diagnostics");
        let runtime_dir = runtime_dir
            .to_str()
            .expect("temp runtime dir should be valid UTF-8");
        set_env("XDG_RUNTIME_DIR", runtime_dir);
        f()
    }

    #[test]
    fn status_file_tracks_stage_updates_and_cleanup() {
        with_runtime_dir(|| {
            let diagnostics = DictationRuntimeDiagnostics::new(&Config::default());
            let status_path = status_file_path();
            assert!(status_path.exists());

            let initial = std::fs::read_to_string(&status_path).expect("read status");
            let initial: MainStatusSnapshot =
                serde_json::from_str(&initial).expect("parse initial status");
            assert_eq!(initial.stage, DictationStage::Starting.as_str());

            diagnostics.enter_stage(
                DictationStage::RecordingStopped,
                DictationStageMetadata {
                    audio_samples: Some(32_000),
                    sample_rate: Some(16_000),
                    audio_duration_ms: Some(2_000),
                    ..DictationStageMetadata::default()
                },
            );

            let updated = std::fs::read_to_string(&status_path).expect("read status");
            let updated: MainStatusSnapshot =
                serde_json::from_str(&updated).expect("parse updated status");
            assert_eq!(updated.stage, DictationStage::RecordingStopped.as_str());
            assert_eq!(updated.metadata.audio_samples, Some(32_000));

            diagnostics.clear_with_stage(DictationStage::Done);
            assert!(!status_path.exists());
        });
    }

    #[test]
    fn status_file_removed_on_drop_without_explicit_completion() {
        with_runtime_dir(|| {
            let status_path = status_file_path();
            {
                let diagnostics = DictationRuntimeDiagnostics::new(&Config::default());
                diagnostics.enter_stage(
                    DictationStage::AsrTranscribe,
                    DictationStageMetadata {
                        transcript_chars: Some(42),
                        ..DictationStageMetadata::default()
                    },
                );
                assert!(status_path.exists());
            }

            assert!(!status_path.exists());
        });
    }
}

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::config::{Config, RewriteBackend, TranscriptionBackend};

const STATUS_FILE_NAME: &str = "main-status.json";
const HANG_DEBUG_ENV: &str = "WHISPERS_HANG_DEBUG";
const DEFAULT_PREPARE_HANG_TIMEOUT: Duration = Duration::from_secs(20);
const DEFAULT_TRANSCRIBE_HANG_TIMEOUT: Duration = Duration::from_secs(90);
const DEFAULT_WATCHDOG_POLL_INTERVAL: Duration = Duration::from_millis(250);

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

#[derive(Debug, Clone, Copy)]
struct HangWatchdogConfig {
    enabled: bool,
    prepare_timeout: Duration,
    transcribe_timeout: Duration,
    poll_interval: Duration,
}

#[derive(Debug)]
struct WatchdogControl {
    #[cfg(test)]
    enabled: bool,
    stop: AtomicBool,
    handle: Mutex<Option<std::thread::JoinHandle<()>>>,
}

#[derive(Clone, Debug)]
pub(crate) struct DictationRuntimeDiagnostics {
    status_path: PathBuf,
    state: Arc<Mutex<State>>,
    ownership: Arc<()>,
    watchdog: Arc<WatchdogControl>,
}

impl DictationRuntimeDiagnostics {
    pub(crate) fn new(config: &Config) -> Self {
        Self::build(config, hang_watchdog_config(config))
    }

    fn build(config: &Config, hang_watchdog: HangWatchdogConfig) -> Self {
        let now = now_ms();
        let watchdog_enabled = hang_watchdog.enabled
            && config.transcription.backend == TranscriptionBackend::WhisperCpp;
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
            ownership: Arc::new(()),
            watchdog: Arc::new(WatchdogControl {
                #[cfg(test)]
                enabled: watchdog_enabled,
                stop: AtomicBool::new(false),
                handle: Mutex::new(None),
            }),
        };
        diagnostics.start_watchdog(HangWatchdogConfig {
            enabled: watchdog_enabled,
            ..hang_watchdog
        });
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

    #[cfg(test)]
    fn new_with_watchdog_config(config: &Config, hang_watchdog: HangWatchdogConfig) -> Self {
        Self::build(config, hang_watchdog)
    }

    #[cfg(test)]
    fn watchdog_enabled(&self) -> bool {
        self.watchdog.enabled
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

    fn start_watchdog(&self, config: HangWatchdogConfig) {
        if !config.enabled {
            return;
        }

        let state = Arc::clone(&self.state);
        let status_path = self.status_path.clone();
        let stop = Arc::clone(&self.watchdog);
        let handle = std::thread::spawn(move || {
            let mut dumped_stage_started_at_ms = None::<u64>;

            loop {
                if stop.stop.load(Ordering::Relaxed) {
                    break;
                }
                std::thread::sleep(config.poll_interval);

                let (stage, snapshot) = match state.lock() {
                    Ok(state) => (state.stage, snapshot_from_state(&state)),
                    Err(_) => break,
                };

                if matches!(stage, DictationStage::Done | DictationStage::Cancelled) {
                    break;
                }

                let Some(timeout) = stage_timeout(stage, config) else {
                    dumped_stage_started_at_ms = None;
                    continue;
                };

                let elapsed = now_ms().saturating_sub(snapshot.stage_started_at_ms);
                if elapsed < timeout.as_millis() as u64 {
                    continue;
                }

                if dumped_stage_started_at_ms == Some(snapshot.stage_started_at_ms) {
                    continue;
                }

                dumped_stage_started_at_ms = Some(snapshot.stage_started_at_ms);
                if let Err(err) = write_hang_bundle(&status_path, stage, &snapshot) {
                    tracing::warn!("failed to write hang diagnostics bundle: {err}");
                }
            }
        });

        match self.watchdog.handle.lock() {
            Ok(mut guard) => {
                *guard = Some(handle);
            }
            Err(_) => {
                tracing::warn!("dictation watchdog lock poisoned; detaching watchdog thread");
            }
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
        if Arc::strong_count(&self.ownership) == 1 {
            self.watchdog.stop.store(true, Ordering::Relaxed);
            if let Ok(mut guard) = self.watchdog.handle.lock()
                && let Some(handle) = guard.take()
            {
                let _ = handle.join();
            }
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

fn hang_watchdog_config(config: &Config) -> HangWatchdogConfig {
    HangWatchdogConfig {
        enabled: std::env::var(HANG_DEBUG_ENV)
            .map(|value| value == "1")
            .unwrap_or(false)
            && config.transcription.backend == TranscriptionBackend::WhisperCpp,
        prepare_timeout: DEFAULT_PREPARE_HANG_TIMEOUT,
        transcribe_timeout: DEFAULT_TRANSCRIBE_HANG_TIMEOUT,
        poll_interval: DEFAULT_WATCHDOG_POLL_INTERVAL,
    }
}

fn stage_timeout(stage: DictationStage, config: HangWatchdogConfig) -> Option<Duration> {
    match stage {
        DictationStage::AsrPrepare => Some(config.prepare_timeout),
        DictationStage::AsrTranscribe => Some(config.transcribe_timeout),
        _ => None,
    }
}

fn hang_bundle_path(status_path: &Path, pid: u32, stage: DictationStage) -> PathBuf {
    status_path
        .parent()
        .unwrap_or_else(|| Path::new("/tmp"))
        .join(format!("hang-{pid}-{}-{}.log", stage.as_str(), now_ms()))
}

fn write_hang_bundle(
    status_path: &Path,
    stage: DictationStage,
    snapshot: &MainStatusSnapshot,
) -> std::io::Result<PathBuf> {
    let bundle_path = hang_bundle_path(status_path, snapshot.pid, stage);
    let mut body = String::new();
    let _ = writeln!(body, "whispers hang diagnostics");
    let _ = writeln!(body, "pid: {}", snapshot.pid);
    let _ = writeln!(body, "stage: {}", stage.as_str());
    let _ = writeln!(body, "status_path: {}", status_path.display());
    let _ = writeln!(body);
    let _ = writeln!(body, "== main-status.json ==");
    match serde_json::to_string_pretty(snapshot) {
        Ok(json) => {
            let _ = writeln!(body, "{json}");
        }
        Err(err) => {
            let _ = writeln!(body, "failed to encode status snapshot: {err}");
        }
    }
    let _ = writeln!(body);
    body.push_str(&capture_stack_trace(snapshot.pid));
    body.push('\n');
    body.push_str(&capture_lsof(snapshot.pid));

    std::fs::write(&bundle_path, body)?;
    tracing::warn!(
        path = %bundle_path.display(),
        stage = stage.as_str(),
        "wrote dictation hang diagnostics bundle"
    );
    Ok(bundle_path)
}

fn capture_stack_trace(pid: u32) -> String {
    let pid = pid.to_string();
    if command_available("gstack") {
        let output = run_command("gstack", &[pid.as_str()]);
        if output.success {
            return format_command_output("gstack", &output);
        }

        let mut body = format_command_output("gstack", &output);
        if command_available("gdb") {
            body.push('\n');
            body.push_str(&format_command_output(
                "gdb",
                &run_command("gdb", &["-batch", "-ex", "thread apply all bt", "-p", &pid]),
            ));
        }
        return body;
    }

    if command_available("gdb") {
        return format_command_output(
            "gdb",
            &run_command("gdb", &["-batch", "-ex", "thread apply all bt", "-p", &pid]),
        );
    }

    "== stack trace ==\nno stack capture tool available; checked gstack and gdb\n".to_string()
}

fn capture_lsof(pid: u32) -> String {
    if !command_available("lsof") {
        return "== lsof ==\nlsof not available\n".to_string();
    }

    format_command_output("lsof", &run_command("lsof", &["-p", &pid.to_string()]))
}

#[derive(Debug)]
struct CommandCapture {
    success: bool,
    stdout: String,
    stderr: String,
    error: Option<String>,
}

fn run_command(program: &str, args: &[&str]) -> CommandCapture {
    match Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .output()
    {
        Ok(output) => CommandCapture {
            success: output.status.success(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            error: None,
        },
        Err(err) => CommandCapture {
            success: false,
            stdout: String::new(),
            stderr: String::new(),
            error: Some(err.to_string()),
        },
    }
}

fn format_command_output(program: &str, capture: &CommandCapture) -> String {
    let mut body = String::new();
    let _ = writeln!(body, "== {program} ==");
    if let Some(error) = capture.error.as_deref() {
        let _ = writeln!(body, "failed to execute: {error}");
        return body;
    }
    let _ = writeln!(body, "success: {}", capture.success);
    if !capture.stdout.trim().is_empty() {
        let _ = writeln!(body, "-- stdout --");
        body.push_str(&capture.stdout);
        if !capture.stdout.ends_with('\n') {
            body.push('\n');
        }
    }
    if !capture.stderr.trim().is_empty() {
        let _ = writeln!(body, "-- stderr --");
        body.push_str(&capture.stderr);
        if !capture.stderr.ends_with('\n') {
            body.push('\n');
        }
    }
    body
}

fn command_available(program: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|path| {
        std::env::split_paths(&path).any(|dir| {
            let candidate = dir.join(program);
            candidate.is_file()
        })
    })
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
    use std::os::unix::fs::PermissionsExt;

    fn with_runtime_dir<T>(f: impl FnOnce() -> T) -> T {
        let _env_lock = env_lock();
        let _guard = EnvVarGuard::capture(&["PATH", "XDG_RUNTIME_DIR"]);
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

    #[test]
    fn watchdog_only_arms_for_local_whisper_cpp() {
        with_runtime_dir(|| {
            let mut cloud = Config::default();
            cloud.transcription.backend = TranscriptionBackend::Cloud;
            let diagnostics = DictationRuntimeDiagnostics::new_with_watchdog_config(
                &cloud,
                HangWatchdogConfig {
                    enabled: true,
                    prepare_timeout: Duration::from_millis(20),
                    transcribe_timeout: Duration::from_millis(20),
                    poll_interval: Duration::from_millis(5),
                },
            );
            assert!(!diagnostics.watchdog_enabled());

            let diagnostics = DictationRuntimeDiagnostics::new_with_watchdog_config(
                &Config::default(),
                HangWatchdogConfig {
                    enabled: true,
                    prepare_timeout: Duration::from_millis(20),
                    transcribe_timeout: Duration::from_millis(20),
                    poll_interval: Duration::from_millis(5),
                },
            );
            assert!(diagnostics.watchdog_enabled());
        });
    }

    #[test]
    fn watchdog_does_not_fire_after_stage_advances() {
        with_runtime_dir(|| {
            let diagnostics = DictationRuntimeDiagnostics::new_with_watchdog_config(
                &Config::default(),
                HangWatchdogConfig {
                    enabled: true,
                    prepare_timeout: Duration::from_millis(40),
                    transcribe_timeout: Duration::from_millis(40),
                    poll_interval: Duration::from_millis(5),
                },
            );
            diagnostics.enter_stage(
                DictationStage::AsrPrepare,
                DictationStageMetadata::default(),
            );
            std::thread::sleep(Duration::from_millis(10));
            diagnostics.enter_stage(
                DictationStage::Postprocess,
                DictationStageMetadata {
                    transcript_chars: Some(12),
                    ..DictationStageMetadata::default()
                },
            );
            std::thread::sleep(Duration::from_millis(80));

            let runtime_dir = status_file_path()
                .parent()
                .expect("runtime dir")
                .to_path_buf();
            let dump_count = std::fs::read_dir(runtime_dir)
                .expect("read runtime dir")
                .flatten()
                .filter(|entry| entry.file_name().to_string_lossy().starts_with("hang-"))
                .count();
            assert_eq!(dump_count, 0);
        });
    }

    #[test]
    fn watchdog_dump_includes_stage_metadata_and_command_output() {
        with_runtime_dir(|| {
            let bin_dir = unique_temp_dir("runtime-diagnostics-bin");
            let gstack_path = bin_dir.join("gstack");
            let lsof_path = bin_dir.join("lsof");
            std::fs::write(&gstack_path, "#!/bin/sh\necho \"fake gstack $@\"\n")
                .expect("write gstack");
            std::fs::write(&lsof_path, "#!/bin/sh\necho \"fake lsof $@\"\n").expect("write lsof");
            std::fs::set_permissions(&gstack_path, std::fs::Permissions::from_mode(0o755))
                .expect("chmod gstack");
            std::fs::set_permissions(&lsof_path, std::fs::Permissions::from_mode(0o755))
                .expect("chmod lsof");
            let original_path = std::env::var("PATH").unwrap_or_default();
            let path = if original_path.is_empty() {
                bin_dir.display().to_string()
            } else {
                format!("{}:{original_path}", bin_dir.display())
            };
            set_env("PATH", &path);

            let diagnostics = DictationRuntimeDiagnostics::new_with_watchdog_config(
                &Config::default(),
                HangWatchdogConfig {
                    enabled: true,
                    prepare_timeout: Duration::from_millis(30),
                    transcribe_timeout: Duration::from_millis(30),
                    poll_interval: Duration::from_millis(5),
                },
            );
            diagnostics.enter_stage(
                DictationStage::AsrPrepare,
                DictationStageMetadata {
                    audio_samples: Some(4_096),
                    sample_rate: Some(16_000),
                    audio_duration_ms: Some(256),
                    ..DictationStageMetadata::default()
                },
            );

            let runtime_dir = status_file_path()
                .parent()
                .expect("runtime dir")
                .to_path_buf();
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            let dump_path = loop {
                let maybe_path = std::fs::read_dir(&runtime_dir)
                    .expect("read runtime dir")
                    .flatten()
                    .map(|entry| entry.path())
                    .find(|path| {
                        path.file_name()
                            .is_some_and(|name| name.to_string_lossy().starts_with("hang-"))
                    });
                if let Some(path) = maybe_path {
                    break path;
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "watchdog did not emit dump"
                );
                std::thread::sleep(Duration::from_millis(10));
            };

            let dump = std::fs::read_to_string(&dump_path).expect("read dump");
            assert!(dump.contains("\"stage\": \"asr_prepare\""));
            assert!(dump.contains("\"audio_samples\": 4096"));
            assert!(dump.contains("fake gstack"));
            assert!(dump.contains("fake lsof"));
        });
    }
}

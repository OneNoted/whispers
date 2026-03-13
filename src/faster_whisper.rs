use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::os::unix::net::UnixStream as StdUnixStream;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use base64::Engine;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::process::Command;

use crate::asr_protocol::{AsrRequest, AsrResponse};
use crate::config::{TranscriptionBackend, TranscriptionConfig, data_dir};
use crate::error::{Result, WhsprError};
use crate::transcribe::Transcript;

const PYTHON_WORKER_SOURCE: &str = include_str!("faster_whisper_worker.py");
const RUNTIME_READY_MARKER: &str = ".runtime-ready";

struct ManagedModelInfo {
    name: &'static str,
    repo_id: &'static str,
}

const MANAGED_MODELS: &[ManagedModelInfo] = &[ManagedModelInfo {
    name: "distil-large-v3.5",
    repo_id: "distil-whisper/distil-large-v3.5-ct2",
}];

pub fn managed_model_local_path(name: &str) -> PathBuf {
    faster_whisper_models_dir().join(name)
}

pub async fn download_managed_model(name: &str) -> Result<()> {
    let model = find_managed_model(name)
        .ok_or_else(|| WhsprError::Download(format!("unknown faster-whisper model '{name}'")))?;
    tokio::task::spawn_blocking(move || {
        ensure_runtime_sync()?;
        let script = ensure_worker_script()?;
        let python = runtime_python_path();
        let model_dir = managed_model_local_path(model.name);
        if model_dir_is_ready(&model_dir) {
            tracing::info!(
                "faster-whisper model '{}' already downloaded at {}",
                model.name,
                model_dir.display()
            );
            println!("{}", crate::ui::ready_message("ASR", model.name));
            return Ok(());
        }
        if model_dir.exists() {
            tracing::warn!(
                "removing incomplete faster-whisper model directory before re-download: {}",
                model_dir.display()
            );
            remove_existing_model_dir(&model_dir).map_err(|e| {
                WhsprError::Download(format!(
                    "failed to clear incomplete faster-whisper model directory {}: {e}",
                    model_dir.display()
                ))
            })?;
        }
        if let Some(parent) = model_dir.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                WhsprError::Download(format!(
                    "failed to create faster-whisper models directory {}: {e}",
                    parent.display()
                ))
            })?;
        }
        let spinner = crate::ui::spinner(format!(
            "Preparing faster-whisper runtime and model {}...",
            model.name
        ));
        let status = std::process::Command::new(&python)
            .arg(&script)
            .arg("download")
            .arg("--repo-id")
            .arg(model.repo_id)
            .arg("--model-dir")
            .arg(&model_dir)
            .stdout(crate::ui::child_stdio())
            .stderr(crate::ui::child_stdio())
            .status()
            .map_err(|e| {
                WhsprError::Download(format!(
                    "failed to start faster-whisper downloader via {}: {e}",
                    python.display()
                ))
            })?;
        spinner.finish_and_clear();
        if !status.success() {
            return Err(WhsprError::Download(format!(
                "faster-whisper model download failed with status {status}"
            )));
        }
        if !model_dir_is_ready(&model_dir) {
            return Err(WhsprError::Download(format!(
                "downloaded faster-whisper model at {} is incomplete; expected a CTranslate2 model with model.bin",
                model_dir.display()
            )));
        }
        println!("{}", crate::ui::ready_message("ASR", model.name));
        Ok(())
    })
    .await
    .map_err(|e| WhsprError::Download(format!("faster-whisper download task failed: {e}")))?
}

pub fn prepare_service(config: &TranscriptionConfig) -> Option<FasterWhisperService> {
    if config.backend != TranscriptionBackend::FasterWhisper {
        return None;
    }

    let model_path = resolve_model_path(config)?;
    Some(FasterWhisperService::new(config, &model_path))
}

pub fn resolve_model_path(config: &TranscriptionConfig) -> Option<PathBuf> {
    if !config.model_path.trim().is_empty() {
        return Some(PathBuf::from(crate::config::expand_tilde(
            &config.model_path,
        )));
    }

    find_managed_model(&config.selected_model)
        .map(|_| managed_model_local_path(&config.selected_model))
}

pub fn model_dir_is_ready(path: &Path) -> bool {
    path.join("model.bin").is_file()
}

#[derive(Debug, Clone)]
pub struct FasterWhisperService {
    socket_path: PathBuf,
    lock_path: PathBuf,
    model_path: PathBuf,
    language: String,
    use_gpu: bool,
    idle_timeout_ms: u64,
}

impl FasterWhisperService {
    pub fn new(config: &TranscriptionConfig, model_path: &Path) -> Self {
        let (socket_path, lock_path) = service_paths(config, model_path);
        Self {
            socket_path,
            lock_path,
            model_path: model_path.to_path_buf(),
            language: config.language.clone(),
            use_gpu: config.use_gpu,
            idle_timeout_ms: config.idle_timeout_ms,
        }
    }

    pub fn prewarm(&self) -> Result<()> {
        if self.is_running() {
            return Ok(());
        }

        let _lock = match StartupLock::try_acquire(&self.lock_path)? {
            Some(lock) => lock,
            None => return Ok(()),
        };
        if self.is_running() {
            return Ok(());
        }
        if self.socket_path.exists() {
            let _ = std::fs::remove_file(&self.socket_path);
        }
        ensure_runtime_sync()?;
        if !self.model_path.exists() {
            return Err(WhsprError::Transcription(format!(
                "faster-whisper model directory not found: {}. Run: whispers asr-model download {}",
                self.model_path.display(),
                self.model_path
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "distil-large-v3.5".into())
            )));
        }
        if !model_dir_is_ready(&self.model_path) {
            return Err(WhsprError::Transcription(format!(
                "faster-whisper model directory is incomplete: {}. Re-run: whispers asr-model download {}",
                self.model_path.display(),
                self.model_path
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "distil-large-v3.5".into())
            )));
        }
        self.spawn_worker()
    }

    async fn ensure_running(&self, timeout: Duration) -> Result<()> {
        if self.is_running() {
            return Ok(());
        }

        self.prewarm()?;

        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            match UnixStream::connect(&self.socket_path).await {
                Ok(stream) => {
                    drop(stream);
                    return Ok(());
                }
                Err(_) if tokio::time::Instant::now() < deadline => {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                Err(err) => {
                    return Err(WhsprError::Transcription(format!(
                        "faster-whisper worker at {} did not become ready: {err}",
                        self.socket_path.display()
                    )));
                }
            }
        }
    }

    fn is_running(&self) -> bool {
        StdUnixStream::connect(&self.socket_path).is_ok()
    }

    pub(crate) fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    fn spawn_worker(&self) -> Result<()> {
        let python = runtime_python_path();
        let script = ensure_worker_script()?;
        let mut command = Command::new(&python);
        command
            .arg(&script)
            .arg("serve")
            .arg("--socket-path")
            .arg(&self.socket_path)
            .arg("--model-dir")
            .arg(&self.model_path)
            .arg("--language")
            .arg(&self.language)
            .arg("--device")
            .arg(if self.use_gpu { "cuda" } else { "cpu" })
            .arg("--compute-type")
            .arg(if self.use_gpu { "float16" } else { "int8" })
            .arg("--idle-timeout-ms")
            .arg(self.idle_timeout_ms.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        if self.use_gpu {
            apply_cuda_library_path(&mut command);
        }

        command.spawn().map_err(|e| {
            WhsprError::Transcription(format!(
                "failed to start faster-whisper worker via {}: {e}",
                python.display()
            ))
        })?;
        Ok(())
    }

    pub async fn transcribe(&self, audio: &[f32], sample_rate: u32) -> Result<Transcript> {
        let timeout = Duration::from_millis(60_000);
        self.ensure_running(timeout).await?;

        let mut stream = tokio::time::timeout(timeout, UnixStream::connect(&self.socket_path))
            .await
            .map_err(|_| {
                WhsprError::Transcription(format!(
                    "faster-whisper worker timed out after {}ms",
                    timeout.as_millis()
                ))
            })?
            .map_err(|e| {
                WhsprError::Transcription(format!(
                    "failed to connect to faster-whisper worker at {}: {e}",
                    self.socket_path.display()
                ))
            })?;

        let mut audio_bytes = Vec::with_capacity(std::mem::size_of_val(audio));
        for sample in audio {
            audio_bytes.extend_from_slice(&sample.to_le_bytes());
        }
        let mut payload = serde_json::to_vec(&AsrRequest::Transcribe {
            audio_f32_b64: base64::engine::general_purpose::STANDARD.encode(audio_bytes),
            sample_rate,
        })
        .map_err(|e| WhsprError::Transcription(format!("failed to encode ASR request: {e}")))?;
        payload.push(b'\n');
        stream
            .write_all(&payload)
            .await
            .map_err(|e| WhsprError::Transcription(format!("failed to send ASR request: {e}")))?;
        stream
            .flush()
            .await
            .map_err(|e| WhsprError::Transcription(format!("failed to flush ASR request: {e}")))?;

        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        tokio::time::timeout(timeout, reader.read_line(&mut line))
            .await
            .map_err(|_| {
                WhsprError::Transcription(format!(
                    "faster-whisper worker timed out after {}ms",
                    timeout.as_millis()
                ))
            })?
            .map_err(|e| WhsprError::Transcription(format!("failed to read ASR response: {e}")))?;

        if line.trim().is_empty() {
            return Err(WhsprError::Transcription(
                "faster-whisper worker exited without sending a response".into(),
            ));
        }

        match serde_json::from_str::<AsrResponse>(&line)
            .map_err(|e| WhsprError::Transcription(format!("invalid ASR worker response: {e}")))?
        {
            AsrResponse::Transcript { transcript } => Ok(transcript),
            AsrResponse::Error { message } => Err(WhsprError::Transcription(message)),
        }
    }
}

fn find_managed_model(name: &str) -> Option<&'static ManagedModelInfo> {
    MANAGED_MODELS.iter().find(|info| info.name == name)
}

fn remove_existing_model_dir(path: &Path) -> std::io::Result<()> {
    if path.is_dir() {
        std::fs::remove_dir_all(path)
    } else {
        std::fs::remove_file(path)
    }
}

fn apply_cuda_library_path(command: &mut Command) {
    let Some(ld_library_path) = discover_cuda_library_path() else {
        return;
    };
    command.env("LD_LIBRARY_PATH", ld_library_path);
}

fn discover_cuda_library_path() -> Option<String> {
    let mut dirs = Vec::new();
    push_unique_existing_dir(&mut dirs, Path::new("/usr/lib"));
    push_unique_existing_dir(&mut dirs, Path::new("/usr/lib64"));
    push_unique_existing_dir(&mut dirs, Path::new("/opt/cuda/lib64"));

    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        collect_cuda_dirs(&home.join(".local/lib"), 4, &mut dirs);
        collect_cuda_dirs(&home.join(".local/share/uv/tools"), 8, &mut dirs);
    }

    if let Ok(existing) = std::env::var("LD_LIBRARY_PATH") {
        for part in existing.split(':') {
            if !part.is_empty() {
                push_unique_existing_dir(&mut dirs, Path::new(part));
            }
        }
    }

    (!dirs.is_empty()).then(|| {
        dirs.into_iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(":")
    })
}

fn collect_cuda_dirs(root: &Path, remaining_depth: usize, dirs: &mut Vec<PathBuf>) {
    if remaining_depth == 0 || !root.is_dir() {
        return;
    }

    let entries = match std::fs::read_dir(root) {
        Ok(entries) => entries,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_cuda_dirs(&path, remaining_depth - 1, dirs);
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if matches!(
            file_name,
            "libcublas.so.12" | "libcublasLt.so.12" | "libcudart.so.12" | "libcudnn.so.9"
        ) {
            if let Some(parent) = path.parent() {
                push_unique_existing_dir(dirs, parent);
            }
        }
    }
}

fn push_unique_existing_dir(dirs: &mut Vec<PathBuf>, path: &Path) {
    if !path.is_dir() {
        return;
    }
    if dirs.iter().any(|existing| existing == path) {
        return;
    }
    dirs.push(path.to_path_buf());
}

fn faster_whisper_runtime_dir() -> PathBuf {
    data_dir().join("faster-whisper")
}

fn faster_whisper_models_dir() -> PathBuf {
    faster_whisper_runtime_dir().join("models")
}

fn runtime_python_path() -> PathBuf {
    faster_whisper_runtime_dir()
        .join("venv")
        .join("bin")
        .join("python")
}

fn worker_script_path() -> PathBuf {
    faster_whisper_runtime_dir().join("faster_whisper_worker.py")
}

fn ensure_worker_script() -> Result<PathBuf> {
    let path = worker_script_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            WhsprError::Transcription(format!(
                "failed to create faster-whisper runtime directory {}: {e}",
                parent.display()
            ))
        })?;
    }
    let write_script = match std::fs::read_to_string(&path) {
        Ok(existing) => existing != PYTHON_WORKER_SOURCE,
        Err(_) => true,
    };
    if write_script {
        std::fs::write(&path, PYTHON_WORKER_SOURCE).map_err(|e| {
            WhsprError::Transcription(format!(
                "failed to write faster-whisper worker helper {}: {e}",
                path.display()
            ))
        })?;
    }
    Ok(path)
}

fn ensure_runtime_sync() -> Result<()> {
    ensure_worker_script()?;
    let python = runtime_python_path();
    let marker = faster_whisper_runtime_dir()
        .join("venv")
        .join(RUNTIME_READY_MARKER);
    if python.exists() && marker.exists() {
        return Ok(());
    }

    let runtime_dir = faster_whisper_runtime_dir();
    std::fs::create_dir_all(&runtime_dir).map_err(|e| {
        WhsprError::Transcription(format!(
            "failed to create faster-whisper runtime dir {}: {e}",
            runtime_dir.display()
        ))
    })?;

    let python3 = system_python()?;
    let venv_dir = runtime_dir.join("venv");
    let status = std::process::Command::new(&python3)
        .arg("-m")
        .arg("venv")
        .arg(&venv_dir)
        .status()
        .map_err(|e| {
            WhsprError::Transcription(format!("failed to create faster-whisper venv: {e}"))
        })?;
    if !status.success() {
        return Err(WhsprError::Transcription(format!(
            "failed to create faster-whisper venv with status {status}"
        )));
    }

    let pip = venv_dir.join("bin").join("pip");
    let status = std::process::Command::new(&pip)
        .arg("install")
        .arg("--upgrade")
        .arg("pip")
        .status()
        .map_err(|e| WhsprError::Transcription(format!("failed to bootstrap pip: {e}")))?;
    if !status.success() {
        return Err(WhsprError::Transcription(format!(
            "failed to upgrade pip for faster-whisper runtime: {status}"
        )));
    }

    let status = std::process::Command::new(&pip)
        .arg("install")
        .arg("faster-whisper")
        .arg("huggingface-hub")
        .arg("numpy")
        .status()
        .map_err(|e| {
            WhsprError::Transcription(format!("failed to install faster-whisper runtime: {e}"))
        })?;
    if !status.success() {
        return Err(WhsprError::Transcription(format!(
            "failed to install faster-whisper runtime packages: {status}"
        )));
    }

    std::fs::write(&marker, b"ready").map_err(|e| {
        WhsprError::Transcription(format!(
            "failed to write faster-whisper runtime marker {}: {e}",
            marker.display()
        ))
    })?;
    Ok(())
}

fn system_python() -> Result<PathBuf> {
    for candidate in ["python3", "python"] {
        match std::process::Command::new(candidate)
            .arg("--version")
            .status()
        {
            Ok(status) if status.success() => return Ok(PathBuf::from(candidate)),
            _ => continue,
        }
    }
    Err(WhsprError::Transcription(
        "faster-whisper backend requires python3 or python on PATH".into(),
    ))
}

fn runtime_dir() -> PathBuf {
    std::env::var("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"))
}

fn service_paths(config: &TranscriptionConfig, model_path: &Path) -> (PathBuf, PathBuf) {
    let mut hasher = DefaultHasher::new();
    env!("CARGO_PKG_VERSION").hash(&mut hasher);
    model_path.display().to_string().hash(&mut hasher);
    config.language.hash(&mut hasher);
    config.use_gpu.hash(&mut hasher);
    config.idle_timeout_ms.hash(&mut hasher);
    let digest = hasher.finish();

    let dir = runtime_dir().join("whispers");
    (
        dir.join(format!("asr-faster-{digest:016x}.sock")),
        dir.join(format!("asr-faster-{digest:016x}.lock")),
    )
}

struct StartupLock {
    path: PathBuf,
    _file: std::fs::File,
}

impl StartupLock {
    fn try_acquire(path: &Path) -> Result<Option<Self>> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                WhsprError::Transcription(format!(
                    "failed to create ASR runtime directory {}: {e}",
                    parent.display()
                ))
            })?;
        }

        match std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(path)
        {
            Ok(file) => Ok(Some(Self {
                path: path.to_path_buf(),
                _file: file,
            })),
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => Ok(None),
            Err(err) => Err(WhsprError::Transcription(format!(
                "failed to create ASR startup lock {}: {err}",
                path.display()
            ))),
        }
    }
}

impl Drop for StartupLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn managed_model_path_uses_data_dir() {
        let path = managed_model_local_path("distil-large-v3.5");
        assert!(path.ends_with("faster-whisper/models/distil-large-v3.5"));
    }

    #[test]
    fn resolve_model_path_prefers_config_override() {
        let _env_lock = crate::test_support::env_lock();
        let _guard = crate::test_support::EnvVarGuard::capture(&["HOME"]);
        let home = crate::test_support::unique_temp_dir("faster-whisper-home");
        crate::test_support::set_env("HOME", &home.to_string_lossy());

        let config = TranscriptionConfig {
            backend: TranscriptionBackend::FasterWhisper,
            model_path: "~/custom-distil".into(),
            ..TranscriptionConfig::default()
        };
        let path = resolve_model_path(&config).expect("path");
        assert_eq!(
            path,
            PathBuf::from(crate::config::expand_tilde("~/custom-distil"))
        );
    }

    #[test]
    fn model_dir_is_ready_requires_model_bin() {
        let dir = crate::test_support::unique_temp_dir("faster-whisper-ready");
        std::fs::write(dir.join("config.json"), b"{}").expect("write placeholder");
        assert!(!model_dir_is_ready(&dir));

        std::fs::write(dir.join("model.bin"), b"ct2").expect("write model bin");
        assert!(model_dir_is_ready(&dir));
    }

    #[test]
    fn collect_cuda_dirs_finds_nested_library_directory() {
        let root = crate::test_support::unique_temp_dir("faster-whisper-cuda-libs");
        let lib_dir = root
            .join("nested")
            .join("nvidia")
            .join("cublas")
            .join("lib");
        std::fs::create_dir_all(&lib_dir).expect("create lib dir");
        std::fs::write(lib_dir.join("libcublas.so.12"), b"stub").expect("write cublas");

        let mut dirs = Vec::new();
        collect_cuda_dirs(&root, 6, &mut dirs);

        assert_eq!(dirs, vec![lib_dir]);
    }
}

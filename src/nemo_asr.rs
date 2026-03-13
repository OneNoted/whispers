use std::collections::hash_map::DefaultHasher;
use std::ffi::OsStr;
use std::hash::{Hash, Hasher};
use std::os::unix::net::UnixStream as StdUnixStream;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use base64::Engine;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::process::Command;

use crate::asr_protocol::{AsrRequest, AsrResponse};
use crate::config::{TranscriptionBackend, TranscriptionConfig, data_dir};
use crate::error::{Result, WhsprError};
use crate::transcribe::Transcript;

const PYTHON_WORKER_SOURCE: &str = include_str!("nemo_asr_worker.py");
const RUNTIME_READY_MARKER: &str = ".runtime-ready";
const MODEL_READY_METADATA: &str = ".model-ready.json";
const STARTUP_LOCK_STALE_AGE: Duration = Duration::from_secs(600);
const STARTUP_READY_TIMEOUT: Duration = Duration::from_secs(240);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NemoModelFamily {
    Parakeet,
    CanaryQwen,
}

impl NemoModelFamily {
    fn as_str(self) -> &'static str {
        match self {
            Self::Parakeet => "parakeet",
            Self::CanaryQwen => "canary_qwen",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NemoRuntimeProfile {
    Asr,
    SpeechLm,
}

impl NemoRuntimeProfile {
    fn as_str(self) -> &'static str {
        match self {
            Self::Asr => "asr",
            Self::SpeechLm => "speechlm",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ManagedModelInfo {
    name: &'static str,
    repo_id: &'static str,
    family: NemoModelFamily,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelReadyMetadata {
    repo_id: String,
    family: NemoModelFamily,
    local_model_path: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct ResolvedModelRef {
    model_ref: String,
    family: NemoModelFamily,
}

const MANAGED_MODELS: &[ManagedModelInfo] = &[
    ManagedModelInfo {
        name: "parakeet-tdt_ctc-1.1b",
        repo_id: "nvidia/parakeet-tdt_ctc-1.1b",
        family: NemoModelFamily::Parakeet,
    },
    ManagedModelInfo {
        name: "canary-qwen-2.5b",
        repo_id: "nvidia/canary-qwen-2.5b",
        family: NemoModelFamily::CanaryQwen,
    },
];

pub fn managed_model_local_path(name: &str) -> PathBuf {
    nemo_models_dir().join(name)
}

pub fn model_dir_is_ready(path: &Path) -> bool {
    path.join(MODEL_READY_METADATA).is_file()
}

pub async fn download_managed_model(name: &str) -> Result<()> {
    let model = find_managed_model(name)
        .ok_or_else(|| WhsprError::Download(format!("unknown NeMo model '{name}'")))?;
    tokio::task::spawn_blocking(move || {
        ensure_runtime_sync(model.family)?;
        let script = ensure_worker_script()?;
        let python = runtime_python_path(runtime_profile_for_family(model.family));
        let model_dir = managed_model_local_path(model.name);
        if model_dir_is_ready(&model_dir) {
            tracing::info!(
                "nemo ASR model '{}' already prepared at {}",
                model.name,
                model_dir.display()
            );
            println!("{}", crate::ui::ready_message("ASR", model.name));
            return Ok(());
        }
        if model_dir.exists() {
            remove_existing_model_dir(&model_dir).map_err(|e| {
                WhsprError::Download(format!(
                    "failed to clear incomplete NeMo model directory {}: {e}",
                    model_dir.display()
                ))
            })?;
        }
        std::fs::create_dir_all(&model_dir).map_err(|e| {
            WhsprError::Download(format!(
                "failed to create NeMo model directory {}: {e}",
                model_dir.display()
            ))
        })?;
        let spinner = crate::ui::spinner(format!(
            "Preparing experimental NeMo runtime and model {}...",
            model.name
        ));
        let status = std::process::Command::new(&python)
            .arg(&script)
            .arg("download")
            .arg("--model-ref")
            .arg(model.repo_id)
            .arg("--family")
            .arg(model.family.as_str())
            .arg("--model-dir")
            .arg(&model_dir)
            .arg("--cache-dir")
            .arg(nemo_cache_dir())
            .stdout(crate::ui::child_stdio())
            .stderr(crate::ui::child_stdio())
            .status()
            .map_err(|e| {
                WhsprError::Download(format!(
                    "failed to start NeMo downloader via {}: {e}",
                    python.display()
                ))
            })?;
        spinner.finish_and_clear();
        if !status.success() {
            return Err(WhsprError::Download(format!(
                "NeMo model download failed with status {status}"
            )));
        }
        if !model_dir_is_ready(&model_dir) {
            return Err(WhsprError::Download(format!(
                "downloaded NeMo model at {} is incomplete",
                model_dir.display()
            )));
        }
        println!("{}", crate::ui::ready_message("ASR", model.name));
        Ok(())
    })
    .await
    .map_err(|e| WhsprError::Download(format!("NeMo download task failed: {e}")))?
}

pub fn prepare_service(config: &TranscriptionConfig) -> Option<NemoAsrService> {
    if config.backend != TranscriptionBackend::Nemo {
        return None;
    }
    let resolved = resolve_model_ref(config)?;
    Some(NemoAsrService::new(config, &resolved))
}

pub fn resolve_model_ref(config: &TranscriptionConfig) -> Option<ResolvedModelRef> {
    if let Some(model) = find_managed_model(&config.selected_model) {
        let model_dir = managed_model_local_path(model.name);
        if let Some(metadata) = load_model_ready_metadata(&model_dir) {
            if let Some(local_model_path) = metadata.local_model_path {
                return Some(ResolvedModelRef {
                    model_ref: local_model_path,
                    family: metadata.family,
                });
            }
        }
        if let Some(local_model_path) = infer_cached_model_path(model.repo_id, model.family) {
            return Some(ResolvedModelRef {
                model_ref: local_model_path,
                family: model.family,
            });
        }
        return Some(ResolvedModelRef {
            model_ref: model.repo_id.to_string(),
            family: model.family,
        });
    }

    if !config.model_path.trim().is_empty() {
        let model_ref = crate::config::expand_tilde(&config.model_path);
        let family = infer_family(&config.selected_model).or_else(|| infer_family(&model_ref))?;
        return Some(ResolvedModelRef { model_ref, family });
    }

    None
}

#[derive(Debug, Clone)]
pub struct NemoAsrService {
    socket_path: PathBuf,
    lock_path: PathBuf,
    model_ref: String,
    family: NemoModelFamily,
    language: String,
    use_gpu: bool,
    idle_timeout_ms: u64,
}

impl NemoAsrService {
    fn new(config: &TranscriptionConfig, model: &ResolvedModelRef) -> Self {
        let (socket_path, lock_path) = service_paths(config, model);
        Self {
            socket_path,
            lock_path,
            model_ref: model.model_ref.clone(),
            family: model.family,
            language: config.language.clone(),
            use_gpu: config.use_gpu,
            idle_timeout_ms: config.idle_timeout_ms,
        }
    }

    pub fn prewarm(&self) -> Result<()> {
        if self.is_running() {
            tracing::debug!(
                socket = %self.socket_path.display(),
                "NeMo ASR worker already running"
            );
            return Ok(());
        }

        let mut lock = match StartupLock::try_acquire(&self.lock_path)? {
            Some(lock) => lock,
            None => {
                tracing::debug!(
                    lock = %self.lock_path.display(),
                    "NeMo ASR worker startup already in progress"
                );
                return Ok(());
            }
        };
        if self.is_running() {
            return Ok(());
        }
        if self.socket_path.exists() {
            let _ = std::fs::remove_file(&self.socket_path);
        }
        ensure_runtime_sync(self.family)?;
        tracing::info!(
            model_ref = %self.model_ref,
            socket = %self.socket_path.display(),
            "starting NeMo ASR worker"
        );
        self.spawn_worker(Some(&lock.path))?;
        lock.disarm();
        Ok(())
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
                    tracing::info!(
                        socket = %self.socket_path.display(),
                        "NeMo ASR worker is ready"
                    );
                    return Ok(());
                }
                Err(_) if tokio::time::Instant::now() < deadline => {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                Err(err) => {
                    let _ = std::fs::remove_file(&self.lock_path);
                    return Err(WhsprError::Transcription(format!(
                        "NeMo worker at {} did not become ready: {err}",
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

    fn spawn_worker(&self, startup_lock_path: Option<&Path>) -> Result<()> {
        let python = runtime_python_path(runtime_profile_for_family(self.family));
        let script = ensure_worker_script()?;
        let mut command = Command::new(&python);
        command
            .arg(&script)
            .arg("serve")
            .arg("--socket-path")
            .arg(&self.socket_path)
            .arg("--model-ref")
            .arg(&self.model_ref)
            .arg("--family")
            .arg(self.family.as_str())
            .arg("--language")
            .arg(&self.language)
            .arg("--device")
            .arg(if self.use_gpu { "cuda" } else { "cpu" })
            .arg("--idle-timeout-ms")
            .arg(self.idle_timeout_ms.to_string())
            .arg("--cache-dir")
            .arg(nemo_cache_dir())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        if let Some(lock_path) = startup_lock_path {
            command.arg("--startup-lock-path").arg(lock_path);
        }

        command.spawn().map_err(|e| {
            WhsprError::Transcription(format!(
                "failed to start NeMo ASR worker via {}: {e}",
                python.display()
            ))
        })?;
        Ok(())
    }

    pub async fn transcribe(&self, audio: &[f32], sample_rate: u32) -> Result<Transcript> {
        self.ensure_running(STARTUP_READY_TIMEOUT).await?;

        let mut stream =
            tokio::time::timeout(REQUEST_TIMEOUT, UnixStream::connect(&self.socket_path))
                .await
                .map_err(|_| {
                    WhsprError::Transcription(format!(
                        "NeMo ASR worker timed out after {}ms",
                        REQUEST_TIMEOUT.as_millis()
                    ))
                })?
                .map_err(|e| {
                    WhsprError::Transcription(format!(
                        "failed to connect to NeMo ASR worker at {}: {e}",
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
        tokio::time::timeout(REQUEST_TIMEOUT, reader.read_line(&mut line))
            .await
            .map_err(|_| {
                WhsprError::Transcription(format!(
                    "NeMo ASR worker timed out after {}ms",
                    REQUEST_TIMEOUT.as_millis()
                ))
            })?
            .map_err(|e| WhsprError::Transcription(format!("failed to read ASR response: {e}")))?;

        if line.trim().is_empty() {
            return Err(WhsprError::Transcription(
                "NeMo ASR worker exited without sending a response".into(),
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

fn load_model_ready_metadata(model_dir: &Path) -> Option<ModelReadyMetadata> {
    let path = model_dir.join(MODEL_READY_METADATA);
    let raw = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&raw).ok()
}

fn infer_cached_model_path(repo_id: &str, family: NemoModelFamily) -> Option<String> {
    let cache_root = nemo_cache_dir()
        .join("hub")
        .join(repo_id_cache_key(repo_id));
    let revision = std::fs::read_to_string(cache_root.join("refs").join("main"))
        .ok()?
        .trim()
        .to_string();
    if revision.is_empty() {
        return None;
    }
    let snapshot_dir = cache_root.join("snapshots").join(revision);
    let candidate = match family {
        NemoModelFamily::Parakeet => snapshot_dir.join("parakeet-tdt_ctc-1.1b.nemo"),
        NemoModelFamily::CanaryQwen => return None,
    };
    candidate.exists().then(|| candidate.display().to_string())
}

fn repo_id_cache_key(repo_id: &str) -> String {
    format!("models--{}", repo_id.replace('/', "--"))
}

fn infer_family(value: &str) -> Option<NemoModelFamily> {
    let lower = value.to_ascii_lowercase();
    if lower.contains("parakeet") {
        Some(NemoModelFamily::Parakeet)
    } else if lower.contains("canary-qwen") || lower.contains("canary_qwen") {
        Some(NemoModelFamily::CanaryQwen)
    } else {
        None
    }
}

fn remove_existing_model_dir(path: &Path) -> std::io::Result<()> {
    if path.is_dir() {
        std::fs::remove_dir_all(path)
    } else {
        std::fs::remove_file(path)
    }
}

fn runtime_profile_for_family(family: NemoModelFamily) -> NemoRuntimeProfile {
    match family {
        NemoModelFamily::Parakeet => NemoRuntimeProfile::Asr,
        NemoModelFamily::CanaryQwen => NemoRuntimeProfile::SpeechLm,
    }
}

fn runtime_python_path(profile: NemoRuntimeProfile) -> PathBuf {
    runtime_env_dir(profile).join("bin").join("python")
}

fn worker_script_path() -> PathBuf {
    nemo_runtime_root().join("nemo_asr_worker.py")
}

fn ensure_worker_script() -> Result<PathBuf> {
    let path = worker_script_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            WhsprError::Transcription(format!(
                "failed to create NeMo runtime directory {}: {e}",
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
                "failed to write NeMo worker helper {}: {e}",
                path.display()
            ))
        })?;
    }
    Ok(path)
}

fn ensure_runtime_sync(family: NemoModelFamily) -> Result<()> {
    ensure_worker_script()?;
    let profile = runtime_profile_for_family(family);
    let python = runtime_python_path(profile);
    let marker = runtime_env_dir(profile).join(RUNTIME_READY_MARKER);
    if python.exists() && marker.exists() {
        return Ok(());
    }

    let runtime_dir = runtime_env_dir(profile);
    std::fs::create_dir_all(&runtime_dir).map_err(|e| {
        WhsprError::Transcription(format!(
            "failed to create NeMo runtime dir {}: {e}",
            runtime_dir.display()
        ))
    })?;

    let python3 = system_python()?;
    let venv_dir = runtime_dir;
    let status = std::process::Command::new(&python3)
        .arg("-m")
        .arg("venv")
        .arg(&venv_dir)
        .status()
        .map_err(|e| WhsprError::Transcription(format!("failed to create NeMo venv: {e}")))?;
    if !status.success() {
        return Err(WhsprError::Transcription(format!(
            "failed to create NeMo venv with status {status}"
        )));
    }

    let pip = venv_dir.join("bin").join("pip");
    let status = std::process::Command::new(&pip)
        .arg("install")
        .arg("--upgrade")
        .arg("pip")
        .arg("setuptools")
        .arg("wheel")
        .status()
        .map_err(|e| WhsprError::Transcription(format!("failed to bootstrap pip: {e}")))?;
    if !status.success() {
        return Err(WhsprError::Transcription(format!(
            "failed to upgrade pip for NeMo runtime: {status}"
        )));
    }

    install_runtime_packages(&pip, profile)?;

    std::fs::write(&marker, b"ready").map_err(|e| {
        WhsprError::Transcription(format!(
            "failed to write NeMo runtime marker {}: {e}",
            marker.display()
        ))
    })?;
    Ok(())
}

fn system_python() -> Result<PathBuf> {
    for candidate in [
        "python3.10",
        "python3.11",
        "python3.12",
        "python3",
        "python",
    ] {
        let Some((major, minor)) = python_version(candidate) else {
            continue;
        };
        if major == 3 && (10..=12).contains(&minor) {
            return Ok(PathBuf::from(candidate));
        }
    }
    Err(WhsprError::Transcription(
        "NeMo experimental backend requires Python 3.10, 3.11, or 3.12 on PATH".into(),
    ))
}

fn python_version<S: AsRef<OsStr>>(python: S) -> Option<(u32, u32)> {
    let output = std::process::Command::new(python)
        .arg("-c")
        .arg("import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let version = String::from_utf8(output.stdout).ok()?;
    let mut parts = version.trim().split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    Some((major, minor))
}

fn install_runtime_packages(pip: &Path, profile: NemoRuntimeProfile) -> Result<()> {
    run_pip_install(
        pip,
        &["install", "Cython", "packaging", "huggingface-hub"],
        "base NeMo runtime helpers",
    )?;
    run_pip_install(
        pip,
        &["install", "torch", "torchaudio"],
        "NeMo runtime torch packages",
    )?;

    match profile {
        NemoRuntimeProfile::Asr => run_pip_install(
            pip,
            &["install", "nemo_toolkit[asr]>=2.5.0"],
            "Parakeet NeMo ASR packages",
        ),
        NemoRuntimeProfile::SpeechLm => run_pip_install(
            pip,
            &[
                "install",
                "nemo_toolkit[asr,tts] @ git+https://github.com/NVIDIA/NeMo.git",
            ],
            "Canary-Qwen NeMo runtime packages",
        ),
    }
}

fn run_pip_install(pip: &Path, args: &[&str], label: &str) -> Result<()> {
    let status = std::process::Command::new(pip)
        .args(args)
        .status()
        .map_err(|e| WhsprError::Transcription(format!("failed to install {label}: {e}")))?;
    if !status.success() {
        return Err(WhsprError::Transcription(format!(
            "failed to install {label}: exit status: {status}"
        )));
    }
    Ok(())
}

fn runtime_dir() -> PathBuf {
    std::env::var("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"))
}

fn nemo_runtime_root() -> PathBuf {
    data_dir().join("nemo")
}

fn runtime_env_dir(profile: NemoRuntimeProfile) -> PathBuf {
    nemo_runtime_root().join(format!("venv-{}", profile.as_str()))
}

fn nemo_models_dir() -> PathBuf {
    nemo_runtime_root().join("models")
}

fn nemo_cache_dir() -> PathBuf {
    nemo_runtime_root().join("hf-cache")
}

fn service_paths(config: &TranscriptionConfig, model: &ResolvedModelRef) -> (PathBuf, PathBuf) {
    let mut hasher = DefaultHasher::new();
    env!("CARGO_PKG_VERSION").hash(&mut hasher);
    model.model_ref.hash(&mut hasher);
    model.family.hash(&mut hasher);
    config.language.hash(&mut hasher);
    config.use_gpu.hash(&mut hasher);
    config.idle_timeout_ms.hash(&mut hasher);
    let digest = hasher.finish();

    let dir = runtime_dir().join("whispers");
    (
        dir.join(format!("asr-nemo-{digest:016x}.sock")),
        dir.join(format!("asr-nemo-{digest:016x}.lock")),
    )
}

struct StartupLock {
    path: PathBuf,
    file: std::fs::File,
}

impl StartupLock {
    fn try_acquire(path: &Path) -> Result<Option<Self>> {
        use std::io::ErrorKind;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                WhsprError::Transcription(format!(
                    "failed to create runtime directory {}: {e}",
                    parent.display()
                ))
            })?;
        }

        if let Ok(metadata) = std::fs::metadata(path) {
            if let Ok(modified) = metadata.modified() {
                if let Ok(elapsed) = modified.elapsed() {
                    if elapsed > STARTUP_LOCK_STALE_AGE {
                        let _ = std::fs::remove_file(path);
                    }
                }
            }
        }

        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
        {
            Ok(file) => Ok(Some(Self {
                path: path.to_path_buf(),
                file,
            })),
            Err(err) if err.kind() == ErrorKind::AlreadyExists => Ok(None),
            Err(err) => Err(WhsprError::Transcription(format!(
                "failed to create startup lock {}: {err}",
                path.display()
            ))),
        }
    }

    fn disarm(&mut self) {
        let _ = self.file.sync_all();
        self.path = PathBuf::new();
    }
}

impl Drop for StartupLock {
    fn drop(&mut self) {
        if self.path.as_os_str().is_empty() {
            return;
        }
        let _ = self.file.sync_all();
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TranscriptionConfig;
    use std::sync::{Mutex, OnceLock};

    fn test_dir_guard() -> std::sync::MutexGuard<'static, ()> {
        static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
        GUARD.get_or_init(|| Mutex::new(())).lock().expect("lock")
    }

    #[test]
    fn managed_model_path_uses_data_dir() {
        let path = managed_model_local_path("parakeet-tdt_ctc-1.1b");
        assert!(path.ends_with("nemo/models/parakeet-tdt_ctc-1.1b"));
    }

    #[test]
    fn model_dir_is_ready_requires_marker() {
        let dir = crate::test_support::unique_temp_dir("nemo-model-ready");
        assert!(!model_dir_is_ready(&dir));
        std::fs::write(dir.join(MODEL_READY_METADATA), b"{}").expect("write marker");
        assert!(model_dir_is_ready(&dir));
    }

    #[test]
    fn resolve_model_ref_uses_managed_repo_ids() {
        let config = TranscriptionConfig {
            backend: TranscriptionBackend::Nemo,
            selected_model: "canary-qwen-2.5b".into(),
            ..TranscriptionConfig::default()
        };
        let resolved = resolve_model_ref(&config).expect("resolve");
        assert_eq!(resolved.model_ref, "nvidia/canary-qwen-2.5b");
        assert_eq!(resolved.family, NemoModelFamily::CanaryQwen);
    }

    #[test]
    fn resolve_model_ref_prefers_cached_local_parakeet_snapshot() {
        let _env_lock = crate::test_support::env_lock();
        let _env_guard = crate::test_support::EnvVarGuard::capture(&["XDG_DATA_HOME"]);
        let _guard = test_dir_guard();
        let runtime_root = crate::test_support::unique_temp_dir("nemo-cache");
        unsafe {
            std::env::set_var("XDG_DATA_HOME", runtime_root.join("data"));
        }
        let cache_root = runtime_root
            .join("data")
            .join("whispers")
            .join("nemo")
            .join("hf-cache")
            .join("hub")
            .join("models--nvidia--parakeet-tdt_ctc-1.1b");
        std::fs::create_dir_all(cache_root.join("refs")).expect("refs");
        std::fs::create_dir_all(cache_root.join("snapshots").join("abc123")).expect("snapshot");
        std::fs::write(cache_root.join("refs").join("main"), "abc123").expect("write ref");
        let local_model = cache_root
            .join("snapshots")
            .join("abc123")
            .join("parakeet-tdt_ctc-1.1b.nemo");
        std::fs::write(&local_model, b"stub").expect("write model");

        let config = TranscriptionConfig {
            backend: TranscriptionBackend::Nemo,
            selected_model: "parakeet-tdt_ctc-1.1b".into(),
            ..TranscriptionConfig::default()
        };
        let resolved = resolve_model_ref(&config).expect("resolve");
        assert_eq!(resolved.model_ref, local_model.display().to_string());
        assert_eq!(resolved.family, NemoModelFamily::Parakeet);
    }

    #[test]
    fn prepare_service_requires_nemo_backend() {
        let config = TranscriptionConfig::default();
        assert!(prepare_service(&config).is_none());
    }
}

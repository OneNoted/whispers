use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::os::unix::net::UnixStream as StdUnixStream;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::process::Command;

use crate::config::RewriteConfig;
use crate::error::{Result, WhsprError};
use crate::rewrite_model;
use crate::rewrite_profile::ResolvedRewriteProfile;
use crate::rewrite_protocol::{RewriteTranscript, WorkerRequest, WorkerResponse};

pub async fn rewrite_transcript(
    config: &RewriteConfig,
    model_path: &Path,
    transcript: &RewriteTranscript,
    custom_instructions: Option<&str>,
) -> Result<String> {
    let service = RewriteService::new(config, model_path);
    rewrite_with_service(&service, config, transcript, custom_instructions).await
}

#[derive(Debug, Clone)]
pub struct RewriteService {
    socket_path: PathBuf,
    lock_path: PathBuf,
    model_path: PathBuf,
    profile: ResolvedRewriteProfile,
    max_tokens: usize,
    max_output_chars: usize,
    idle_timeout_ms: u64,
}

impl RewriteService {
    pub fn new(config: &RewriteConfig, model_path: &Path) -> Self {
        let selected_managed_model = if config.model_path.trim().is_empty()
            && rewrite_model::managed_profile(&config.selected_model).is_some()
        {
            Some(config.selected_model.as_str())
        } else {
            None
        };
        let profile = config.profile.resolve(selected_managed_model, model_path);
        let (socket_path, lock_path) = service_paths(model_path, profile, config);

        Self {
            socket_path,
            lock_path,
            model_path: model_path.to_path_buf(),
            profile,
            max_tokens: config.max_tokens,
            max_output_chars: config.max_output_chars,
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
                Err(err) if tokio::time::Instant::now() < deadline => {
                    let _ = err;
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                Err(err) => {
                    return Err(WhsprError::Rewrite(format!(
                        "rewrite worker at {} did not become ready: {err}",
                        self.socket_path.display()
                    )));
                }
            }
        }
    }

    fn is_running(&self) -> bool {
        StdUnixStream::connect(&self.socket_path).is_ok()
    }

    fn spawn_worker(&self) -> Result<()> {
        let worker_path = worker_executable_path();
        let mut command = Command::new(&worker_path);
        command
            .arg("--model-path")
            .arg(&self.model_path)
            .arg("--socket-path")
            .arg(&self.socket_path)
            .arg("--profile")
            .arg(self.profile.as_str())
            .arg("--max-tokens")
            .arg(self.max_tokens.to_string())
            .arg("--max-output-chars")
            .arg(self.max_output_chars.to_string())
            .arg("--idle-timeout-ms")
            .arg(self.idle_timeout_ms.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());

        command.spawn().map_err(|e| {
            WhsprError::Rewrite(format!(
                "failed to start rewrite worker from {}: {e}",
                worker_path.display()
            ))
        })?;

        Ok(())
    }
}

pub async fn rewrite_with_service(
    service: &RewriteService,
    config: &RewriteConfig,
    transcript: &RewriteTranscript,
    custom_instructions: Option<&str>,
) -> Result<String> {
    let timeout = Duration::from_millis(config.timeout_ms);
    tracing::trace!(
        candidates = transcript.rewrite_candidates.len(),
        hypotheses = transcript.edit_hypotheses.len(),
        has_recommended = transcript.recommended_candidate.is_some(),
        "sending rewrite request to worker"
    );
    service.ensure_running(timeout).await?;

    let mut stream = tokio::time::timeout(timeout, UnixStream::connect(&service.socket_path))
        .await
        .map_err(|_| {
            WhsprError::Rewrite(format!(
                "rewrite worker timed out after {}ms",
                timeout.as_millis()
            ))
        })?
        .map_err(|e| {
            WhsprError::Rewrite(format!(
                "failed to connect to rewrite worker at {}: {e}",
                service.socket_path.display()
            ))
        })?;

    let mut payload = serde_json::to_vec(&WorkerRequest::Rewrite {
        transcript: transcript.clone(),
        custom_instructions: custom_instructions.map(str::to_owned),
    })
    .map_err(|e| WhsprError::Rewrite(format!("failed to encode rewrite request: {e}")))?;
    payload.push(b'\n');
    stream
        .write_all(&payload)
        .await
        .map_err(|e| WhsprError::Rewrite(format!("failed to send rewrite request: {e}")))?;
    stream
        .flush()
        .await
        .map_err(|e| WhsprError::Rewrite(format!("failed to flush rewrite request: {e}")))?;

    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    tokio::time::timeout(timeout, reader.read_line(&mut line))
        .await
        .map_err(|_| {
            WhsprError::Rewrite(format!(
                "rewrite worker timed out after {}ms",
                timeout.as_millis()
            ))
        })?
        .map_err(|e| WhsprError::Rewrite(format!("failed to read rewrite response: {e}")))?;

    if line.trim().is_empty() {
        return Err(WhsprError::Rewrite(
            "rewrite worker exited without sending a response".into(),
        ));
    }

    match serde_json::from_str::<WorkerResponse>(&line)
        .map_err(|e| WhsprError::Rewrite(format!("invalid rewrite worker response: {e}")))?
    {
        WorkerResponse::Result { text } => {
            tracing::trace!(
                output_len = text.len(),
                "received rewrite response from worker"
            );
            Ok(text)
        }
        WorkerResponse::Error { message } => Err(WhsprError::Rewrite(message)),
    }
}

fn runtime_dir() -> PathBuf {
    std::env::var("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"))
}

fn service_paths(
    model_path: &Path,
    profile: ResolvedRewriteProfile,
    config: &RewriteConfig,
) -> (PathBuf, PathBuf) {
    let mut hasher = DefaultHasher::new();
    env!("CARGO_PKG_VERSION").hash(&mut hasher);
    model_path.display().to_string().hash(&mut hasher);
    profile.as_str().hash(&mut hasher);
    config.max_tokens.hash(&mut hasher);
    config.max_output_chars.hash(&mut hasher);
    let digest = hasher.finish();

    let dir = runtime_dir().join("whispers");
    let socket_path = dir.join(format!("rewrite-{digest:016x}.sock"));
    let lock_path = dir.join(format!("rewrite-{digest:016x}.lock"));
    (socket_path, lock_path)
}

fn worker_executable_path() -> PathBuf {
    if let Ok(path) = std::env::var("WHISPERS_REWRITE_WORKER") {
        return PathBuf::from(path);
    }

    std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|dir| dir.join("whispers-rewrite-worker")))
        .filter(|path| path.exists())
        .unwrap_or_else(|| PathBuf::from("whispers-rewrite-worker"))
}

struct StartupLock {
    path: PathBuf,
    _file: std::fs::File,
}

impl StartupLock {
    fn try_acquire(path: &Path) -> Result<Option<Self>> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                WhsprError::Rewrite(format!(
                    "failed to create rewrite runtime directory {}: {e}",
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
            Err(err) => Err(WhsprError::Rewrite(format!(
                "failed to create rewrite startup lock {}: {err}",
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
    use crate::rewrite_profile::RewriteProfile;

    #[test]
    fn service_paths_change_when_profile_changes() {
        let config = RewriteConfig::default();
        let qwen = service_paths(
            Path::new("/models/qwen.gguf"),
            ResolvedRewriteProfile::Qwen,
            &config,
        );
        let generic = service_paths(
            Path::new("/models/qwen.gguf"),
            ResolvedRewriteProfile::Generic,
            &config,
        );
        assert_ne!(qwen.0, generic.0);
    }

    #[test]
    fn manual_model_override_uses_filename_for_auto_profile_resolution() {
        let config = RewriteConfig {
            profile: RewriteProfile::Auto,
            model_path: "/models/CustomLlama.gguf".into(),
            ..RewriteConfig::default()
        };
        let service = RewriteService::new(&config, Path::new("/models/CustomLlama.gguf"));
        assert_eq!(service.profile, ResolvedRewriteProfile::LlamaCompat);
    }

    #[test]
    fn managed_model_uses_selected_model_for_auto_profile_resolution() {
        let config = RewriteConfig {
            selected_model: "qwen-3.5-4b-q4_k_m".into(),
            profile: RewriteProfile::Auto,
            ..RewriteConfig::default()
        };
        let service = RewriteService::new(&config, Path::new("/models/custom.gguf"));
        assert_eq!(service.profile, ResolvedRewriteProfile::Qwen);
    }
}

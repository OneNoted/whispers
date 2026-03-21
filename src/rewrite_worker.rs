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

    async fn ensure_running(
        &self,
        deadline: tokio::time::Instant,
        timeout: Duration,
    ) -> Result<()> {
        if self.is_running() {
            return Ok(());
        }

        self.prewarm()?;

        loop {
            let remaining = remaining_rewrite_budget(deadline, timeout)?;
            match tokio::time::timeout(remaining, UnixStream::connect(&self.socket_path)).await {
                Ok(Ok(stream)) => {
                    drop(stream);
                    return Ok(());
                }
                Ok(Err(err)) if tokio::time::Instant::now() < deadline => {
                    let _ = err;
                    tokio::time::sleep(Duration::from_millis(50).min(remaining)).await;
                }
                Ok(Err(err)) => {
                    return Err(WhsprError::Rewrite(format!(
                        "rewrite worker at {} did not become ready: {err}",
                        self.socket_path.display()
                    )));
                }
                Err(_) => return Err(rewrite_timeout_error(timeout)),
            };
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

fn remaining_rewrite_budget(deadline: tokio::time::Instant, timeout: Duration) -> Result<Duration> {
    deadline
        .checked_duration_since(tokio::time::Instant::now())
        .ok_or_else(|| rewrite_timeout_error(timeout))
}

fn rewrite_timeout_error(timeout: Duration) -> WhsprError {
    WhsprError::Rewrite(format!(
        "rewrite worker timed out after {}ms",
        timeout.as_millis()
    ))
}

async fn rewrite_request(
    service: &RewriteService,
    payload: &[u8],
    deadline: tokio::time::Instant,
    timeout: Duration,
) -> Result<String> {
    let mut stream = tokio::time::timeout(
        remaining_rewrite_budget(deadline, timeout)?,
        UnixStream::connect(&service.socket_path),
    )
    .await
    .map_err(|_| rewrite_timeout_error(timeout))?
    .map_err(|e| {
        WhsprError::Rewrite(format!(
            "failed to connect to rewrite worker at {}: {e}",
            service.socket_path.display()
        ))
    })?;

    tokio::time::timeout(
        remaining_rewrite_budget(deadline, timeout)?,
        stream.write_all(payload),
    )
    .await
    .map_err(|_| rewrite_timeout_error(timeout))?
    .map_err(|e| WhsprError::Rewrite(format!("failed to send rewrite request: {e}")))?;

    tokio::time::timeout(remaining_rewrite_budget(deadline, timeout)?, stream.flush())
        .await
        .map_err(|_| rewrite_timeout_error(timeout))?
        .map_err(|e| WhsprError::Rewrite(format!("failed to flush rewrite request: {e}")))?;

    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    tokio::time::timeout(
        remaining_rewrite_budget(deadline, timeout)?,
        reader.read_line(&mut line),
    )
    .await
    .map_err(|_| rewrite_timeout_error(timeout))?
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

pub async fn rewrite_with_service(
    service: &RewriteService,
    config: &RewriteConfig,
    transcript: &RewriteTranscript,
    custom_instructions: Option<&str>,
) -> Result<String> {
    let timeout = Duration::from_millis(config.timeout_ms);
    let deadline = tokio::time::Instant::now() + timeout;
    tracing::trace!(
        candidates = transcript.rewrite_candidates.len(),
        hypotheses = transcript.edit_hypotheses.len(),
        has_recommended = transcript.recommended_candidate.is_some(),
        "sending rewrite request to worker"
    );
    service.ensure_running(deadline, timeout).await?;

    let mut payload = serde_json::to_vec(&WorkerRequest::Rewrite {
        transcript: transcript.clone(),
        custom_instructions: custom_instructions.map(str::to_owned),
    })
    .map_err(|e| WhsprError::Rewrite(format!("failed to encode rewrite request: {e}")))?;
    payload.push(b'\n');
    rewrite_request(service, &payload, deadline, timeout).await
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
    use crate::error::WhsprError;
    use crate::rewrite_profile::RewriteProfile;
    use crate::rewrite_protocol::{
        RewriteCorrectionPolicy, RewritePolicyContext, RewriteTranscript,
    };
    use crate::test_support::unique_temp_dir;
    use std::path::PathBuf;

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

    #[tokio::test]
    async fn rewrite_with_service_times_out_when_request_write_stalls() {
        let runtime_dir = unique_temp_dir("rewrite-request-timeout");
        let socket_path = runtime_dir.join("rewrite.sock");
        let listener =
            tokio::net::UnixListener::bind(&socket_path).expect("bind stalled rewrite socket");

        let server = tokio::spawn(async move {
            let (probe, _) = listener.accept().await.expect("accept readiness probe");
            drop(probe);
            let (_stalled_request, _) = listener.accept().await.expect("accept rewrite request");
            tokio::time::sleep(Duration::from_secs(1)).await;
        });

        let service = RewriteService {
            socket_path: socket_path.clone(),
            lock_path: runtime_dir.join("rewrite.lock"),
            model_path: PathBuf::from("/tmp/test.gguf"),
            profile: ResolvedRewriteProfile::Generic,
            max_tokens: 256,
            max_output_chars: 1200,
            idle_timeout_ms: 0,
        };
        let config = RewriteConfig {
            timeout_ms: 50,
            ..RewriteConfig::default()
        };

        let err = rewrite_with_service(
            &service,
            &config,
            &oversized_transcript(2 * 1024 * 1024),
            None,
        )
        .await
        .expect_err("stalled rewrite request should time out");
        let message = match err {
            WhsprError::Rewrite(message) => message,
            other => panic!("unexpected error: {other:?}"),
        };
        assert!(message.contains("rewrite worker timed out"));

        server.abort();
    }

    fn oversized_transcript(size: usize) -> RewriteTranscript {
        let text = "word ".repeat(size / 5);
        RewriteTranscript {
            raw_text: text.clone(),
            correction_aware_text: text,
            aggressive_correction_text: None,
            detected_language: Some("en".into()),
            typing_context: None,
            recent_session_entries: Vec::new(),
            session_backtrack_candidates: Vec::new(),
            recommended_session_candidate: None,
            segments: Vec::new(),
            edit_intents: Vec::new(),
            edit_signals: Vec::new(),
            edit_hypotheses: Vec::new(),
            rewrite_candidates: Vec::new(),
            recommended_candidate: None,
            edit_context: Default::default(),
            policy_context: RewritePolicyContext {
                correction_policy: RewriteCorrectionPolicy::Balanced,
                matched_rule_names: Vec::new(),
                effective_rule_instructions: Vec::new(),
                active_glossary_terms: Vec::new(),
                glossary_candidates: Vec::new(),
            },
        }
    }
}

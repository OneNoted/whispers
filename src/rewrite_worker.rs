use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

use crate::branding;
use crate::cleanup;
use crate::config::RewriteConfig;
use crate::error::{Result, WhsprError};
use crate::rewrite_protocol::{
    RewriteTranscript, RewriteTranscriptSegment, WorkerRequest, WorkerResponse,
};
use crate::transcribe::Transcript;

pub async fn rewrite_transcript(
    config: &RewriteConfig,
    model_path: &Path,
    transcript: &Transcript,
) -> Result<String> {
    let mut worker = RewriteWorker::spawn(config, model_path)?;
    rewrite_with_worker(&mut worker, config, transcript).await
}

pub struct RewriteWorker {
    _child: Child,
    stdin: ChildStdin,
    stdout: Lines<BufReader<ChildStdout>>,
    ready: bool,
}

impl RewriteWorker {
    pub fn spawn(config: &RewriteConfig, model_path: &Path) -> Result<Self> {
        let worker_path = worker_executable_path();

        let mut command = Command::new(&worker_path);
        command
            .arg("--model-path")
            .arg(model_path)
            .arg("--max-tokens")
            .arg(config.max_tokens.to_string())
            .arg("--max-output-chars")
            .arg(config.max_output_chars.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);

        let mut child = command.spawn().map_err(|e| {
            WhsprError::Rewrite(format!(
                "failed to start rewrite worker from {}: {e}",
                worker_path.display()
            ))
        })?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| WhsprError::Rewrite("rewrite worker did not expose stdin".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| WhsprError::Rewrite("rewrite worker did not expose stdout".into()))?;

        Ok(Self {
            _child: child,
            stdin,
            stdout: BufReader::new(stdout).lines(),
            ready: false,
        })
    }

    async fn ensure_ready(&mut self, timeout: Duration) -> Result<()> {
        if self.ready {
            return Ok(());
        }

        match self.read_response(timeout).await? {
            WorkerResponse::Ready => {
                self.ready = true;
                Ok(())
            }
            WorkerResponse::Error { message } => Err(WhsprError::Rewrite(message)),
            WorkerResponse::Result { .. } => Err(WhsprError::Rewrite(
                "rewrite worker sent result before ready".into(),
            )),
        }
    }

    async fn send_request(&mut self, request: &WorkerRequest) -> Result<()> {
        let mut payload = serde_json::to_vec(request)
            .map_err(|e| WhsprError::Rewrite(format!("failed to encode rewrite request: {e}")))?;
        payload.push(b'\n');

        self.stdin
            .write_all(&payload)
            .await
            .map_err(|e| WhsprError::Rewrite(format!("failed to send rewrite request: {e}")))?;
        self.stdin
            .flush()
            .await
            .map_err(|e| WhsprError::Rewrite(format!("failed to flush rewrite request: {e}")))?;
        Ok(())
    }

    async fn read_response(&mut self, timeout: Duration) -> Result<WorkerResponse> {
        let line = tokio::time::timeout(timeout, self.stdout.next_line())
            .await
            .map_err(|_| {
                WhsprError::Rewrite(format!(
                    "rewrite worker timed out after {}ms",
                    timeout.as_millis()
                ))
            })?
            .map_err(|e| WhsprError::Rewrite(format!("failed to read rewrite response: {e}")))?;

        let line = line.ok_or_else(|| {
            WhsprError::Rewrite("rewrite worker exited without sending a response".into())
        })?;

        serde_json::from_str(&line)
            .map_err(|e| WhsprError::Rewrite(format!("invalid rewrite worker response: {e}")))
    }
}

pub async fn rewrite_with_worker(
    worker: &mut RewriteWorker,
    config: &RewriteConfig,
    transcript: &Transcript,
) -> Result<String> {
    let timeout = Duration::from_millis(config.timeout_ms);
    worker.ensure_ready(timeout).await?;

    let request = WorkerRequest::Rewrite {
        transcript: to_protocol_transcript(transcript),
    };
    worker.send_request(&request).await?;

    match worker.read_response(timeout).await? {
        WorkerResponse::Result { text } => Ok(text),
        WorkerResponse::Error { message } => Err(WhsprError::Rewrite(message)),
        WorkerResponse::Ready => Err(WhsprError::Rewrite(
            "rewrite worker sent duplicate ready response".into(),
        )),
    }
}

fn worker_executable_path() -> PathBuf {
    if let Ok(path) = std::env::var(branding::REWRITE_WORKER_ENV) {
        return PathBuf::from(path);
    }

    branding::resolve_sidecar_executable(&[branding::REWRITE_WORKER_BINARY])
}

fn to_protocol_transcript(transcript: &Transcript) -> RewriteTranscript {
    RewriteTranscript {
        raw_text: transcript.raw_text.clone(),
        correction_aware_text: cleanup::correction_aware_text(transcript),
        detected_language: transcript.detected_language.clone(),
        segments: transcript
            .segments
            .iter()
            .map(|segment| RewriteTranscriptSegment {
                text: segment.text.clone(),
                start_ms: segment.start_ms,
                end_ms: segment.end_ms,
            })
            .collect(),
    }
}

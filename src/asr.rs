use crate::cloud::CloudService;
use crate::config::{Config, TranscriptionBackend, TranscriptionConfig, TranscriptionFallback};
use crate::error::{Result, WhsprError};
use crate::faster_whisper::{self, FasterWhisperService};
use crate::model;
use crate::nemo_asr::{self, NemoAsrService};
use crate::transcribe::{
    Transcript, TranscriptionBackend as SyncTranscriptionBackend, WhisperLocal,
};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub enum PreparedTranscriber {
    Whisper(tokio::task::JoinHandle<Result<WhisperLocal>>),
    Faster(FasterWhisperService),
    Nemo(NemoAsrService),
    Cloud(CloudService),
}

pub fn prepare_transcriber(config: &Config) -> Result<PreparedTranscriber> {
    cleanup_stale_transcribers(config)?;

    match config.transcription.backend {
        TranscriptionBackend::WhisperCpp => {
            let whisper_config = config.transcription.clone();
            let model_path = config.resolved_model_path();
            Ok(PreparedTranscriber::Whisper(tokio::task::spawn_blocking(
                move || WhisperLocal::new(&whisper_config, &model_path),
            )))
        }
        TranscriptionBackend::FasterWhisper => {
            faster_whisper::prepare_service(&config.transcription)
                .map(PreparedTranscriber::Faster)
                .ok_or_else(|| {
                    WhsprError::Transcription(
                        "faster-whisper backend selected but no model path could be resolved"
                            .into(),
                    )
                })
        }
        TranscriptionBackend::Nemo => nemo_asr::prepare_service(&config.transcription)
            .map(PreparedTranscriber::Nemo)
            .ok_or_else(|| {
                WhsprError::Transcription(
                    "nemo backend selected but no model reference could be resolved".into(),
                )
            }),
        TranscriptionBackend::Cloud => Ok(PreparedTranscriber::Cloud(CloudService::new(config)?)),
    }
}

pub fn cleanup_stale_transcribers(config: &Config) -> Result<()> {
    let retained = retained_socket_paths(config);
    let stale_workers = collect_stale_asr_workers(&retained)?;
    for worker in stale_workers {
        tracing::info!(
            pid = worker.pid,
            kind = worker.kind,
            socket = %worker.socket_path.display(),
            "terminating stale ASR worker"
        );
        let result = unsafe { libc::kill(worker.pid, libc::SIGTERM) };
        if result == 0 {
            continue;
        }
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::ESRCH) {
            continue;
        }
        return Err(WhsprError::Transcription(format!(
            "failed to terminate stale {} worker (pid {}): {err}",
            worker.kind, worker.pid
        )));
    }
    Ok(())
}

pub fn prewarm_transcriber(prepared: &PreparedTranscriber, phase: &str) {
    match prepared {
        PreparedTranscriber::Faster(service) => match service.prewarm() {
            Ok(()) => tracing::info!("prewarming faster-whisper worker via {}", phase),
            Err(err) => tracing::warn!("failed to prewarm faster-whisper worker: {err}"),
        },
        PreparedTranscriber::Nemo(service) => match service.prewarm() {
            Ok(()) => tracing::info!("prewarming NeMo ASR worker via {}", phase),
            Err(err) => tracing::warn!("failed to prewarm NeMo ASR worker: {err}"),
        },
        _ => {}
    }
}

pub async fn transcribe_audio(
    config: &Config,
    prepared: PreparedTranscriber,
    audio: Vec<f32>,
    sample_rate: u32,
) -> Result<Transcript> {
    match prepared {
        PreparedTranscriber::Whisper(handle) => {
            let backend = handle.await.map_err(|e| {
                WhsprError::Transcription(format!("model loading task failed: {e}"))
            })??;
            tokio::task::spawn_blocking(move || backend.transcribe(&audio, sample_rate))
                .await
                .map_err(|e| WhsprError::Transcription(format!("transcription task failed: {e}")))?
        }
        PreparedTranscriber::Faster(service) => match service.transcribe(&audio, sample_rate).await
        {
            Ok(transcript) => Ok(transcript),
            Err(err) => {
                tracing::warn!("faster-whisper transcription failed: {err}");
                fallback_whisper_cpp_transcribe(config, audio, sample_rate).await
            }
        },
        PreparedTranscriber::Nemo(service) => match service.transcribe(&audio, sample_rate).await {
            Ok(transcript) => Ok(transcript),
            Err(err) => {
                tracing::warn!("NeMo ASR transcription failed: {err}");
                fallback_whisper_cpp_transcribe(config, audio, sample_rate).await
            }
        },
        PreparedTranscriber::Cloud(service) => {
            match service.transcribe_audio(config, &audio, sample_rate).await {
                Ok(transcript) => Ok(transcript),
                Err(err) => {
                    tracing::warn!("cloud transcription failed: {err}");
                    fallback_local_transcribe(config, audio, sample_rate).await
                }
            }
        }
    }
}

async fn fallback_local_transcribe(
    config: &Config,
    audio: Vec<f32>,
    sample_rate: u32,
) -> Result<Transcript> {
    if config.transcription.backend == TranscriptionBackend::Cloud
        && config.transcription.fallback == TranscriptionFallback::None
    {
        return Err(WhsprError::Transcription(
            "cloud transcription failed and [transcription].fallback = \"none\"".into(),
        ));
    }

    let mut local_config = config.transcription.clone();
    local_config.backend = config.transcription.resolved_local_backend();
    let model_path = config.resolved_model_path();
    tracing::warn!(
        "falling back to local ASR backend '{}' using {}",
        local_config.backend.as_str(),
        model_path.display()
    );
    let prepared = match local_config.backend {
        TranscriptionBackend::WhisperCpp => {
            let whisper_config = local_config.clone();
            Ok(PreparedTranscriber::Whisper(tokio::task::spawn_blocking(
                move || WhisperLocal::new(&whisper_config, &model_path),
            )))
        }
        TranscriptionBackend::FasterWhisper => faster_whisper::prepare_service(&local_config)
            .map(PreparedTranscriber::Faster)
            .ok_or_else(|| {
                WhsprError::Transcription(
                    "faster-whisper fallback selected but no model path could be resolved".into(),
                )
            }),
        TranscriptionBackend::Nemo => nemo_asr::prepare_service(&local_config)
            .map(PreparedTranscriber::Nemo)
            .ok_or_else(|| {
                WhsprError::Transcription(
                    "nemo fallback selected but no model reference could be resolved".into(),
                )
            }),
        TranscriptionBackend::Cloud => Err(WhsprError::Transcription(
            "cloud backend cannot be prepared as a local transcriber".into(),
        )),
    }?;
    match prepared {
        PreparedTranscriber::Whisper(handle) => {
            let backend = handle.await.map_err(|e| {
                WhsprError::Transcription(format!("fallback model loading task failed: {e}"))
            })??;
            tokio::task::spawn_blocking(move || backend.transcribe(&audio, sample_rate))
                .await
                .map_err(|e| {
                    WhsprError::Transcription(format!("fallback transcription task failed: {e}"))
                })?
        }
        PreparedTranscriber::Faster(service) => service.transcribe(&audio, sample_rate).await,
        PreparedTranscriber::Nemo(service) => service.transcribe(&audio, sample_rate).await,
        PreparedTranscriber::Cloud(_) => Err(WhsprError::Transcription(
            "cloud fallback resolved to cloud backend".into(),
        )),
    }
}

async fn fallback_whisper_cpp_transcribe(
    config: &Config,
    audio: Vec<f32>,
    sample_rate: u32,
) -> Result<Transcript> {
    let Some(model_path) = fallback_whisper_model_path() else {
        return Err(WhsprError::Transcription(
            "faster-whisper failed and no local large-v3-turbo fallback model is available".into(),
        ));
    };
    tracing::warn!("falling back to whisper_cpp using {}", model_path.display());
    let whisper_config = whisper_fallback_config(&config.transcription);
    let backend =
        tokio::task::spawn_blocking(move || WhisperLocal::new(&whisper_config, &model_path))
            .await
            .map_err(|e| {
                WhsprError::Transcription(format!("fallback model loading task failed: {e}"))
            })??;
    tokio::task::spawn_blocking(move || backend.transcribe(&audio, sample_rate))
        .await
        .map_err(|e| {
            WhsprError::Transcription(format!("fallback transcription task failed: {e}"))
        })?
}

fn whisper_fallback_config(config: &TranscriptionConfig) -> TranscriptionConfig {
    let mut fallback = config.clone();
    fallback.backend = TranscriptionBackend::WhisperCpp;
    fallback.local_backend = TranscriptionBackend::WhisperCpp;
    fallback.selected_model = "large-v3-turbo".into();
    fallback.model_path = model::model_path_for_config("ggml-large-v3-turbo.bin");
    fallback
}

fn fallback_whisper_model_path() -> Option<std::path::PathBuf> {
    let path = model::selected_model_local_path("large-v3-turbo")?;
    path.exists().then_some(path)
}

fn retained_socket_paths(config: &Config) -> HashSet<PathBuf> {
    let mut retained = HashSet::new();
    match config.transcription.backend {
        TranscriptionBackend::FasterWhisper => {
            if let Some(service) = faster_whisper::prepare_service(&config.transcription) {
                retained.insert(service.socket_path().to_path_buf());
            }
        }
        TranscriptionBackend::Nemo => {
            if let Some(service) = nemo_asr::prepare_service(&config.transcription) {
                retained.insert(service.socket_path().to_path_buf());
            }
        }
        TranscriptionBackend::WhisperCpp | TranscriptionBackend::Cloud => {}
    }
    retained
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AsrWorkerProcess {
    pid: libc::pid_t,
    kind: &'static str,
    socket_path: PathBuf,
}

fn collect_stale_asr_workers(retained: &HashSet<PathBuf>) -> Result<Vec<AsrWorkerProcess>> {
    let proc_dir = std::fs::read_dir("/proc")
        .map_err(|e| WhsprError::Transcription(format!("failed to inspect /proc: {e}")))?;
    let mut stale = Vec::new();
    for entry in proc_dir {
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => continue,
        };
        let file_name = entry.file_name();
        let Some(pid) = file_name.to_string_lossy().parse::<libc::pid_t>().ok() else {
            continue;
        };
        let cmdline = match std::fs::read(entry.path().join("cmdline")) {
            Ok(cmdline) => cmdline,
            Err(_) => continue,
        };
        let Some((kind, socket_path)) = parse_asr_worker_cmdline(&cmdline) else {
            continue;
        };
        if retained.contains(&socket_path) {
            continue;
        }
        stale.push(AsrWorkerProcess {
            pid,
            kind,
            socket_path,
        });
    }
    Ok(stale)
}

fn parse_asr_worker_cmdline(cmdline: &[u8]) -> Option<(&'static str, PathBuf)> {
    let args: Vec<String> = cmdline
        .split(|byte| *byte == 0)
        .filter(|arg| !arg.is_empty())
        .map(|arg| String::from_utf8_lossy(arg).into_owned())
        .collect();
    if args.is_empty() || !args.iter().any(|arg| arg == "serve") {
        return None;
    }

    let kind = if args.iter().any(|arg| {
        Path::new(arg)
            .file_name()
            .is_some_and(|name| name == "faster_whisper_worker.py")
    }) {
        "faster_whisper"
    } else if args.iter().any(|arg| {
        Path::new(arg)
            .file_name()
            .is_some_and(|name| name == "nemo_asr_worker.py")
    }) {
        "nemo"
    } else {
        return None;
    };

    let socket_index = args.iter().position(|arg| arg == "--socket-path")?;
    let socket_path = PathBuf::from(args.get(socket_index + 1)?);
    let runtime_scope = asr_runtime_scope_dir();
    if !socket_path.starts_with(&runtime_scope) {
        return None;
    }
    let file_name = socket_path.file_name()?.to_string_lossy();
    if !file_name.starts_with("asr-") || !file_name.ends_with(".sock") {
        return None;
    }

    Some((kind, socket_path))
}

fn asr_runtime_scope_dir() -> PathBuf {
    let base = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(base).join("whispers")
}

pub fn validate_transcription_config(config: &Config) -> Result<()> {
    if config.transcription.backend == TranscriptionBackend::Cloud {
        crate::cloud::validate_config(config)?;
    }

    if config.transcription.resolved_local_backend() == TranscriptionBackend::FasterWhisper
        && !config.transcription.language.eq_ignore_ascii_case("en")
        && !config.transcription.language.eq_ignore_ascii_case("auto")
    {
        return Err(WhsprError::Config(
            "faster-whisper managed models are currently English-focused; set [transcription].language = \"en\" or \"auto\"".into(),
        ));
    }

    if config.transcription.resolved_local_backend() == TranscriptionBackend::FasterWhisper
        && config.transcription.language.eq_ignore_ascii_case("auto")
    {
        tracing::warn!(
            "faster-whisper backend is configured with language = \"auto\"; English dictation is recommended"
        );
    }

    if config.transcription.resolved_local_backend() == TranscriptionBackend::Nemo
        && !config.transcription.language.eq_ignore_ascii_case("en")
        && !config.transcription.language.eq_ignore_ascii_case("auto")
    {
        return Err(WhsprError::Config(
            "NeMo experimental ASR models are currently English-only; set [transcription].language = \"en\" or \"auto\"".into(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_faster_worker_cmdline_extracts_socket_path() {
        let socket = asr_runtime_scope_dir().join("asr-faster-123.sock");
        let cmdline = format!(
            "/home/user/.local/share/whispers/faster-whisper/venv/bin/python\0/home/user/.local/share/whispers/faster-whisper/faster_whisper_worker.py\0serve\0--socket-path\0{}\0--model-dir\0/tmp/model\0",
            socket.display()
        );
        let parsed = parse_asr_worker_cmdline(cmdline.as_bytes()).expect("parse worker");
        assert_eq!(parsed.0, "faster_whisper");
        assert_eq!(parsed.1, socket);
    }

    #[test]
    fn parse_nemo_worker_cmdline_extracts_socket_path() {
        let socket = asr_runtime_scope_dir().join("asr-nemo-456.sock");
        let cmdline = format!(
            "/home/user/.local/share/whispers/nemo/venv-asr/bin/python\0/home/user/.local/share/whispers/nemo/nemo_asr_worker.py\0serve\0--socket-path\0{}\0--model-ref\0/tmp/model.nemo\0",
            socket.display()
        );
        let parsed = parse_asr_worker_cmdline(cmdline.as_bytes()).expect("parse worker");
        assert_eq!(parsed.0, "nemo");
        assert_eq!(parsed.1, socket);
    }

    #[test]
    fn parse_asr_worker_cmdline_ignores_unrelated_processes() {
        let socket = asr_runtime_scope_dir().join("asr-other.sock");
        let cmdline = format!(
            "/usr/bin/python\0/home/user/script.py\0serve\0--socket-path\0{}\0",
            socket.display()
        );
        assert!(parse_asr_worker_cmdline(cmdline.as_bytes()).is_none());
    }

    #[test]
    fn parse_asr_worker_cmdline_ignores_socket_outside_runtime_scope() {
        let cmdline = b"/home/user/.local/share/whispers/nemo/venv-asr/bin/python\0/home/user/.local/share/whispers/nemo/nemo_asr_worker.py\0serve\0--socket-path\0/var/run/asr-nemo.sock\0";
        assert!(parse_asr_worker_cmdline(cmdline).is_none());
    }
}

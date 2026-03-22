use crate::config::{Config, TranscriptionBackend};
use crate::error::{Result, WhsprError};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

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

fn retained_socket_paths(config: &Config) -> HashSet<PathBuf> {
    let mut retained = HashSet::new();
    match config.transcription.backend {
        TranscriptionBackend::FasterWhisper => {
            if let Some(service) = crate::faster_whisper::prepare_service(&config.transcription) {
                retained.insert(service.socket_path().to_path_buf());
            }
        }
        TranscriptionBackend::Nemo => {
            if let Some(service) = crate::nemo_asr::prepare_service(&config.transcription) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{EnvVarGuard, env_lock, set_env, unique_temp_dir};

    fn with_runtime_dir<T>(f: impl FnOnce(PathBuf) -> T) -> T {
        let _env_lock = env_lock();
        let _guard = EnvVarGuard::capture(&["XDG_RUNTIME_DIR"]);
        let runtime_dir = unique_temp_dir("asr-cleanup-runtime");
        set_env(
            "XDG_RUNTIME_DIR",
            runtime_dir
                .to_str()
                .expect("runtime dir should be valid UTF-8"),
        );
        f(runtime_dir.join("whispers"))
    }

    #[test]
    fn parse_faster_worker_cmdline_extracts_socket_path() {
        with_runtime_dir(|runtime_scope| {
            let socket = runtime_scope.join("asr-faster-123.sock");
            let cmdline = format!(
                "/home/user/.local/share/whispers/faster-whisper/venv/bin/python\0/home/user/.local/share/whispers/faster-whisper/faster_whisper_worker.py\0serve\0--socket-path\0{}\0--model-dir\0/tmp/model\0",
                socket.display()
            );
            let parsed = parse_asr_worker_cmdline(cmdline.as_bytes()).expect("parse worker");
            assert_eq!(parsed.0, "faster_whisper");
            assert_eq!(parsed.1, socket);
        });
    }

    #[test]
    fn parse_nemo_worker_cmdline_extracts_socket_path() {
        with_runtime_dir(|runtime_scope| {
            let socket = runtime_scope.join("asr-nemo-456.sock");
            let cmdline = format!(
                "/home/user/.local/share/whispers/nemo/venv-asr/bin/python\0/home/user/.local/share/whispers/nemo/nemo_asr_worker.py\0serve\0--socket-path\0{}\0--model-ref\0/tmp/model.nemo\0",
                socket.display()
            );
            let parsed = parse_asr_worker_cmdline(cmdline.as_bytes()).expect("parse worker");
            assert_eq!(parsed.0, "nemo");
            assert_eq!(parsed.1, socket);
        });
    }

    #[test]
    fn parse_asr_worker_cmdline_ignores_unrelated_processes() {
        with_runtime_dir(|runtime_scope| {
            let socket = runtime_scope.join("asr-other.sock");
            let cmdline = format!(
                "/usr/bin/python\0/home/user/script.py\0serve\0--socket-path\0{}\0",
                socket.display()
            );
            assert!(parse_asr_worker_cmdline(cmdline.as_bytes()).is_none());
        });
    }

    #[test]
    fn parse_asr_worker_cmdline_ignores_socket_outside_runtime_scope() {
        let cmdline = b"/home/user/.local/share/whispers/nemo/venv-asr/bin/python\0/home/user/.local/share/whispers/nemo/nemo_asr_worker.py\0serve\0--socket-path\0/var/run/asr-nemo.sock\0";
        assert!(parse_asr_worker_cmdline(cmdline).is_none());
    }
}

use std::path::{Path, PathBuf};

use tracing_subscriber::EnvFilter;

use crate::branding;
use crate::error::{Result, WhsprError};
use crate::ui;

pub struct PidLock {
    path: PathBuf,
    _file: std::fs::File,
}

impl Drop for PidLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn pid_file_path() -> PathBuf {
    let runtime_dir = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(runtime_dir).join(branding::MAIN_PID_FILE)
}

fn read_pid_from_lock(path: &Path) -> Option<libc::pid_t> {
    let contents = std::fs::read_to_string(path).ok()?;
    contents.trim().parse().ok()
}

fn process_exists(pid: libc::pid_t) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

fn pid_belongs_to_whspr(pid: libc::pid_t) -> bool {
    if !process_exists(pid) {
        return false;
    }

    let current_exe = std::env::current_exe()
        .ok()
        .and_then(|p| std::fs::canonicalize(p).ok());
    let target_exe = std::fs::canonicalize(format!("/proc/{pid}/exe")).ok();

    if let (Some(current), Some(target)) = (current_exe.as_ref(), target_exe.as_ref()) {
        if current == target {
            return true;
        }
    }

    let current_name = std::env::current_exe()
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_else(|| branding::MAIN_BINARY.into());
    let cmdline = match std::fs::read(format!("/proc/{pid}/cmdline")) {
        Ok(bytes) => bytes,
        Err(_) => return false,
    };
    let Some(first_arg) = cmdline.split(|b| *b == 0).next() else {
        return false;
    };
    if first_arg.is_empty() {
        return false;
    }
    let first_arg = String::from_utf8_lossy(first_arg);
    Path::new(first_arg.as_ref())
        .file_name()
        .map(|name| name.to_string_lossy() == current_name)
        .unwrap_or(false)
}

fn try_acquire_pid_lock(path: &Path) -> std::io::Result<PidLock> {
    use std::io::Write;

    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    writeln!(file, "{}", std::process::id())?;

    Ok(PidLock {
        path: path.to_path_buf(),
        _file: file,
    })
}

fn signal_existing_instance(path: &Path) -> Result<bool> {
    let Some(pid) = read_pid_from_lock(path) else {
        tracing::warn!("stale pid lock at {}, removing", path.display());
        let _ = std::fs::remove_file(path);
        return Ok(false);
    };

    if !pid_belongs_to_whspr(pid) {
        tracing::warn!(
            "pid lock at {} points to non-whspr process ({pid}), removing",
            path.display()
        );
        let _ = std::fs::remove_file(path);
        return Ok(false);
    }

    tracing::info!("sending toggle signal to running instance (pid {pid})");
    let ret = unsafe { libc::kill(pid, libc::SIGUSR1) };
    if ret == 0 {
        return Ok(true);
    }

    let err = std::io::Error::last_os_error();
    tracing::warn!("failed to signal pid {pid}: {err}");
    if err.raw_os_error() == Some(libc::ESRCH) {
        let _ = std::fs::remove_file(path);
        return Ok(false);
    }

    Err(err.into())
}

pub fn acquire_or_signal_lock() -> Result<Option<PidLock>> {
    let path = pid_file_path();

    for _ in 0..2 {
        match try_acquire_pid_lock(&path) {
            Ok(lock) => return Ok(Some(lock)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                if signal_existing_instance(&path)? {
                    return Ok(None);
                }
            }
            Err(e) => return Err(e.into()),
        }
    }

    Err(WhsprError::Config(format!(
        "failed to acquire pid lock at {}",
        path.display()
    )))
}

pub fn init_tracing(verbose: u8) {
    ui::configure_terminal_colors();
    ui::set_verbosity(verbose);
    let filter = match verbose {
        0 => "whispers=warn",
        1 => "whispers=info",
        2 => "whispers=debug",
        _ => "whispers=trace",
    };

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(filter)),
        )
        .compact()
        .init();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_lock_path(suffix: &str) -> PathBuf {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "whispers-test-{suffix}-{}-{now}.pid",
            std::process::id()
        ))
    }

    #[test]
    fn signal_existing_instance_cleans_invalid_pid_file() {
        let path = temp_lock_path("invalid");
        std::fs::write(&path, "not-a-pid").unwrap();
        assert!(!signal_existing_instance(&path).unwrap());
        assert!(!path.exists());
    }

    #[test]
    fn signal_existing_instance_cleans_missing_process_pid_file() {
        let path = temp_lock_path("missing-process");
        std::fs::write(&path, "99999999").unwrap();
        assert!(!signal_existing_instance(&path).unwrap());
        assert!(!path.exists());
    }

    #[test]
    fn try_acquire_pid_lock_uses_create_new_semantics() {
        let path = temp_lock_path("acquire");
        let lock = try_acquire_pid_lock(&path).unwrap();
        assert!(path.exists());

        let err = match try_acquire_pid_lock(&path) {
            Ok(_) => panic!("lock acquisition should fail when file already exists"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);

        drop(lock);
        assert!(!path.exists());

        let lock2 = try_acquire_pid_lock(&path).unwrap();
        drop(lock2);
        assert!(!path.exists());
    }
}

#![allow(dead_code)]

use std::path::PathBuf;

pub const APP_NAME: &str = "whispers";
pub const MAIN_BINARY: &str = APP_NAME;

#[cfg(feature = "osd")]
pub const OSD_BINARY: &str = "whispers-osd";

pub const REWRITE_WORKER_BINARY: &str = "whispers-rewrite-worker";

pub const REWRITE_WORKER_ENV: &str = "WHISPERS_REWRITE_WORKER";

pub const MAIN_PID_FILE: &str = "whispers.pid";
#[cfg(feature = "osd")]
pub const OSD_PID_FILE: &str = "whispers-osd.pid";

pub const LOG_TARGET: &str = "whispers";
pub const UINPUT_KEYBOARD_NAME: &str = "whispers-keyboard";

pub fn resolve_sidecar_executable(candidates: &[&str]) -> PathBuf {
    if let Ok(current_exe) = std::env::current_exe() {
        if let Some(dir) = current_exe.parent() {
            for candidate in candidates {
                let path = dir.join(candidate);
                if path.exists() {
                    return path;
                }
            }
        }
    }

    for candidate in candidates {
        if let Some(path) = executable_in_path(candidate) {
            return path;
        }
    }

    PathBuf::from(candidates[0])
}

fn executable_in_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

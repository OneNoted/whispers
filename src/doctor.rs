use std::env;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use crate::config::Config;
use crate::{paths, setup};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Level {
    Pass,
    Warning,
    Failure,
}

impl Level {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Warning => "WARN",
            Self::Failure => "FAIL",
        }
    }
}

pub struct Check {
    pub level: Level,
    pub name: &'static str,
    pub detail: String,
}

pub struct Report {
    pub checks: Vec<Check>,
}

impl Report {
    pub fn healthy(&self) -> bool {
        !self
            .checks
            .iter()
            .any(|check| check.level == Level::Failure)
    }
}

pub fn run() -> Report {
    let mut checks = Vec::new();
    let config = match Config::load(None) {
        Ok(config) => {
            checks.push(pass("configuration", "version 2 configuration is valid"));
            Some(config)
        }
        Err(error) => {
            checks.push(fail("configuration", error.to_string()));
            None
        }
    };
    match paths::data_dir().and_then(|path| setup::validate_installation(&path)) {
        Ok(()) => checks.push(pass(
            "managed runtimes",
            "sherpa-onnx, Nemotron, and llama.cpp are installed",
        )),
        Err(error) => checks.push(fail("managed runtimes", error.to_string())),
    }
    if let Ok(library) = paths::sherpa_library() {
        // SAFETY: Doctor only asks the dynamic loader to validate this pinned shared object.
        match unsafe { libloading::Library::new(&library) } {
            Ok(handle) => {
                drop(handle);
                checks.push(pass("ASR loader", "sherpa-onnx shared library loads"));
            }
            Err(error) => checks.push(fail(
                "ASR loader",
                format!("{}: {error}", library.display()),
            )),
        }
    }
    command_check(&mut checks, "audio capture", "pw-cat", true);
    if find_command("wtype").is_some() {
        checks.push(pass("text delivery", "wtype is available"));
    } else if find_command("wl-copy").is_some() {
        checks.push(warn(
            "text delivery",
            "wtype is missing; clipboard delivery will be used",
        ));
    } else {
        checks.push(fail("text delivery", "install wtype or wl-clipboard"));
    }
    command_check(&mut checks, "notifications", "notify-send", false);
    command_check(&mut checks, "audio feedback", "pw-play", false);
    command_check(&mut checks, "fallback ASR", "whisper-cli", false);
    if env::var_os("XDG_RUNTIME_DIR").is_some() {
        checks.push(pass("runtime directory", "XDG_RUNTIME_DIR is set"));
    } else {
        checks.push(fail("runtime directory", "XDG_RUNTIME_DIR is not set"));
    }
    if env::var_os("WAYLAND_DISPLAY").is_some() {
        checks.push(pass("desktop session", "Wayland is available"));
    } else {
        checks.push(warn(
            "desktop session",
            "WAYLAND_DISPLAY is not set in this shell",
        ));
    }
    if let Some(config) = config {
        match paths::fallback_model(&config.recognition.fallback_model) {
            Ok(path) if path.is_file() => checks.push(pass(
                "fallback model",
                format!("{} is available", path.display()),
            )),
            Ok(path) => checks.push(warn(
                "fallback model",
                format!(
                    "{} is missing; streaming ASR remains primary",
                    path.display()
                ),
            )),
            Err(error) => checks.push(warn("fallback model", error.to_string())),
        }
        if config.enhancement.enabled {
            match paths::enhancement_model(&config.enhancement.model) {
                Ok(path) if path.is_file() => checks.push(pass(
                    "enhancement model",
                    format!("{} is available", path.display()),
                )),
                Ok(path) => checks.push(warn(
                    "enhancement model",
                    format!(
                        "{} is missing; deterministic cleanup will remain available",
                        path.display()
                    ),
                )),
                Err(error) => checks.push(warn("enhancement model", error.to_string())),
            }
        }
    }
    Report { checks }
}

fn command_check(
    checks: &mut Vec<Check>,
    name: &'static str,
    command: &'static str,
    required: bool,
) {
    if let Some(path) = find_command(command) {
        checks.push(pass(name, format!("{}", path.display())));
    } else if required {
        checks.push(fail(name, format!("{command} is not installed")));
    } else {
        checks.push(warn(
            name,
            format!("{command} is not installed; this feature is disabled"),
        ));
    }
}

fn find_command(command: &str) -> Option<PathBuf> {
    let candidate = Path::new(command);
    if candidate.components().count() > 1 {
        return executable(candidate).then(|| candidate.to_path_buf());
    }
    env::var_os("PATH")?
        .to_string_lossy()
        .split(':')
        .find_map(|dir| {
            let path = Path::new(dir).join(command);
            executable(&path).then_some(path)
        })
}

fn executable(path: &Path) -> bool {
    path.metadata()
        .is_ok_and(|metadata| metadata.is_file() && metadata.permissions().mode() & 0o111 != 0)
}

fn pass(name: &'static str, detail: impl Into<String>) -> Check {
    Check {
        level: Level::Pass,
        name,
        detail: detail.into(),
    }
}

fn warn(name: &'static str, detail: impl Into<String>) -> Check {
    Check {
        level: Level::Warning,
        name,
        detail: detail.into(),
    }
}

fn fail(name: &'static str, detail: impl Into<String>) -> Check {
    Check {
        level: Level::Failure,
        name,
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_health_depends_only_on_failures() {
        assert!(
            Report {
                checks: vec![warn("optional", "missing")]
            }
            .healthy()
        );
        assert!(
            !Report {
                checks: vec![fail("required", "missing")]
            }
            .healthy()
        );
    }
}

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use crate::{Error, Result};

pub const MANAGED_LLAMA_RUNTIME: &str = "runtimes/llama.cpp/b10258-vulkan";

/// Writes `contents` to `path` through an atomic rename, following symbolic
/// links so that dotfile-managed files stay links to their real location.
pub(crate) fn write_atomic(path: &Path, contents: &[u8]) -> Result<()> {
    let target = resolve_symlink(path);
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut temporary = target.as_os_str().to_os_string();
    temporary.push(format!(".{}.tmp", std::process::id()));
    let temporary = PathBuf::from(temporary);
    fs::write(&temporary, contents)?;
    fs::rename(&temporary, &target)?;
    Ok(())
}

fn resolve_symlink(path: &Path) -> PathBuf {
    if !path.is_symlink() {
        return path.to_path_buf();
    }
    if let Ok(canonical) = fs::canonicalize(path) {
        return canonical;
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    match fs::read_link(path) {
        Ok(link) if link.is_absolute() => link,
        Ok(link) => parent.join(link),
        Err(_) => path.to_path_buf(),
    }
}

pub fn config_dir() -> Result<PathBuf> {
    xdg_path("XDG_CONFIG_HOME", ".config").map(|path| path.join("whispers"))
}

pub fn data_dir() -> Result<PathBuf> {
    xdg_path("XDG_DATA_HOME", ".local/share").map(|path| path.join("whispers"))
}

pub fn runtime_dir() -> Result<PathBuf> {
    env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .map(|path| path.join("whispers"))
        .ok_or_else(|| Error::message("XDG_RUNTIME_DIR is not set"))
}

pub fn control_socket() -> Result<PathBuf> {
    runtime_dir().map(|path| path.join("control.sock"))
}

pub fn vocabulary() -> Result<PathBuf> {
    data_dir().map(|path| path.join("vocabulary.toml"))
}

pub fn osd_config() -> Result<PathBuf> {
    data_dir().map(|path| path.join("ui/shell.qml"))
}

pub fn recognition_model_dir(model: &str) -> Result<PathBuf> {
    if let Some(path) = env::var_os("WHISPERS_MODEL_DIR") {
        return Ok(PathBuf::from(path));
    }
    data_dir().map(|path| path.join("models").join(model))
}

pub fn sherpa_library() -> Result<PathBuf> {
    if let Some(path) = env::var_os("WHISPERS_SHERPA_LIBRARY") {
        return Ok(PathBuf::from(path));
    }
    data_dir().map(|path| {
        path.join("runtimes/sherpa-onnx/current/lib")
            .join("libsherpa-onnx-c-api.so")
    })
}

pub fn llama_server() -> Result<PathBuf> {
    if let Some(path) = env::var_os("WHISPERS_LLAMA_SERVER") {
        return Ok(PathBuf::from(path));
    }
    data_dir().map(|path| path.join(MANAGED_LLAMA_RUNTIME).join("llama-server"))
}

pub fn fallback_model(model: &str) -> Result<PathBuf> {
    let candidate = PathBuf::from(model);
    if candidate.is_absolute() {
        return Ok(candidate);
    }
    data_dir().map(|path| path.join(model))
}

pub fn enhancement_model(model: &str) -> Result<PathBuf> {
    let candidate = PathBuf::from(model);
    if candidate.is_absolute() {
        return Ok(candidate);
    }
    data_dir().map(|path| path.join("rewrite").join(model))
}

fn xdg_path(variable: &str, fallback: &str) -> Result<PathBuf> {
    if let Some(path) = env::var_os(variable) {
        return Ok(PathBuf::from(path));
    }
    env::var_os("HOME")
        .map(PathBuf::from)
        .map(|path| path.join(fallback))
        .ok_or_else(|| Error::message(format!("neither {variable} nor HOME is set")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_writes_follow_symlinks() {
        let directory =
            env::temp_dir().join(format!("whispers-write-atomic-{}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        assert!(fs::create_dir_all(&directory).is_ok());
        let target = directory.join("target.toml");
        let link = directory.join("link.toml");
        let dangling = directory.join("dangling.toml");
        assert!(fs::write(&target, "old").is_ok());
        assert!(std::os::unix::fs::symlink(&target, &link).is_ok());
        assert!(std::os::unix::fs::symlink("missing.toml", &dangling).is_ok());

        assert!(write_atomic(&link, b"new").is_ok());
        assert!(write_atomic(&dangling, b"created").is_ok());

        assert!(link.is_symlink() && dangling.is_symlink());
        assert_eq!(fs::read_to_string(&target).ok().as_deref(), Some("new"));
        assert_eq!(
            fs::read_to_string(directory.join("missing.toml"))
                .ok()
                .as_deref(),
            Some("created")
        );
        let _ = fs::remove_dir_all(&directory);
    }
}

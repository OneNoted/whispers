use std::env;
use std::path::PathBuf;

use crate::{Error, Result};

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

fn xdg_path(variable: &str, fallback: &str) -> Result<PathBuf> {
    if let Some(path) = env::var_os(variable) {
        return Ok(PathBuf::from(path));
    }
    env::var_os("HOME")
        .map(PathBuf::from)
        .map(|path| path.join(fallback))
        .ok_or_else(|| Error::message(format!("neither {variable} nor HOME is set")))
}

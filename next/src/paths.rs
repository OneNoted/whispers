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

pub fn vocabulary() -> Result<PathBuf> {
    data_dir().map(|path| path.join("vocabulary.toml"))
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
    data_dir().map(|path| path.join("runtimes/llama.cpp/current/llama-server"))
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

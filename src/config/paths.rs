use std::path::{Path, PathBuf};

use super::Config;

pub fn default_config_path() -> PathBuf {
    xdg_dir("config").join("whispers").join("config.toml")
}

pub fn resolve_config_path(path: Option<&Path>) -> PathBuf {
    match path {
        Some(path) => path.to_path_buf(),
        None => default_config_path(),
    }
}

pub fn data_dir() -> PathBuf {
    xdg_dir("data").join("whispers")
}

fn xdg_dir(kind: &str) -> PathBuf {
    match kind {
        "config" => {
            if let Ok(dir) = std::env::var("XDG_CONFIG_HOME") {
                PathBuf::from(dir)
            } else if let Ok(home) = std::env::var("HOME") {
                PathBuf::from(home).join(".config")
            } else {
                tracing::warn!("neither XDG_CONFIG_HOME nor HOME is set, falling back to /tmp");
                PathBuf::from("/tmp")
            }
        }
        "data" => {
            if let Ok(dir) = std::env::var("XDG_DATA_HOME") {
                PathBuf::from(dir)
            } else if let Ok(home) = std::env::var("HOME") {
                PathBuf::from(home).join(".local").join("share")
            } else {
                tracing::warn!("neither XDG_DATA_HOME nor HOME is set, falling back to /tmp");
                PathBuf::from("/tmp")
            }
        }
        _ => {
            tracing::warn!("unknown XDG directory kind '{kind}', falling back to /tmp");
            PathBuf::from("/tmp")
        }
    }
}

pub fn expand_tilde(path: &str) -> String {
    match path.strip_prefix("~/") {
        Some(rest) => {
            if let Ok(home) = std::env::var("HOME") {
                return format!("{home}/{rest}");
            }
            tracing::warn!("HOME is not set, cannot expand tilde in path: {path}");
        }
        None if path == "~" => {
            if let Ok(home) = std::env::var("HOME") {
                return home;
            }
            tracing::warn!("HOME is not set, cannot expand tilde in path: {path}");
        }
        _ => {}
    }
    path.to_string()
}

impl Config {
    pub fn resolved_model_path(&self) -> PathBuf {
        PathBuf::from(expand_tilde(&self.transcription.model_path))
    }

    pub fn resolved_rewrite_model_path(&self) -> Option<PathBuf> {
        (!self.rewrite.model_path.trim().is_empty())
            .then(|| PathBuf::from(expand_tilde(&self.rewrite.model_path)))
    }

    pub fn resolved_rewrite_instructions_path(&self) -> Option<PathBuf> {
        (!self.rewrite.instructions_path.trim().is_empty())
            .then(|| PathBuf::from(expand_tilde(&self.rewrite.instructions_path)))
    }

    pub fn resolved_dictionary_path(&self) -> PathBuf {
        PathBuf::from(expand_tilde(&self.personalization.dictionary_path))
    }

    pub fn resolved_snippets_path(&self) -> PathBuf {
        PathBuf::from(expand_tilde(&self.personalization.snippets_path))
    }

    pub fn resolved_agentic_policy_path(&self) -> PathBuf {
        PathBuf::from(expand_tilde(&self.agentic_rewrite.policy_path))
    }

    pub fn resolved_agentic_glossary_path(&self) -> PathBuf {
        PathBuf::from(expand_tilde(&self.agentic_rewrite.glossary_path))
    }
}

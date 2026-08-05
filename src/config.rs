use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{Result, paths};

pub const CONFIG_VERSION: u16 = 2;
pub const DEFAULT_FINAL_MODEL: &str = "nemo-parakeet-tdt-0.6b-v3-int8";
pub const LEGACY_ENGLISH_FINAL_MODEL: &str = "nemo-parakeet-unified-en-0.6b-int8";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct Config {
    pub version: u16,
    pub language: String,
    pub audio: AudioConfig,
    pub recognition: RecognitionConfig,
    pub enhancement: EnhancementConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct AudioConfig {
    pub device: String,
    pub feedback: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct RecognitionConfig {
    pub runtime: String,
    pub model: String,
    pub final_model: String,
    pub fallback_model: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct EnhancementConfig {
    pub enabled: bool,
    pub command: String,
    pub model: String,
    pub timeout_ms: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            version: CONFIG_VERSION,
            language: "auto".into(),
            audio: AudioConfig::default(),
            recognition: RecognitionConfig::default(),
            enhancement: EnhancementConfig::default(),
        }
    }
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            device: String::new(),
            feedback: true,
        }
    }
}

impl Default for RecognitionConfig {
    fn default() -> Self {
        Self {
            runtime: "sherpa-onnx".into(),
            model: "nemotron-3.5-asr-streaming-0.6b-560ms-int8".into(),
            final_model: DEFAULT_FINAL_MODEL.into(),
            fallback_model: "ggml-large-v3-turbo.bin".into(),
        }
    }
}

impl Default for EnhancementConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            command: "managed".into(),
            model: "Qwen_Qwen3.5-4B-Q4_K_M.gguf".into(),
            timeout_ms: 10_000,
        }
    }
}

impl Config {
    pub fn path() -> Result<PathBuf> {
        paths::config_dir().map(|path| path.join("config.toml"))
    }

    pub fn load(path: Option<&Path>) -> Result<Self> {
        let path = path.map_or_else(Self::path, |path| Ok(path.to_path_buf()))?;
        match fs::read_to_string(path) {
            Ok(contents) => {
                let value: toml::Value = toml::from_str(&contents)?;
                let version = value
                    .get("version")
                    .and_then(toml::Value::as_integer)
                    .unwrap_or_default();
                if version != i64::from(CONFIG_VERSION) {
                    return Err(crate::Error::message(format!(
                        "configuration version {version} is not supported; run `whispers setup`"
                    )));
                }
                let config: Self = toml::from_str(&contents)?;
                Ok(config)
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(error) => Err(error.into()),
        }
    }

    pub fn save(&self) -> Result<()> {
        self.save_to(&Self::path()?)
    }

    pub fn save_to(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut temporary = path.as_os_str().to_os_string();
        temporary.push(format!(".{}.tmp", std::process::id()));
        let temporary = PathBuf::from(temporary);
        fs::write(&temporary, toml::to_string_pretty(self)?)?;
        fs::rename(temporary, path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn existing_unversioned_configuration_requires_migration() {
        let path = std::env::temp_dir().join(format!(
            "whispers-config-version-test-{}.toml",
            std::process::id()
        ));
        let _ = fs::remove_file(&path);
        assert!(fs::write(&path, "[audio]\ndevice = \"legacy\"\n").is_ok());
        let result = Config::load(Some(&path));
        assert!(result.is_err());
        assert!(result.is_err_and(|error| error.to_string().contains("whispers setup")));
        let _ = fs::remove_file(path);
    }
}

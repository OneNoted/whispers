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
    pub delivery: DeliveryConfig,
    pub recognition: RecognitionConfig,
    pub enhancement: EnhancementConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct AudioConfig {
    pub device: String,
    pub feedback: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct DeliveryConfig {
    pub typing_delay_ms: u64,
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
            delivery: DeliveryConfig::default(),
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
        paths::write_atomic(path, toml::to_string_pretty(self)?.as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saving_through_a_symlink_updates_the_link_target() {
        let directory =
            std::env::temp_dir().join(format!("whispers-config-symlink-{}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        assert!(fs::create_dir_all(&directory).is_ok());
        let target = directory.join("dotfiles-config.toml");
        let link = directory.join("config.toml");
        assert!(fs::write(&target, "version = 1\n").is_ok());
        assert!(std::os::unix::fs::symlink(&target, &link).is_ok());

        assert!(Config::default().save_to(&link).is_ok());

        assert!(link.is_symlink());
        assert!(fs::read_to_string(&target).is_ok_and(|text| text.contains("version = 2")));
        let _ = fs::remove_dir_all(&directory);
    }

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

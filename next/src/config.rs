use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{Result, paths};

pub const CONFIG_VERSION: u16 = 2;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct Config {
    pub version: u16,
    pub language: String,
    pub audio: AudioConfig,
    pub recognition: RecognitionConfig,
    pub enhancement: EnhancementConfig,
    pub delivery: DeliveryConfig,
    pub privacy: PrivacyConfig,
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
    pub fallback_model: String,
    pub idle_timeout_seconds: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct EnhancementConfig {
    pub enabled: bool,
    pub command: String,
    pub model: String,
    pub timeout_ms: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct DeliveryConfig {
    pub strategy: String,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct PrivacyConfig {
    pub retain_audio: bool,
    pub log_transcripts: bool,
    pub allow_network: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            version: CONFIG_VERSION,
            language: "auto".into(),
            audio: AudioConfig::default(),
            recognition: RecognitionConfig::default(),
            enhancement: EnhancementConfig::default(),
            delivery: DeliveryConfig::default(),
            privacy: PrivacyConfig::default(),
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
            fallback_model: "ggml-large-v3-turbo.bin".into(),
            idle_timeout_seconds: 600,
        }
    }
}

impl Default for EnhancementConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            command: "llama-server".into(),
            model: "Qwen_Qwen3.5-4B-Q4_K_M.gguf".into(),
            timeout_ms: 1_500,
        }
    }
}

impl Default for DeliveryConfig {
    fn default() -> Self {
        Self {
            strategy: "auto".into(),
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
                let config: Self = toml::from_str(&contents)?;
                if config.version != CONFIG_VERSION {
                    return Err(crate::Error::message(format!(
                        "configuration version {} is not supported; run `whispers setup`",
                        config.version
                    )));
                }
                Ok(config)
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(error) => Err(error.into()),
        }
    }
}

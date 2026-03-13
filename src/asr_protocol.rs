use serde::{Deserialize, Serialize};

use crate::transcribe::Transcript;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AsrRequest {
    Transcribe {
        audio_f32_b64: String,
        sample_rate: u32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AsrResponse {
    Transcript { transcript: Transcript },
    Error { message: String },
}

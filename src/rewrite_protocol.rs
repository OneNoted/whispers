use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RewriteTranscript {
    pub raw_text: String,
    pub correction_aware_text: String,
    pub detected_language: Option<String>,
    pub segments: Vec<RewriteTranscriptSegment>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RewriteTranscriptSegment {
    pub text: String,
    pub start_ms: u32,
    pub end_ms: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkerRequest {
    Rewrite { transcript: RewriteTranscript },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkerResponse {
    Ready,
    Result { text: String },
    Error { message: String },
}

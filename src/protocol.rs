use std::io::{Read, Write};

use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::{Error, Result};

pub const ENGINE_PROTOCOL_VERSION: u16 = 1;
pub const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum FrameKind {
    Control = 1,
    Audio = 2,
}

impl TryFrom<u8> for FrameKind {
    type Error = Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            1 => Ok(Self::Control),
            2 => Ok(Self::Audio),
            _ => Err(Error::message(format!(
                "unknown engine frame kind: {value}"
            ))),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EngineRequest {
    Hello {
        protocol_version: u16,
    },
    Start {
        session_id: u64,
        sample_rate: u32,
        language: String,
        vocabulary: Vec<String>,
    },
    Finish,
    Cancel,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EngineResponse {
    Hello {
        protocol_version: u16,
        engine: String,
        capabilities: Vec<String>,
    },
    Partial {
        revision: u64,
        text: String,
    },
    Final {
        text: String,
        language: Option<String>,
        confidence: Option<f32>,
    },
    Error {
        code: String,
        message: String,
        retryable: bool,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "command", rename_all = "snake_case")]
pub enum ControlRequest {
    Toggle,
    Cancel,
    Status,
    Shutdown,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ControlResponse {
    pub ok: bool,
    pub state: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

pub fn write_json_frame<W: Write, T: Serialize>(writer: &mut W, value: &T) -> Result<()> {
    let payload = serde_json::to_vec(value)?;
    write_frame(writer, FrameKind::Control, &payload)
}

pub fn write_frame<W: Write>(writer: &mut W, kind: FrameKind, payload: &[u8]) -> Result<()> {
    if payload.len() > MAX_FRAME_BYTES {
        return Err(Error::FrameTooLarge(payload.len()));
    }
    writer.write_all(&[kind as u8])?;
    let length = u32::try_from(payload.len()).map_err(|_| Error::FrameTooLarge(payload.len()))?;
    writer.write_all(&length.to_be_bytes())?;
    writer.write_all(payload)?;
    writer.flush()?;
    Ok(())
}

pub fn read_frame<R: Read>(reader: &mut R) -> Result<(FrameKind, Vec<u8>)> {
    let mut header = [0_u8; 5];
    reader.read_exact(&mut header)?;
    let kind = FrameKind::try_from(header[0])?;
    let length = u32::from_be_bytes([header[1], header[2], header[3], header[4]]) as usize;
    if length > MAX_FRAME_BYTES {
        return Err(Error::FrameTooLarge(length));
    }
    let mut payload = vec![0; length];
    reader.read_exact(&mut payload)?;
    Ok((kind, payload))
}

pub fn read_json_frame<R: Read, T: DeserializeOwned>(reader: &mut R) -> Result<T> {
    let (kind, payload) = read_frame(reader)?;
    if kind != FrameKind::Control {
        return Err(Error::message("expected a control frame"));
    }
    Ok(serde_json::from_slice(&payload)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_control_frames() {
        let request = EngineRequest::Hello {
            protocol_version: ENGINE_PROTOCOL_VERSION,
        };
        let mut bytes = Vec::new();
        assert!(write_json_frame(&mut bytes, &request).is_ok());
        let decoded: Result<EngineRequest> = read_json_frame(&mut bytes.as_slice());
        assert_eq!(decoded.ok(), Some(request));
    }

    #[test]
    fn rejects_oversized_frames_before_allocation() {
        let mut bytes = vec![FrameKind::Audio as u8];
        let oversized = u32::try_from(MAX_FRAME_BYTES).unwrap_or(u32::MAX) + 1;
        bytes.extend_from_slice(&oversized.to_be_bytes());
        assert!(matches!(
            read_frame(&mut bytes.as_slice()),
            Err(Error::FrameTooLarge(_))
        ));
    }
}

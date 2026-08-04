use std::io;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Message(String),
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid configuration: {0}")]
    Config(#[from] toml::de::Error),
    #[error("could not encode configuration: {0}")]
    ConfigEncode(#[from] toml::ser::Error),
    #[error("invalid command line: {0}")]
    Cli(#[from] lexopt::Error),
    #[error("protocol frame is too large: {0} bytes")]
    FrameTooLarge(usize),
    #[error("protocol version {actual} is incompatible; expected {expected}")]
    ProtocolVersion { expected: u16, actual: u16 },
}

impl Error {
    pub fn message(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

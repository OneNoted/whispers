pub mod audio;
pub mod cli;
pub mod config;
pub mod daemon;
pub mod desktop;
pub mod doctor;
pub mod engine;
pub mod enhance;
pub mod error;
pub mod normalize;
pub mod paths;
pub mod pipeline;
pub mod protocol;
pub mod service;
pub mod setup;
pub mod vocabulary;

pub use error::{Error, Result};

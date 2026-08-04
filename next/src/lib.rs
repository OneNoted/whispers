pub mod audio;
pub mod cli;
pub mod config;
pub mod daemon;
pub mod desktop;
pub mod engine;
pub mod error;
pub mod normalize;
pub mod paths;
pub mod pipeline;
pub mod protocol;
pub mod service;

pub use error::{Error, Result};

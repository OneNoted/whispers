use std::io::BufReader;
use std::path::Path;

use rodio::Decoder;
use rodio::source::UniformSourceIterator;

use crate::error::{Result, WhsprError};

pub const TARGET_SAMPLE_RATE: u32 = 16000;

/// Decode an audio file to mono 16 kHz f32 samples suitable for Whisper.
pub fn decode_audio_file(path: &Path) -> Result<Vec<f32>> {
    let file = std::fs::File::open(path)
        .map_err(|e| WhsprError::Audio(format!("failed to open {}: {e}", path.display())))?;

    let reader = BufReader::new(file);

    let decoder = Decoder::new(reader)
        .map_err(|e| WhsprError::Audio(format!("failed to decode {}: {e}", path.display())))?;

    let resampled = UniformSourceIterator::<Decoder<BufReader<std::fs::File>>>::new(
        decoder,
        1,
        TARGET_SAMPLE_RATE,
    );

    let samples: Vec<f32> = resampled.collect();

    if samples.is_empty() {
        return Err(WhsprError::Audio(format!(
            "no audio samples decoded from {}",
            path.display()
        )));
    }

    tracing::info!(
        "decoded {}: {:.1}s, {} samples",
        path.display(),
        samples.len() as f64 / TARGET_SAMPLE_RATE as f64,
        samples.len()
    );

    Ok(samples)
}

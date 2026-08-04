use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};

use crate::{Error, Result};

pub const SAMPLE_RATE: u32 = 16_000;

pub struct PipewireRecording {
    child: Child,
    samples: Arc<Mutex<Vec<f32>>>,
    reader: Option<JoinHandle<Result<()>>>,
}

impl PipewireRecording {
    pub fn start(device: &str) -> Result<(Self, mpsc::Receiver<Vec<f32>>)> {
        let mut command = Command::new("pw-cat");
        command.args([
            "--record",
            "--raw",
            "--format",
            "f32",
            "--rate",
            "16000",
            "--channels",
            "1",
        ]);
        if !device.is_empty() {
            command.args(["--target", device]);
        }
        command
            .arg("-")
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        let mut child = command.spawn().map_err(|error| {
            Error::message(format!("could not start PipeWire capture: {error}"))
        })?;
        let mut stdout = child
            .stdout
            .take()
            .ok_or_else(|| Error::message("pw-cat did not provide an audio stream"))?;
        let samples = Arc::new(Mutex::new(Vec::new()));
        let stored = Arc::clone(&samples);
        let (sender, receiver) = mpsc::channel();
        let reader = thread::spawn(move || {
            let mut sender = Some(sender);
            let mut pending = Vec::new();
            let mut buffer = [0_u8; 16 * 1024];
            loop {
                let read = stdout.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                pending.extend_from_slice(&buffer[..read]);
                let complete = pending.len() - (pending.len() % 4);
                if complete == 0 {
                    continue;
                }
                let chunk = pending[..complete]
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                    .collect::<Vec<_>>();
                pending.drain(..complete);
                stored
                    .lock()
                    .map_err(|_| Error::message("audio buffer lock is poisoned"))?
                    .extend_from_slice(&chunk);
                if sender
                    .as_ref()
                    .is_some_and(|sender| sender.send(chunk).is_err())
                {
                    sender = None;
                }
            }
            Ok(())
        });
        Ok((
            Self {
                child,
                samples,
                reader: Some(reader),
            },
            receiver,
        ))
    }

    pub fn stop(mut self) -> Result<Vec<f32>> {
        self.stop_child()?;
        if let Some(reader) = self.reader.take() {
            reader
                .join()
                .map_err(|_| Error::message("audio reader thread panicked"))??;
        }
        Arc::try_unwrap(self.samples)
            .map_err(|_| Error::message("audio buffer is still in use"))?
            .into_inner()
            .map_err(|_| Error::message("audio buffer lock is poisoned"))
    }

    pub fn cancel(mut self) -> Result<()> {
        self.stop_child()?;
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
        Ok(())
    }

    fn stop_child(&mut self) -> Result<()> {
        if self.child.try_wait()?.is_some() {
            return Ok(());
        }
        self.child.kill()?;
        let _ = self.child.wait();
        Ok(())
    }
}

pub fn has_speech(samples: &[f32]) -> bool {
    if samples.len() < (SAMPLE_RATE as usize / 5) {
        return false;
    }
    let count = u32::try_from(samples.len()).unwrap_or(u32::MAX);
    let energy = samples
        .iter()
        .map(|sample| f64::from(*sample) * f64::from(*sample))
        .sum::<f64>()
        / f64::from(count);
    energy.sqrt() >= 0.003
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_is_not_speech() {
        assert!(!has_speech(&vec![0.0; SAMPLE_RATE as usize]));
    }

    #[test]
    fn rejects_clicks_that_are_too_short() {
        assert!(!has_speech(&[1.0; 32]));
    }

    #[test]
    fn sustained_audio_is_speech() {
        assert!(has_speech(&vec![0.02; SAMPLE_RATE as usize]));
    }
}

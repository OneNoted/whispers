use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::{Error, Result};

pub const SAMPLE_RATE: u32 = 16_000;
const MAX_RECORDING_SAMPLES: usize = SAMPLE_RATE as usize * 60 * 10;
const SPEECH_FRAME_SAMPLES: usize = SAMPLE_RATE as usize / 50;
const MIN_SPEECH_FRAMES: usize = 5;
const SPEECH_RMS_THRESHOLD: f64 = 0.003;
const DISCARD_BOUNDARY_TIMEOUT: Duration = Duration::from_secs(5);

pub struct PipewireRecording {
    child: Child,
    samples: Arc<Mutex<Vec<f32>>>,
    reader: Option<JoinHandle<Result<()>>>,
    reader_control: mpsc::Sender<ReaderCommand>,
}

enum ReaderCommand {
    Discard {
        acknowledge: mpsc::SyncSender<()>,
        resume: mpsc::Receiver<()>,
    },
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
        let Some(mut stdout) = child.stdout.take() else {
            let _ = child.kill();
            let _ = child.wait();
            return Err(Error::message("pw-cat did not provide an audio stream"));
        };
        let samples = Arc::new(Mutex::new(Vec::new()));
        let stored = Arc::clone(&samples);
        let (sender, receiver) = mpsc::channel();
        let (reader_control, control_receiver) = mpsc::channel();
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
                if let Ok(ReaderCommand::Discard {
                    acknowledge,
                    resume,
                }) = control_receiver.try_recv()
                {
                    pending.clear();
                    stored
                        .lock()
                        .map_err(|_| Error::message("audio buffer lock is poisoned"))?
                        .clear();
                    let _ = acknowledge.send(());
                    let _ = resume.recv_timeout(DISCARD_BOUNDARY_TIMEOUT);
                    continue;
                }
                let mut stored = stored
                    .lock()
                    .map_err(|_| Error::message("audio buffer lock is poisoned"))?;
                if stored.len().saturating_add(chunk.len()) > MAX_RECORDING_SAMPLES {
                    return Err(Error::message("dictation exceeded the 10 minute limit"));
                }
                stored.extend_from_slice(&chunk);
                drop(stored);
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
                reader_control,
            },
            receiver,
        ))
    }

    pub fn discard_buffered(&self, receiver: &mpsc::Receiver<Vec<f32>>) -> Result<()> {
        let (acknowledge, acknowledged) = mpsc::sync_channel(0);
        let (resume, resumed) = mpsc::sync_channel(0);
        self.reader_control
            .send(ReaderCommand::Discard {
                acknowledge,
                resume: resumed,
            })
            .map_err(|_| Error::message("audio reader stopped before the discard boundary"))?;
        acknowledged
            .recv_timeout(DISCARD_BOUNDARY_TIMEOUT)
            .map_err(|_| Error::message("audio reader did not acknowledge the discard boundary"))?;
        while receiver.try_recv().is_ok() {}
        resume
            .send(())
            .map_err(|_| Error::message("audio reader stopped at the discard boundary"))
    }

    pub fn stop(mut self) -> Result<Vec<f32>> {
        self.stop_child()?;
        if let Some(reader) = self.reader.take() {
            reader
                .join()
                .map_err(|_| Error::message("audio reader thread panicked"))??;
        }
        self.samples
            .lock()
            .map_err(|_| Error::message("audio buffer lock is poisoned"))
            .map(|samples| samples.clone())
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

impl Drop for PipewireRecording {
    fn drop(&mut self) {
        let _ = self.stop_child();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

pub fn has_speech(samples: &[f32]) -> bool {
    if samples.len() < (SAMPLE_RATE as usize / 5) {
        return false;
    }
    let frame_sample_count = f64::from(SAMPLE_RATE / 50);
    samples
        .chunks_exact(SPEECH_FRAME_SAMPLES)
        .map(|frame| {
            let energy = frame
                .iter()
                .map(|sample| f64::from(*sample) * f64::from(*sample))
                .sum::<f64>()
                / frame_sample_count;
            energy.sqrt() >= SPEECH_RMS_THRESHOLD
        })
        .scan(0, |consecutive, voiced| {
            *consecutive = if voiced { *consecutive + 1 } else { 0 };
            Some(*consecutive)
        })
        .any(|consecutive| consecutive >= MIN_SPEECH_FRAMES)
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

    #[test]
    fn speech_survives_a_long_trailing_pause() {
        let mut samples = vec![0.02; SAMPLE_RATE as usize];
        samples.extend(vec![0.0; SAMPLE_RATE as usize * 60]);
        assert!(has_speech(&samples));
    }

    #[test]
    fn isolated_click_in_a_long_recording_is_not_speech() {
        let mut samples = vec![0.0; SAMPLE_RATE as usize];
        samples[1_000..1_032].fill(1.0);
        assert!(!has_speech(&samples));
    }
}

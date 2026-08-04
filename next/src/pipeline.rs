use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use crate::audio::{self, PipewireRecording};
use crate::config::Config;
use crate::desktop::{self, Focus};
use crate::engine::EngineClient;
use crate::normalize::normalize_spoken_text;
use crate::service::{ActiveDictation, CompletedDictation, Pipeline};
use crate::{Error, Result, paths};

pub struct LocalPipeline {
    config: Config,
    engine: Arc<Mutex<Option<EngineClient>>>,
    next_session: AtomicU64,
}

struct LocalDictation {
    recording: Option<PipewireRecording>,
    feeder: Option<JoinHandle<()>>,
    engine: Arc<Mutex<Option<EngineClient>>>,
    target: Option<Focus>,
    config: Config,
    streaming: bool,
}

impl LocalPipeline {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            engine: Arc::new(Mutex::new(None)),
            next_session: AtomicU64::new(1),
        }
    }

    fn start_engine(&self) -> bool {
        let Ok(mut engine) = self.engine.lock() else {
            return false;
        };
        if engine.as_mut().is_none_or(|client| !client.is_healthy()) {
            *engine = EngineClient::spawn(&self.config.recognition.runtime).ok();
        }
        let Some(client) = engine.as_mut() else {
            return false;
        };
        client
            .start(
                self.next_session.fetch_add(1, Ordering::Relaxed),
                &self.config.language,
                Vec::new(),
            )
            .is_ok()
    }
}

impl Pipeline for LocalPipeline {
    fn start(&self) -> Result<Box<dyn ActiveDictation>> {
        if self.config.audio.feedback {
            desktop::play_cue("start");
        }
        let target = desktop::capture_focus();
        let streaming = self.start_engine();
        let (recording, receiver) = PipewireRecording::start(&self.config.audio.device)?;
        let engine = Arc::clone(&self.engine);
        let feeder = thread::spawn(move || {
            while let Ok(samples) = receiver.recv() {
                let Ok(mut engine) = engine.lock() else {
                    break;
                };
                let Some(client) = engine.as_mut() else {
                    break;
                };
                if client.push_audio(&samples).is_err() {
                    break;
                }
            }
        });
        desktop::show_listening();
        Ok(Box::new(LocalDictation {
            recording: Some(recording),
            feeder: Some(feeder),
            engine: Arc::clone(&self.engine),
            target,
            config: self.config.clone(),
            streaming,
        }))
    }

    fn deliver(&self, dictation: &CompletedDictation) -> Result<()> {
        desktop::deliver(&dictation.text, dictation.target.as_deref())
    }
}

impl ActiveDictation for LocalDictation {
    fn finish(mut self: Box<Self>, cancelled: &AtomicBool) -> Result<CompletedDictation> {
        let recording = self
            .recording
            .take()
            .ok_or_else(|| Error::message("recording was already consumed"))?;
        let samples = recording.stop()?;
        if let Some(feeder) = self.feeder.take() {
            let _ = feeder.join();
        }
        if self.config.audio.feedback {
            desktop::play_cue("stop");
        }
        if cancelled.load(Ordering::Acquire) || !audio::has_speech(&samples) {
            return Ok(CompletedDictation {
                text: String::new(),
                target: None,
            });
        }
        let transcript = if self.streaming {
            self.engine
                .lock()
                .map_err(|_| Error::message("engine lock is poisoned"))?
                .as_mut()
                .ok_or_else(|| Error::message("streaming engine disappeared"))?
                .finish()
                .or_else(|_| transcribe_with_whisper(&self.config, &samples))?
        } else {
            transcribe_with_whisper(&self.config, &samples)?
        };
        Ok(CompletedDictation {
            text: normalize_spoken_text(&transcript),
            target: self.target.map(|focus| focus.identity),
        })
    }

    fn cancel(mut self: Box<Self>) -> Result<()> {
        let recording = self
            .recording
            .take()
            .ok_or_else(|| Error::message("recording was already consumed"))?;
        recording.cancel()?;
        if let Some(feeder) = self.feeder.take() {
            let _ = feeder.join();
        }
        if self.streaming {
            if let Ok(mut engine) = self.engine.lock() {
                if let Some(client) = engine.as_mut() {
                    let _ = client.cancel();
                }
            }
        }
        Ok(())
    }
}

fn transcribe_with_whisper(config: &Config, samples: &[f32]) -> Result<String> {
    let runtime_dir = paths::runtime_dir()?;
    fs::create_dir_all(&runtime_dir)?;
    let stem = format!("fallback-{}", std::process::id());
    let wav_path = runtime_dir.join(format!("{stem}.wav"));
    write_wav(&wav_path, samples)?;
    let model = paths::fallback_model(&config.recognition.fallback_model)?;
    let output = Command::new("whisper-cli")
        .args([
            "-m",
            &model.to_string_lossy(),
            "-f",
            &wav_path.to_string_lossy(),
        ])
        .args([
            "--no-timestamps",
            "--no-prints",
            "--language",
            &config.language,
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output();
    let _ = fs::remove_file(&wav_path);
    let output =
        output.map_err(|error| Error::message(format!("could not run whisper-cli: {error}")))?;
    if !output.status.success() {
        return Err(Error::message("whisper-cli transcription failed"));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

#[allow(clippy::cast_possible_truncation)]
fn write_wav(path: &Path, samples: &[f32]) -> Result<()> {
    let data_length = u32::try_from(samples.len().saturating_mul(2))
        .map_err(|_| Error::message("recording is too large for WAV"))?;
    let mut file = fs::File::create(path)?;
    file.write_all(b"RIFF")?;
    file.write_all(&(36_u32.saturating_add(data_length)).to_le_bytes())?;
    file.write_all(b"WAVEfmt ")?;
    file.write_all(&16_u32.to_le_bytes())?;
    file.write_all(&1_u16.to_le_bytes())?;
    file.write_all(&1_u16.to_le_bytes())?;
    file.write_all(&16_000_u32.to_le_bytes())?;
    file.write_all(&32_000_u32.to_le_bytes())?;
    file.write_all(&2_u16.to_le_bytes())?;
    file.write_all(&16_u16.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&data_length.to_le_bytes())?;
    for sample in samples {
        let scaled = (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16;
        file.write_all(&scaled.to_le_bytes())?;
    }
    Ok(())
}

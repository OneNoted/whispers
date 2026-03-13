use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};

use crate::config::AudioConfig;
use crate::error::{Result, WhsprError};

const PREALLOC_SECONDS: usize = 120;

pub struct AudioRecorder {
    config: AudioConfig,
    buffer: Arc<Mutex<Vec<f32>>>,
    stream: Option<cpal::Stream>,
}

impl AudioRecorder {
    pub fn new(config: &AudioConfig) -> Self {
        Self {
            config: config.clone(),
            buffer: Arc::new(Mutex::new(Vec::new())),
            stream: None,
        }
    }

    pub fn start(&mut self) -> Result<()> {
        let host = cpal::default_host();

        let device = if self.config.device.is_empty() {
            host.default_input_device()
                .ok_or_else(|| WhsprError::Audio("no default input device found".into()))?
        } else {
            host.input_devices()
                .map_err(|e| WhsprError::Audio(format!("failed to enumerate input devices: {e}")))?
                .find(|d| {
                    d.description()
                        .map(|desc| desc.name().contains(&self.config.device))
                        .unwrap_or(false)
                })
                .ok_or_else(|| {
                    WhsprError::Audio(format!("input device '{}' not found", self.config.device))
                })?
        };

        let device_name = device
            .description()
            .map(|d| d.name().to_string())
            .unwrap_or_else(|_| "unknown".into());
        tracing::info!("using input device: {device_name}");

        let (stream_config, sample_format) = choose_input_config(&device, self.config.sample_rate)?;
        if stream_config.channels != 1 {
            tracing::warn!(
                "device input has {} channels; downmixing to mono",
                stream_config.channels
            );
        }
        tracing::info!(
            "audio stream config: {} Hz, {} channels, {:?}",
            stream_config.sample_rate,
            stream_config.channels,
            sample_format
        );

        let buffer = Arc::clone(&self.buffer);
        {
            let mut guard = buffer
                .lock()
                .map_err(|_| WhsprError::Audio("audio buffer lock poisoned".into()))?;
            guard.clear();
            let prealloc_samples =
                (self.config.sample_rate as usize).saturating_mul(PREALLOC_SECONDS);
            let capacity = guard.capacity();
            if capacity < prealloc_samples {
                guard.reserve_exact(prealloc_samples - capacity);
            }
        }
        let channels = stream_config.channels as usize;

        // The callback still takes a mutex on the realtime thread. Preallocation
        // and reserve calls reduce realloc pressure, but a lock-free buffer would
        // be the next step for strict realtime guarantees.
        let err_fn = |err: cpal::StreamError| {
            tracing::error!("audio stream error: {err}");
        };

        let stream = match sample_format {
            SampleFormat::F32 => device
                .build_input_stream(
                    &stream_config,
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        if let Ok(mut buf) = buffer.lock() {
                            append_mono_f32(data, channels, &mut buf);
                        }
                    },
                    err_fn,
                    None,
                )
                .map_err(|e| WhsprError::Audio(format!("failed to build input stream: {e}")))?,
            SampleFormat::I16 => device
                .build_input_stream(
                    &stream_config,
                    move |data: &[i16], _: &cpal::InputCallbackInfo| {
                        if let Ok(mut buf) = buffer.lock() {
                            append_mono_i16(data, channels, &mut buf);
                        }
                    },
                    err_fn,
                    None,
                )
                .map_err(|e| WhsprError::Audio(format!("failed to build input stream: {e}")))?,
            SampleFormat::U16 => device
                .build_input_stream(
                    &stream_config,
                    move |data: &[u16], _: &cpal::InputCallbackInfo| {
                        if let Ok(mut buf) = buffer.lock() {
                            append_mono_u16(data, channels, &mut buf);
                        }
                    },
                    err_fn,
                    None,
                )
                .map_err(|e| WhsprError::Audio(format!("failed to build input stream: {e}")))?,
            other => {
                return Err(WhsprError::Audio(format!(
                    "unsupported input sample format: {other:?}"
                )));
            }
        };

        stream
            .play()
            .map_err(|e| WhsprError::Audio(format!("failed to start audio stream: {e}")))?;

        // Leak any previous stream to avoid the ALSA/PipeWire click artifact
        // (see stop() comment for rationale).
        if let Some(old) = self.stream.take() {
            let _ = old.pause();
            std::mem::forget(old);
        }
        self.stream = Some(stream);
        tracing::info!("audio recording started");
        Ok(())
    }

    pub fn stop(&mut self) -> Result<Vec<f32>> {
        // Take and leak the stream — cpal's ALSA backend calls snd_pcm_close()
        // on drop without draining first, which causes an audible click on
        // PipeWire when the stream is still "warm".  The OS reclaims file
        // descriptors on process exit.
        if let Some(stream) = self.stream.take() {
            let _ = stream.pause();
            std::mem::forget(stream);
        }

        let mut buffer = std::mem::take(
            &mut *self
                .buffer
                .lock()
                .map_err(|_| WhsprError::Audio("audio buffer lock poisoned".into()))?,
        );
        tracing::info!("audio recording stopped, captured {} samples", buffer.len());

        if buffer.is_empty() {
            return Err(WhsprError::Audio("no audio data captured".into()));
        }

        // Fade out the last few ms to remove any trailing click artifact.
        let fade_samples = (self.config.sample_rate as usize * 5) / 1000; // 5ms
        let fade_len = fade_samples.min(buffer.len());
        let start = buffer.len() - fade_len;
        for i in 0..fade_len {
            let gain = 1.0 - (i as f32 / fade_len as f32);
            buffer[start + i] *= gain;
        }

        Ok(buffer)
    }
}

fn choose_input_config(
    device: &cpal::Device,
    sample_rate: u32,
) -> Result<(StreamConfig, SampleFormat)> {
    let supported = device
        .supported_input_configs()
        .map_err(|e| WhsprError::Audio(format!("failed to get supported configs: {e}")))?;

    let mut best: Option<(u8, StreamConfig, SampleFormat)> = None;

    for cfg in supported {
        if cfg.min_sample_rate() > sample_rate || cfg.max_sample_rate() < sample_rate {
            continue;
        }
        let format_score = match cfg.sample_format() {
            SampleFormat::F32 => 3,
            SampleFormat::I16 => 2,
            SampleFormat::U16 => 1,
            _ => 0,
        };
        if format_score == 0 {
            continue;
        }
        // Prefer mono (20), then fewer channels over more (penalty scales with count)
        let channel_score: u8 = if cfg.channels() == 1 {
            20
        } else {
            10u8.saturating_sub(cfg.channels() as u8)
        };
        let score = channel_score + format_score;

        let config = StreamConfig {
            channels: cfg.channels(),
            sample_rate,
            buffer_size: cpal::BufferSize::Default,
        };

        let replace = best
            .as_ref()
            .map(|(best_score, _, _)| score > *best_score)
            .unwrap_or(true);
        if replace {
            best = Some((score, config, cfg.sample_format()));
        }
    }

    best.map(|(_, config, format)| (config, format))
        .ok_or_else(|| {
            WhsprError::Audio(format!(
                "no supported input config for {} Hz (supported formats must be f32, i16, or u16)",
                sample_rate
            ))
        })
}

fn append_mono_f32(data: &[f32], channels: usize, out: &mut Vec<f32>) {
    if channels <= 1 {
        out.extend_from_slice(data);
        return;
    }
    out.reserve(data.len() / channels);
    for frame in data.chunks(channels) {
        let sum: f32 = frame.iter().copied().sum();
        out.push(sum / frame.len() as f32);
    }
}

fn append_mono_i16(data: &[i16], channels: usize, out: &mut Vec<f32>) {
    const I16_SCALE: f32 = 32768.0;
    if channels <= 1 {
        out.extend(data.iter().map(|s| *s as f32 / I16_SCALE));
        return;
    }
    out.reserve(data.len() / channels);
    for frame in data.chunks(channels) {
        let sum: f32 = frame.iter().map(|s| *s as f32 / I16_SCALE).sum();
        out.push(sum / frame.len() as f32);
    }
}

fn append_mono_u16(data: &[u16], channels: usize, out: &mut Vec<f32>) {
    if channels <= 1 {
        out.extend(
            data.iter()
                .map(|s| (*s as f32 / u16::MAX as f32) * 2.0 - 1.0),
        );
        return;
    }
    out.reserve(data.len() / channels);
    for frame in data.chunks(channels) {
        let sum: f32 = frame
            .iter()
            .map(|s| (*s as f32 / u16::MAX as f32) * 2.0 - 1.0)
            .sum();
        out.push(sum / frame.len() as f32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn append_mono_f32_passthrough_for_single_channel() {
        let mut out = Vec::new();
        append_mono_f32(&[0.1, -0.2, 0.3], 1, &mut out);
        assert_eq!(out, vec![0.1, -0.2, 0.3]);
    }

    #[test]
    fn append_mono_f32_downmixes_stereo() {
        let mut out = Vec::new();
        append_mono_f32(&[1.0, -1.0, 0.5, 0.5], 2, &mut out);
        assert!(approx_eq(out[0], 0.0, 1e-6));
        assert!(approx_eq(out[1], 0.5, 1e-6));
    }

    #[test]
    fn append_mono_i16_converts_to_f32() {
        let mut out = Vec::new();
        append_mono_i16(&[i16::MAX, i16::MIN], 1, &mut out);
        assert!(approx_eq(out[0], 1.0, 1e-4));
        assert!(out[1] < -0.99);
    }

    #[test]
    fn append_mono_u16_downmixes_and_converts() {
        let mut out = Vec::new();
        append_mono_u16(&[0, u16::MAX], 2, &mut out);
        assert!(approx_eq(out[0], 0.0, 0.01));
    }
}

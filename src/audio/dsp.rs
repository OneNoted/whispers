const HIGHPASS_CUTOFF_HZ: f32 = 80.0;
const TRIM_FRAME_MS: usize = 10;
const TRIM_PADDING_MS: usize = 40;
const TRIM_MIN_RMS: f32 = 0.002;
const TRIM_RELATIVE_RMS: f32 = 0.08;
pub(super) const NORMALIZE_TARGET_PEAK: f32 = 0.85;
const NORMALIZE_MAX_GAIN: f32 = 2.5;
const NORMALIZE_MIN_PEAK: f32 = 0.005;

pub fn preprocess_audio(samples: &mut Vec<f32>, sample_rate: u32) {
    if samples.is_empty() || sample_rate == 0 {
        return;
    }

    let before_len = samples.len();
    let before = audio_stats(samples);

    remove_dc_offset(samples);
    apply_highpass(samples, sample_rate, HIGHPASS_CUTOFF_HZ);
    trim_silence(samples, sample_rate);
    let gain = normalize_peak(samples);

    let after = audio_stats(samples);
    tracing::debug!(
        "audio preprocessing: len {} -> {}, rms {:.4} -> {:.4}, peak {:.4} -> {:.4}, gain {:.2}x",
        before_len,
        samples.len(),
        before.rms,
        after.rms,
        before.peak,
        after.peak,
        gain
    );
}

#[derive(Clone, Copy)]
struct AudioStats {
    rms: f32,
    peak: f32,
}

fn audio_stats(samples: &[f32]) -> AudioStats {
    if samples.is_empty() {
        return AudioStats {
            rms: 0.0,
            peak: 0.0,
        };
    }

    let mut peak = 0.0f32;
    let mut energy = 0.0f32;
    for sample in samples {
        peak = peak.max(sample.abs());
        energy += sample * sample;
    }

    AudioStats {
        rms: (energy / samples.len() as f32).sqrt(),
        peak,
    }
}

pub(super) fn remove_dc_offset(samples: &mut [f32]) {
    if samples.is_empty() {
        return;
    }

    let mean = samples.iter().copied().sum::<f32>() / samples.len() as f32;
    if mean.abs() < 1e-6 {
        return;
    }

    for sample in samples {
        *sample -= mean;
    }
}

fn apply_highpass(samples: &mut [f32], sample_rate: u32, cutoff_hz: f32) {
    if samples.len() < 2 || sample_rate == 0 || cutoff_hz <= 0.0 {
        return;
    }

    let dt = 1.0 / sample_rate as f32;
    let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
    let alpha = rc / (rc + dt);

    let mut previous_input = samples[0];
    let mut previous_output = 0.0f32;
    samples[0] = 0.0;

    for sample in samples.iter_mut().skip(1) {
        let input = *sample;
        let output = alpha * (previous_output + input - previous_input);
        *sample = output;
        previous_input = input;
        previous_output = output;
    }
}

pub(super) fn trim_silence(samples: &mut Vec<f32>, sample_rate: u32) {
    if samples.is_empty() || sample_rate == 0 {
        return;
    }

    let frame_len = ((sample_rate as usize * TRIM_FRAME_MS) / 1000).max(1);
    if samples.len() <= frame_len * 2 {
        return;
    }

    let frame_rms: Vec<f32> = samples.chunks(frame_len).map(frame_rms).collect();
    let peak_rms = frame_rms.iter().copied().fold(0.0f32, f32::max);
    if peak_rms <= 0.0 {
        return;
    }

    let threshold = (peak_rms * TRIM_RELATIVE_RMS).max(TRIM_MIN_RMS);
    let Some(first_active) = frame_rms.iter().position(|rms| *rms >= threshold) else {
        return;
    };
    let Some(last_active) = frame_rms.iter().rposition(|rms| *rms >= threshold) else {
        return;
    };

    let padding_samples = (sample_rate as usize * TRIM_PADDING_MS) / 1000;
    let padding_frames = padding_samples.div_ceil(frame_len);
    let start_frame = first_active.saturating_sub(padding_frames);
    let end_frame = (last_active + 1 + padding_frames).min(frame_rms.len());

    let start = start_frame.saturating_mul(frame_len);
    let end = (end_frame.saturating_mul(frame_len)).min(samples.len());
    if start == 0 && end == samples.len() {
        return;
    }
    if start >= end {
        return;
    }

    *samples = samples[start..end].to_vec();
}

fn frame_rms(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }

    let energy = frame.iter().map(|sample| sample * sample).sum::<f32>();
    (energy / frame.len() as f32).sqrt()
}

pub(super) fn normalize_peak(samples: &mut [f32]) -> f32 {
    let peak = samples.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
    if !(NORMALIZE_MIN_PEAK..NORMALIZE_TARGET_PEAK).contains(&peak) {
        return 1.0;
    }

    let gain = (NORMALIZE_TARGET_PEAK / peak).min(NORMALIZE_MAX_GAIN);
    if gain <= 1.0 {
        return 1.0;
    }

    for sample in samples {
        *sample = (*sample * gain).clamp(-1.0, 1.0);
    }

    gain
}

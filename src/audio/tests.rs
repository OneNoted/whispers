use super::{dsp, preprocess_audio, recorder};

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}

#[test]
fn append_mono_f32_passthrough_for_single_channel() {
    let mut out = Vec::new();
    recorder::append_mono_f32(&[0.1, -0.2, 0.3], 1, &mut out);
    assert_eq!(out, vec![0.1, -0.2, 0.3]);
}

#[test]
fn append_mono_f32_downmixes_stereo() {
    let mut out = Vec::new();
    recorder::append_mono_f32(&[1.0, -1.0, 0.5, 0.5], 2, &mut out);
    assert!(approx_eq(out[0], 0.0, 1e-6));
    assert!(approx_eq(out[1], 0.5, 1e-6));
}

#[test]
fn append_mono_i16_converts_to_f32() {
    let mut out = Vec::new();
    recorder::append_mono_i16(&[i16::MAX, i16::MIN], 1, &mut out);
    assert!(approx_eq(out[0], 1.0, 1e-4));
    assert!(out[1] < -0.99);
}

#[test]
fn append_mono_u16_downmixes_and_converts() {
    let mut out = Vec::new();
    recorder::append_mono_u16(&[0, u16::MAX], 2, &mut out);
    assert!(approx_eq(out[0], 0.0, 0.01));
}

#[test]
fn remove_dc_offset_centers_signal() {
    let mut samples = vec![0.3, 0.5, 0.7];
    dsp::remove_dc_offset(&mut samples);
    let mean = samples.iter().copied().sum::<f32>() / samples.len() as f32;
    assert!(mean.abs() < 1e-6);
}

#[test]
fn trim_silence_removes_outer_quiet_sections() {
    let sample_rate = 1000;
    let mut samples = vec![0.0; 120];
    samples.extend(std::iter::repeat_n(0.2, 200));
    samples.extend(vec![0.0; 120]);

    dsp::trim_silence(&mut samples, sample_rate);

    assert!(samples.len() < 440);
    assert!(samples.len() >= 200);
    assert!(samples.iter().any(|sample| sample.abs() >= 0.19));
}

#[test]
fn normalize_peak_boosts_quiet_audio_without_clipping() {
    let mut samples = vec![0.2, -0.3, 0.4];
    let gain = dsp::normalize_peak(&mut samples);
    let peak = samples.iter().copied().map(f32::abs).fold(0.0f32, f32::max);

    assert!(gain > 1.0);
    assert!(approx_eq(peak, dsp::NORMALIZE_TARGET_PEAK, 1e-4));
    assert!(samples.iter().all(|sample| sample.abs() <= 1.0));
}

#[test]
fn preprocess_audio_reduces_leading_and_trailing_silence() {
    let sample_rate = 16000;
    let mut samples = vec![0.0; 1600];
    samples.extend((0..3200).map(|idx| if idx % 2 == 0 { 0.08 } else { -0.08 }));
    samples.extend(vec![0.0; 1600]);

    preprocess_audio(&mut samples, sample_rate);

    assert!(samples.len() < 6400);
    assert!(samples.iter().any(|sample| sample.abs() > 0.1));
}

#[path = "../branding.rs"]
mod branding;
#[path = "../osd_protocol.rs"]
mod osd_protocol;

use std::ffi::CString;
use std::io::{BufRead, BufReader};
use std::os::fd::AsRawFd;
use std::os::unix::io::{AsFd, FromRawFd};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Instant;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use font8x8::{BASIC_FONTS, UnicodeFonts};
use fontdue::Font;
use osd_protocol::{OsdEvent, VoiceOsdStatus, VoiceOsdUpdate};
use wayland_client::protocol::{
    wl_buffer, wl_compositor, wl_registry, wl_shm, wl_shm_pool, wl_surface,
};
use wayland_client::{Connection, Dispatch, QueueHandle, delegate_noop};
use wayland_protocols_wlr::layer_shell::v1::client::{zwlr_layer_shell_v1, zwlr_layer_surface_v1};

const NUM_BARS: usize = 12;
const BAR_WIDTH: u32 = 3;
const BAR_GAP: u32 = 2;
const PAD_X: u32 = 12;
const PAD_Y: u32 = 6;
const BAR_MIN_HEIGHT: f32 = 2.0;
const BAR_MAX_HEIGHT: f32 = 20.0;
const METER_WIDTH: u32 = PAD_X * 2 + NUM_BARS as u32 * BAR_WIDTH + (NUM_BARS as u32 - 1) * BAR_GAP;
const METER_HEIGHT: u32 = BAR_MAX_HEIGHT as u32 + PAD_Y * 2;
const VOICE_WIDTH: u32 = 760;
const VOICE_HEIGHT: u32 = 248;
const MARGIN_BOTTOM: i32 = 40;
const CORNER_RADIUS: u32 = 16;
const BORDER_WIDTH: u32 = 1;
const RISE_RATE: f32 = 0.40;
const DECAY_RATE: f32 = 0.92;
const FPS: i32 = 30;
const FRAME_MS: i32 = 1000 / FPS;

const BG_R: u8 = 15;
const BG_G: u8 = 16;
const BG_B: u8 = 24;
const BG_A: u8 = 160;
const BORDER_R: u8 = 200;
const BORDER_G: u8 = 215;
const BORDER_B: u8 = 255;
const BORDER_A: u8 = 22;

const BAR_REST_R: f32 = 0.863;
const BAR_REST_G: f32 = 0.882;
const BAR_REST_B: f32 = 0.922;
const BAR_REST_A: f32 = 0.706;
const BAR_PEAK_R: f32 = 0.392;
const BAR_PEAK_G: f32 = 0.608;
const BAR_PEAK_B: f32 = 1.0;
const BAR_PEAK_A: f32 = 0.941;

const TEXT_PRIMARY: (u8, u8, u8, u8) = (242, 246, 255, 230);
const TEXT_MUTED: (u8, u8, u8, u8) = (185, 196, 220, 200);
const TEXT_UNSTABLE: (u8, u8, u8, u8) = (115, 235, 226, 240);
const TEXT_REWRITE: (u8, u8, u8, u8) = (255, 208, 126, 220);
const TEXT_REWRITE_PRIMARY: (u8, u8, u8, u8) = (255, 236, 205, 240);
const TEXT_WARNING: (u8, u8, u8, u8) = (255, 153, 134, 235);
const DIVIDER: (u8, u8, u8, u8) = (120, 150, 205, 42);

static SHOULD_EXIT: AtomicBool = AtomicBool::new(false);
static OSD_FONT: OnceLock<Option<Font>> = OnceLock::new();

#[derive(Clone, Copy, PartialEq, Eq)]
enum OverlayMode {
    Meter,
    Voice,
}

struct AudioLevel {
    rms_bits: AtomicU32,
}

impl AudioLevel {
    fn new() -> Self {
        Self {
            rms_bits: AtomicU32::new(0),
        }
    }

    fn set(&self, val: f32) {
        self.rms_bits.store(val.to_bits(), Ordering::Relaxed);
    }

    fn get(&self) -> f32 {
        f32::from_bits(self.rms_bits.load(Ordering::Relaxed))
    }
}

struct BarState {
    heights: [f32; NUM_BARS],
    smooth_rms: f32,
}

impl BarState {
    fn new() -> Self {
        Self {
            heights: [BAR_MIN_HEIGHT; NUM_BARS],
            smooth_rms: 0.0,
        }
    }

    fn update(&mut self, rms: f32, time: f32) {
        let level = (rms * 5.0).min(1.0);
        let rms_target = (rms * 4.0).min(1.0);
        self.smooth_rms = self.smooth_rms * 0.85 + rms_target * 0.15;

        for i in 0..NUM_BARS {
            let t = i as f32 / NUM_BARS as f32;
            let wave1 = (t * std::f32::consts::PI * 2.0 + time * 2.0).sin() * 0.5 + 0.5;
            let wave2 = (t * std::f32::consts::PI * 1.5 - time * 1.2).sin() * 0.3 + 0.5;
            let wave3 = (t * std::f32::consts::PI * 3.5 + time * 4.0).sin() * 0.15 + 0.5;

            let combined = (wave1 * 0.55 + wave2 * 0.30 + wave3 * 0.15) * level;
            let target = BAR_MIN_HEIGHT + combined * (BAR_MAX_HEIGHT - BAR_MIN_HEIGHT);

            if target > self.heights[i] {
                self.heights[i] += (target - self.heights[i]) * RISE_RATE;
            } else {
                self.heights[i] = self.heights[i] * DECAY_RATE + target * (1.0 - DECAY_RATE);
            }
            self.heights[i] = self.heights[i].clamp(BAR_MIN_HEIGHT, BAR_MAX_HEIGHT);
        }
    }
}

#[derive(Debug, Clone)]
struct VoiceOverlayState {
    status: VoiceOsdStatus,
    stable_text: String,
    unstable_text: String,
    rewrite_preview: Option<String>,
    live_inject: bool,
    frozen: bool,
}

impl Default for VoiceOverlayState {
    fn default() -> Self {
        Self {
            status: VoiceOsdStatus::Listening,
            stable_text: String::new(),
            unstable_text: String::new(),
            rewrite_preview: None,
            live_inject: false,
            frozen: false,
        }
    }
}

impl From<VoiceOsdUpdate> for VoiceOverlayState {
    fn from(update: VoiceOsdUpdate) -> Self {
        Self {
            status: update.status,
            stable_text: update.stable_text,
            unstable_text: update.unstable_text,
            rewrite_preview: update.rewrite_preview,
            live_inject: update.live_inject,
            frozen: update.frozen,
        }
    }
}

struct OsdState {
    running: bool,
    width: u32,
    height: u32,
    compositor: Option<wl_compositor::WlCompositor>,
    shm: Option<wl_shm::WlShm>,
    layer_shell: Option<zwlr_layer_shell_v1::ZwlrLayerShellV1>,
    surface: Option<wl_surface::WlSurface>,
    layer_surface: Option<zwlr_layer_surface_v1::ZwlrLayerSurfaceV1>,
    buffer: Option<wl_buffer::WlBuffer>,
    configured: bool,
}

fn pid_file_path() -> PathBuf {
    let runtime_dir = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(runtime_dir).join(branding::OSD_PID_FILE)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    unsafe {
        libc::signal(
            libc::SIGTERM,
            handle_signal as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGINT,
            handle_signal as *const () as libc::sighandler_t,
        );
    }

    let mode = if std::env::args().any(|arg| arg == "--voice") {
        OverlayMode::Voice
    } else {
        OverlayMode::Meter
    };
    let (width, height) = match mode {
        OverlayMode::Meter => (METER_WIDTH, METER_HEIGHT),
        OverlayMode::Voice => (VOICE_WIDTH, VOICE_HEIGHT),
    };

    let _ = std::fs::write(pid_file_path(), std::process::id().to_string());

    let audio_level = Arc::new(AudioLevel::new());
    let _audio_stream = start_audio_capture(Arc::clone(&audio_level));

    let voice_state = matches!(mode, OverlayMode::Voice)
        .then(|| Arc::new(std::sync::Mutex::new(VoiceOverlayState::default())));
    let _voice_reader = voice_state
        .as_ref()
        .map(|state| start_voice_event_reader(Arc::clone(state)));

    let conn = Connection::connect_to_env()?;
    let mut event_queue = conn.new_event_queue();
    let qh = event_queue.handle();
    conn.display().get_registry(&qh, ());

    let mut state = OsdState {
        running: true,
        width,
        height,
        compositor: None,
        shm: None,
        layer_shell: None,
        surface: None,
        layer_surface: None,
        buffer: None,
        configured: false,
    };

    event_queue.roundtrip(&mut state)?;

    let compositor = state
        .compositor
        .as_ref()
        .ok_or("compositor not advertised by wayland server")?;
    let layer_shell = state
        .layer_shell
        .as_ref()
        .ok_or("zwlr_layer_shell_v1 not supported by compositor")?;

    let surface = compositor.create_surface(&qh, ());
    let layer_surface = layer_shell.get_layer_surface(
        &surface,
        None,
        zwlr_layer_shell_v1::Layer::Overlay,
        branding::OSD_BINARY.to_string(),
        &qh,
        (),
    );

    layer_surface.set_size(width, height);
    layer_surface.set_anchor(zwlr_layer_surface_v1::Anchor::Bottom);
    layer_surface.set_margin(0, 0, MARGIN_BOTTOM, 0);
    layer_surface.set_exclusive_zone(-1);
    layer_surface.set_keyboard_interactivity(zwlr_layer_surface_v1::KeyboardInteractivity::None);
    surface.commit();

    state.surface = Some(surface);
    state.layer_surface = Some(layer_surface);
    event_queue.roundtrip(&mut state)?;

    let mut bars = BarState::new();
    let start_time = Instant::now();
    let mut pixels = vec![0u8; (width * height * 4) as usize];

    let stride = width * 4;
    let shm_size = (stride * height) as i32;
    let shm_name = CString::new(branding::OSD_BINARY).expect("valid OSD memfd name");
    let shm_fd = unsafe { libc::memfd_create(shm_name.as_ptr(), libc::MFD_CLOEXEC) };
    if shm_fd < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let shm_file = unsafe { std::fs::File::from_raw_fd(shm_fd) };
    shm_file.set_len(shm_size as u64)?;
    let shm = state
        .shm
        .as_ref()
        .ok_or("wl_shm not advertised by wayland server")?;
    let pool = shm.create_pool(shm_file.as_fd(), shm_size, &qh, ());

    while state.running && !SHOULD_EXIT.load(Ordering::Relaxed) {
        conn.flush()?;

        let Some(read_guard) = event_queue.prepare_read() else {
            event_queue.dispatch_pending(&mut state)?;
            continue;
        };
        let mut pollfd = libc::pollfd {
            fd: read_guard.connection_fd().as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };

        let ret = unsafe { libc::poll(&mut pollfd, 1, FRAME_MS) };
        if ret > 0 {
            let _ = read_guard.read();
        } else {
            drop(read_guard);
        }
        event_queue.dispatch_pending(&mut state)?;

        if !state.configured {
            continue;
        }

        let time = start_time.elapsed().as_secs_f32();
        let rms = audio_level.get();
        bars.update(rms, time);

        let voice_snapshot = voice_state
            .as_ref()
            .and_then(|shared| shared.lock().ok().map(|state| state.clone()));

        pixels.fill(0);
        render_frame(
            &mut pixels,
            state.width,
            state.height,
            &bars,
            time,
            mode,
            voice_snapshot.as_ref(),
        );

        if let Err(err) = present_frame(&mut state, &qh, &pool, &shm_file, &pixels, width, height) {
            eprintln!("frame dropped: {err}");
        }
    }

    pool.destroy();
    if let Some(ls) = state.layer_surface.take() {
        ls.destroy();
    }
    if let Some(s) = state.surface.take() {
        s.destroy();
    }
    if let Some(b) = state.buffer.take() {
        b.destroy();
    }
    let _ = std::fs::remove_file(pid_file_path());
    Ok(())
}

unsafe extern "C" fn handle_signal(_sig: libc::c_int) {
    SHOULD_EXIT.store(true, Ordering::Relaxed);
}

fn start_voice_event_reader(
    state: Arc<std::sync::Mutex<VoiceOverlayState>>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let reader = BufReader::new(stdin.lock());
        for line in reader.lines() {
            let Ok(line) = line else {
                break;
            };
            if line.trim().is_empty() {
                continue;
            }
            let Ok(event) = serde_json::from_str::<OsdEvent>(&line) else {
                continue;
            };
            let OsdEvent::VoiceUpdate(update) = event;
            if let Ok(mut guard) = state.lock() {
                *guard = update.into();
            }
        }
    })
}

fn start_audio_capture(level: Arc<AudioLevel>) -> Option<cpal::Stream> {
    let host = cpal::default_host();
    let device = host.default_input_device()?;
    let config = device
        .supported_input_configs()
        .ok()
        .and_then(|configs| {
            configs
                .filter(|c| c.min_sample_rate() <= 16000 && c.max_sample_rate() >= 16000)
                .filter(|c| c.sample_format() == cpal::SampleFormat::F32)
                .min_by_key(|c| c.channels())
                .map(|c| cpal::StreamConfig {
                    channels: c.channels(),
                    sample_rate: 16000,
                    buffer_size: cpal::BufferSize::Default,
                })
        })
        .unwrap_or(cpal::StreamConfig {
            channels: 1,
            sample_rate: 16000,
            buffer_size: cpal::BufferSize::Default,
        });

    let channels = config.channels as usize;
    let stream = device
        .build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if data.is_empty() {
                    return;
                }
                let sample_count = data.len() / channels.max(1);
                if sample_count == 0 {
                    return;
                }
                let sum: f32 = if channels <= 1 {
                    data.iter().map(|s| s * s).sum()
                } else {
                    data.chunks(channels)
                        .map(|frame| {
                            let mono: f32 = frame.iter().sum::<f32>() / frame.len() as f32;
                            mono * mono
                        })
                        .sum()
                };
                let rms = (sum / sample_count as f32).sqrt();
                level.set(rms);
            },
            |err| eprintln!("audio capture error: {err}"),
            None,
        )
        .ok()?;

    stream.play().ok()?;
    Some(stream)
}

fn render_frame(
    pixels: &mut [u8],
    w: u32,
    h: u32,
    bars: &BarState,
    _time: f32,
    mode: OverlayMode,
    voice_state: Option<&VoiceOverlayState>,
) {
    if matches!(mode, OverlayMode::Voice) {
        draw_rounded_rect(
            pixels,
            w,
            h,
            0,
            0,
            w,
            h,
            CORNER_RADIUS,
            BG_R,
            BG_G,
            BG_B,
            BG_A,
        );
        draw_rounded_border(
            pixels,
            w,
            h,
            CORNER_RADIUS,
            BORDER_WIDTH,
            BORDER_R,
            BORDER_G,
            BORDER_B,
            BORDER_A,
        );
        for x in (CORNER_RADIUS + 2)..(w.saturating_sub(CORNER_RADIUS + 2)) {
            set_pixel_blend(pixels, w, h, x, 1, 255, 255, 255, 12);
        }
    }

    match mode {
        OverlayMode::Meter => render_meter_overlay(pixels, w, h, bars),
        OverlayMode::Voice => render_voice_overlay(
            pixels,
            w,
            h,
            bars,
            voice_state.unwrap_or(&VoiceOverlayState::default()),
        ),
    }
}

fn render_meter_overlay(pixels: &mut [u8], w: u32, h: u32, bars: &BarState) {
    render_meter_bars(pixels, w, h, bars, h / 2);
}

fn render_voice_overlay(
    pixels: &mut [u8],
    w: u32,
    h: u32,
    bars: &BarState,
    voice: &VoiceOverlayState,
) {
    let pad = 20i32;
    let header_y = 16i32;
    let transcript_y = 50i32;
    let status_font = 16.0;
    let badge_font = 13.0;
    let transcript_font = 20.0;
    let footer_font = 14.0;
    let transcript_line_height = line_height(transcript_font) + 4;
    let footer_line_height = line_height(footer_font) + 2;
    let transcript_width = w.saturating_sub((pad as u32) * 2);
    let raw_live_text = combined_voice_text(&voice.stable_text, &voice.unstable_text);
    let rewrite_available = voice
        .rewrite_preview
        .as_deref()
        .map(str::trim)
        .filter(|text| !text.is_empty());

    let status_label = status_label(voice.status);
    draw_text(
        pixels,
        w,
        h,
        pad,
        header_y,
        status_font,
        status_label,
        TEXT_PRIMARY,
    );

    let badge_text = if voice.live_inject {
        "LIVE INJECT"
    } else {
        "PREVIEW ONLY"
    };
    let badge_width = text_width(badge_text, badge_font);
    draw_text(
        pixels,
        w,
        h,
        w.saturating_sub(pad as u32 + badge_width) as i32,
        header_y,
        badge_font,
        badge_text,
        if voice.live_inject {
            TEXT_UNSTABLE
        } else {
            TEXT_MUTED
        },
    );

    if let Some(rewrite_text) = rewrite_available {
        let preview_label = "Live rewrite preview";
        draw_text(
            pixels,
            w,
            h,
            pad,
            transcript_y - footer_line_height - 2,
            footer_font,
            preview_label,
            TEXT_REWRITE,
        );

        let rewrite_lines = wrap_text(rewrite_text, transcript_width, 4, transcript_font);
        for (idx, line) in rewrite_lines.iter().enumerate() {
            draw_text(
                pixels,
                w,
                h,
                pad,
                transcript_y + idx as i32 * transcript_line_height,
                transcript_font,
                line,
                TEXT_REWRITE_PRIMARY,
            );
        }
    } else {
        let stable_lines = wrap_text(&voice.stable_text, transcript_width, 3, transcript_font);
        let unstable_lines = wrap_text(&voice.unstable_text, transcript_width, 2, transcript_font);

        if stable_lines.is_empty() && unstable_lines.is_empty() {
            draw_text(
                pixels,
                w,
                h,
                pad,
                transcript_y,
                transcript_font,
                "Listening for speech...",
                TEXT_MUTED,
            );
        } else {
            for (idx, line) in stable_lines.iter().enumerate() {
                draw_text(
                    pixels,
                    w,
                    h,
                    pad,
                    transcript_y + idx as i32 * transcript_line_height,
                    transcript_font,
                    line,
                    TEXT_PRIMARY,
                );
            }

            let unstable_y = transcript_y + stable_lines.len() as i32 * transcript_line_height;
            for (idx, line) in unstable_lines.iter().enumerate() {
                draw_text(
                    pixels,
                    w,
                    h,
                    pad,
                    unstable_y + idx as i32 * transcript_line_height,
                    transcript_font,
                    line,
                    TEXT_UNSTABLE,
                );
            }
        }
    }

    let divider_y = h.saturating_sub(74);
    for x in pad as u32..w.saturating_sub(pad as u32) {
        set_pixel_blend(
            pixels, w, h, x, divider_y, DIVIDER.0, DIVIDER.1, DIVIDER.2, DIVIDER.3,
        );
    }

    if voice.frozen {
        let warning_lines = wrap_text(
            "Focus changed. Live injection is frozen for this take.",
            transcript_width,
            2,
            footer_font,
        );
        for (idx, line) in warning_lines.iter().enumerate() {
            draw_text(
                pixels,
                w,
                h,
                pad,
                divider_y as i32 + 10 + idx as i32 * footer_line_height,
                footer_font,
                line,
                TEXT_WARNING,
            );
        }
    } else if rewrite_available.is_some() {
        draw_text(
            pixels,
            w,
            h,
            pad,
            divider_y as i32 + 10,
            footer_font,
            "Raw live hypothesis",
            TEXT_MUTED,
        );
        let raw_lines = wrap_text(&raw_live_text, transcript_width, 2, footer_font);
        for (idx, line) in raw_lines.iter().enumerate() {
            draw_text(
                pixels,
                w,
                h,
                pad,
                divider_y as i32 + 10 + footer_line_height + idx as i32 * footer_line_height,
                footer_font,
                line,
                TEXT_MUTED,
            );
        }
    }

    render_voice_bars(pixels, w, h, bars, h.saturating_sub(24));
}

fn render_meter_bars(pixels: &mut [u8], w: u32, h: u32, bars: &BarState, center_y: u32) {
    let color_t = bars.smooth_rms.clamp(0.0, 1.0);
    let cr = (lerp(BAR_REST_R, BAR_PEAK_R, color_t) * 255.0) as u8;
    let cg = (lerp(BAR_REST_G, BAR_PEAK_G, color_t) * 255.0) as u8;
    let cb = (lerp(BAR_REST_B, BAR_PEAK_B, color_t) * 255.0) as u8;
    let base_alpha = lerp(BAR_REST_A, BAR_PEAK_A, color_t);

    let glow_expand = 1 + (color_t * 2.0) as u32;
    let glow_alpha = (15.0 + color_t * 25.0) as u8;

    let total_width = NUM_BARS as u32 * BAR_WIDTH + (NUM_BARS as u32 - 1) * BAR_GAP;
    let start_x = w.saturating_sub(total_width) / 2;
    for i in 0..NUM_BARS {
        let bx = start_x + i as u32 * (BAR_WIDTH + BAR_GAP);
        let bar_h = bars.heights[i] as u32;
        let half_h = bar_h / 2;
        let top_y = center_y.saturating_sub(half_h);

        for gy in top_y.saturating_sub(glow_expand)..=(top_y + bar_h + glow_expand).min(h.saturating_sub(1)) {
            for gx in bx.saturating_sub(glow_expand)..=(bx + BAR_WIDTH + glow_expand).min(w.saturating_sub(1)) {
                set_pixel_blend(pixels, w, h, gx, gy, cr, cg, cb, glow_alpha);
            }
        }

        for y in top_y..(top_y + bar_h).min(h) {
            let vy = (y as f32 - top_y as f32) / bar_h.max(1) as f32;
            let brightness = 1.0 - (vy - 0.5).abs() * 0.5;
            let a = (brightness * base_alpha * 255.0) as u8;
            for x in bx..(bx + BAR_WIDTH).min(w) {
                set_pixel_blend(pixels, w, h, x, y, cr, cg, cb, a);
            }
        }
    }
}

fn render_voice_bars(pixels: &mut [u8], w: u32, h: u32, bars: &BarState, center_y: u32) {
    const VOICE_BAR_LEFT_R: f32 = 0.0;
    const VOICE_BAR_LEFT_G: f32 = 0.82;
    const VOICE_BAR_LEFT_B: f32 = 0.75;
    const VOICE_BAR_RIGHT_R: f32 = 0.65;
    const VOICE_BAR_RIGHT_G: f32 = 0.35;
    const VOICE_BAR_RIGHT_B: f32 = 1.0;

    let total_width = NUM_BARS as u32 * BAR_WIDTH + (NUM_BARS as u32 - 1) * BAR_GAP;
    let start_x = w.saturating_sub(total_width) / 2;
    for i in 0..NUM_BARS {
        let bx = start_x + i as u32 * (BAR_WIDTH + BAR_GAP);
        let bar_h = bars.heights[i] as u32;
        let half_h = bar_h / 2;
        let top_y = center_y.saturating_sub(half_h);

        let t = i as f32 / (NUM_BARS - 1) as f32;
        let r = lerp(VOICE_BAR_LEFT_R, VOICE_BAR_RIGHT_R, t);
        let g = lerp(VOICE_BAR_LEFT_G, VOICE_BAR_RIGHT_G, t);
        let b = lerp(VOICE_BAR_LEFT_B, VOICE_BAR_RIGHT_B, t);
        let cr = (r * 255.0) as u8;
        let cg = (g * 255.0) as u8;
        let cb = (b * 255.0) as u8;

        for gy in top_y.saturating_sub(2)..=(top_y + bar_h + 2).min(h.saturating_sub(1)) {
            for gx in bx.saturating_sub(1)..=(bx + BAR_WIDTH).min(w.saturating_sub(1)) {
                set_pixel_blend(pixels, w, h, gx, gy, cr, cg, cb, 25);
            }
        }

        for y in top_y..(top_y + bar_h).min(h) {
            let vy = (y as f32 - top_y as f32) / bar_h.max(1) as f32;
            let brightness = 1.0 - (vy - 0.5).abs() * 0.6;
            let a = (brightness * 230.0) as u8;
            for x in bx..(bx + BAR_WIDTH).min(w) {
                set_pixel_blend(pixels, w, h, x, y, cr, cg, cb, a);
            }
        }
    }
}

fn status_label(status: VoiceOsdStatus) -> &'static str {
    match status {
        VoiceOsdStatus::Listening => "Listening",
        VoiceOsdStatus::Transcribing => "Transcribing",
        VoiceOsdStatus::Rewriting => "Rewriting",
        VoiceOsdStatus::Finalizing => "Finalizing",
        VoiceOsdStatus::Frozen => "Frozen",
    }
}

fn load_osd_font() -> Option<Font> {
    const FONT_CANDIDATES: &[&str] = &[
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/noto/NotoSans-Medium.ttf",
        "/usr/share/fonts/TTF/NotoSansMNerdFont-Regular.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/ArimoNerdFont-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMNerdFont-Regular.ttf",
    ];

    FONT_CANDIDATES
        .iter()
        .find_map(|path| load_font_from_path(path))
}

fn load_font_from_path(path: &str) -> Option<Font> {
    let bytes = std::fs::read(Path::new(path)).ok()?;
    Font::from_bytes(bytes, fontdue::FontSettings::default()).ok()
}

fn osd_font() -> Option<&'static Font> {
    OSD_FONT.get_or_init(load_osd_font).as_ref()
}

fn line_height(px_size: f32) -> i32 {
    if let Some(font) = osd_font() {
        if let Some(metrics) = font.horizontal_line_metrics(px_size) {
            return metrics.new_line_size.ceil().max(px_size.ceil()) as i32;
        }
    }
    let scale = ((px_size / 8.0).round() as i32).max(1);
    8 * scale + scale + 2
}

fn wrap_text(text: &str, max_width_px: u32, max_lines: usize, px_size: f32) -> Vec<String> {
    if max_width_px == 0 || max_lines == 0 {
        return Vec::new();
    }

    let text = sanitize_text(text);
    if text.is_empty() {
        return Vec::new();
    }

    let mut lines = Vec::new();
    let mut current = String::new();

    for word in text.split_whitespace() {
        if word.is_empty() {
            continue;
        }
        let candidate = if current.is_empty() {
            word.to_string()
        } else {
            format!("{current} {word}")
        };

        if text_width(&candidate, px_size) <= max_width_px {
            current = candidate;
            continue;
        }

        if !current.is_empty() {
            lines.push(current);
        }

        current = truncate_word_to_width(word, max_width_px, px_size);
    }

    if !current.is_empty() {
        lines.push(current);
    }

    if lines.len() > max_lines {
        let mut tail = lines.split_off(lines.len() - max_lines);
        if let Some(first) = tail.first_mut() {
            *first = fit_text_to_width(&format!("…{first}"), max_width_px, px_size, true);
        }
        return tail;
    }

    lines
}

fn combined_voice_text(stable_text: &str, unstable_text: &str) -> String {
    match (stable_text.trim(), unstable_text.trim()) {
        ("", "") => String::new(),
        ("", unstable) => unstable.to_string(),
        (stable, "") => stable.to_string(),
        (stable, unstable) => format!("{stable} {unstable}"),
    }
}

fn sanitize_text(text: &str) -> String {
    let mut normalized = String::new();
    let mut pending_space = false;

    for ch in text.chars() {
        if ch.is_control() && !ch.is_whitespace() {
            continue;
        }
        if ch.is_whitespace() {
            if !normalized.is_empty() {
                pending_space = true;
            }
            continue;
        }
        if pending_space {
            normalized.push(' ');
            pending_space = false;
        }
        normalized.push(ch);
    }

    normalized
}

fn truncate_word_to_width(word: &str, max_width_px: u32, px_size: f32) -> String {
    fit_text_to_width(word, max_width_px, px_size, false)
}

fn fit_text_to_width(text: &str, max_width_px: u32, px_size: f32, keep_tail: bool) -> String {
    let sanitized = sanitize_text(text);
    if sanitized.is_empty() {
        return String::new();
    }
    if text_width(&sanitized, px_size) <= max_width_px {
        return sanitized;
    }

    let ellipsis = "…";
    let chars: Vec<char> = sanitized.chars().collect();
    if keep_tail {
        for start in 0..chars.len() {
            let candidate: String = std::iter::once('…')
                .chain(chars[start..].iter().copied())
                .collect();
            if text_width(&candidate, px_size) <= max_width_px {
                return candidate;
            }
        }
    } else {
        let mut candidate = String::new();
        for ch in chars {
            let next = if candidate.is_empty() {
                format!("{ch}{ellipsis}")
            } else {
                format!("{candidate}{ch}{ellipsis}")
            };
            if text_width(&next, px_size) > max_width_px {
                break;
            }
            candidate.push(ch);
        }
        if !candidate.is_empty() {
            candidate.push('…');
            return candidate;
        }
    }

    ellipsis.to_string()
}

fn text_width(text: &str, px_size: f32) -> u32 {
    let sanitized = sanitize_text(text);
    if sanitized.is_empty() {
        return 0;
    }
    if let Some(font) = osd_font() {
        let width: f32 = sanitized
            .chars()
            .map(|ch| font.metrics(ch, px_size).advance_width)
            .sum();
        return width.ceil().max(0.0) as u32;
    }

    let scale = ((px_size / 8.0).round() as i32).max(1);
    let glyph_w = (8 * scale + scale).max(1) as u32;
    sanitized.chars().count() as u32 * glyph_w
}

fn draw_text(
    pixels: &mut [u8],
    w: u32,
    h: u32,
    x: i32,
    y: i32,
    px_size: f32,
    text: &str,
    color: (u8, u8, u8, u8),
) {
    let sanitized = sanitize_text(text);
    if sanitized.is_empty() {
        return;
    }

    if let Some(font) = osd_font() {
        let baseline = y as f32
            + font
                .horizontal_line_metrics(px_size)
                .map(|metrics| metrics.ascent)
                .unwrap_or(px_size * 0.92);
        let mut pen_x = x as f32;
        for ch in sanitized.chars() {
            let (metrics, bitmap) = font.rasterize(ch, px_size);
            let glyph_x = pen_x.round() as i32 + metrics.xmin;
            // fontdue reports ymin from the baseline to the glyph's bottom edge.
            // In our positive-Y-down compositor space, the top pixel row sits at
            // baseline - height - ymin.
            let glyph_y = (baseline - metrics.height as f32 - metrics.ymin as f32).round() as i32;
            for row in 0..metrics.height {
                for col in 0..metrics.width {
                    let alpha = bitmap[row * metrics.width + col];
                    if alpha == 0 {
                        continue;
                    }
                    let px = glyph_x + col as i32;
                    let py = glyph_y + row as i32;
                    if px >= 0 && py >= 0 {
                        let blended_alpha = ((alpha as u16 * color.3 as u16) / 255) as u8;
                        set_pixel_blend(
                            pixels,
                            w,
                            h,
                            px as u32,
                            py as u32,
                            color.0,
                            color.1,
                            color.2,
                            blended_alpha,
                        );
                    }
                }
            }
            pen_x += metrics.advance_width;
        }
        return;
    }

    let scale = ((px_size / 8.0).round() as i32).max(1);
    let mut cursor_x = x;
    let glyph_advance = (8 * scale + scale).max(1);
    for ch in sanitized.chars() {
        draw_bitmap_char(pixels, w, h, cursor_x, y, scale, ch, color);
        cursor_x += glyph_advance;
    }
}

fn draw_bitmap_char(
    pixels: &mut [u8],
    w: u32,
    h: u32,
    x: i32,
    y: i32,
    scale: i32,
    ch: char,
    color: (u8, u8, u8, u8),
) {
    let glyph = BASIC_FONTS.get(ch).or_else(|| BASIC_FONTS.get('?'));
    let Some(glyph) = glyph else {
        return;
    };

    for (row_idx, row_bits) in glyph.iter().enumerate() {
        for col_idx in 0..8 {
            if (row_bits >> col_idx) & 1 == 0 {
                continue;
            }
            for sy in 0..scale.max(1) {
                for sx in 0..scale.max(1) {
                    let px = x + (col_idx * scale as usize + sx as usize) as i32;
                    let py = y + (row_idx * scale as usize + sy as usize) as i32;
                    if px >= 0 && py >= 0 {
                        set_pixel_blend(
                            pixels, w, h, px as u32, py as u32, color.0, color.1, color.2, color.3,
                        );
                    }
                }
            }
        }
    }
}

fn present_frame(
    state: &mut OsdState,
    qh: &QueueHandle<OsdState>,
    pool: &wl_shm_pool::WlShmPool,
    shm_file: &std::fs::File,
    pixels: &[u8],
    w: u32,
    h: u32,
) -> std::io::Result<()> {
    let stride = w * 4;

    use std::io::{Seek, Write};
    let mut writer = shm_file;
    writer.seek(std::io::SeekFrom::Start(0))?;
    writer.write_all(pixels)?;

    if let Some(old) = state.buffer.take() {
        old.destroy();
    }

    let buffer = pool.create_buffer(
        0,
        w as i32,
        h as i32,
        stride as i32,
        wl_shm::Format::Argb8888,
        qh,
        (),
    );

    let surface = state
        .surface
        .as_ref()
        .ok_or_else(|| std::io::Error::other("wayland surface not initialized"))?;
    surface.attach(Some(&buffer), 0, 0);
    surface.damage_buffer(0, 0, w as i32, h as i32);
    surface.commit();

    state.buffer = Some(buffer);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn set_pixel_blend(pixels: &mut [u8], w: u32, h: u32, x: u32, y: u32, r: u8, g: u8, b: u8, a: u8) {
    if x >= w || y >= h || a == 0 {
        return;
    }
    let idx = ((y * w + x) * 4) as usize;
    if a == 255 {
        pixels[idx] = b;
        pixels[idx + 1] = g;
        pixels[idx + 2] = r;
        pixels[idx + 3] = 255;
        return;
    }
    let sa = a as u32;
    let inv = 255 - sa;
    pixels[idx] = ((sa * b as u32 + inv * pixels[idx] as u32) / 255) as u8;
    pixels[idx + 1] = ((sa * g as u32 + inv * pixels[idx + 1] as u32) / 255) as u8;
    pixels[idx + 2] = ((sa * r as u32 + inv * pixels[idx + 2] as u32) / 255) as u8;
    pixels[idx + 3] = ((sa * 255 + inv * pixels[idx + 3] as u32) / 255) as u8;
}

#[allow(clippy::too_many_arguments)]
fn draw_rounded_rect(
    pixels: &mut [u8],
    pw: u32,
    ph: u32,
    x0: u32,
    y0: u32,
    w: u32,
    h: u32,
    radius: u32,
    r: u8,
    g: u8,
    b: u8,
    a: u8,
) {
    for y in y0..y0 + h {
        for x in x0..x0 + w {
            let lx = x - x0;
            let ly = y - y0;
            if is_inside_rounded_rect(lx, ly, w, h, radius) {
                set_pixel_blend(pixels, pw, ph, x, y, r, g, b, a);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_rounded_border(
    pixels: &mut [u8],
    w: u32,
    h: u32,
    radius: u32,
    thickness: u32,
    r: u8,
    g: u8,
    b: u8,
    a: u8,
) {
    for y in 0..h {
        for x in 0..w {
            let inside_outer = is_inside_rounded_rect(x, y, w, h, radius);
            let inside_inner = x >= thickness
                && y >= thickness
                && x < w - thickness
                && y < h - thickness
                && is_inside_rounded_rect(
                    x - thickness,
                    y - thickness,
                    w - 2 * thickness,
                    h - 2 * thickness,
                    radius.saturating_sub(thickness),
                );
            if inside_outer && !inside_inner {
                set_pixel_blend(pixels, w, h, x, y, r, g, b, a);
            }
        }
    }
}

fn is_inside_rounded_rect(x: u32, y: u32, w: u32, h: u32, r: u32) -> bool {
    if r == 0 || w == 0 || h == 0 {
        return x < w && y < h;
    }
    let in_left = x < r;
    let in_right = x >= w - r;
    let in_top = y < r;
    let in_bottom = y >= h - r;

    if in_left && in_top {
        let dx = r - 1 - x;
        let dy = r - 1 - y;
        return dx * dx + dy * dy <= (r - 1) * (r - 1);
    }
    if in_right && in_top {
        let dx = x - (w - r);
        let dy = r - 1 - y;
        return dx * dx + dy * dy <= (r - 1) * (r - 1);
    }
    if in_left && in_bottom {
        let dx = r - 1 - x;
        let dy = y - (h - r);
        return dx * dx + dy * dy <= (r - 1) * (r - 1);
    }
    if in_right && in_bottom {
        let dx = x - (w - r);
        let dy = y - (h - r);
        return dx * dx + dy * dy <= (r - 1) * (r - 1);
    }
    true
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

impl Dispatch<wl_registry::WlRegistry, ()> for OsdState {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global {
            name, interface, ..
        } = event
        {
            match &interface[..] {
                "wl_compositor" => {
                    state.compositor =
                        Some(registry.bind::<wl_compositor::WlCompositor, _, _>(name, 6, qh, ()));
                }
                "wl_shm" => {
                    state.shm = Some(registry.bind::<wl_shm::WlShm, _, _>(name, 1, qh, ()));
                }
                "zwlr_layer_shell_v1" => {
                    state.layer_shell = Some(
                        registry.bind::<zwlr_layer_shell_v1::ZwlrLayerShellV1, _, _>(
                            name,
                            1,
                            qh,
                            (),
                        ),
                    );
                }
                _ => {}
            }
        }
    }
}

impl Dispatch<zwlr_layer_surface_v1::ZwlrLayerSurfaceV1, ()> for OsdState {
    fn event(
        state: &mut Self,
        layer_surface: &zwlr_layer_surface_v1::ZwlrLayerSurfaceV1,
        event: zwlr_layer_surface_v1::Event,
        _: &(),
        _: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        match event {
            zwlr_layer_surface_v1::Event::Configure {
                serial,
                width,
                height,
            } => {
                layer_surface.ack_configure(serial);
                if width > 0 {
                    state.width = width;
                }
                if height > 0 {
                    state.height = height;
                }
                state.configured = true;
            }
            zwlr_layer_surface_v1::Event::Closed => {
                state.running = false;
            }
            _ => {}
        }
    }
}

delegate_noop!(OsdState: ignore wl_compositor::WlCompositor);
delegate_noop!(OsdState: ignore wl_surface::WlSurface);
delegate_noop!(OsdState: ignore wl_shm::WlShm);
delegate_noop!(OsdState: ignore wl_shm_pool::WlShmPool);
delegate_noop!(OsdState: ignore wl_buffer::WlBuffer);
delegate_noop!(OsdState: ignore zwlr_layer_shell_v1::ZwlrLayerShellV1);

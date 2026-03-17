use std::ffi::CString;
use std::os::fd::AsRawFd;
use std::os::unix::io::{AsFd, FromRawFd};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Instant;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use wayland_client::protocol::{
    wl_buffer, wl_compositor, wl_registry, wl_shm, wl_shm_pool, wl_surface,
};
use wayland_client::{Connection, Dispatch, QueueHandle, delegate_noop};
use wayland_protocols_wlr::layer_shell::v1::client::{zwlr_layer_shell_v1, zwlr_layer_surface_v1};
use whispers::branding;

// --- Layout ---
const NUM_BARS: usize = 21;
const BAR_WIDTH: u32 = 4;
const BAR_GAP: u32 = 4;
const BAR_MIN_HEIGHT: f32 = 5.0;
const BAR_MAX_HEIGHT: f32 = 24.0;
const TRACK_HEIGHT: u32 = 34;
const TRACK_BAR_PAD_X: u32 = 8;
const STATUS_RADIUS: u32 = 5;
const STATUS_X_OFFSET: u32 = 28;
const TRACK_X_OFFSET: u32 = 54;
const PILL_WIDTH: u32 = 248;
const PILL_HEIGHT: u32 = 58;
const PILL_RADIUS: u32 = 29;
const OSD_WIDTH: u32 = 276;
const OSD_HEIGHT: u32 = 82;
const MARGIN_BOTTOM: i32 = 26;
const BORDER_WIDTH: u32 = 1;
const SHADOW_SPREAD: u32 = 8;
const RISE_RATE: f32 = 0.48;
const DECAY_RATE: f32 = 0.84;

// --- Animation ---
const FPS: i32 = 30;
const FRAME_MS: i32 = 1000 / FPS;

// --- Colors ---
const BG_R: u8 = 12;
const BG_G: u8 = 16;
const BG_B: u8 = 22;
const BG_A: u8 = 224;

const BORDER_R: u8 = 236;
const BORDER_G: u8 = 242;
const BORDER_B: u8 = 255;
const BORDER_A: u8 = 28;

const TRACK_R: u8 = 26;
const TRACK_G: u8 = 33;
const TRACK_B: u8 = 44;
const TRACK_A: u8 = 214;

const STATUS_R: u8 = 108;
const STATUS_G: u8 = 236;
const STATUS_B: u8 = 196;

const BAR_EDGE_R: f32 = 132.0;
const BAR_EDGE_G: f32 = 179.0;
const BAR_EDGE_B: f32 = 230.0;
const BAR_CENTER_R: f32 = 235.0;
const BAR_CENTER_G: f32 = 245.0;
const BAR_CENTER_B: f32 = 255.0;

static SHOULD_EXIT: AtomicBool = AtomicBool::new(false);

// --- Audio state (shared with capture thread) ---
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

// --- Bar animation state ---
struct BarState {
    heights: [f32; NUM_BARS],
}

impl BarState {
    fn new() -> Self {
        Self {
            heights: [BAR_MIN_HEIGHT; NUM_BARS],
        }
    }

    fn update(&mut self, rms: f32, time: f32) {
        let level = (rms * 4.8).min(1.0);
        let center = (NUM_BARS.saturating_sub(1)) as f32 / 2.0;
        let idle = 0.14 + (time * 1.05).sin().abs() * 0.05;

        for i in 0..NUM_BARS {
            let offset = ((i as f32 - center) / center.max(1.0)).abs();
            let envelope = 0.28 + (1.0 - offset.powf(1.6)) * 0.72;
            let t = i as f32 / NUM_BARS as f32;
            let wave1 = (t * std::f32::consts::PI * 2.3 + time * 2.6).sin() * 0.5 + 0.5;
            let wave2 = (t * std::f32::consts::PI * 4.6 - time * 1.35).sin() * 0.5 + 0.5;
            let wave3 = (t * std::f32::consts::PI * 7.4 + time * 4.8).sin() * 0.5 + 0.5;
            let motion = wave1 * 0.42 + wave2 * 0.33 + wave3 * 0.25;
            let combined = envelope * (idle + level * motion);
            let target = BAR_MIN_HEIGHT + combined.min(1.0) * (BAR_MAX_HEIGHT - BAR_MIN_HEIGHT);

            // Smooth: fast rise, slow decay
            if target > self.heights[i] {
                self.heights[i] += (target - self.heights[i]) * RISE_RATE;
            } else {
                self.heights[i] = self.heights[i] * DECAY_RATE + target * (1.0 - DECAY_RATE);
            }
            self.heights[i] = self.heights[i].clamp(BAR_MIN_HEIGHT, BAR_MAX_HEIGHT);
        }
    }
}

#[derive(Clone, Copy)]
struct Layout {
    pill_x: u32,
    pill_y: u32,
    pill_width: u32,
    pill_height: u32,
    status_x: u32,
    status_y: u32,
    track_x: u32,
    track_y: u32,
    track_width: u32,
    track_height: u32,
    wave_x: u32,
    wave_y: u32,
    wave_height: u32,
}

impl Layout {
    fn new(canvas_width: u32, canvas_height: u32) -> Self {
        let pill_width = PILL_WIDTH.min(canvas_width.saturating_sub(16));
        let pill_height = PILL_HEIGHT.min(canvas_height.saturating_sub(16));
        let pill_x = (canvas_width.saturating_sub(pill_width)) / 2;
        let pill_y = (canvas_height.saturating_sub(pill_height)) / 2;
        let wave_width =
            NUM_BARS as u32 * BAR_WIDTH + (NUM_BARS.saturating_sub(1) as u32) * BAR_GAP;
        let track_width = wave_width + TRACK_BAR_PAD_X * 2;
        let track_height = TRACK_HEIGHT.min(pill_height.saturating_sub(10));
        let track_x = pill_x + TRACK_X_OFFSET.min(pill_width.saturating_sub(track_width + 8));
        let track_y = pill_y + (pill_height.saturating_sub(track_height)) / 2;
        let wave_x = track_x + TRACK_BAR_PAD_X;
        let wave_y = track_y + (track_height.saturating_sub(BAR_MAX_HEIGHT as u32)) / 2;

        Self {
            pill_x,
            pill_y,
            pill_width,
            pill_height,
            status_x: pill_x + STATUS_X_OFFSET.min(pill_width.saturating_sub(STATUS_RADIUS + 8)),
            status_y: pill_y + pill_height / 2,
            track_x,
            track_y,
            track_width,
            track_height,
            wave_x,
            wave_y,
            wave_height: BAR_MAX_HEIGHT as u32,
        }
    }
}

// --- Wayland state ---
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

    let _ = std::fs::write(pid_file_path(), std::process::id().to_string());

    // Start audio capture for visualization
    let audio_level = Arc::new(AudioLevel::new());
    let _audio_stream = start_audio_capture(Arc::clone(&audio_level));

    // Wayland setup
    let conn = Connection::connect_to_env()?;
    let mut event_queue = conn.new_event_queue();
    let qh = event_queue.handle();

    conn.display().get_registry(&qh, ());

    let mut state = OsdState {
        running: true,
        width: OSD_WIDTH,
        height: OSD_HEIGHT,
        compositor: None,
        shm: None,
        layer_shell: None,
        surface: None,
        layer_surface: None,
        buffer: None,
        configured: false,
    };

    event_queue.roundtrip(&mut state)?;

    // Create layer surface
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

    layer_surface.set_size(OSD_WIDTH, OSD_HEIGHT);
    layer_surface.set_anchor(zwlr_layer_surface_v1::Anchor::Bottom);
    layer_surface.set_margin(0, 0, MARGIN_BOTTOM, 0);
    layer_surface.set_exclusive_zone(-1);
    layer_surface.set_keyboard_interactivity(zwlr_layer_surface_v1::KeyboardInteractivity::None);
    surface.commit();

    state.surface = Some(surface);
    state.layer_surface = Some(layer_surface);

    event_queue.roundtrip(&mut state)?;

    // Animation state
    let mut bars = BarState::new();
    let start_time = Instant::now();

    // Reusable pixel buffer (avoids alloc/dealloc per frame)
    let mut pixels = vec![0u8; (OSD_WIDTH * OSD_HEIGHT * 4) as usize];

    // Persistent shm pool: create memfd + pool once, reuse each frame
    let stride = OSD_WIDTH * 4;
    let shm_size = (stride * OSD_HEIGHT) as i32;
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

    // Main animation loop
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

        // Update animation
        let time = start_time.elapsed().as_secs_f32();
        let rms = audio_level.get();
        bars.update(rms, time);

        // Render frame into reusable buffer
        let w = state.width;
        let h = state.height;
        pixels.fill(0);
        render_frame(&mut pixels, w, h, &bars, time);

        // Present frame using persistent shm pool
        if let Err(e) = present_frame(&mut state, &qh, &pool, &shm_file, &pixels, w, h) {
            eprintln!("frame dropped: {e}");
        }
    }

    // Cleanup
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

// --- Audio capture ---

fn start_audio_capture(level: Arc<AudioLevel>) -> Option<cpal::Stream> {
    let host = cpal::default_host();
    let device = host.default_input_device()?;

    // Try to find a supported config at 16kHz, preferring mono then fewer channels
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
                // Downmix to mono if needed, then compute RMS
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

// --- Rendering ---

fn render_frame(pixels: &mut [u8], w: u32, h: u32, bars: &BarState, time: f32) {
    let layout = Layout::new(w, h);

    draw_shadow(pixels, w, h, &layout);
    draw_rounded_rect(
        pixels,
        w,
        h,
        layout.pill_x,
        layout.pill_y,
        layout.pill_width,
        layout.pill_height,
        PILL_RADIUS,
        BG_R,
        BG_G,
        BG_B,
        BG_A,
    );
    draw_rounded_border(
        pixels,
        w,
        h,
        layout.pill_x,
        layout.pill_y,
        layout.pill_width,
        layout.pill_height,
        PILL_RADIUS,
        BORDER_WIDTH,
        BORDER_R,
        BORDER_G,
        BORDER_B,
        BORDER_A,
    );

    for y in layout.pill_y + 2..layout.pill_y + layout.pill_height / 2 {
        let highlight = 18u8.saturating_sub(((y - layout.pill_y) * 2) as u8);
        for x in
            layout.pill_x + PILL_RADIUS / 2..layout.pill_x + layout.pill_width - PILL_RADIUS / 2
        {
            set_pixel_blend(pixels, w, h, x, y, 255, 255, 255, highlight);
        }
    }

    draw_rounded_rect(
        pixels,
        w,
        h,
        layout.track_x,
        layout.track_y,
        layout.track_width,
        layout.track_height,
        layout.track_height / 2,
        TRACK_R,
        TRACK_G,
        TRACK_B,
        TRACK_A,
    );
    draw_rounded_border(
        pixels,
        w,
        h,
        layout.track_x,
        layout.track_y,
        layout.track_width,
        layout.track_height,
        layout.track_height / 2,
        BORDER_WIDTH,
        255,
        255,
        255,
        14,
    );

    let connector_y = layout.status_y.saturating_sub(1);
    draw_rounded_rect(
        pixels,
        w,
        h,
        layout.status_x + STATUS_RADIUS + 7,
        connector_y,
        layout
            .track_x
            .saturating_sub(layout.status_x + STATUS_RADIUS + 13),
        2,
        1,
        121,
        147,
        173,
        52,
    );

    let status_glow = 18 + ((time * 3.1).sin().abs() * 10.0) as u32;
    draw_soft_circle(
        pixels,
        w,
        h,
        layout.status_x as i32,
        layout.status_y as i32,
        (STATUS_RADIUS + 8) as i32,
        STATUS_R,
        STATUS_G,
        STATUS_B,
        status_glow as u8,
    );
    draw_soft_circle(
        pixels,
        w,
        h,
        layout.status_x as i32,
        layout.status_y as i32,
        (STATUS_RADIUS + 3) as i32,
        STATUS_R,
        STATUS_G,
        STATUS_B,
        64,
    );
    draw_circle(
        pixels,
        w,
        h,
        layout.status_x as i32,
        layout.status_y as i32,
        STATUS_RADIUS as i32,
        STATUS_R,
        STATUS_G,
        STATUS_B,
        255,
    );
    draw_circle(
        pixels,
        w,
        h,
        layout.status_x as i32 - 1,
        layout.status_y as i32 - 1,
        2,
        236,
        255,
        247,
        224,
    );

    let center_y = layout.wave_y + layout.wave_height / 2;
    for i in 0..NUM_BARS {
        let bx = layout.wave_x + i as u32 * (BAR_WIDTH + BAR_GAP);
        let bar_h = bars.heights[i] as u32;
        let half_h = bar_h / 2;
        let top_y = center_y.saturating_sub(half_h);

        let center = (NUM_BARS.saturating_sub(1)) as f32 / 2.0;
        let distance = ((i as f32 - center) / center.max(1.0)).abs();
        let focus = 1.0 - distance.powf(1.5);
        let r = lerp(BAR_EDGE_R, BAR_CENTER_R, focus);
        let g = lerp(BAR_EDGE_G, BAR_CENTER_G, focus);
        let b = lerp(BAR_EDGE_B, BAR_CENTER_B, focus);
        let cr = r as u8;
        let cg = g as u8;
        let cb = b as u8;

        draw_rounded_rect(
            pixels,
            w,
            h,
            bx.saturating_sub(1),
            top_y.saturating_sub(1),
            BAR_WIDTH + 2,
            (bar_h + 2).min(layout.wave_height + 2),
            (BAR_WIDTH + 2) / 2,
            cr,
            cg,
            cb,
            34,
        );
        draw_rounded_rect(
            pixels,
            w,
            h,
            bx,
            top_y,
            BAR_WIDTH,
            bar_h.max(2),
            BAR_WIDTH / 2,
            cr,
            cg,
            cb,
            228,
        );
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

    // Destroy previous buffer
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

// --- Drawing primitives ---

fn draw_shadow(pixels: &mut [u8], w: u32, h: u32, layout: &Layout) {
    for i in (1..=SHADOW_SPREAD).rev() {
        let spread = i;
        let alpha = 4 + (SHADOW_SPREAD - i) as u8 * 3;
        draw_rounded_rect(
            pixels,
            w,
            h,
            layout.pill_x.saturating_sub(spread),
            layout.pill_y + spread / 2,
            layout.pill_width + spread * 2,
            layout.pill_height + spread,
            PILL_RADIUS + spread,
            3,
            6,
            10,
            alpha,
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn set_pixel_blend(pixels: &mut [u8], w: u32, h: u32, x: u32, y: u32, r: u8, g: u8, b: u8, a: u8) {
    if x >= w || y >= h || a == 0 {
        return;
    }
    let idx = ((y * w + x) * 4) as usize;
    if a == 255 {
        // Premultiplied: BGRA
        pixels[idx] = b;
        pixels[idx + 1] = g;
        pixels[idx + 2] = r;
        pixels[idx + 3] = 255;
        return;
    }
    let sa = a as u32;
    let inv = 255 - sa;
    // Premultiply source, blend with existing premultiplied dest
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
    x0: u32,
    y0: u32,
    rect_w: u32,
    rect_h: u32,
    radius: u32,
    thickness: u32,
    r: u8,
    g: u8,
    b: u8,
    a: u8,
) {
    for y in y0..y0 + rect_h {
        for x in x0..x0 + rect_w {
            let inside_outer = is_inside_rounded_rect(x - x0, y - y0, rect_w, rect_h, radius);
            let inside_inner = x >= x0 + thickness
                && y >= y0 + thickness
                && x < x0 + rect_w - thickness
                && y < y0 + rect_h - thickness
                && is_inside_rounded_rect(
                    x - x0 - thickness,
                    y - y0 - thickness,
                    rect_w - 2 * thickness,
                    rect_h - 2 * thickness,
                    radius.saturating_sub(thickness),
                );
            if inside_outer && !inside_inner {
                set_pixel_blend(pixels, w, h, x, y, r, g, b, a);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_circle(
    pixels: &mut [u8],
    w: u32,
    h: u32,
    cx: i32,
    cy: i32,
    radius: i32,
    r: u8,
    g: u8,
    b: u8,
    a: u8,
) {
    if radius <= 0 {
        return;
    }
    let radius_sq = radius * radius;
    for y in (cy - radius)..=(cy + radius) {
        for x in (cx - radius)..=(cx + radius) {
            let dx = x - cx;
            let dy = y - cy;
            if dx * dx + dy * dy <= radius_sq && x >= 0 && y >= 0 {
                set_pixel_blend(pixels, w, h, x as u32, y as u32, r, g, b, a);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_soft_circle(
    pixels: &mut [u8],
    w: u32,
    h: u32,
    cx: i32,
    cy: i32,
    radius: i32,
    r: u8,
    g: u8,
    b: u8,
    a: u8,
) {
    if radius <= 0 || a == 0 {
        return;
    }
    let radius_sq = (radius * radius) as f32;
    for y in (cy - radius)..=(cy + radius) {
        for x in (cx - radius)..=(cx + radius) {
            let dx = (x - cx) as f32;
            let dy = (y - cy) as f32;
            let dist_sq = dx * dx + dy * dy;
            if dist_sq <= radius_sq && x >= 0 && y >= 0 {
                let falloff = 1.0 - (dist_sq.sqrt() / radius as f32);
                let alpha = (a as f32 * falloff * falloff) as u8;
                set_pixel_blend(pixels, w, h, x as u32, y as u32, r, g, b, alpha);
            }
        }
    }
}

fn is_inside_rounded_rect(x: u32, y: u32, w: u32, h: u32, r: u32) -> bool {
    if r == 0 || w == 0 || h == 0 {
        return x < w && y < h;
    }
    // Check only corner regions
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
    a + (b - a) * t.clamp(0.0, 1.0)
}

// --- Dispatch implementations ---

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

#[cfg(test)]
mod tests {
    use super::{
        BAR_GAP, BAR_WIDTH, Layout, NUM_BARS, OSD_HEIGHT, OSD_WIDTH, STATUS_RADIUS, TRACK_BAR_PAD_X,
    };

    #[test]
    fn waveform_track_contains_all_bars() {
        let layout = Layout::new(OSD_WIDTH, OSD_HEIGHT);
        let bars_width = NUM_BARS as u32 * BAR_WIDTH + (NUM_BARS as u32 - 1) * BAR_GAP;
        assert_eq!(layout.wave_x, layout.track_x + TRACK_BAR_PAD_X);
        assert!(layout.wave_x + bars_width <= layout.track_x + layout.track_width);
    }

    #[test]
    fn status_indicator_stays_left_of_waveform() {
        let layout = Layout::new(OSD_WIDTH, OSD_HEIGHT);
        assert!(layout.status_x + STATUS_RADIUS + 8 < layout.track_x);
        assert!(layout.track_x + layout.track_width <= layout.pill_x + layout.pill_width);
    }
}

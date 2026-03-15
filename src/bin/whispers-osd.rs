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
const NUM_BARS: usize = 28;
const BAR_WIDTH: u32 = 3;
const BAR_GAP: u32 = 2;
const PAD_X: u32 = 10;
const PAD_Y: u32 = 8;
const BAR_MIN_HEIGHT: f32 = 2.0;
const BAR_MAX_HEIGHT: f32 = 30.0;
const OSD_WIDTH: u32 = PAD_X * 2 + NUM_BARS as u32 * BAR_WIDTH + (NUM_BARS as u32 - 1) * BAR_GAP;
const OSD_HEIGHT: u32 = BAR_MAX_HEIGHT as u32 + PAD_Y * 2;
const MARGIN_BOTTOM: i32 = 40;
const CORNER_RADIUS: u32 = 12;
const BORDER_WIDTH: u32 = 1;
const RISE_RATE: f32 = 0.55;
const DECAY_RATE: f32 = 0.88;

// --- Animation ---
const FPS: i32 = 30;
const FRAME_MS: i32 = 1000 / FPS;

// --- Colors ---
const BG_R: u8 = 18;
const BG_G: u8 = 18;
const BG_B: u8 = 30;
const BG_A: u8 = 185;

const BORDER_R: u8 = 140;
const BORDER_G: u8 = 180;
const BORDER_B: u8 = 255;
const BORDER_A: u8 = 40;

// Bar gradient: teal → violet
const BAR_LEFT_R: f32 = 0.0;
const BAR_LEFT_G: f32 = 0.82;
const BAR_LEFT_B: f32 = 0.75;
const BAR_RIGHT_R: f32 = 0.65;
const BAR_RIGHT_G: f32 = 0.35;
const BAR_RIGHT_B: f32 = 1.0;

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
        // Amplify RMS for visual impact
        let level = (rms * 5.0).min(1.0);

        for i in 0..NUM_BARS {
            let t = i as f32 / NUM_BARS as f32;
            // Create wave pattern across bars, driven by audio level
            let wave1 = (t * std::f32::consts::PI * 2.5 + time * 3.0).sin() * 0.5 + 0.5;
            let wave2 = (t * std::f32::consts::PI * 1.3 - time * 1.8).sin() * 0.3 + 0.5;
            let wave3 = (t * std::f32::consts::PI * 4.0 + time * 5.5).sin() * 0.2 + 0.5;

            let combined = (wave1 * 0.5 + wave2 * 0.3 + wave3 * 0.2) * level;
            let target = BAR_MIN_HEIGHT + combined * (BAR_MAX_HEIGHT - BAR_MIN_HEIGHT);

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

fn render_frame(pixels: &mut [u8], w: u32, h: u32, bars: &BarState, _time: f32) {
    // Glassmorphic background
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

    // Top highlight (glass reflection)
    for x in (CORNER_RADIUS + 2)..(w.saturating_sub(CORNER_RADIUS + 2)) {
        set_pixel_blend(pixels, w, h, x, 1, 255, 255, 255, 18);
    }

    // Visualizer bars
    let center_y = h / 2;
    for i in 0..NUM_BARS {
        let bx = PAD_X + i as u32 * (BAR_WIDTH + BAR_GAP);
        let bar_h = bars.heights[i] as u32;
        let half_h = bar_h / 2;
        let top_y = center_y.saturating_sub(half_h);

        let t = i as f32 / (NUM_BARS - 1) as f32;
        let r = lerp(BAR_LEFT_R, BAR_RIGHT_R, t);
        let g = lerp(BAR_LEFT_G, BAR_RIGHT_G, t);
        let b = lerp(BAR_LEFT_B, BAR_RIGHT_B, t);
        let cr = (r * 255.0) as u8;
        let cg = (g * 255.0) as u8;
        let cb = (b * 255.0) as u8;

        // Glow
        for gy in top_y.saturating_sub(2)..=(top_y + bar_h + 2).min(h - 1) {
            for gx in bx.saturating_sub(1)..=(bx + BAR_WIDTH).min(w - 1) {
                set_pixel_blend(pixels, w, h, gx, gy, cr, cg, cb, 25);
            }
        }

        // Bar body with vertical brightness gradient
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
    a + (b - a) * t
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

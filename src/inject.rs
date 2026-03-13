use std::process::{Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use evdev::uinput::VirtualDevice;
use evdev::{AttributeSet, EventType, InputEvent, KeyCode};

use crate::context::{SurfaceKind, TypingContext};
use crate::error::{Result, WhsprError};

const DEVICE_READY_DELAY: Duration = Duration::from_millis(45);
const PASTE_KEY_DELAY: Duration = Duration::from_millis(4);

static INJECT_DEVICE: OnceLock<Mutex<Option<VirtualDevice>>> = OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PasteShortcut {
    CtrlV,
    CtrlShiftV,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InjectionPolicy {
    paste_shortcut: PasteShortcut,
    surface_label: &'static str,
    backspace_key_delay: Duration,
    backspace_burst_len: usize,
    backspace_burst_pause: Duration,
    clipboard_ready_delay: Duration,
    post_delete_settle_delay: Duration,
    live_destructive_delete_limit: usize,
    destructive_correction_confirmations: usize,
}

pub struct TextInjector {
    wl_copy_bin: String,
    wl_copy_args: Vec<String>,
}

impl TextInjector {
    pub fn new() -> Self {
        Self {
            wl_copy_bin: "wl-copy".to_string(),
            wl_copy_args: Vec::new(),
        }
    }

    #[cfg(test)]
    fn with_wl_copy_command(bin: &str, args: &[&str]) -> Self {
        Self {
            wl_copy_bin: bin.to_string(),
            wl_copy_args: args.iter().map(|arg| (*arg).to_string()).collect(),
        }
    }

    pub async fn inject(&self, text: &str, context: &TypingContext) -> Result<()> {
        if text.is_empty() {
            tracing::warn!("empty text, nothing to inject");
            return Ok(());
        }

        let text = text.to_string();
        let text_len = text.len();
        let wl_copy_bin = self.wl_copy_bin.clone();
        let wl_copy_args = self.wl_copy_args.clone();
        let policy = InjectionPolicy::for_context(context);
        tokio::task::spawn_blocking(move || {
            inject_sync(&wl_copy_bin, &wl_copy_args, &text, policy)
        })
        .await
        .map_err(|e| WhsprError::Injection(format!("injection task panicked: {e}")))??;

        tracing::info!(
            paste_shortcut = policy.paste_shortcut().label(),
            surface_policy = policy.label(),
            "injected {} chars via wl-copy + paste shortcut",
            text_len
        );
        Ok(())
    }

    pub async fn replace_recent_text(
        &self,
        delete_graphemes: usize,
        text: &str,
        context: &TypingContext,
    ) -> Result<()> {
        if delete_graphemes == 0 {
            return self.inject(text, context).await;
        }

        let text = text.to_string();
        let wl_copy_bin = self.wl_copy_bin.clone();
        let wl_copy_args = self.wl_copy_args.clone();
        let policy = InjectionPolicy::for_context(context);
        tokio::task::spawn_blocking(move || {
            replace_recent_text_sync(&wl_copy_bin, &wl_copy_args, delete_graphemes, &text, policy)
        })
        .await
        .map_err(|e| WhsprError::Injection(format!("replace task panicked: {e}")))??;

        tracing::info!(
            delete_graphemes,
            paste_shortcut = policy.paste_shortcut().label(),
            surface_policy = policy.label(),
            "replaced graphemes via backspace + wl-copy paste"
        );
        Ok(())
    }
}

impl InjectionPolicy {
    pub(crate) fn for_context(context: &TypingContext) -> Self {
        match context.surface_kind {
            SurfaceKind::Terminal => Self {
                paste_shortcut: PasteShortcut::CtrlShiftV,
                surface_label: "terminal",
                backspace_key_delay: Duration::from_millis(2),
                backspace_burst_len: 48,
                backspace_burst_pause: Duration::from_millis(4),
                clipboard_ready_delay: Duration::from_millis(50),
                post_delete_settle_delay: Duration::from_millis(6),
                live_destructive_delete_limit: usize::MAX,
                destructive_correction_confirmations: 2,
            },
            SurfaceKind::Editor => Self {
                paste_shortcut: PasteShortcut::CtrlV,
                surface_label: "editor",
                backspace_key_delay: Duration::from_millis(3),
                backspace_burst_len: 32,
                backspace_burst_pause: Duration::from_millis(6),
                clipboard_ready_delay: Duration::from_millis(55),
                post_delete_settle_delay: Duration::from_millis(8),
                live_destructive_delete_limit: 24,
                destructive_correction_confirmations: 2,
            },
            SurfaceKind::Browser => Self {
                paste_shortcut: PasteShortcut::CtrlV,
                surface_label: "browser",
                backspace_key_delay: Duration::from_millis(5),
                backspace_burst_len: 16,
                backspace_burst_pause: Duration::from_millis(12),
                clipboard_ready_delay: Duration::from_millis(70),
                post_delete_settle_delay: Duration::from_millis(12),
                live_destructive_delete_limit: 12,
                destructive_correction_confirmations: 3,
            },
            SurfaceKind::GenericText => Self {
                paste_shortcut: PasteShortcut::CtrlV,
                surface_label: "generic_text",
                backspace_key_delay: Duration::from_millis(5),
                backspace_burst_len: 12,
                backspace_burst_pause: Duration::from_millis(14),
                clipboard_ready_delay: Duration::from_millis(75),
                post_delete_settle_delay: Duration::from_millis(14),
                live_destructive_delete_limit: 8,
                destructive_correction_confirmations: 2,
            },
            SurfaceKind::Unknown => Self {
                paste_shortcut: PasteShortcut::CtrlV,
                surface_label: "unknown",
                backspace_key_delay: Duration::from_millis(6),
                backspace_burst_len: 10,
                backspace_burst_pause: Duration::from_millis(16),
                clipboard_ready_delay: Duration::from_millis(80),
                post_delete_settle_delay: Duration::from_millis(16),
                live_destructive_delete_limit: 0,
                destructive_correction_confirmations: usize::MAX,
            },
        }
    }

    pub(crate) fn destructive_correction_confirmations(self) -> usize {
        self.destructive_correction_confirmations
    }

    pub(crate) fn allows_live_destructive_correction(self, delete_graphemes: usize) -> bool {
        self.live_destructive_delete_limit > 0
            && delete_graphemes <= self.live_destructive_delete_limit
    }

    pub(crate) fn label(self) -> &'static str {
        self.surface_label
    }

    fn paste_shortcut(self) -> PasteShortcut {
        self.paste_shortcut
    }
}

impl PasteShortcut {
    fn label(self) -> &'static str {
        match self {
            Self::CtrlV => "Ctrl+V",
            Self::CtrlShiftV => "Ctrl+Shift+V",
        }
    }
}

fn inject_sync(
    wl_copy_bin: &str,
    wl_copy_args: &[String],
    text: &str,
    policy: InjectionPolicy,
) -> Result<()> {
    with_virtual_device(|device, created| {
        if created {
            std::thread::sleep(DEVICE_READY_DELAY);
        }
        run_wl_copy(wl_copy_bin, wl_copy_args, text)?;
        std::thread::sleep(policy.clipboard_ready_delay);
        emit_paste_combo(device, policy.paste_shortcut())
    })
}

fn replace_recent_text_sync(
    wl_copy_bin: &str,
    wl_copy_args: &[String],
    delete_graphemes: usize,
    text: &str,
    policy: InjectionPolicy,
) -> Result<()> {
    with_virtual_device(|device, created| {
        if created {
            // The first use needs a short registration window so the compositor
            // doesn't miss the initial backspaces.
            std::thread::sleep(DEVICE_READY_DELAY);
        }
        emit_backspaces(device, delete_graphemes, policy)?;

        if !text.is_empty() {
            std::thread::sleep(policy.post_delete_settle_delay);
            run_wl_copy(wl_copy_bin, wl_copy_args, text)?;
            std::thread::sleep(policy.clipboard_ready_delay);
            emit_paste_combo(device, policy.paste_shortcut())?;
        }

        Ok(())
    })
}

fn with_virtual_device<T>(f: impl FnOnce(&mut VirtualDevice, bool) -> Result<T>) -> Result<T> {
    let device_store = INJECT_DEVICE.get_or_init(|| Mutex::new(None));
    let mut guard = device_store
        .lock()
        .map_err(|_| WhsprError::Injection("uinput device lock poisoned".into()))?;
    let created = if guard.is_none() {
        *guard = Some(build_virtual_device()?);
        true
    } else {
        false
    };
    let device = guard
        .as_mut()
        .ok_or_else(|| WhsprError::Injection("uinput device unavailable".into()))?;
    f(device, created)
}

fn build_virtual_device() -> Result<VirtualDevice> {
    let mut keys = AttributeSet::<KeyCode>::new();
    keys.insert(KeyCode::KEY_LEFTCTRL);
    keys.insert(KeyCode::KEY_LEFTSHIFT);
    keys.insert(KeyCode::KEY_BACKSPACE);
    keys.insert(KeyCode::KEY_V);

    VirtualDevice::builder()
        .map_err(|e| WhsprError::Injection(format!("uinput: {e}")))?
        .name("whispers-keyboard")
        .with_keys(&keys)
        .map_err(|e| WhsprError::Injection(format!("uinput keys: {e}")))?
        .build()
        .map_err(|e| WhsprError::Injection(format!("uinput build: {e}")))
}

fn run_wl_copy(wl_copy_bin: &str, wl_copy_args: &[String], text: &str) -> Result<()> {
    run_wl_copy_with_timeout(wl_copy_bin, wl_copy_args, text, Duration::from_secs(2))
}

fn run_wl_copy_with_timeout(
    wl_copy_bin: &str,
    wl_copy_args: &[String],
    text: &str,
    timeout: Duration,
) -> Result<()> {
    let mut wl_copy = Command::new(wl_copy_bin)
        .args(wl_copy_args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| WhsprError::Injection(format!("failed to spawn wl-copy: {e}")))?;

    {
        use std::io::Write;
        let mut stdin = wl_copy
            .stdin
            .take()
            .ok_or_else(|| WhsprError::Injection("wl-copy stdin unavailable".into()))?;
        stdin
            .write_all(text.as_bytes())
            .map_err(|e| WhsprError::Injection(format!("wl-copy stdin write: {e}")))?;
    }

    let deadline = std::time::Instant::now() + timeout;
    let status = loop {
        if let Some(status) = wl_copy
            .try_wait()
            .map_err(|e| WhsprError::Injection(format!("wl-copy wait: {e}")))?
        {
            break status;
        }
        if std::time::Instant::now() >= deadline {
            let _ = wl_copy.kill();
            let _ = wl_copy.wait();
            return Err(WhsprError::Injection(format!(
                "wl-copy timed out after {}ms",
                timeout.as_millis()
            )));
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    if !status.success() {
        return Err(WhsprError::Injection(format!(
            "wl-copy exited with {status}"
        )));
    }
    Ok(())
}

fn emit_paste_combo(device: &mut VirtualDevice, shortcut: PasteShortcut) -> Result<()> {
    let mut modifier_events = vec![InputEvent::new(
        EventType::KEY.0,
        KeyCode::KEY_LEFTCTRL.0,
        1,
    )];
    if matches!(shortcut, PasteShortcut::CtrlShiftV) {
        modifier_events.push(InputEvent::new(
            EventType::KEY.0,
            KeyCode::KEY_LEFTSHIFT.0,
            1,
        ));
    }
    device
        .emit(&modifier_events)
        .map_err(|e| WhsprError::Injection(format!("paste modifier press: {e}")))?;
    std::thread::sleep(PASTE_KEY_DELAY);

    device
        .emit(&[
            InputEvent::new(EventType::KEY.0, KeyCode::KEY_V.0, 1),
            InputEvent::new(EventType::KEY.0, KeyCode::KEY_V.0, 0),
        ])
        .map_err(|e| WhsprError::Injection(format!("paste key press: {e}")))?;
    std::thread::sleep(PASTE_KEY_DELAY);

    let mut release_events = Vec::new();
    if matches!(shortcut, PasteShortcut::CtrlShiftV) {
        release_events.push(InputEvent::new(
            EventType::KEY.0,
            KeyCode::KEY_LEFTSHIFT.0,
            0,
        ));
    }
    release_events.push(InputEvent::new(
        EventType::KEY.0,
        KeyCode::KEY_LEFTCTRL.0,
        0,
    ));
    device
        .emit(&release_events)
        .map_err(|e| WhsprError::Injection(format!("paste modifier release: {e}")))?;

    Ok(())
}

fn emit_backspaces(
    device: &mut VirtualDevice,
    count: usize,
    policy: InjectionPolicy,
) -> Result<()> {
    for index in 0..count {
        device
            .emit(&[
                InputEvent::new(EventType::KEY.0, KeyCode::KEY_BACKSPACE.0, 1),
                InputEvent::new(EventType::KEY.0, KeyCode::KEY_BACKSPACE.0, 0),
            ])
            .map_err(|e| WhsprError::Injection(format!("backspace key press: {e}")))?;
        std::thread::sleep(policy.backspace_key_delay);
        let next = index + 1;
        if next < count
            && next % policy.backspace_burst_len == 0
            && !policy.backspace_burst_pause.is_zero()
        {
            std::thread::sleep(policy.backspace_burst_pause);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::WhsprError;

    fn context(surface_kind: SurfaceKind) -> TypingContext {
        TypingContext {
            focus_fingerprint: "focus".into(),
            app_id: Some("app".into()),
            window_title: Some("window".into()),
            surface_kind,
            browser_domain: None,
            captured_at_ms: 0,
        }
    }

    #[test]
    fn run_wl_copy_reports_spawn_failure() {
        let err = run_wl_copy("/definitely/missing/wl-copy", &[], "hello")
            .expect_err("missing binary should fail");
        match err {
            WhsprError::Injection(msg) => {
                assert!(msg.contains("failed to spawn wl-copy"), "unexpected: {msg}");
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn run_wl_copy_reports_non_zero_exit() {
        let err = run_wl_copy(
            "/bin/sh",
            &[String::from("-c"), String::from("exit 7")],
            "hello",
        )
        .expect_err("non-zero exit should fail");
        match err {
            WhsprError::Injection(msg) => {
                assert!(msg.contains("wl-copy exited"), "unexpected: {msg}");
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn run_wl_copy_reports_timeout() {
        let err = run_wl_copy_with_timeout(
            "/bin/sh",
            &[String::from("-c"), String::from("sleep 1")],
            "hello",
            Duration::from_millis(80),
        )
        .expect_err("sleep should time out");
        match err {
            WhsprError::Injection(msg) => {
                assert!(msg.contains("timed out"), "unexpected: {msg}");
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[tokio::test]
    async fn inject_empty_text_is_noop() {
        let injector = TextInjector::with_wl_copy_command("/bin/true", &[]);
        injector
            .inject("", &TypingContext::unknown())
            .await
            .expect("empty text should no-op");
    }

    #[test]
    fn terminal_policy_uses_terminal_paste_shortcut() {
        let policy = InjectionPolicy::for_context(&context(SurfaceKind::Terminal));
        assert_eq!(policy.paste_shortcut(), PasteShortcut::CtrlShiftV);
        assert!(policy.allows_live_destructive_correction(64));
        assert_eq!(policy.destructive_correction_confirmations(), 2);
    }

    #[test]
    fn unknown_policy_disables_live_destructive_corrections() {
        let policy = InjectionPolicy::for_context(&context(SurfaceKind::Unknown));
        assert_eq!(policy.paste_shortcut(), PasteShortcut::CtrlV);
        assert!(!policy.allows_live_destructive_correction(1));
    }

    #[test]
    fn browser_policy_requires_more_confirmation_and_smaller_live_rewrites() {
        let policy = InjectionPolicy::for_context(&context(SurfaceKind::Browser));
        assert_eq!(policy.destructive_correction_confirmations(), 3);
        assert!(policy.allows_live_destructive_correction(12));
        assert!(!policy.allows_live_destructive_correction(13));
    }
}

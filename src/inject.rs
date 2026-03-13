use std::process::{Command, Stdio};
use std::time::Duration;

use evdev::uinput::VirtualDevice;
use evdev::{AttributeSet, EventType, InputEvent, KeyCode};

use crate::error::{Result, WhsprError};

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

    pub async fn inject(&self, text: &str) -> Result<()> {
        if text.is_empty() {
            tracing::warn!("empty text, nothing to inject");
            return Ok(());
        }

        let text = text.to_string();
        let text_len = text.len();
        let wl_copy_bin = self.wl_copy_bin.clone();
        let wl_copy_args = self.wl_copy_args.clone();
        tokio::task::spawn_blocking(move || inject_sync(&wl_copy_bin, &wl_copy_args, &text))
            .await
            .map_err(|e| WhsprError::Injection(format!("injection task panicked: {e}")))??;

        tracing::info!("injected {} chars via wl-copy + Ctrl+Shift+V", text_len);
        Ok(())
    }
}

fn inject_sync(wl_copy_bin: &str, wl_copy_args: &[String], text: &str) -> Result<()> {
    // Create uinput device early so it registers with the compositor
    // while wl-copy + clipboard delay run in parallel.
    let mut keys = AttributeSet::<KeyCode>::new();
    keys.insert(KeyCode::KEY_LEFTCTRL);
    keys.insert(KeyCode::KEY_LEFTSHIFT);
    keys.insert(KeyCode::KEY_V);

    let mut device = VirtualDevice::builder()
        .map_err(|e| WhsprError::Injection(format!("uinput: {e}")))?
        .name(crate::branding::UINPUT_KEYBOARD_NAME)
        .with_keys(&keys)
        .map_err(|e| WhsprError::Injection(format!("uinput keys: {e}")))?
        .build()
        .map_err(|e| WhsprError::Injection(format!("uinput build: {e}")))?;

    run_wl_copy(wl_copy_bin, wl_copy_args, text)?;

    // Wait for compositor to process the clipboard offer.
    // The uinput device was created above, so it has already been
    // registering during the wl-copy write.
    std::thread::sleep(Duration::from_millis(180));
    emit_paste_combo(&mut device)?;

    Ok(())
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

fn emit_paste_combo(device: &mut VirtualDevice) -> Result<()> {
    device
        .emit(&[
            InputEvent::new(EventType::KEY.0, KeyCode::KEY_LEFTCTRL.0, 1),
            InputEvent::new(EventType::KEY.0, KeyCode::KEY_LEFTSHIFT.0, 1),
        ])
        .map_err(|e| WhsprError::Injection(format!("paste modifier press: {e}")))?;
    std::thread::sleep(Duration::from_millis(12));

    device
        .emit(&[
            InputEvent::new(EventType::KEY.0, KeyCode::KEY_V.0, 1),
            InputEvent::new(EventType::KEY.0, KeyCode::KEY_V.0, 0),
        ])
        .map_err(|e| WhsprError::Injection(format!("paste key press: {e}")))?;
    std::thread::sleep(Duration::from_millis(12));

    device
        .emit(&[
            InputEvent::new(EventType::KEY.0, KeyCode::KEY_LEFTSHIFT.0, 0),
            InputEvent::new(EventType::KEY.0, KeyCode::KEY_LEFTCTRL.0, 0),
        ])
        .map_err(|e| WhsprError::Injection(format!("paste modifier release: {e}")))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::WhsprError;

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
        injector.inject("").await.expect("empty text should no-op");
    }
}

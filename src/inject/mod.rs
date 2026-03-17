mod clipboard;
mod keyboard;

#[cfg(test)]
mod tests;

use crate::error::{Result, WhsprError};

const DEVICE_READY_DELAY: std::time::Duration = std::time::Duration::from_millis(120);
const CLIPBOARD_READY_DELAY: std::time::Duration = std::time::Duration::from_millis(180);
const POST_DELETE_SETTLE_DELAY: std::time::Duration = std::time::Duration::from_millis(30);

pub struct TextInjector {
    wl_copy_bin: String,
    wl_copy_args: Vec<String>,
}

impl Default for TextInjector {
    fn default() -> Self {
        Self::new()
    }
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

    pub async fn replace_recent_text(&self, delete_graphemes: usize, text: &str) -> Result<()> {
        if delete_graphemes == 0 {
            return self.inject(text).await;
        }

        let text = text.to_string();
        let wl_copy_bin = self.wl_copy_bin.clone();
        let wl_copy_args = self.wl_copy_args.clone();
        tokio::task::spawn_blocking(move || {
            replace_recent_text_sync(&wl_copy_bin, &wl_copy_args, delete_graphemes, &text)
        })
        .await
        .map_err(|e| WhsprError::Injection(format!("replace task panicked: {e}")))??;

        tracing::info!(
            "replaced {} graphemes via backspace + wl-copy paste",
            delete_graphemes
        );
        Ok(())
    }
}

fn inject_sync(wl_copy_bin: &str, wl_copy_args: &[String], text: &str) -> Result<()> {
    let mut keyboard = keyboard::VirtualKeyboardAdapter::new()?;
    let clipboard = clipboard::ClipboardAdapter::new(wl_copy_bin, wl_copy_args);

    clipboard.copy(text)?;

    // Wait for compositor to process the clipboard offer.
    // The uinput device was created above, so it has already been
    // registering during the wl-copy write.
    std::thread::sleep(CLIPBOARD_READY_DELAY);
    keyboard.emit_paste_combo()?;

    Ok(())
}

fn replace_recent_text_sync(
    wl_copy_bin: &str,
    wl_copy_args: &[String],
    delete_graphemes: usize,
    text: &str,
) -> Result<()> {
    let mut keyboard = keyboard::VirtualKeyboardAdapter::new()?;
    let clipboard = clipboard::ClipboardAdapter::new(wl_copy_bin, wl_copy_args);

    // Unlike plain injection, replacement can try to backspace immediately
    // after creating the uinput device. Give the compositor a moment to
    // register it first so the initial backspaces are not dropped.
    std::thread::sleep(DEVICE_READY_DELAY);
    keyboard.emit_backspaces(delete_graphemes)?;

    if !text.is_empty() {
        std::thread::sleep(POST_DELETE_SETTLE_DELAY);
        clipboard.copy(text)?;
        std::thread::sleep(CLIPBOARD_READY_DELAY);
        keyboard.emit_paste_combo()?;
    }

    Ok(())
}

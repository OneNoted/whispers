use std::process::{Command, Stdio};
use std::time::Duration;

use crate::error::{Result, WhsprError};
use crate::runtime_guards::wait_child_with_timeout;

pub(super) struct ClipboardAdapter<'a> {
    wl_copy_bin: &'a str,
    wl_copy_args: &'a [String],
}

impl<'a> ClipboardAdapter<'a> {
    pub(super) fn new(wl_copy_bin: &'a str, wl_copy_args: &'a [String]) -> Self {
        Self {
            wl_copy_bin,
            wl_copy_args,
        }
    }

    pub(super) fn copy(&self, text: &str) -> Result<()> {
        run_wl_copy_with_timeout(
            self.wl_copy_bin,
            self.wl_copy_args,
            text,
            Duration::from_secs(2),
        )
    }
}

#[cfg(test)]
pub(super) fn run_wl_copy(wl_copy_bin: &str, wl_copy_args: &[String], text: &str) -> Result<()> {
    run_wl_copy_with_timeout(wl_copy_bin, wl_copy_args, text, Duration::from_secs(2))
}

pub(super) fn run_wl_copy_with_timeout(
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

    let mut saw_broken_pipe = false;
    {
        use std::io::Write;
        let mut stdin = wl_copy
            .stdin
            .take()
            .ok_or_else(|| WhsprError::Injection("wl-copy stdin unavailable".into()))?;
        if let Err(err) = stdin.write_all(text.as_bytes()) {
            if err.kind() == std::io::ErrorKind::BrokenPipe {
                saw_broken_pipe = true;
            } else {
                return Err(WhsprError::Injection(format!(
                    "wl-copy stdin write: {e}",
                    e = err
                )));
            }
        }
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
            let _ = wait_child_with_timeout(&mut wl_copy, Duration::from_millis(100));
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
    if saw_broken_pipe {
        return Err(WhsprError::Injection(
            "wl-copy closed stdin before consuming the clipboard payload".into(),
        ));
    }
    Ok(())
}

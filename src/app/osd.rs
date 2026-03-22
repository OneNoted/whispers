use std::process::Child;
use std::time::Duration;

#[cfg(feature = "osd")]
use std::process::Command;

use crate::runtime_guards::wait_child_with_timeout;

const OSD_SHUTDOWN_TIMEOUT: Duration = Duration::from_millis(250);

#[cfg(feature = "osd")]
pub(super) fn spawn_osd() -> Option<Child> {
    // Look for whispers-osd next to our own binary first, then fall back to PATH
    let osd_path = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|dir| dir.join("whispers-osd")))
        .filter(|p| p.exists())
        .unwrap_or_else(|| "whispers-osd".into());

    match Command::new(&osd_path).spawn() {
        Ok(child) => {
            tracing::debug!("spawned whispers-osd (pid {})", child.id());
            Some(child)
        }
        Err(e) => {
            tracing::warn!(
                "failed to spawn whispers-osd from {}: {e}",
                osd_path.display()
            );
            None
        }
    }
}

#[cfg(not(feature = "osd"))]
pub(super) fn spawn_osd() -> Option<Child> {
    None
}

pub(super) fn kill_osd(child: &mut Option<Child>) {
    if let Some(mut c) = child.take() {
        let pid = c.id() as libc::pid_t;
        unsafe {
            libc::kill(pid, libc::SIGTERM);
        }
        match wait_child_with_timeout(&mut c, OSD_SHUTDOWN_TIMEOUT) {
            Ok(Some(_)) => {
                tracing::debug!("whispers-osd (pid {pid}) terminated");
            }
            Ok(None) => {
                unsafe {
                    libc::kill(pid, libc::SIGKILL);
                }
                let _ = wait_child_with_timeout(&mut c, OSD_SHUTDOWN_TIMEOUT);
                tracing::warn!("whispers-osd (pid {pid}) did not exit after SIGTERM; killed");
            }
            Err(err) => {
                tracing::warn!("failed to wait for whispers-osd (pid {pid}) to exit: {err}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::kill_osd;
    use std::process::Child;

    #[test]
    fn kill_osd_none_is_noop() {
        let mut child: Option<Child> = None;
        kill_osd(&mut child);
        assert!(child.is_none());
    }
}

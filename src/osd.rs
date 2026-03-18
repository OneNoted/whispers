#[cfg(feature = "osd")]
use std::io::Write;
use std::process::Child;

#[cfg(feature = "osd")]
use std::process::{ChildStdin, Command, Stdio};

#[cfg(feature = "osd")]
use crate::branding;
#[cfg(feature = "osd")]
use crate::osd_protocol::OsdEvent;

pub enum OsdMode {
    Meter,
    Voice,
}

pub struct OsdHandle {
    child: Option<Child>,
    #[cfg(feature = "osd")]
    stdin: Option<ChildStdin>,
}

impl OsdHandle {
    pub fn spawn(mode: OsdMode) -> Self {
        #[cfg(feature = "osd")]
        {
            let osd_path = branding::resolve_sidecar_executable(&[branding::OSD_BINARY]);
            let mut command = Command::new(&osd_path);
            if matches!(mode, OsdMode::Voice) {
                command.arg("--voice").stdin(Stdio::piped());
            }

            match command.spawn() {
                Ok(mut child) => {
                    tracing::debug!("spawned whispers-osd (pid {})", child.id());
                    return Self {
                        stdin: child.stdin.take(),
                        child: Some(child),
                    };
                }
                Err(e) => {
                    tracing::warn!(
                        "failed to spawn whispers-osd from {}: {e}",
                        osd_path.display()
                    );
                }
            }
        }

        let _ = mode;
        Self {
            child: None,
            #[cfg(feature = "osd")]
            stdin: None,
        }
    }

    pub fn send_voice_update(&mut self, update: &crate::osd_protocol::VoiceOsdUpdate) {
        #[cfg(feature = "osd")]
        {
            let Some(stdin) = self.stdin.as_mut() else {
                return;
            };
            let Ok(payload) = serde_json::to_string(&OsdEvent::VoiceUpdate(update.clone())) else {
                tracing::warn!("failed to encode voice OSD update");
                return;
            };
            if let Err(err) = stdin.write_all(payload.as_bytes()) {
                tracing::warn!("failed to write voice OSD update: {err}");
                return;
            }
            if let Err(err) = stdin.write_all(b"\n") {
                tracing::warn!("failed to terminate voice OSD update: {err}");
                return;
            }
            if let Err(err) = stdin.flush() {
                tracing::warn!("failed to flush voice OSD update: {err}");
            }
        }

        let _ = update;
    }

    pub fn kill(&mut self) {
        if let Some(mut child) = self.child.take() {
            let pid = child.id() as libc::pid_t;
            unsafe {
                libc::kill(pid, libc::SIGTERM);
            }
            let _ = child.wait();
            tracing::debug!("whispers-osd (pid {pid}) terminated");
        }
    }
}

impl Drop for OsdHandle {
    fn drop(&mut self) {
        self.kill();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kill_without_child_is_noop() {
        let mut handle = OsdHandle {
            child: None,
            #[cfg(feature = "osd")]
            stdin: None,
        };
        handle.kill();
        assert!(handle.child.is_none());
    }
}

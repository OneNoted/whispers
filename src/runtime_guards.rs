use std::io::{self, Read};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::process::{Child, Command, ExitStatus, Output, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

const POLL_INTERVAL: Duration = Duration::from_millis(10);
const KILL_WAIT_TIMEOUT: Duration = Duration::from_millis(100);

#[derive(Debug)]
pub(crate) enum TimedTaskError {
    Spawn(io::Error),
    Timeout(Duration),
    Panic,
    ChannelClosed,
}

impl TimedTaskError {
    pub(crate) fn describe(&self, phase: &str) -> String {
        match self {
            Self::Spawn(err) => format!("failed to spawn {phase} thread: {err}"),
            Self::Timeout(timeout) => {
                format!("{phase} timed out after {}ms", timeout.as_millis())
            }
            Self::Panic => format!("{phase} thread panicked"),
            Self::ChannelClosed => format!("{phase} thread exited without a result"),
        }
    }
}

pub(crate) fn run_detached_with_timeout<T, F>(
    timeout: Duration,
    phase: &'static str,
    task: F,
) -> Result<T, TimedTaskError>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    let (tx, rx) = mpsc::sync_channel(1);
    thread::Builder::new()
        .name(format!("whispers-{}", phase.replace(' ', "-")))
        .spawn(move || {
            let result = catch_unwind(AssertUnwindSafe(task)).map_err(|_| TimedTaskError::Panic);
            let _ = tx.send(result);
        })
        .map_err(TimedTaskError::Spawn)?;

    match rx.recv_timeout(timeout) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(err)) => Err(err),
        Err(mpsc::RecvTimeoutError::Timeout) => Err(TimedTaskError::Timeout(timeout)),
        Err(mpsc::RecvTimeoutError::Disconnected) => Err(TimedTaskError::ChannelClosed),
    }
}

pub(crate) async fn run_detached_with_timeout_async<T, F>(
    timeout: Duration,
    phase: &'static str,
    task: F,
) -> Result<T, TimedTaskError>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    let (tx, rx) = tokio::sync::oneshot::channel();
    thread::Builder::new()
        .name(format!("whispers-{}", phase.replace(' ', "-")))
        .spawn(move || {
            let result = catch_unwind(AssertUnwindSafe(task)).map_err(|_| TimedTaskError::Panic);
            let _ = tx.send(result);
        })
        .map_err(TimedTaskError::Spawn)?;

    match tokio::time::timeout(timeout, rx).await {
        Ok(Ok(Ok(value))) => Ok(value),
        Ok(Ok(Err(err))) => Err(err),
        Ok(Err(_)) => Err(TimedTaskError::ChannelClosed),
        Err(_) => Err(TimedTaskError::Timeout(timeout)),
    }
}

pub(crate) fn wait_child_with_timeout(
    child: &mut Child,
    timeout: Duration,
) -> io::Result<Option<ExitStatus>> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Some(status));
        }
        if Instant::now() >= deadline {
            return Ok(None);
        }
        thread::sleep(POLL_INTERVAL.min(deadline.saturating_duration_since(Instant::now())));
    }
}

pub(crate) fn run_command_output_with_timeout(
    command: &mut Command,
    timeout: Duration,
) -> io::Result<Output> {
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn()?;
    let stdout_reader = spawn_pipe_reader(child.stdout.take());
    let stderr_reader = spawn_pipe_reader(child.stderr.take());
    let status = match wait_child_with_timeout(&mut child, timeout)? {
        Some(status) => status,
        None => {
            let _ = child.kill();
            let _ = wait_child_with_timeout(&mut child, KILL_WAIT_TIMEOUT);
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!("command timed out after {}ms", timeout.as_millis()),
            ));
        }
    };

    let stdout = stdout_reader.join().unwrap_or_default();
    let stderr = stderr_reader.join().unwrap_or_default();
    Ok(Output {
        status,
        stdout,
        stderr,
    })
}

pub(crate) fn run_command_status_with_timeout(
    command: &mut Command,
    timeout: Duration,
) -> io::Result<ExitStatus> {
    let mut child = command.spawn()?;
    match wait_child_with_timeout(&mut child, timeout)? {
        Some(status) => Ok(status),
        None => {
            let _ = child.kill();
            let _ = wait_child_with_timeout(&mut child, KILL_WAIT_TIMEOUT);
            Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!("command timed out after {}ms", timeout.as_millis()),
            ))
        }
    }
}

fn spawn_pipe_reader<R>(reader: Option<R>) -> thread::JoinHandle<Vec<u8>>
where
    R: Read + Send + 'static,
{
    thread::spawn(move || {
        let mut bytes = Vec::new();
        if let Some(mut reader) = reader {
            let _ = reader.read_to_end(&mut bytes);
        }
        bytes
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detached_timeout_returns_result() {
        let value =
            run_detached_with_timeout(Duration::from_millis(20), "test task", || 7).expect("ok");
        assert_eq!(value, 7);
    }

    #[test]
    fn detached_timeout_returns_timeout_error() {
        let err = run_detached_with_timeout(Duration::from_millis(10), "test task", || {
            thread::sleep(Duration::from_millis(50));
        })
        .expect_err("timeout");
        assert_eq!(err.describe("test task"), "test task timed out after 10ms");
    }

    #[tokio::test]
    async fn detached_async_timeout_returns_result() {
        let value = run_detached_with_timeout_async(Duration::from_millis(20), "async task", || 9)
            .await
            .expect("ok");
        assert_eq!(value, 9);
    }

    #[test]
    fn command_output_timeout_captures_successful_output() {
        let mut command = Command::new("/bin/sh");
        command.args(["-c", "printf 'hello'; printf 'warn' >&2"]);
        let output =
            run_command_output_with_timeout(&mut command, Duration::from_secs(1)).expect("output");
        assert!(output.status.success());
        assert_eq!(String::from_utf8_lossy(&output.stdout), "hello");
        assert_eq!(String::from_utf8_lossy(&output.stderr), "warn");
    }

    #[test]
    fn command_output_timeout_kills_hung_process() {
        let mut command = Command::new("/bin/sh");
        command.args(["-c", "sleep 5"]);
        let err = run_command_output_with_timeout(&mut command, Duration::from_millis(20))
            .expect_err("timeout");
        assert_eq!(err.kind(), io::ErrorKind::TimedOut);
    }

    #[test]
    fn command_status_timeout_returns_exit_status() {
        let mut command = Command::new("/bin/sh");
        command.args(["-c", "exit 7"]);
        let status =
            run_command_status_with_timeout(&mut command, Duration::from_secs(1)).expect("status");
        assert_eq!(status.code(), Some(7));
    }

    #[test]
    fn command_status_timeout_kills_hung_process() {
        let mut command = Command::new("/bin/sh");
        command.args(["-c", "sleep 5"]);
        let err = run_command_status_with_timeout(&mut command, Duration::from_millis(20))
            .expect_err("timeout");
        assert_eq!(err.kind(), io::ErrorKind::TimedOut);
    }
}

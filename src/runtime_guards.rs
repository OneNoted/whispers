use std::io::{self, Read};
#[cfg(unix)]
use std::os::fd::{AsRawFd, RawFd};
use std::process::{Child, Command, ExitStatus, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::process::CommandExt;

const POLL_INTERVAL: Duration = Duration::from_millis(10);
const KILL_WAIT_TIMEOUT: Duration = Duration::from_millis(100);

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
    configure_command_for_timeout(command);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn()?;

    #[cfg(unix)]
    {
        run_command_output_with_timeout_unix(&mut child, timeout)
    }

    #[cfg(not(unix))]
    {
        run_command_output_with_timeout_fallback(&mut child, timeout)
    }
}

#[cfg(unix)]
fn run_command_output_with_timeout_unix(
    child: &mut Child,
    timeout: Duration,
) -> io::Result<Output> {
    let mut stdout = child.stdout.take().map(NonBlockingPipe::new).transpose()?;
    let mut stderr = child.stderr.take().map(NonBlockingPipe::new).transpose()?;
    let deadline = Instant::now() + timeout;
    let mut cleanup_deadline = None::<Instant>;
    let mut status = None::<ExitStatus>;

    loop {
        if status.is_none() {
            status = child.try_wait()?;
        }

        if let Some(pipe) = stdout.as_mut() {
            pipe.drain()?;
        }
        if let Some(pipe) = stderr.as_mut() {
            pipe.drain()?;
        }

        let pipes_closed = stdout.as_ref().is_none_or(NonBlockingPipe::is_closed)
            && stderr.as_ref().is_none_or(NonBlockingPipe::is_closed);
        if let Some(status) = status
            && pipes_closed
        {
            if cleanup_deadline.is_some() {
                return timed_out_error(timeout);
            }
            return Ok(Output {
                status,
                stdout: stdout.map_or_else(Vec::new, NonBlockingPipe::into_bytes),
                stderr: stderr.map_or_else(Vec::new, NonBlockingPipe::into_bytes),
            });
        }

        let now = Instant::now();
        if cleanup_deadline.is_none() && now >= deadline {
            kill_child_with_descendants(child);
            let _ = wait_child_with_timeout(child, KILL_WAIT_TIMEOUT);
            cleanup_deadline = Some(now + KILL_WAIT_TIMEOUT);
        } else if cleanup_deadline.is_some_and(|limit| now >= limit) {
            return timed_out_error(timeout);
        }

        let sleep_until = cleanup_deadline.unwrap_or(deadline);
        thread::sleep(POLL_INTERVAL.min(sleep_until.saturating_duration_since(now)));
    }
}

#[cfg(not(unix))]
fn run_command_output_with_timeout_fallback(
    child: &mut Child,
    timeout: Duration,
) -> io::Result<Output> {
    let stdout_reader = spawn_pipe_reader(child.stdout.take());
    let stderr_reader = spawn_pipe_reader(child.stderr.take());
    let status = match wait_child_with_timeout(child, timeout)? {
        Some(status) => status,
        None => {
            kill_child_with_descendants(child);
            let _ = wait_child_with_timeout(child, KILL_WAIT_TIMEOUT);
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            return timed_out_error(timeout);
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
    configure_command_for_timeout(command);
    let mut child = command.spawn()?;
    match wait_child_with_timeout(&mut child, timeout)? {
        Some(status) => Ok(status),
        None => {
            kill_child_with_descendants(&mut child);
            let _ = wait_child_with_timeout(&mut child, KILL_WAIT_TIMEOUT);
            timed_out_error(timeout)
        }
    }
}

fn timed_out_error<T>(timeout: Duration) -> io::Result<T> {
    Err(io::Error::new(
        io::ErrorKind::TimedOut,
        format!("command timed out after {}ms", timeout.as_millis()),
    ))
}

#[cfg(unix)]
struct NonBlockingPipe<R> {
    reader: R,
    bytes: Vec<u8>,
    closed: bool,
}

#[cfg(unix)]
impl<R> NonBlockingPipe<R>
where
    R: Read + AsRawFd,
{
    fn new(reader: R) -> io::Result<Self> {
        set_nonblocking(reader.as_raw_fd())?;
        Ok(Self {
            reader,
            bytes: Vec::new(),
            closed: false,
        })
    }

    fn drain(&mut self) -> io::Result<()> {
        if self.closed {
            return Ok(());
        }

        let mut chunk = [0_u8; 8192];
        loop {
            match self.reader.read(&mut chunk) {
                Ok(0) => {
                    self.closed = true;
                    return Ok(());
                }
                Ok(read) => self.bytes.extend_from_slice(&chunk[..read]),
                Err(err) if err.kind() == io::ErrorKind::WouldBlock => return Ok(()),
                Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
                Err(err) => return Err(err),
            }
        }
    }

    fn is_closed(&self) -> bool {
        self.closed
    }

    fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

#[cfg(unix)]
fn set_nonblocking(fd: RawFd) -> io::Result<()> {
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags == -1 {
        return Err(io::Error::last_os_error());
    }

    let result = unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) };
    if result == -1 {
        return Err(io::Error::last_os_error());
    }

    Ok(())
}

#[cfg(not(unix))]
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

fn configure_command_for_timeout(command: &mut Command) {
    #[cfg(unix)]
    command.process_group(0);
}

fn kill_child_with_descendants(child: &mut Child) {
    #[cfg(unix)]
    {
        // Timed commands are spawned into their own process group so shell
        // wrappers and background descendants cannot keep inherited pipes open
        // past the timeout.
        let _ = unsafe { libc::killpg(child.id() as i32, libc::SIGKILL) };
    }

    #[cfg(not(unix))]
    {
        let _ = child.kill();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::unique_temp_dir;

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
        command.args(["-c", "/bin/sleep 5"]);
        let err = run_command_output_with_timeout(&mut command, Duration::from_millis(20))
            .expect_err("timeout");
        assert_eq!(err.kind(), io::ErrorKind::TimedOut);
    }

    #[test]
    fn command_output_timeout_kills_shell_descendants_holding_pipes_open() {
        let temp_dir = unique_temp_dir("runtime-guards-output-timeout");
        let pid_path = temp_dir.join("child.pid");
        let script = format!("/bin/sleep 5 & echo $! > '{}' ; wait", pid_path.display());
        let mut command = Command::new("/bin/sh");
        command.args(["-c", &script]);

        let started = Instant::now();
        let err = run_command_output_with_timeout(&mut command, Duration::from_millis(20))
            .expect_err("timeout");

        assert_eq!(err.kind(), io::ErrorKind::TimedOut);
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "timeout path should return promptly"
        );
        let child_pid = std::fs::read_to_string(&pid_path)
            .expect("pid file")
            .trim()
            .parse::<i32>()
            .expect("pid should parse");
        assert!(
            process_is_gone(child_pid),
            "timed command descendants should be terminated"
        );
    }

    #[test]
    fn command_output_timeout_applies_while_draining_pipes() {
        let temp_dir = unique_temp_dir("runtime-guards-output-drain-timeout");
        let pid_path = temp_dir.join("child.pid");
        let script = format!("/bin/sleep 5 & echo $! > '{}' ; exit 0", pid_path.display());
        let mut command = Command::new("/bin/sh");
        command.args(["-c", &script]);

        let started = Instant::now();
        let err = run_command_output_with_timeout(&mut command, Duration::from_millis(20))
            .expect_err("timeout");

        assert_eq!(err.kind(), io::ErrorKind::TimedOut);
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "drain timeout should return promptly"
        );
        let child_pid = std::fs::read_to_string(&pid_path)
            .expect("pid file")
            .trim()
            .parse::<i32>()
            .expect("pid should parse");
        assert!(
            process_is_gone(child_pid),
            "drain timeout should terminate descendants holding inherited pipes"
        );
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
        command.args(["-c", "/bin/sleep 5"]);
        let err = run_command_status_with_timeout(&mut command, Duration::from_millis(20))
            .expect_err("timeout");
        assert_eq!(err.kind(), io::ErrorKind::TimedOut);
    }

    #[test]
    fn command_status_timeout_kills_shell_descendants() {
        let temp_dir = unique_temp_dir("runtime-guards-status-timeout");
        let pid_path = temp_dir.join("child.pid");
        let script = format!("/bin/sleep 5 & echo $! > '{}' ; wait", pid_path.display());
        let mut command = Command::new("/bin/sh");
        command.args(["-c", &script]);

        let err = run_command_status_with_timeout(&mut command, Duration::from_millis(20))
            .expect_err("timeout");

        assert_eq!(err.kind(), io::ErrorKind::TimedOut);
        let child_pid = std::fs::read_to_string(&pid_path)
            .expect("pid file")
            .trim()
            .parse::<i32>()
            .expect("pid should parse");
        assert!(
            process_is_gone(child_pid),
            "timed command descendants should be terminated"
        );
    }

    fn process_is_gone(pid: i32) -> bool {
        #[cfg(unix)]
        {
            let result = unsafe { libc::kill(pid, 0) };
            if result == 0 {
                false
            } else {
                let err = io::Error::last_os_error();
                err.raw_os_error() == Some(libc::ESRCH)
            }
        }

        #[cfg(not(unix))]
        {
            let _ = pid;
            true
        }
    }
}

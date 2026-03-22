use std::io::{self, Read};
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
    let stdout_reader = spawn_pipe_reader(child.stdout.take());
    let stderr_reader = spawn_pipe_reader(child.stderr.take());
    let status = match wait_child_with_timeout(&mut child, timeout)? {
        Some(status) => status,
        None => {
            kill_child_with_descendants(&mut child);
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
    configure_command_for_timeout(command);
    let mut child = command.spawn()?;
    match wait_child_with_timeout(&mut child, timeout)? {
        Some(status) => Ok(status),
        None => {
            kill_child_with_descendants(&mut child);
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

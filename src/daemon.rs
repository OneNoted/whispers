use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use crate::config::Config;
use crate::pipeline::LocalPipeline;
use crate::protocol::{ControlRequest, ControlResponse};
use crate::service::Service;
use crate::{Error, Result, paths};

pub fn run() -> Result<()> {
    let runtime_dir = paths::runtime_dir()?;
    fs::create_dir_all(&runtime_dir)?;
    fs::set_permissions(&runtime_dir, fs::Permissions::from_mode(0o700))?;
    let socket = paths::control_socket()?;
    if socket.exists() {
        if UnixStream::connect(&socket).is_ok() {
            return Err(Error::message("the whispers service is already running"));
        }
        fs::remove_file(&socket)?;
    }
    let listener = UnixListener::bind(&socket)?;
    fs::set_permissions(&socket, fs::Permissions::from_mode(0o600))?;
    let _guard = SocketGuard(socket);
    let service = Service::new(LocalPipeline::new(Config::load(None)?));
    for connection in listener.incoming() {
        let mut connection = match connection {
            Ok(connection) => connection,
            Err(error) => {
                eprintln!("whispers: rejected control connection: {error}");
                continue;
            }
        };
        let request = match read_request(&connection) {
            Ok(request) => request,
            Err(error) => {
                eprintln!("whispers: ignored invalid control request: {error}");
                continue;
            }
        };
        let shutdown = request == ControlRequest::Shutdown;
        let response = if shutdown {
            ControlResponse {
                ok: true,
                state: "idle".into(),
                message: Some("service stopped".into()),
            }
        } else {
            service.handle(&request)
        };
        if let Err(error) = write_response(&mut connection, &response) {
            eprintln!("whispers: could not send control response: {error}");
        }
        if shutdown {
            break;
        }
    }
    Ok(())
}

pub fn request(request: &ControlRequest, autostart: bool) -> Result<ControlResponse> {
    let socket = paths::control_socket()?;
    match request_once(&socket, request) {
        Ok(response) => Ok(response),
        Err(first_error) if autostart => {
            start_daemon()?;
            let deadline = Instant::now() + Duration::from_secs(3);
            loop {
                match request_once(&socket, request) {
                    Ok(response) => return Ok(response),
                    Err(_) if Instant::now() < deadline => {
                        thread::sleep(Duration::from_millis(20));
                    }
                    Err(error) => {
                        return Err(Error::message(format!(
                            "service did not start: {error}; initial connection: {first_error}"
                        )));
                    }
                }
            }
        }
        Err(error) => Err(error),
    }
}

fn request_once(socket: &std::path::Path, request: &ControlRequest) -> Result<ControlResponse> {
    let mut stream = UnixStream::connect(socket)?;
    serde_json::to_writer(&mut stream, request)?;
    stream.write_all(b"\n")?;
    stream.flush()?;
    let mut payload = Vec::new();
    BufReader::new(stream)
        .take(64 * 1024 + 1)
        .read_until(b'\n', &mut payload)?;
    if payload.is_empty() {
        return Err(Error::message("service closed the control connection"));
    }
    if payload.len() > 64 * 1024 {
        return Err(Error::message("control response is too large"));
    }
    Ok(serde_json::from_slice(&payload)?)
}

fn start_daemon() -> Result<()> {
    Command::new(std::env::current_exe()?)
        .arg("daemon")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map(|_| ())
        .map_err(Into::into)
}

fn read_request(stream: &UnixStream) -> Result<ControlRequest> {
    let mut payload = Vec::new();
    BufReader::new(stream)
        .take(64 * 1024 + 1)
        .read_until(b'\n', &mut payload)?;
    if payload.len() > 64 * 1024 {
        return Err(Error::message("control request is too large"));
    }
    Ok(serde_json::from_slice(&payload)?)
}

fn write_response(stream: &mut UnixStream, response: &ControlResponse) -> Result<()> {
    serde_json::to_writer(&mut *stream, response)?;
    stream.write_all(b"\n")?;
    stream.flush()?;
    Ok(())
}

struct SocketGuard(std::path::PathBuf);

impl Drop for SocketGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_messages_are_line_delimited_json() {
        let mut bytes = Vec::new();
        assert!(serde_json::to_writer(&mut bytes, &ControlRequest::Status).is_ok());
        bytes.push(b'\n');
        assert_eq!(
            serde_json::from_slice::<ControlRequest>(&bytes).ok(),
            Some(ControlRequest::Status)
        );
    }
}

use std::process::ExitCode;

use whispers_next::protocol::ControlRequest;
use whispers_next::{cli, daemon, engine};

fn main() -> ExitCode {
    match cli::parse(std::env::args_os().skip(1)) {
        Ok(cli::Command::Help) => {
            print!("{}", cli::HELP);
            ExitCode::SUCCESS
        }
        Ok(cli::Command::Version) => {
            println!("whispers {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Ok(cli::Command::Toggle) => control(&ControlRequest::Toggle, true),
        Ok(cli::Command::Cancel) => control(&ControlRequest::Cancel, false),
        Ok(cli::Command::Status) => control(&ControlRequest::Status, false),
        Ok(cli::Command::Daemon) => result(daemon::run()),
        Ok(cli::Command::EngineHost { runtime }) => result(engine::run_host(&runtime)),
        Ok(command) => result(Err(whispers_next::Error::message(format!(
            "{command:?} is not wired yet"
        )))),
        Err(error) => {
            eprintln!("whispers: {error}");
            ExitCode::FAILURE
        }
    }
}

fn control(request: &ControlRequest, autostart: bool) -> ExitCode {
    match daemon::request(request, autostart) {
        Ok(response) => {
            if let Some(message) = response.message {
                println!("{message}");
            } else {
                println!("{}", response.state);
            }
            if response.ok {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(error) => {
            if matches!(request, ControlRequest::Status | ControlRequest::Cancel) {
                println!("idle");
                ExitCode::SUCCESS
            } else {
                eprintln!("whispers: {error}");
                ExitCode::FAILURE
            }
        }
    }
}

fn result(result: whispers_next::Result<()>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("whispers: {error}");
            ExitCode::FAILURE
        }
    }
}

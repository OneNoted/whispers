use std::process::ExitCode;

use whispers_next::cli;

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
        Ok(command) => {
            eprintln!("whispers: {command:?} is not wired yet");
            ExitCode::FAILURE
        }
        Err(error) => {
            eprintln!("whispers: {error}");
            ExitCode::FAILURE
        }
    }
}

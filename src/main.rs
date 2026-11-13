use std::process::ExitCode;

use whispers::protocol::ControlRequest;
use whispers::{cli, daemon, doctor, engine, setup};

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
        Ok(cli::Command::Setup {
            offline_pack,
            no_service,
        }) => {
            result(setup::run(offline_pack.as_deref(), !no_service).map(|report| {
                println!("runtime and model are ready");
                if report.migrated_config {
                    println!("migrated the legacy configuration (the original was backed up)");
                }
                println!("{} vocabulary entries", report.vocabulary_entries);
                if !no_service && !report.service_started {
                    println!("user service was installed but could not be started; toggle will still autostart it");
                }
            }))
        }
        Ok(cli::Command::Doctor) => doctor(),
        Ok(cli::Command::Vocabulary(command)) => vocabulary(command),
        Ok(cli::Command::Daemon) => result(daemon::run()),
        Ok(cli::Command::EngineHost { runtime }) => result(engine::run_host(&runtime)),
        Err(error) => {
            eprintln!("whispers: {error}");
            ExitCode::FAILURE
        }
    }
}

fn doctor() -> ExitCode {
    let report = doctor::run();
    for check in &report.checks {
        println!(
            "{:<4} {:<18} {}",
            check.level.label(),
            check.name,
            check.detail
        );
    }
    if report.healthy() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn vocabulary(command: cli::VocabularyCommand) -> ExitCode {
    let operation = (|| {
        let mut vocabulary = whispers::vocabulary::Vocabulary::load()?;
        match command {
            cli::VocabularyCommand::List => {
                for entry in &vocabulary.entries {
                    println!("{}\t{}", entry.term, entry.aliases.join(", "));
                }
            }
            cli::VocabularyCommand::Add { term, aliases } => {
                vocabulary.add(&term, aliases)?;
                vocabulary.save()?;
            }
            cli::VocabularyCommand::Remove { term } => {
                if !vocabulary.remove(&term) {
                    return Err(whispers::Error::message(format!(
                        "vocabulary term not found: {term}"
                    )));
                }
                vocabulary.save()?;
            }
        }
        Ok(())
    })();
    result(operation)
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

fn result(result: whispers::Result<()>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("whispers: {error}");
            ExitCode::FAILURE
        }
    }
}

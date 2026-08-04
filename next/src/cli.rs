use std::ffi::OsString;
use std::path::PathBuf;

use lexopt::prelude::*;

use crate::{Error, Result};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Command {
    Toggle,
    Cancel,
    Status,
    Setup { offline_pack: Option<PathBuf> },
    Doctor,
    Vocabulary(VocabularyCommand),
    Daemon,
    EngineHost { runtime: String },
    Help,
    Version,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VocabularyCommand {
    List,
    Add { term: String, aliases: Vec<String> },
    Remove { term: String },
}

pub fn parse<I, T>(args: I) -> Result<Command>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString>,
{
    let mut parser = lexopt::Parser::from_args(args);
    let Some(argument) = parser.next()? else {
        return Ok(Command::Toggle);
    };
    let command = match argument {
        Short('h') | Long("help") => Command::Help,
        Short('V') | Long("version") => Command::Version,
        Value(value) if value == "toggle" => Command::Toggle,
        Value(value) if value == "cancel" => Command::Cancel,
        Value(value) if value == "status" => Command::Status,
        Value(value) if value == "doctor" => Command::Doctor,
        Value(value) if value == "daemon" => Command::Daemon,
        Value(value) if value == "__engine-host" => {
            let runtime = parser.value()?.string()?;
            Command::EngineHost { runtime }
        }
        Value(value) if value == "setup" => {
            let mut offline_pack = None;
            while let Some(argument) = parser.next()? {
                match argument {
                    Long("runtime-pack") => offline_pack = Some(parser.value()?.into()),
                    _ => return Err(argument.unexpected().into()),
                }
            }
            Command::Setup { offline_pack }
        }
        Value(value) if value == "vocabulary" => parse_vocabulary(&mut parser)?,
        Value(value) => {
            return Err(Error::message(format!(
                "unknown command: {}",
                value.to_string_lossy()
            )));
        }
        other => return Err(other.unexpected().into()),
    };
    if parser.next()?.is_some() {
        return Err(Error::message("unexpected trailing arguments"));
    }
    Ok(command)
}

fn parse_vocabulary(parser: &mut lexopt::Parser) -> Result<Command> {
    let action = parser.value()?.string()?;
    let command = match action.as_str() {
        "list" => VocabularyCommand::List,
        "remove" => VocabularyCommand::Remove {
            term: parser.value()?.string()?,
        },
        "add" => {
            let term = parser.value()?.string()?;
            let mut aliases = Vec::new();
            while let Some(argument) = parser.next()? {
                match argument {
                    Long("alias") => aliases.push(parser.value()?.string()?),
                    _ => return Err(argument.unexpected().into()),
                }
            }
            if aliases.is_empty() {
                return Err(Error::message(
                    "vocabulary add requires at least one --alias",
                ));
            }
            VocabularyCommand::Add { term, aliases }
        }
        _ => {
            return Err(Error::message(format!(
                "unknown vocabulary command: {action}"
            )));
        }
    };
    Ok(Command::Vocabulary(command))
}

pub const HELP: &str = "whispers - intended-text dictation for Wayland

Usage:
  whispers                         Start or stop dictation
  whispers toggle                  Start or stop dictation
  whispers cancel                  Cancel the active dictation
  whispers status                  Show service state
  whispers setup [--runtime-pack PATH]
  whispers doctor
  whispers vocabulary list
  whispers vocabulary add TERM --alias SPOKEN [--alias SPOKEN ...]
  whispers vocabulary remove TERM
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_arguments_is_toggle() {
        assert_eq!(parse(Vec::<String>::new()).ok(), Some(Command::Toggle));
    }

    #[test]
    fn parses_vocabulary_aliases() {
        let command = parse([
            "vocabulary",
            "add",
            "serde_json",
            "--alias",
            "surdy json",
            "--alias",
            "serde json",
        ]);
        assert_eq!(
            command.ok(),
            Some(Command::Vocabulary(VocabularyCommand::Add {
                term: "serde_json".into(),
                aliases: vec!["surdy json".into(), "serde json".into()],
            }))
        );
    }
}

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser, Debug)]
#[command(
    name = "whispers",
    version,
    about = "Speech-to-text dictation tool for Wayland"
)]
pub struct Cli {
    /// Path to config file
    #[arg(short, long, global = true)]
    pub config: Option<PathBuf>,

    /// Increase log verbosity (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq)]
pub enum CompletionShell {
    Bash,
    Zsh,
    Fish,
    Nushell,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Interactive first-time setup wizard
    Setup,

    /// Transcribe an audio file (wav, mp3, flac, ogg, mp4/m4a)
    Transcribe {
        /// Path to the audio file
        file: PathBuf,

        /// Write output to a file instead of stdout
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Print the raw Whisper transcript without post-processing
        #[arg(long)]
        raw: bool,
    },

    /// Manage whisper models
    Model {
        #[command(subcommand)]
        action: ModelAction,
    },

    /// Manage advanced local rewrite models
    RewriteModel {
        #[command(subcommand)]
        action: RewriteModelAction,
    },

    /// Print shell completion script to stdout
    Completions {
        /// Shell name (auto-detected if omitted)
        shell: Option<CompletionShell>,
    },
}

#[derive(Subcommand, Debug)]
pub enum ModelAction {
    /// List available models and their status
    List,

    /// Download a model
    Download {
        /// Model name (e.g. large-v3-turbo, tiny, base)
        name: String,
    },

    /// Select a downloaded model as active
    Select {
        /// Model name to use
        name: String,
    },
}

#[derive(Subcommand, Debug)]
pub enum RewriteModelAction {
    /// List available rewrite models and their status
    List,

    /// Download a rewrite model
    Download {
        /// Model name (e.g. llama-3.2-3b-q4_k_m)
        name: String,
    },

    /// Select a downloaded rewrite model as active
    Select {
        /// Model name to use
        name: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_completions_without_shell() {
        let cli = Cli::try_parse_from(["whispers", "completions"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Completions { shell: None })
        ));
    }

    #[test]
    fn parses_completions_with_supported_shell() {
        let cli = Cli::try_parse_from(["whispers", "completions", "zsh"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Completions {
                shell: Some(CompletionShell::Zsh)
            })
        ));
    }

    #[test]
    fn rejects_invalid_completion_shell() {
        let err = Cli::try_parse_from(["whispers", "completions", "tcsh"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::InvalidValue);
    }

    #[test]
    fn parses_transcribe_raw_flag() {
        let cli = Cli::try_parse_from(["whispers", "transcribe", "clip.wav", "--raw"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Transcribe { raw: true, .. })
        ));
    }

    #[test]
    fn parses_rewrite_model_subcommand() {
        let cli = Cli::try_parse_from(["whispers", "rewrite-model", "list"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::RewriteModel {
                action: RewriteModelAction::List
            })
        ));
    }
}

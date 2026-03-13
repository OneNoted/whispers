use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser, Debug)]
#[command(
    name = "whispers",
    version,
    about = "Local-first speech-to-text dictation for Wayland"
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
    /// Guided setup for local, cloud, and experimental dictation paths
    Setup,

    /// Transcribe an audio file (wav, mp3, flac, ogg, mp4/m4a)
    Transcribe {
        /// Path to the audio file
        file: PathBuf,

        /// Write output to a file instead of stdout
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Print the raw transcription output without post-processing
        #[arg(long)]
        raw: bool,
    },

    /// Legacy whisper_cpp-only model commands (prefer `asr-model`)
    Model {
        #[command(subcommand)]
        action: ModelAction,
    },

    /// Manage ASR models across recommended and experimental backends
    AsrModel {
        #[command(subcommand)]
        action: AsrModelAction,
    },

    /// Manage local rewrite models used by advanced post-processing
    RewriteModel {
        #[command(subcommand)]
        action: RewriteModelAction,
    },

    /// Manage deterministic dictionary replacements
    Dictionary {
        #[command(subcommand)]
        action: DictionaryAction,
    },

    /// Check optional cloud ASR/rewrite configuration and connectivity
    Cloud {
        #[command(subcommand)]
        action: CloudAction,
    },

    /// Manage spoken snippets
    Snippets {
        #[command(subcommand)]
        action: SnippetAction,
    },

    /// Print the configured custom rewrite instructions file path
    RewriteInstructionsPath,

    /// Print shell completion script to stdout
    Completions {
        /// Shell name (auto-detected if omitted)
        shell: Option<CompletionShell>,
    },
}

#[derive(Subcommand, Debug)]
pub enum ModelAction {
    /// List legacy whisper_cpp models and their status
    List,

    /// Download a legacy whisper_cpp model
    Download {
        /// Model name (e.g. large-v3-turbo, tiny, base)
        name: String,
    },

    /// Select a downloaded legacy whisper_cpp model as active
    Select {
        /// Model name to use
        name: String,
    },
}

#[derive(Subcommand, Debug)]
pub enum AsrModelAction {
    /// List available ASR models, including recommended and experimental options
    List,

    /// Download an ASR model
    Download {
        /// Model name (e.g. large-v3-turbo, distil-large-v3.5, parakeet-tdt_ctc-1.1b)
        name: String,
    },

    /// Select a downloaded ASR model as active
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
        /// Model name (e.g. qwen-3.5-4b-q4_k_m)
        name: String,
    },

    /// Select a downloaded rewrite model as active
    Select {
        /// Model name to use
        name: String,
    },
}

#[derive(Subcommand, Debug)]
pub enum DictionaryAction {
    /// List dictionary entries
    List,

    /// Add or update a dictionary replacement
    Add {
        /// Phrase to match after normalization
        phrase: String,

        /// Replacement text to emit
        replace: String,
    },

    /// Remove a dictionary replacement by phrase
    Remove {
        /// Phrase to remove
        phrase: String,
    },
}

#[derive(Subcommand, Debug)]
pub enum SnippetAction {
    /// List snippets
    List,

    /// Add or update a snippet
    Add {
        /// Spoken snippet name used after the trigger phrase
        name: String,

        /// Text inserted when the snippet is expanded
        text: String,
    },

    /// Remove a snippet by name
    Remove {
        /// Snippet name to remove
        name: String,
    },
}

#[derive(Subcommand, Debug)]
pub enum CloudAction {
    /// Check cloud config, env vars, and provider connectivity
    Check,
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

    #[test]
    fn parses_asr_model_subcommand() {
        let cli = Cli::try_parse_from(["whispers", "asr-model", "list"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::AsrModel {
                action: AsrModelAction::List
            })
        ));
    }

    #[test]
    fn parses_dictionary_add_subcommand() {
        let cli = Cli::try_parse_from(["whispers", "dictionary", "add", "foo", "bar"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Dictionary {
                action: DictionaryAction::Add { .. }
            })
        ));
    }

    #[test]
    fn parses_snippet_add_subcommand() {
        let cli = Cli::try_parse_from(["whispers", "snippets", "add", "sig", "Best"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Snippets {
                action: SnippetAction::Add { .. }
            })
        ));
    }

    #[test]
    fn parses_cloud_check_subcommand() {
        let cli = Cli::try_parse_from(["whispers", "cloud", "check"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Cloud {
                action: CloudAction::Check
            })
        ));
    }
}

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

use crate::rewrite_protocol::{RewriteCorrectionPolicy, RewriteSurfaceKind};

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

impl CompletionShell {
    pub const fn all() -> [Self; 4] {
        [Self::Bash, Self::Zsh, Self::Fish, Self::Nushell]
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bash => "bash",
            Self::Zsh => "zsh",
            Self::Fish => "fish",
            Self::Nushell => "nushell",
        }
    }

    pub fn binary_names(self) -> &'static [&'static str] {
        match self {
            Self::Bash => &["bash"],
            Self::Zsh => &["zsh"],
            Self::Fish => &["fish"],
            Self::Nushell => &["nu", "nushell"],
        }
    }
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Guided setup for local, cloud, and experimental dictation paths
    Setup,

    /// Show the active configuration, selected models, and runtime options
    Status,

    /// Experimental live voice mode with partial transcription preview
    Voice,

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

    /// Manage app-aware rewrite policy rules
    AppRule {
        #[command(subcommand)]
        action: AppRuleAction,
    },

    /// Manage technical glossary entries for agentic rewrite
    Glossary {
        #[command(subcommand)]
        action: GlossaryAction,
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
pub enum AppRuleAction {
    /// Print the configured app rule file path
    Path,

    /// List configured app rules
    List,

    /// Add or update an app rule
    Add {
        /// Stable rule name used for updates and removals
        name: String,

        /// Instructions appended to the effective rewrite prompt
        instructions: String,

        /// Match on the active surface kind
        #[arg(long)]
        surface_kind: Option<RewriteSurfaceKind>,

        /// Match on the exact app ID
        #[arg(long)]
        app_id: Option<String>,

        /// Case-insensitive substring match on the window title
        #[arg(long)]
        window_title_contains: Option<String>,

        /// Case-insensitive substring match on the browser domain
        #[arg(long)]
        browser_domain_contains: Option<String>,

        /// Override the effective correction policy
        #[arg(long)]
        correction_policy: Option<RewriteCorrectionPolicy>,
    },

    /// Remove an app rule by name
    Remove {
        /// Rule name to remove
        name: String,
    },
}

#[derive(Subcommand, Debug)]
pub enum GlossaryAction {
    /// Print the configured glossary file path
    Path,

    /// List configured glossary entries
    List,

    /// Add or update a glossary entry
    Add {
        /// Canonical output term
        term: String,

        /// Alias that should map to the canonical term
        #[arg(long = "alias", required = true)]
        aliases: Vec<String>,

        /// Match on the active surface kind
        #[arg(long)]
        surface_kind: Option<RewriteSurfaceKind>,

        /// Match on the exact app ID
        #[arg(long)]
        app_id: Option<String>,

        /// Case-insensitive substring match on the window title
        #[arg(long)]
        window_title_contains: Option<String>,

        /// Case-insensitive substring match on the browser domain
        #[arg(long)]
        browser_domain_contains: Option<String>,
    },

    /// Remove a glossary entry by canonical term
    Remove {
        /// Canonical term to remove
        term: String,
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
    fn parses_voice_command() {
        let cli = Cli::try_parse_from(["whispers", "voice"]).unwrap();
        assert!(matches!(cli.command, Some(Command::Voice)));
    }

    #[test]
    fn parses_status_command() {
        let cli = Cli::try_parse_from(["whispers", "status"]).unwrap();
        assert!(matches!(cli.command, Some(Command::Status)));
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
    fn parses_app_rule_add_subcommand() {
        let cli = Cli::try_parse_from([
            "whispers",
            "app-rule",
            "add",
            "zed",
            "Preserve identifiers.",
            "--app-id",
            "dev.zed.Zed",
            "--correction-policy",
            "balanced",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::AppRule {
                action: AppRuleAction::Add { .. }
            })
        ));
    }

    #[test]
    fn parses_glossary_add_subcommand() {
        let cli = Cli::try_parse_from([
            "whispers",
            "glossary",
            "add",
            "TypeScript",
            "--alias",
            "type script",
            "--surface-kind",
            "editor",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Glossary {
                action: GlossaryAction::Add { .. }
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

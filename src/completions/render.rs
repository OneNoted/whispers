use std::io::Write;

use clap::CommandFactory;
use clap_complete::{generate, shells};
use clap_complete_nushell::Nushell;

use crate::cli::{Cli, CompletionShell};

pub(super) fn write_completions(shell: CompletionShell, out: &mut dyn Write) {
    let mut cmd = Cli::command();
    let bin_name = cmd.get_name().to_string();

    match shell {
        CompletionShell::Bash => generate(shells::Bash, &mut cmd, &bin_name, out),
        CompletionShell::Zsh => generate(shells::Zsh, &mut cmd, &bin_name, out),
        CompletionShell::Fish => generate(shells::Fish, &mut cmd, &bin_name, out),
        CompletionShell::Nushell => generate(Nushell, &mut cmd, &bin_name, out),
    }
}

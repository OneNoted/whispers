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
        CompletionShell::Fish => {
            generate(shells::Fish, &mut cmd, &bin_name, out);
            write_fish_completion_shell_values(out);
        }
        CompletionShell::Nushell => generate(Nushell, &mut cmd, &bin_name, out),
    }
}

fn write_fish_completion_shell_values(out: &mut dyn Write) {
    // clap_complete omits optional positional ValueEnum arguments for fish, so fish
    // otherwise falls back to filename completion after `whispers completions`.
    writeln!(
        out,
        "complete -c whispers -n \"__fish_whispers_using_subcommand completions\" -f -a \"bash zsh fish nushell\" -d 'Shell completion format'"
    )
    .expect("writing to the completion output must succeed");
}

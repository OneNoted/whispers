mod detect;
mod render;

#[cfg(test)]
mod tests;

use crate::cli::CompletionShell;
use crate::error::{Result, WhsprError};

const SUPPORTED_SHELLS: &str = "bash|zsh|fish|nushell";

pub fn run_completions(shell_arg: Option<CompletionShell>) -> Result<()> {
    let shell = shell_arg.or_else(detect::detect_shell).ok_or_else(|| {
        WhsprError::Config(format!(
            "could not detect shell automatically. Specify one manually: whispers completions <{SUPPORTED_SHELLS}>"
        ))
    })?;

    let mut stdout = std::io::stdout();
    render::write_completions(shell, &mut stdout);
    Ok(())
}

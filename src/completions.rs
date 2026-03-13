use std::io::Write;
use std::path::Path;

use clap::CommandFactory;
use clap_complete::{generate, shells};
use clap_complete_nushell::Nushell;

use crate::cli::{Cli, CompletionShell};
use crate::error::{Result, WhsprError};

const SUPPORTED_SHELLS: &str = "bash|zsh|fish|nushell";

pub fn run_completions(shell_arg: Option<CompletionShell>) -> Result<()> {
    let shell = shell_arg.or_else(detect_shell).ok_or_else(|| {
        WhsprError::Config(format!(
            "could not detect shell automatically. Specify one manually: whispers completions <{SUPPORTED_SHELLS}>"
        ))
    })?;

    let mut stdout = std::io::stdout();
    write_completions(shell, &mut stdout);
    Ok(())
}

fn detect_shell() -> Option<CompletionShell> {
    detect_shell_from_env().or_else(detect_shell_from_parent_process)
}

fn detect_shell_from_env() -> Option<CompletionShell> {
    std::env::var("SHELL")
        .ok()
        .and_then(|value| shell_from_path_like(&value))
}

fn detect_shell_from_parent_process() -> Option<CompletionShell> {
    let ppid = unsafe { libc::getppid() };
    if ppid <= 0 {
        return None;
    }

    detect_shell_from_pid(ppid)
}

fn detect_shell_from_pid(pid: libc::pid_t) -> Option<CompletionShell> {
    let cmdline_path = format!("/proc/{pid}/cmdline");
    if let Ok(cmdline) = std::fs::read(cmdline_path) {
        if let Some(shell) = shell_from_cmdline_bytes(&cmdline) {
            return Some(shell);
        }
    }

    let comm_path = format!("/proc/{pid}/comm");
    std::fs::read_to_string(comm_path)
        .ok()
        .and_then(|comm| shell_from_token(comm.trim()))
}

fn shell_from_cmdline_bytes(bytes: &[u8]) -> Option<CompletionShell> {
    let first = bytes.split(|b| *b == 0).next()?;
    if first.is_empty() {
        return None;
    }

    let token = String::from_utf8_lossy(first);
    shell_from_path_like(&token)
}

fn shell_from_path_like(value: &str) -> Option<CompletionShell> {
    let name = Path::new(value).file_name()?.to_string_lossy();
    shell_from_token(&name)
}

fn shell_from_token(value: &str) -> Option<CompletionShell> {
    let normalized = value.trim().trim_start_matches('-').to_ascii_lowercase();

    match normalized.as_str() {
        "bash" => Some(CompletionShell::Bash),
        "zsh" => Some(CompletionShell::Zsh),
        "fish" => Some(CompletionShell::Fish),
        "nu" | "nushell" => Some(CompletionShell::Nushell),
        _ => None,
    }
}

fn write_completions(shell: CompletionShell, out: &mut dyn Write) {
    let mut cmd = Cli::command();
    let bin_name = cmd.get_name().to_string();

    match shell {
        CompletionShell::Bash => generate(shells::Bash, &mut cmd, &bin_name, out),
        CompletionShell::Zsh => generate(shells::Zsh, &mut cmd, &bin_name, out),
        CompletionShell::Fish => generate(shells::Fish, &mut cmd, &bin_name, out),
        CompletionShell::Nushell => generate(Nushell, &mut cmd, &bin_name, out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_detection_from_env_value_supports_paths() {
        assert_eq!(
            shell_from_path_like("/usr/bin/zsh"),
            Some(CompletionShell::Zsh)
        );
        assert_eq!(
            shell_from_path_like("/bin/bash"),
            Some(CompletionShell::Bash)
        );
    }

    #[test]
    fn shell_detection_accepts_login_shell_prefix_and_nu_alias() {
        assert_eq!(shell_from_token("-fish"), Some(CompletionShell::Fish));
        assert_eq!(shell_from_token("nu"), Some(CompletionShell::Nushell));
    }

    #[test]
    fn shell_detection_from_cmdline_uses_first_argv_entry() {
        let cmdline = b"/usr/bin/fish\0-l\0";
        assert_eq!(
            shell_from_cmdline_bytes(cmdline),
            Some(CompletionShell::Fish)
        );
    }

    #[test]
    fn shell_detection_returns_none_for_unknown_values() {
        assert_eq!(shell_from_path_like("/bin/tcsh"), None);
        assert_eq!(shell_from_token("xonsh"), None);
        assert_eq!(shell_from_cmdline_bytes(b""), None);
    }

    fn generate_to_string(shell: CompletionShell) -> String {
        let mut output = Vec::new();
        write_completions(shell, &mut output);
        String::from_utf8(output).unwrap()
    }

    #[test]
    fn generates_bash_completion_script() {
        let script = generate_to_string(CompletionShell::Bash);
        assert!(script.contains("whispers"));
        assert!(script.contains("complete"));
    }

    #[test]
    fn generates_zsh_completion_script() {
        let script = generate_to_string(CompletionShell::Zsh);
        assert!(script.contains("whispers"));
        assert!(script.contains("compdef"));
    }

    #[test]
    fn generates_fish_completion_script() {
        let script = generate_to_string(CompletionShell::Fish);
        assert!(script.contains("whispers"));
        assert!(script.contains("complete -c"));
    }

    #[test]
    fn generates_nushell_completion_script() {
        let script = generate_to_string(CompletionShell::Nushell);
        assert!(script.contains("whispers"));
        assert!(script.contains("export extern"));
    }
}

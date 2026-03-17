use std::path::Path;

use crate::cli::CompletionShell;

pub(super) fn detect_shell() -> Option<CompletionShell> {
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

pub(super) fn shell_from_cmdline_bytes(bytes: &[u8]) -> Option<CompletionShell> {
    let first = bytes.split(|b| *b == 0).next()?;
    if first.is_empty() {
        return None;
    }

    let token = String::from_utf8_lossy(first);
    shell_from_path_like(&token)
}

pub(super) fn shell_from_path_like(value: &str) -> Option<CompletionShell> {
    let name = Path::new(value).file_name()?.to_string_lossy();
    shell_from_token(&name)
}

pub(super) fn shell_from_token(value: &str) -> Option<CompletionShell> {
    let normalized = value.trim().trim_start_matches('-').to_ascii_lowercase();

    match normalized.as_str() {
        "bash" => Some(CompletionShell::Bash),
        "zsh" => Some(CompletionShell::Zsh),
        "fish" => Some(CompletionShell::Fish),
        "nu" | "nushell" => Some(CompletionShell::Nushell),
        _ => None,
    }
}

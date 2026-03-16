use crate::cli::CompletionShell;

use super::{detect, render};

#[test]
fn shell_detection_from_env_value_supports_paths() {
    assert_eq!(
        detect::shell_from_path_like("/usr/bin/zsh"),
        Some(CompletionShell::Zsh)
    );
    assert_eq!(
        detect::shell_from_path_like("/bin/bash"),
        Some(CompletionShell::Bash)
    );
}

#[test]
fn shell_detection_accepts_login_shell_prefix_and_nu_alias() {
    assert_eq!(
        detect::shell_from_token("-fish"),
        Some(CompletionShell::Fish)
    );
    assert_eq!(
        detect::shell_from_token("nu"),
        Some(CompletionShell::Nushell)
    );
}

#[test]
fn shell_detection_from_cmdline_uses_first_argv_entry() {
    let cmdline = b"/usr/bin/fish\0-l\0";
    assert_eq!(
        detect::shell_from_cmdline_bytes(cmdline),
        Some(CompletionShell::Fish)
    );
}

#[test]
fn shell_detection_returns_none_for_unknown_values() {
    assert_eq!(detect::shell_from_path_like("/bin/tcsh"), None);
    assert_eq!(detect::shell_from_token("xonsh"), None);
    assert_eq!(detect::shell_from_cmdline_bytes(b""), None);
}

fn generate_to_string(shell: CompletionShell) -> String {
    let mut output = Vec::new();
    render::write_completions(shell, &mut output);
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

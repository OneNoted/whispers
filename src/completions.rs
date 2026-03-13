use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

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

pub fn detect_shell() -> Option<CompletionShell> {
    detect_shell_from_env().or_else(detect_shell_from_parent_process)
}

pub fn detect_installed_shells() -> Vec<CompletionShell> {
    detect_installed_shells_in_path(std::env::var_os("PATH").as_deref())
}

pub fn install_completions(shell: CompletionShell) -> Result<PathBuf> {
    let path = install_path(shell);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            WhsprError::Config(format!(
                "failed to create completion directory {}: {e}",
                parent.display()
            ))
        })?;
    }
    std::fs::write(&path, render_completions(shell)).map_err(|e| {
        WhsprError::Config(format!(
            "failed to write {} completions to {}: {e}",
            shell.as_str(),
            path.display()
        ))
    })?;
    finalize_completion_install(shell, &path)?;
    Ok(path)
}

pub fn install_note(shell: CompletionShell, path: &Path) -> Option<String> {
    match shell {
        CompletionShell::Bash => {
            Some("Restart bash or run `source ~/.bashrc` to refresh completions.".to_string())
        }
        CompletionShell::Zsh if path.parent() == Some(home_dir().join(".zfunc").as_path()) => {
            Some("Ensure your zsh fpath includes ~/.zfunc before running compinit.".to_string())
        }
        CompletionShell::Zsh => Some(
            "Restart zsh or run `autoload -Uz compinit && compinit` to refresh completions."
                .to_string(),
        ),
        CompletionShell::Nushell => Some("Restart Nushell to refresh completions.".to_string()),
        _ => None,
    }
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

fn detect_installed_shells_in_path(path: Option<&std::ffi::OsStr>) -> Vec<CompletionShell> {
    CompletionShell::all()
        .into_iter()
        .filter(|shell| {
            shell
                .binary_names()
                .iter()
                .any(|binary| binary_exists_in_path(binary, path))
        })
        .collect()
}

fn binary_exists_in_path(binary: &str, path: Option<&std::ffi::OsStr>) -> bool {
    let Some(path) = path else {
        return false;
    };

    std::env::split_paths(path).any(|dir| dir.join(binary).is_file())
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
    out.write_all(render_completions(shell).as_bytes())
        .expect("writing completions to stdout should succeed");
}

pub fn render_completions(shell: CompletionShell) -> String {
    let mut cmd = Cli::command();
    let bin_name = cmd.get_name().to_string();
    let mut output = Vec::new();

    match shell {
        CompletionShell::Bash => generate(shells::Bash, &mut cmd, &bin_name, &mut output),
        CompletionShell::Zsh => generate(shells::Zsh, &mut cmd, &bin_name, &mut output),
        CompletionShell::Fish => generate(shells::Fish, &mut cmd, &bin_name, &mut output),
        CompletionShell::Nushell => generate(Nushell, &mut cmd, &bin_name, &mut output),
    }

    String::from_utf8(output).expect("completion scripts should be valid UTF-8")
}

fn install_path(shell: CompletionShell) -> PathBuf {
    match shell {
        CompletionShell::Bash => xdg_data_home()
            .join("bash-completion")
            .join("completions")
            .join("whispers"),
        CompletionShell::Zsh => zsh_install_dir().join("_whispers"),
        CompletionShell::Fish => xdg_config_home()
            .join("fish")
            .join("completions")
            .join("whispers.fish"),
        CompletionShell::Nushell => nushell_autoload_dir().join("whispers.nu"),
    }
}

fn finalize_completion_install(shell: CompletionShell, path: &Path) -> Result<()> {
    match shell {
        CompletionShell::Bash => ensure_bashrc_sources_completion(path),
        CompletionShell::Nushell => ensure_nushell_config_uses_completion(path),
        _ => Ok(()),
    }
}

fn ensure_bashrc_sources_completion(path: &Path) -> Result<()> {
    let bashrc_path = home_dir().join(".bashrc");
    let existing = std::fs::read_to_string(&bashrc_path).unwrap_or_default();
    let start_marker = "# >>> whispers bash completions >>>";
    let end_marker = "# <<< whispers bash completions <<<";
    let source_path = shell_double_quote(path);
    let block = format!(
        "{start_marker}\nif [[ $- == *i* ]] && [[ -r \"{source_path}\" ]]; then\n  source \"{source_path}\"\nfi\n{end_marker}\n"
    );

    let updated = if let Some(start) = existing.find(start_marker) {
        if let Some(end_rel) = existing[start..].find(end_marker) {
            let end = start + end_rel + end_marker.len();
            let mut next = String::new();
            next.push_str(&existing[..start]);
            if !next.ends_with('\n') && !next.is_empty() {
                next.push('\n');
            }
            next.push_str(&block);
            let suffix = existing[end..]
                .strip_prefix('\n')
                .unwrap_or(&existing[end..]);
            next.push_str(suffix);
            next
        } else {
            existing.clone()
        }
    } else if existing.contains(&format!("source \"{source_path}\"")) {
        existing
    } else {
        let mut next = existing;
        if !next.ends_with('\n') && !next.is_empty() {
            next.push('\n');
        }
        if !next.is_empty() {
            next.push('\n');
        }
        next.push_str(&block);
        next
    };

    if updated != std::fs::read_to_string(&bashrc_path).unwrap_or_default() {
        std::fs::write(&bashrc_path, updated).map_err(|e| {
            WhsprError::Config(format!(
                "failed to update bash init at {}: {e}",
                bashrc_path.display()
            ))
        })?;
    }

    Ok(())
}

fn ensure_nushell_config_uses_completion(path: &Path) -> Result<()> {
    let config_path = xdg_config_home().join("nushell").join("config.nu");
    let existing = std::fs::read_to_string(&config_path).unwrap_or_default();
    let start_marker = "# >>> whispers nushell completions >>>";
    let end_marker = "# <<< whispers nushell completions <<<";
    let source_path = nushell_string_literal(path);
    let block = format!("{start_marker}\nuse {source_path} *\n{end_marker}\n");

    let updated = replace_or_prepend_block(&existing, start_marker, end_marker, &block);
    if updated != existing {
        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                WhsprError::Config(format!(
                    "failed to create Nushell config directory {}: {e}",
                    parent.display()
                ))
            })?;
        }
        std::fs::write(&config_path, updated).map_err(|e| {
            WhsprError::Config(format!(
                "failed to update Nushell config at {}: {e}",
                config_path.display()
            ))
        })?;
    }

    Ok(())
}

fn replace_or_prepend_block(
    existing: &str,
    start_marker: &str,
    end_marker: &str,
    block: &str,
) -> String {
    if let Some(start) = existing.find(start_marker) {
        if let Some(end_rel) = existing[start..].find(end_marker) {
            let end = start + end_rel + end_marker.len();
            let mut next = String::new();
            next.push_str(&existing[..start]);
            if !next.ends_with('\n') && !next.is_empty() {
                next.push('\n');
            }
            next.push_str(block);
            let suffix = existing[end..]
                .strip_prefix('\n')
                .unwrap_or(&existing[end..]);
            next.push_str(suffix);
            return next;
        }

        return existing.to_string();
    }

    if existing.contains(block.trim()) {
        return existing.to_string();
    }

    let mut next = String::new();
    next.push_str(block);
    if !existing.is_empty() && !block.ends_with('\n') {
        next.push('\n');
    }
    next.push_str(existing);
    next
}

fn zsh_install_dir() -> PathBuf {
    detect_zsh_completion_dir().unwrap_or_else(|| home_dir().join(".zfunc"))
}

fn detect_zsh_completion_dir() -> Option<PathBuf> {
    let output = Command::new("zsh")
        .arg("-ic")
        .arg("print -l -- $fpath")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    preferred_zsh_completion_dir_from_fpath(&String::from_utf8_lossy(&output.stdout))
}

fn preferred_zsh_completion_dir_from_fpath(output: &str) -> Option<PathBuf> {
    let home = home_dir();
    let candidates: Vec<PathBuf> = output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(PathBuf::from)
        .filter(|path| path.starts_with(&home))
        .collect();

    candidates
        .iter()
        .find(|path| !path.starts_with(home.join(".cache")))
        .cloned()
        .or_else(|| candidates.into_iter().next())
}

fn nushell_autoload_dir() -> PathBuf {
    detect_nushell_autoload_dir()
        .unwrap_or_else(|| xdg_config_home().join("nushell").join("autoload"))
}

fn detect_nushell_autoload_dir() -> Option<PathBuf> {
    let output = Command::new("nu")
        .arg("-c")
        .arg("print (($nu.user-autoload-dirs | first) | to text)")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if path.is_empty() {
        None
    } else {
        Some(PathBuf::from(path))
    }
}

fn shell_double_quote(path: &Path) -> String {
    path.display()
        .to_string()
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
}

fn nushell_string_literal(path: &Path) -> String {
    format!(
        "\"{}\"",
        path.display()
            .to_string()
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
    )
}

fn xdg_config_home() -> PathBuf {
    if let Ok(dir) = std::env::var("XDG_CONFIG_HOME") {
        PathBuf::from(dir)
    } else {
        home_dir().join(".config")
    }
}

fn xdg_data_home() -> PathBuf {
    if let Ok(dir) = std::env::var("XDG_DATA_HOME") {
        PathBuf::from(dir)
    } else {
        home_dir().join(".local").join("share")
    }
}

fn home_dir() -> PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home)
    } else {
        tracing::warn!("HOME is not set, falling back to /tmp for shell completion install path");
        PathBuf::from("/tmp")
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
        render_completions(shell)
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

    #[test]
    fn install_path_uses_expected_xdg_locations() {
        let _env_lock = crate::test_support::env_lock();
        let _guard = crate::test_support::EnvVarGuard::capture(&[
            "HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
        ]);
        let root = crate::test_support::unique_temp_dir("completions-paths");
        crate::test_support::set_env("HOME", &root.to_string_lossy());
        crate::test_support::remove_env("XDG_CONFIG_HOME");
        crate::test_support::remove_env("XDG_DATA_HOME");

        assert_eq!(
            install_path(CompletionShell::Fish),
            root.join(".config/fish/completions/whispers.fish")
        );
        assert_eq!(
            install_path(CompletionShell::Bash),
            root.join(".local/share/bash-completion/completions/whispers")
        );
        assert_eq!(
            install_path(CompletionShell::Zsh),
            root.join(".zfunc/_whispers")
        );
        assert_eq!(
            install_path(CompletionShell::Nushell),
            root.join(".config/nushell/autoload/whispers.nu")
        );
    }

    #[test]
    fn install_completions_writes_script_to_target_path() {
        let _env_lock = crate::test_support::env_lock();
        let _guard = crate::test_support::EnvVarGuard::capture(&[
            "HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
        ]);
        let root = crate::test_support::unique_temp_dir("completions-install");
        crate::test_support::set_env("HOME", &root.to_string_lossy());
        crate::test_support::remove_env("XDG_CONFIG_HOME");
        crate::test_support::remove_env("XDG_DATA_HOME");

        let path = install_completions(CompletionShell::Fish).expect("install fish completions");
        let script = std::fs::read_to_string(&path).expect("read installed fish completions");
        assert!(script.contains("complete -c whispers"));
        assert_eq!(path, root.join(".config/fish/completions/whispers.fish"));
    }

    #[test]
    fn detect_installed_shells_finds_supported_binaries_on_path() {
        let root = crate::test_support::unique_temp_dir("completions-shell-detect");
        for binary in ["fish", "zsh", "nu"] {
            let path = root.join(binary);
            std::fs::write(path, "#!/bin/sh\n").expect("write fake shell");
        }

        let detected = detect_installed_shells_in_path(Some(root.as_os_str()));
        assert_eq!(
            detected,
            vec![
                CompletionShell::Zsh,
                CompletionShell::Fish,
                CompletionShell::Nushell,
            ]
        );
    }

    #[test]
    fn preferred_zsh_completion_dir_uses_active_user_fpath_over_fallback() {
        let _env_lock = crate::test_support::env_lock();
        let _guard = crate::test_support::EnvVarGuard::capture(&["HOME"]);
        let root = crate::test_support::unique_temp_dir("completions-zsh-fpath");
        crate::test_support::set_env("HOME", &root.to_string_lossy());

        let fpath = format!(
            "{}/.cache/zinit/completions\n{}/.local/share/zinit/completions\n/usr/share/zsh/site-functions\n",
            root.display(),
            root.display()
        );
        assert_eq!(
            preferred_zsh_completion_dir_from_fpath(&fpath),
            Some(root.join(".local/share/zinit/completions"))
        );
    }

    #[test]
    fn ensure_bashrc_sources_completion_adds_guarded_block() {
        let _env_lock = crate::test_support::env_lock();
        let _guard = crate::test_support::EnvVarGuard::capture(&["HOME"]);
        let root = crate::test_support::unique_temp_dir("completions-bashrc");
        crate::test_support::set_env("HOME", &root.to_string_lossy());

        let completion_path = root.join(".local/share/bash-completion/completions/whispers");
        std::fs::create_dir_all(completion_path.parent().expect("parent")).expect("mkdir");
        std::fs::write(&completion_path, "# test\n").expect("write completion");

        ensure_bashrc_sources_completion(&completion_path).expect("update bashrc");
        let bashrc = std::fs::read_to_string(root.join(".bashrc")).expect("read bashrc");
        assert!(bashrc.contains("# >>> whispers bash completions >>>"));
        assert!(bashrc.contains("source \"/"));
        assert!(bashrc.contains("bash-completion/completions/whispers"));
    }

    #[test]
    fn ensure_nushell_config_uses_completion_adds_guarded_block() {
        let _env_lock = crate::test_support::env_lock();
        let _guard = crate::test_support::EnvVarGuard::capture(&["HOME", "XDG_CONFIG_HOME"]);
        let root = crate::test_support::unique_temp_dir("completions-nushell-config");
        crate::test_support::set_env("HOME", &root.to_string_lossy());
        crate::test_support::set_env("XDG_CONFIG_HOME", &root.join(".config").to_string_lossy());

        let completion_path = root.join(".config/nushell/autoload/whispers.nu");
        std::fs::create_dir_all(completion_path.parent().expect("parent")).expect("mkdir");
        std::fs::write(&completion_path, "# test\n").expect("write completion");

        ensure_nushell_config_uses_completion(&completion_path).expect("update nushell config");
        let config =
            std::fs::read_to_string(root.join(".config/nushell/config.nu")).expect("read config");
        assert!(config.contains("# >>> whispers nushell completions >>>"));
        assert!(config.contains("use \""));
        assert!(config.contains("autoload/whispers.nu"));
    }
}

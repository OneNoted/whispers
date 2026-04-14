use std::env;
use std::fs;
use std::fs::OpenOptions;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use crate::error::{Result, WhsprError};

const UINPUT_PATH: &str = "/dev/uinput";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InjectionReadinessReport {
    issues: Vec<InjectionReadinessIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum InjectionReadinessIssue {
    MissingWlCopy,
    MissingUinputDevice,
    UinputPermissionDenied,
    UinputUnavailable(String),
}

impl InjectionReadinessReport {
    pub fn collect() -> Self {
        let mut issues = Vec::new();

        if !command_on_path("wl-copy") {
            issues.push(InjectionReadinessIssue::MissingWlCopy);
        }

        if let Some(issue) = probe_uinput() {
            issues.push(issue);
        }

        Self { issues }
    }

    pub fn is_ready(&self) -> bool {
        self.issues.is_empty()
    }

    pub(crate) fn has_uinput_issue(&self) -> bool {
        self.issues.iter().any(|issue| {
            matches!(
                issue,
                InjectionReadinessIssue::MissingUinputDevice
                    | InjectionReadinessIssue::UinputPermissionDenied
                    | InjectionReadinessIssue::UinputUnavailable(_)
            )
        })
    }

    pub fn issue_lines(&self) -> Vec<String> {
        self.issues
            .iter()
            .map(InjectionReadinessIssue::issue_line)
            .collect()
    }

    pub fn fix_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        for issue in &self.issues {
            for line in issue.fix_lines() {
                if !lines.iter().any(|existing| existing == &line) {
                    lines.push(line);
                }
            }
        }
        lines
    }

    pub(crate) fn as_error(&self) -> Option<WhsprError> {
        if self.is_ready() {
            return None;
        }

        let details = self
            .issues
            .iter()
            .map(InjectionReadinessIssue::runtime_detail)
            .collect::<Vec<_>>()
            .join("; ");
        Some(WhsprError::Injection(format!(
            "paste injection is not ready: {details}"
        )))
    }

    #[cfg(test)]
    pub(crate) fn from_issues(issues: Vec<InjectionReadinessIssue>) -> Self {
        Self { issues }
    }
}

impl InjectionReadinessIssue {
    fn issue_line(&self) -> String {
        match self {
            Self::MissingWlCopy => "`wl-copy` is not available on PATH.".into(),
            Self::MissingUinputDevice => {
                format!("{UINPUT_PATH} is missing, so whispers cannot create its virtual keyboard.")
            }
            Self::UinputPermissionDenied => {
                format!("The current user cannot open {UINPUT_PATH}.")
            }
            Self::UinputUnavailable(detail) => {
                format!("{UINPUT_PATH} exists but could not be opened: {detail}.")
            }
        }
    }

    fn fix_lines(&self) -> Vec<String> {
        match self {
            Self::MissingWlCopy => vec!["Install the `wl-clipboard` package.".into()],
            Self::MissingUinputDevice => vec![
                "Load the `uinput` kernel module: sudo modprobe uinput".into(),
                "Persist it across reboots if needed: create `/etc/modules-load.d/whispers-uinput.conf` with `uinput`.".into(),
            ],
            Self::UinputPermissionDenied => vec![
                "Add your user to the `input` group and create a `udev` rule for `/dev/uinput`.".into(),
                "Log out and back in after changing group membership.".into(),
            ],
            Self::UinputUnavailable(_) => {
                vec!["Check that `/dev/uinput` exists and is writable by the current user.".into()]
            }
        }
    }

    fn runtime_detail(&self) -> String {
        match self {
            Self::MissingWlCopy => "`wl-copy` was not found; install `wl-clipboard`".into(),
            Self::MissingUinputDevice => {
                "`/dev/uinput` is missing; load the `uinput` kernel module".into()
            }
            Self::UinputPermissionDenied => {
                "`/dev/uinput` is present but not writable by the current user; add a `udev` rule, add your user to the `input` group, then log out and back in".into()
            }
            Self::UinputUnavailable(detail) => {
                format!("`/dev/uinput` could not be opened: {detail}")
            }
        }
    }
}

pub fn validate_injection_prerequisites() -> Result<()> {
    let report = InjectionReadinessReport::collect();
    if let Some(err) = report.as_error() {
        return Err(err);
    }
    Ok(())
}

fn probe_uinput() -> Option<InjectionReadinessIssue> {
    let path = Path::new(UINPUT_PATH);
    if !path.exists() {
        return Some(InjectionReadinessIssue::MissingUinputDevice);
    }

    match OpenOptions::new().write(true).open(path) {
        Ok(_) => None,
        Err(err) if err.kind() == std::io::ErrorKind::PermissionDenied => {
            Some(InjectionReadinessIssue::UinputPermissionDenied)
        }
        Err(err) => Some(InjectionReadinessIssue::UinputUnavailable(err.to_string())),
    }
}

fn command_on_path(program: &str) -> bool {
    let Some(path) = env::var_os("PATH") else {
        return false;
    };

    env::split_paths(&path).any(|dir| {
        let candidate = dir.join(program);
        match fs::metadata(candidate) {
            Ok(metadata) => metadata.is_file() && metadata.permissions().mode() & 0o111 != 0,
            Err(_) => false,
        }
    })
}

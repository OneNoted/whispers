use crate::error::WhsprError;

use super::preflight::InjectionReadinessIssue;
use super::*;

#[test]
fn run_wl_copy_reports_spawn_failure() {
    let err = clipboard::run_wl_copy("/definitely/missing/wl-copy", &[], "hello")
        .expect_err("missing binary should fail");
    match err {
        WhsprError::Injection(msg) => {
            assert!(msg.contains("failed to spawn wl-copy"), "unexpected: {msg}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn run_wl_copy_reports_non_zero_exit() {
    let err = clipboard::run_wl_copy(
        "/bin/sh",
        &[String::from("-c"), String::from("exit 7")],
        "hello",
    )
    .expect_err("non-zero exit should fail");
    match err {
        WhsprError::Injection(msg) => {
            assert!(msg.contains("wl-copy exited"), "unexpected: {msg}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn run_wl_copy_reports_timeout() {
    let err = clipboard::run_wl_copy_with_timeout(
        "/bin/sleep",
        &[String::from("1")],
        "hello",
        std::time::Duration::from_millis(80),
    )
    .expect_err("sleep should time out");
    match err {
        WhsprError::Injection(msg) => {
            assert!(msg.contains("timed out"), "unexpected: {msg}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[tokio::test]
async fn inject_empty_text_is_noop() {
    let injector = TextInjector::with_wl_copy_command("/bin/true", &[]);
    injector.inject("").await.expect("empty text should no-op");
}

#[test]
fn readiness_report_formats_missing_uinput_guidance() {
    let report =
        InjectionReadinessReport::from_issues(vec![InjectionReadinessIssue::MissingUinputDevice]);
    assert!(!report.is_ready());
    assert!(
        report
            .issue_lines()
            .iter()
            .any(|line| line.contains("/dev/uinput"))
    );
    assert!(
        report
            .fix_lines()
            .iter()
            .any(|line| line.contains("sudo modprobe uinput"))
    );

    let err = report.as_error().expect("missing uinput should fail");
    match err {
        WhsprError::Injection(msg) => {
            assert!(
                msg.contains("paste injection is not ready"),
                "unexpected: {msg}"
            );
            assert!(msg.contains("uinput` kernel module"), "unexpected: {msg}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn readiness_report_formats_permission_guidance() {
    let report = InjectionReadinessReport::from_issues(vec![
        InjectionReadinessIssue::UinputPermissionDenied,
    ]);
    assert!(report.has_uinput_issue());
    assert!(
        report
            .fix_lines()
            .iter()
            .any(|line| line.contains("`uinput` group"))
    );
}

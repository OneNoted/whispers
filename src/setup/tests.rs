use crate::config::Config;
use crate::error::WhsprError;
use crate::inject::{InjectionReadinessIssue, InjectionReadinessReport};

use super::{CloudSetup, apply, report, side_effects};
use crate::config::{self, TranscriptionBackend, TranscriptionFallback};

#[cfg(not(feature = "local-rewrite"))]
use crate::config::{RewriteBackend, RewriteFallback};

#[test]
fn runtime_selection_resets_cloud_asr_when_disabled() {
    let config_path = crate::test_support::unique_temp_path("setup-runtime-reset", "toml");
    config::write_default_config(&config_path, "~/model.bin").expect("write config");
    config::update_config_transcription_runtime(
        &config_path,
        TranscriptionBackend::Cloud,
        TranscriptionFallback::None,
    )
    .expect("set cloud runtime");

    let cloud = CloudSetup::default();
    apply::apply_runtime_backend_selection(&config_path, TranscriptionBackend::WhisperCpp, &cloud)
        .expect("reset runtime");

    let config = Config::load(Some(&config_path)).expect("load config");
    assert_eq!(
        config.transcription.backend,
        TranscriptionBackend::WhisperCpp
    );
    assert_eq!(
        config.transcription.fallback,
        TranscriptionFallback::ConfiguredLocal
    );
}

#[cfg(not(feature = "local-rewrite"))]
#[test]
fn runtime_selection_disables_local_rewrite_fallback_when_build_lacks_local_rewrite() {
    let config_path = crate::test_support::unique_temp_path("setup-rewrite-fallback-reset", "toml");
    config::write_default_config(&config_path, "~/model.bin").expect("write config");

    let cloud = CloudSetup {
        rewrite_enabled: true,
        rewrite_fallback: RewriteFallback::Local,
        ..CloudSetup::default()
    };
    apply::apply_runtime_backend_selection(&config_path, TranscriptionBackend::WhisperCpp, &cloud)
        .expect("apply runtime");

    let config = Config::load(Some(&config_path)).expect("load config");
    assert_eq!(config.rewrite.backend, RewriteBackend::Cloud);
    assert_eq!(config.rewrite.fallback, RewriteFallback::None);
}

#[test]
fn group_membership_failures_become_warnings_without_marking_success() {
    let mut outcome = side_effects::InjectionSetupOutcome::default();
    let warning = side_effects::record_group_membership_change_result(
        &mut outcome,
        "uinput",
        Err(WhsprError::Config("group add failed".into())),
    )
    .expect("errors should become warnings");

    assert!(!outcome.changed_groups);
    assert!(!outcome.group_membership_ready);
    assert!(!outcome.uinput_rule_ready);
    assert!(warning.contains("Failed to add the current user"));
    assert!(warning.contains("group add failed"));
}

#[test]
fn setup_rejects_root_before_uinput_readiness_short_circuit() {
    let err = side_effects::validate_setup_user(0).expect_err("root should be rejected");
    match err {
        WhsprError::Config(message) => {
            assert!(message.contains("run `whispers setup` as your normal user, not as root"));
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn setup_rejects_root_before_any_setup_side_effects() {
    let err = super::validate_setup_invoker(0).expect_err("root should be rejected up front");
    match err {
        WhsprError::Config(message) => {
            assert!(message.contains("run `whispers setup` as your normal user, not as root"));
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn group_membership_success_marks_logout_as_needed() {
    let mut outcome = side_effects::InjectionSetupOutcome::default();
    let warning =
        side_effects::record_group_membership_change_result(&mut outcome, "uinput", Ok(true));

    assert!(warning.is_none());
    assert!(outcome.changed_groups);
    assert!(outcome.group_membership_ready);
    assert!(!outcome.uinput_rule_ready);
    assert!(!outcome.udev_reload_succeeded);
}

#[test]
fn group_exists_uses_nss_sources() {
    assert!(side_effects::group_exists("root").expect("resolve root group"));

    let missing = format!("whispers-missing-group-{}", std::process::id());
    assert!(!side_effects::group_exists(&missing).expect("resolve missing group"));
}

#[test]
fn existing_group_membership_marks_relogin_as_possible_without_new_group_change() {
    let mut outcome = side_effects::InjectionSetupOutcome::default();
    let warning =
        side_effects::record_group_membership_change_result(&mut outcome, "uinput", Ok(false));

    assert!(warning.is_none());
    assert!(!outcome.changed_groups);
    assert!(outcome.group_membership_ready);
    assert!(!outcome.uinput_rule_ready);
}

#[test]
fn group_change_messages_follow_recorded_reload_status() {
    let success = side_effects::InjectionSetupOutcome {
        changed_groups: true,
        group_membership_ready: true,
        uinput_rule_ready: true,
        udev_reload_succeeded: true,
    };
    let failed_reload = side_effects::InjectionSetupOutcome {
        changed_groups: true,
        group_membership_ready: true,
        uinput_rule_ready: true,
        udev_reload_succeeded: false,
    };

    assert_eq!(
        success.setup_group_change_message(),
        Some("Group membership changed. Log out and back in before testing dictation."),
    );
    assert_eq!(
        failed_reload.setup_group_change_message(),
        Some(
            "Group membership changed. Log out and back in after finishing the remaining paste injection steps.",
        ),
    );
    assert_eq!(
        success.report_group_change_message(),
        Some(
            "If you were just added to the `uinput` group, log out and back in after finishing the remaining paste injection steps.",
        ),
    );
    assert_eq!(
        failed_reload.report_group_change_message(),
        Some(
            "If you were just added to the `uinput` group, log out and back in after finishing the remaining paste injection steps.",
        ),
    );
}

#[test]
fn relogin_only_readiness_collapses_to_logout_instruction() {
    let readiness = InjectionReadinessReport::from_issues(vec![
        InjectionReadinessIssue::UinputPermissionDenied,
    ]);
    let setup = side_effects::InjectionSetupOutcome {
        changed_groups: true,
        group_membership_ready: true,
        uinput_rule_ready: true,
        udev_reload_succeeded: true,
    };

    assert_eq!(
        report::injection_readiness_info_message(&readiness, &setup),
        Some("Log out and back in before testing.")
    );
    assert!(report::injection_readiness_fix_lines(&readiness, &setup).is_empty());
}

#[test]
fn relogin_only_completion_allows_reruns_when_group_is_already_configured() {
    let rerun = side_effects::InjectionSetupOutcome {
        changed_groups: false,
        group_membership_ready: true,
        uinput_rule_ready: true,
        udev_reload_succeeded: true,
    };
    let failed_group_update = side_effects::InjectionSetupOutcome {
        changed_groups: false,
        group_membership_ready: false,
        uinput_rule_ready: true,
        udev_reload_succeeded: true,
    };
    let missing_rule = side_effects::InjectionSetupOutcome {
        changed_groups: false,
        group_membership_ready: true,
        uinput_rule_ready: false,
        udev_reload_succeeded: true,
    };

    assert!(rerun.can_finish_with_relogin_only(true));
    assert!(!failed_group_update.can_finish_with_relogin_only(true));
    assert!(!missing_rule.can_finish_with_relogin_only(true));
    assert!(!rerun.can_finish_with_relogin_only(false));
}

#[test]
fn udev_reload_requires_group_membership_and_rule() {
    let ready = side_effects::InjectionSetupOutcome {
        changed_groups: false,
        group_membership_ready: true,
        uinput_rule_ready: true,
        udev_reload_succeeded: false,
    };
    let missing_group = side_effects::InjectionSetupOutcome {
        changed_groups: false,
        group_membership_ready: false,
        uinput_rule_ready: true,
        udev_reload_succeeded: false,
    };
    let missing_rule = side_effects::InjectionSetupOutcome {
        changed_groups: false,
        group_membership_ready: true,
        uinput_rule_ready: false,
        udev_reload_succeeded: false,
    };

    assert!(ready.should_reload_udev());
    assert!(!missing_group.should_reload_udev());
    assert!(!missing_rule.should_reload_udev());
}

#[test]
fn incomplete_setup_keeps_manual_fix_lines_visible() {
    let readiness = InjectionReadinessReport::from_issues(vec![
        InjectionReadinessIssue::UinputPermissionDenied,
    ]);
    let setup = side_effects::InjectionSetupOutcome {
        changed_groups: true,
        group_membership_ready: true,
        uinput_rule_ready: false,
        udev_reload_succeeded: true,
    };

    assert_eq!(
        report::injection_readiness_info_message(&readiness, &setup),
        Some(
            "If you were just added to the `uinput` group, log out and back in after finishing the remaining paste injection steps.",
        )
    );
    assert!(!report::injection_readiness_fix_lines(&readiness, &setup).is_empty());
}

#[test]
fn current_username_lookup_retries_after_erange() {
    let expected_name = std::ffi::CString::new("whispers-test-user").expect("c string");
    let mut attempts = 0;

    let username =
        side_effects::current_username_for_uid_with(4242, |uid, passwd, _buffer, result| {
            attempts += 1;
            if attempts == 1 {
                return libc::ERANGE;
            }

            unsafe {
                *passwd = libc::passwd {
                    pw_name: expected_name.as_ptr() as *mut _,
                    pw_passwd: std::ptr::null_mut(),
                    pw_uid: uid,
                    pw_gid: 0,
                    pw_gecos: std::ptr::null_mut(),
                    pw_dir: std::ptr::null_mut(),
                    pw_shell: std::ptr::null_mut(),
                };
                *result = passwd;
            }

            0
        })
        .expect("retry should succeed");

    assert_eq!(username, "whispers-test-user");
    assert_eq!(attempts, 2);
}

#[test]
fn udev_trigger_waits_for_settle_before_rechecking() {
    assert!(side_effects::UDEV_TRIGGER_ARGS.contains(&"--settle"));
}

#[test]
fn setup_complete_message_stays_aligned_with_remaining_steps() {
    assert_eq!(
        report::setup_complete_message(false, true, true),
        "Log out and back in, then use whispers."
    );
    assert_eq!(
        report::setup_complete_message(false, true, false),
        "Log out and back in, then finish any remaining paste injection steps above before using whispers."
    );
    assert_eq!(
        report::setup_complete_message(false, true, false),
        "Log out and back in, then finish any remaining paste injection steps above before using whispers."
    );
    assert_eq!(
        report::setup_complete_message(false, false, true),
        "Log out and back in, then use whispers."
    );
    assert_eq!(
        report::setup_complete_message(false, false, false),
        "Finish the paste injection steps above, then use whispers."
    );
    assert_eq!(
        report::setup_complete_message(true, false, false),
        "You can now use whispers."
    );
}

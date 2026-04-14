use crate::config::Config;
use crate::error::WhsprError;

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
    assert!(warning.contains("Failed to add the current user"));
    assert!(warning.contains("group add failed"));
}

#[test]
fn group_membership_success_marks_logout_as_needed() {
    let mut outcome = side_effects::InjectionSetupOutcome::default();
    let warning =
        side_effects::record_group_membership_change_result(&mut outcome, "uinput", Ok(true));

    assert!(warning.is_none());
    assert!(outcome.changed_groups);
    assert!(!outcome.udev_reload_succeeded);
}

#[test]
fn group_change_messages_follow_recorded_reload_status() {
    let success = side_effects::InjectionSetupOutcome {
        changed_groups: true,
        udev_reload_succeeded: true,
    };
    let failed_reload = side_effects::InjectionSetupOutcome {
        changed_groups: true,
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
        Some("If you were just added to the `uinput` group, log out and back in before testing."),
    );
    assert_eq!(
        failed_reload.report_group_change_message(),
        Some(
            "If you were just added to the `uinput` group, log out and back in after finishing the remaining paste injection steps.",
        ),
    );
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
        report::setup_complete_message(false, false, false),
        "Finish the paste injection steps above, then use whispers."
    );
    assert_eq!(
        report::setup_complete_message(true, false, false),
        "You can now use whispers."
    );
}

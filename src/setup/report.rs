use crate::config::TranscriptionBackend;
use crate::inject::InjectionReadinessReport;
use crate::ui::SetupUi;

use super::{SetupSelections, side_effects::InjectionSetupOutcome};

pub(super) fn print_setup_intro(ui: &SetupUi) {
    ui.print_subtle(
        "Recommended models are listed first. Experimental backends are available, but not the default recommendation.",
    );
    ui.blank();
}

pub(super) fn print_setup_summary(ui: &SetupUi, selections: &SetupSelections) {
    ui.print_section("Setup summary");
    println!(
        "  {}: {} ({}, {}, {})",
        crate::ui::summary_key("ASR"),
        crate::ui::value(selections.asr_model.name),
        crate::ui::backend_token(selections.asr_model.backend.as_str()),
        crate::ui::scope_token(selections.asr_model.language_scope.as_str()),
        crate::ui::tier_token(selections.asr_model.support_tier.as_str())
    );

    if let Some(note) = selections.asr_model.setup_note {
        println!("  {}: {}", crate::ui::summary_key("ASR note"), note);
    }

    if selections.cloud.asr_enabled {
        println!(
            "  {}: enabled via {} (fallback: {})",
            crate::ui::summary_key("Cloud ASR"),
            crate::ui::provider_token(selections.cloud.provider.as_str()),
            selections.cloud.asr_fallback.as_str()
        );
    } else {
        println!("  {}: disabled", crate::ui::summary_key("Cloud ASR"));
    }

    match (selections.cloud.rewrite_enabled, selections.rewrite_model) {
        (true, Some(model)) => println!(
            "  {}: cloud with local fallback ({}, mode: {})",
            crate::ui::summary_key("Rewrite"),
            crate::ui::value(model),
            selections.postprocess_mode.as_str()
        ),
        (true, None) => println!(
            "  {}: cloud only (fallback: {}, mode: {})",
            crate::ui::summary_key("Rewrite"),
            selections.cloud.rewrite_fallback.as_str(),
            selections.postprocess_mode.as_str()
        ),
        (false, Some(model)) => println!(
            "  {}: local ({}, mode: {})",
            crate::ui::summary_key("Rewrite"),
            crate::ui::value(model),
            selections.postprocess_mode.as_str()
        ),
        (false, None) => println!(
            "  {}: disabled (raw mode)",
            crate::ui::summary_key("Rewrite")
        ),
    }

    if selections.asr_model.backend == TranscriptionBackend::Nemo && !selections.cloud.asr_enabled {
        println!(
            "  {}: first use may be slower than steady-state while the worker warms.",
            crate::ui::summary_key("NeMo note")
        );
    }
}

pub(super) fn print_injection_readiness(
    ui: &SetupUi,
    readiness: &InjectionReadinessReport,
    setup: &InjectionSetupOutcome,
) {
    ui.print_section("Paste injection");

    if readiness.is_ready() {
        ui.print_ok("Clipboard and virtual keyboard access look ready.");
        return;
    }

    ui.print_warn("Paste injection still needs one-time system setup.");
    for line in readiness.issue_lines() {
        println!("  - {line}");
    }

    if setup.changed_groups {
        if setup.udev_reload_succeeded {
            ui.print_info(
                "If you were just added to the `uinput` group, log out and back in before testing.",
            );
        } else {
            ui.print_info(
                "If you were just added to the `uinput` group, log out and back in after finishing the remaining paste injection steps.",
            );
        }
    }

    for line in readiness.fix_lines() {
        println!("  - {line}");
    }
}

pub(super) fn print_setup_complete(
    ui: &SetupUi,
    readiness: &InjectionReadinessReport,
    setup: &InjectionSetupOutcome,
) {
    ui.print_header("Setup complete");
    println!(
        "{}",
        setup_complete_message(
            readiness.is_ready(),
            setup.changed_groups,
            readiness.only_requires_relogin(),
            setup.udev_reload_succeeded,
        )
    );
    ui.print_section("Example keybind");
    ui.print_subtle("Bind it to a key in your compositor, e.g. for Hyprland:");
    println!("  bind = SUPER ALT, D, exec, whispers");
}

pub(super) fn setup_complete_message(
    is_ready: bool,
    changed_groups: bool,
    only_requires_relogin: bool,
    udev_reload_succeeded: bool,
) -> &'static str {
    if is_ready {
        "You can now use whispers."
    } else if changed_groups && only_requires_relogin && udev_reload_succeeded {
        "Log out and back in, then use whispers."
    } else if changed_groups {
        "Log out and back in, then finish any remaining paste injection steps above before using whispers."
    } else {
        "Finish the paste injection steps above, then use whispers."
    }
}

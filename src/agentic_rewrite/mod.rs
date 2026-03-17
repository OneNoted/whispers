mod admin;
mod runtime;
mod store;

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::error::Result;
use crate::rewrite_protocol::{RewriteCorrectionPolicy, RewriteSurfaceKind, RewriteTranscript};

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
pub struct ContextMatcher {
    pub surface_kind: Option<RewriteSurfaceKind>,
    pub app_id: Option<String>,
    pub window_title_contains: Option<String>,
    pub browser_domain_contains: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
struct AppRule {
    name: String,
    #[serde(flatten)]
    matcher: ContextMatcher,
    instructions: String,
    correction_policy: Option<RewriteCorrectionPolicy>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
struct GlossaryEntry {
    term: String,
    aliases: Vec<String>,
    #[serde(flatten)]
    matcher: ContextMatcher,
}

#[derive(Debug, Clone)]
struct PreparedGlossaryEntry {
    term: String,
    aliases: Vec<String>,
    matcher: ContextMatcher,
    normalized_aliases: Vec<Vec<String>>,
}

pub use runtime::conservative_output_allowed;

pub fn default_policy_path() -> &'static str {
    store::default_policy_path()
}

pub fn default_glossary_path() -> &'static str {
    store::default_glossary_path()
}

pub fn apply_runtime_policy(config: &Config, transcript: &mut RewriteTranscript) {
    let policy_rules = store::load_policy_file_for_runtime(&config.resolved_agentic_policy_path());
    let glossary_entries =
        store::load_glossary_file_for_runtime(&config.resolved_agentic_glossary_path());

    let policy_context = runtime::resolve_policy_context(
        config.agentic_rewrite.default_correction_policy,
        transcript.typing_context.as_ref(),
        &transcript.rewrite_candidates,
        &policy_rules,
        &glossary_entries,
    );

    for candidate in &policy_context.glossary_candidates {
        if transcript
            .rewrite_candidates
            .iter()
            .any(|existing| existing.text == candidate.text)
        {
            continue;
        }
        transcript.rewrite_candidates.push(candidate.clone());
    }

    transcript.policy_context = policy_context;
}

pub fn ensure_starter_files(config: &Config) -> Result<Vec<String>> {
    store::ensure_starter_files(config)
}

pub fn print_app_rule_path(config_override: Option<&Path>) -> Result<()> {
    admin::print_app_rule_path(config_override)
}

pub fn print_glossary_path(config_override: Option<&Path>) -> Result<()> {
    admin::print_glossary_path(config_override)
}

pub fn list_app_rules(config_override: Option<&Path>) -> Result<()> {
    admin::list_app_rules(config_override)
}

pub fn add_app_rule(
    config_override: Option<&Path>,
    name: &str,
    instructions: &str,
    matcher: ContextMatcher,
    correction_policy: Option<RewriteCorrectionPolicy>,
) -> Result<()> {
    admin::add_app_rule(
        config_override,
        name,
        instructions,
        matcher,
        correction_policy,
    )
}

pub fn remove_app_rule(config_override: Option<&Path>, name: &str) -> Result<()> {
    admin::remove_app_rule(config_override, name)
}

pub fn list_glossary(config_override: Option<&Path>) -> Result<()> {
    admin::list_glossary(config_override)
}

pub fn add_glossary_entry(
    config_override: Option<&Path>,
    term: &str,
    aliases: &[String],
    matcher: ContextMatcher,
) -> Result<()> {
    admin::add_glossary_entry(config_override, term, aliases, matcher)
}

pub fn remove_glossary_entry(config_override: Option<&Path>, term: &str) -> Result<()> {
    admin::remove_glossary_entry(config_override, term)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::rewrite_protocol::{
        RewriteCandidate, RewriteCandidateKind, RewritePolicyContext, RewriteTranscript,
        RewriteTypingContext,
    };

    fn typing_context(surface_kind: RewriteSurfaceKind) -> RewriteTypingContext {
        RewriteTypingContext {
            focus_fingerprint: "focus".into(),
            app_id: Some("dev.zed.Zed".into()),
            window_title: Some("docs.rs - serde_json".into()),
            surface_kind,
            browser_domain: Some("docs.rs".into()),
            captured_at_ms: 42,
        }
    }

    fn transcript_with_candidates(surface_kind: RewriteSurfaceKind) -> RewriteTranscript {
        RewriteTranscript {
            raw_text: "type script and sir dee json".into(),
            correction_aware_text: "type script and sir dee json".into(),
            aggressive_correction_text: None,
            detected_language: Some("en".into()),
            typing_context: Some(typing_context(surface_kind)),
            recent_session_entries: Vec::new(),
            session_backtrack_candidates: Vec::new(),
            recommended_session_candidate: None,
            segments: Vec::new(),
            edit_intents: Vec::new(),
            edit_signals: Vec::new(),
            edit_hypotheses: Vec::new(),
            rewrite_candidates: vec![RewriteCandidate {
                kind: RewriteCandidateKind::ConservativeCorrection,
                text: "type script and sir dee json".into(),
            }],
            recommended_candidate: None,
            policy_context: RewritePolicyContext::default(),
        }
    }

    #[test]
    fn apply_runtime_policy_adds_glossary_candidates() {
        let _env_lock = crate::test_support::env_lock();
        let _guard = crate::test_support::EnvVarGuard::capture(&[
            "HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
        ]);
        let home = crate::test_support::unique_temp_dir("agentic-runtime-home");
        crate::test_support::set_env("HOME", &home.to_string_lossy());
        crate::test_support::remove_env("XDG_CONFIG_HOME");
        crate::test_support::remove_env("XDG_DATA_HOME");

        let config = Config::default();
        let glossary_path = config.resolved_agentic_glossary_path();
        store::write_glossary_file(
            &glossary_path,
            &[GlossaryEntry {
                term: "TypeScript".into(),
                aliases: vec!["type script".into()],
                matcher: ContextMatcher {
                    surface_kind: Some(RewriteSurfaceKind::Editor),
                    ..ContextMatcher::default()
                },
            }],
        )
        .expect("write glossary");

        let mut transcript = transcript_with_candidates(RewriteSurfaceKind::Editor);
        apply_runtime_policy(&config, &mut transcript);
        assert!(
            transcript
                .rewrite_candidates
                .iter()
                .any(|candidate| candidate.text == "TypeScript and sir dee json")
        );
    }

    #[test]
    fn add_and_remove_roundtrip_for_policy_and_glossary() {
        let _env_lock = crate::test_support::env_lock();
        let _guard = crate::test_support::EnvVarGuard::capture(&[
            "HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
        ]);
        let home = crate::test_support::unique_temp_dir("agentic-cli-home");
        crate::test_support::set_env("HOME", &home.to_string_lossy());
        crate::test_support::remove_env("XDG_CONFIG_HOME");
        crate::test_support::remove_env("XDG_DATA_HOME");

        add_app_rule(
            None,
            "zed",
            "Preserve Rust identifiers.",
            ContextMatcher {
                app_id: Some("dev.zed.Zed".into()),
                ..ContextMatcher::default()
            },
            Some(RewriteCorrectionPolicy::Balanced),
        )
        .expect("add app rule");
        let config = Config::load(None).expect("config");
        let rules = store::read_policy_file(&config.resolved_agentic_policy_path()).expect("rules");
        assert_eq!(rules.len(), 1);

        add_glossary_entry(
            None,
            "serde_json",
            &[String::from("sir dee json")],
            ContextMatcher::default(),
        )
        .expect("add glossary entry");
        let entries =
            store::read_glossary_file(&config.resolved_agentic_glossary_path()).expect("entries");
        assert_eq!(entries.len(), 1);

        remove_app_rule(None, "zed").expect("remove app rule");
        remove_glossary_entry(None, "serde_json").expect("remove glossary entry");

        let rules = store::read_policy_file(&config.resolved_agentic_policy_path()).expect("rules");
        let entries =
            store::read_glossary_file(&config.resolved_agentic_glossary_path()).expect("entries");
        assert!(rules.is_empty());
        assert!(entries.is_empty());
    }
}

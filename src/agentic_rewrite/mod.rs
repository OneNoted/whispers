mod runtime;

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::config::{Config, PostprocessMode};
use crate::error::{Result, WhsprError};
use crate::rewrite_protocol::{RewriteCorrectionPolicy, RewriteSurfaceKind, RewriteTranscript};

const DEFAULT_POLICY_PATH: &str = "~/.local/share/whispers/app-rewrite-policy.toml";
const DEFAULT_GLOSSARY_PATH: &str = "~/.local/share/whispers/technical-glossary.toml";

const POLICY_STARTER: &str = r#"# App-aware rewrite policy for whispers agentic_rewrite mode.
# Rules are layered, not first-match. Matching rules apply in this order:
# global defaults, surface_kind, app_id, window_title_contains, browser_domain_contains.
# Later, more specific rules override earlier fields.
#
# Uncomment and edit the examples below.
#
# [[rules]]
# name = "terminal-shell"
# surface_kind = "terminal"
# correction_policy = "conservative"
# instructions = "Preserve commands, flags, paths, package names, and environment variables."
#
# [[rules]]
# name = "docs-rs-browser"
# surface_kind = "browser"
# browser_domain_contains = "docs.rs"
# instructions = "Preserve Rust crate names, module paths, and type identifiers."
#
# [[rules]]
# name = "zed-rust"
# app_id = "dev.zed.Zed"
# instructions = "Preserve identifiers, filenames, snake_case, camelCase, and Rust terminology."
"#;

const GLOSSARY_STARTER: &str = r#"# Technical glossary for whispers agentic_rewrite mode.
# Each entry defines a canonical term plus likely spoken or mis-transcribed aliases.
#
# Uncomment and edit the examples below.
#
# [[entries]]
# term = "TypeScript"
# aliases = ["type script", "types script"]
# surface_kind = "editor"
#
# [[entries]]
# term = "pyproject.toml"
# aliases = ["pie project dot toml", "pie project toml"]
# surface_kind = "terminal"
#
# [[entries]]
# term = "serde_json"
# aliases = ["sir dee json", "serdy json"]
# browser_domain_contains = "docs.rs"
"#;

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
struct PolicyFile {
    rules: Vec<AppRule>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
struct GlossaryFile {
    entries: Vec<GlossaryEntry>,
}

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
    DEFAULT_POLICY_PATH
}

pub fn default_glossary_path() -> &'static str {
    DEFAULT_GLOSSARY_PATH
}

pub fn apply_runtime_policy(config: &Config, transcript: &mut RewriteTranscript) {
    let policy_rules = load_policy_file_for_runtime(&config.resolved_agentic_policy_path());
    let glossary_entries = load_glossary_file_for_runtime(&config.resolved_agentic_glossary_path());

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
    if config.postprocess.mode != PostprocessMode::AgenticRewrite {
        return Ok(Vec::new());
    }

    let mut created = Vec::new();
    let policy_path = config.resolved_agentic_policy_path();
    if ensure_text_file(&policy_path, POLICY_STARTER)? {
        created.push(policy_path.display().to_string());
    }

    let glossary_path = config.resolved_agentic_glossary_path();
    if ensure_text_file(&glossary_path, GLOSSARY_STARTER)? {
        created.push(glossary_path.display().to_string());
    }

    Ok(created)
}

pub fn print_app_rule_path(config_override: Option<&Path>) -> Result<()> {
    let config = Config::load(config_override)?;
    println!("{}", config.resolved_agentic_policy_path().display());
    Ok(())
}

pub fn print_glossary_path(config_override: Option<&Path>) -> Result<()> {
    let config = Config::load(config_override)?;
    println!("{}", config.resolved_agentic_glossary_path().display());
    Ok(())
}

pub fn list_app_rules(config_override: Option<&Path>) -> Result<()> {
    let config = Config::load(config_override)?;
    let rules = read_policy_file(&config.resolved_agentic_policy_path())?;
    if rules.is_empty() {
        println!("No app rules configured.");
        return Ok(());
    }

    for rule in rules {
        println!(
            "{} | match: {} | correction_policy: {} | instructions: {}",
            rule.name,
            render_matcher(&rule.matcher),
            rule.correction_policy
                .map(|policy| policy.as_str())
                .unwrap_or("inherit"),
            single_line(&rule.instructions)
        );
    }

    Ok(())
}

pub fn add_app_rule(
    config_override: Option<&Path>,
    name: &str,
    instructions: &str,
    matcher: ContextMatcher,
    correction_policy: Option<RewriteCorrectionPolicy>,
) -> Result<()> {
    let config = Config::load(config_override)?;
    let path = config.resolved_agentic_policy_path();
    let mut rules = read_policy_file(&path)?;
    upsert_app_rule(
        &mut rules,
        AppRule {
            name: name.to_string(),
            matcher,
            instructions: instructions.to_string(),
            correction_policy,
        },
    );
    write_policy_file(&path, &rules)?;
    println!("Added app rule: {name}");
    println!("App rules updated: {}", path.display());
    Ok(())
}

pub fn remove_app_rule(config_override: Option<&Path>, name: &str) -> Result<()> {
    let config = Config::load(config_override)?;
    let path = config.resolved_agentic_policy_path();
    let mut rules = read_policy_file(&path)?;
    let removed = remove_app_rule_entry(&mut rules, name);
    write_policy_file(&path, &rules)?;
    if removed {
        println!("Removed app rule: {name}");
    } else {
        println!("No app rule matched: {name}");
    }
    println!("App rules updated: {}", path.display());
    Ok(())
}

pub fn list_glossary(config_override: Option<&Path>) -> Result<()> {
    let config = Config::load(config_override)?;
    let entries = read_glossary_file(&config.resolved_agentic_glossary_path())?;
    if entries.is_empty() {
        println!("No glossary entries configured.");
        return Ok(());
    }

    for entry in entries {
        let aliases = if entry.aliases.is_empty() {
            "-".to_string()
        } else {
            entry.aliases.join(", ")
        };
        println!(
            "{} | aliases: {} | match: {}",
            entry.term,
            aliases,
            render_matcher(&entry.matcher)
        );
    }

    Ok(())
}

pub fn add_glossary_entry(
    config_override: Option<&Path>,
    term: &str,
    aliases: &[String],
    matcher: ContextMatcher,
) -> Result<()> {
    let config = Config::load(config_override)?;
    let path = config.resolved_agentic_glossary_path();
    let mut entries = read_glossary_file(&path)?;
    upsert_glossary_entry(
        &mut entries,
        GlossaryEntry {
            term: term.to_string(),
            aliases: aliases.to_vec(),
            matcher,
        },
    );
    write_glossary_file(&path, &entries)?;
    println!("Added glossary entry: {term}");
    println!("Glossary updated: {}", path.display());
    Ok(())
}

pub fn remove_glossary_entry(config_override: Option<&Path>, term: &str) -> Result<()> {
    let config = Config::load(config_override)?;
    let path = config.resolved_agentic_glossary_path();
    let mut entries = read_glossary_file(&path)?;
    let removed = remove_glossary_entry_by_term(&mut entries, term);
    write_glossary_file(&path, &entries)?;
    if removed {
        println!("Removed glossary entry: {term}");
    } else {
        println!("No glossary entry matched: {term}");
    }
    println!("Glossary updated: {}", path.display());
    Ok(())
}

fn single_line(text: &str) -> String {
    text.trim().replace('\n', "\\n")
}

fn render_matcher(matcher: &ContextMatcher) -> String {
    let mut parts = Vec::new();
    if let Some(surface_kind) = matcher.surface_kind {
        parts.push(format!("surface_kind={}", surface_kind.as_str()));
    }
    if let Some(app_id) = matcher.app_id.as_deref() {
        parts.push(format!("app_id={app_id}"));
    }
    if let Some(window_title) = matcher.window_title_contains.as_deref() {
        parts.push(format!("window_title_contains={window_title}"));
    }
    if let Some(browser_domain) = matcher.browser_domain_contains.as_deref() {
        parts.push(format!("browser_domain_contains={browser_domain}"));
    }
    if parts.is_empty() {
        "global".to_string()
    } else {
        parts.join(", ")
    }
}

fn ensure_text_file(path: &Path, contents: &str) -> Result<bool> {
    if path.exists() {
        return Ok(false);
    }

    write_parent(path)?;
    std::fs::write(path, contents).map_err(|e| {
        WhsprError::Config(format!(
            "failed to write starter file {}: {e}",
            path.display()
        ))
    })?;
    Ok(true)
}

fn read_policy_file(path: &Path) -> Result<Vec<AppRule>> {
    if !path.exists() {
        return Ok(Vec::new());
    }

    let contents = std::fs::read_to_string(path).map_err(|e| {
        WhsprError::Config(format!("failed to read app rules {}: {e}", path.display()))
    })?;
    if contents.trim().is_empty() {
        return Ok(Vec::new());
    }
    let file: PolicyFile = toml::from_str(&contents).map_err(|e| {
        WhsprError::Config(format!("failed to parse app rules {}: {e}", path.display()))
    })?;
    Ok(file.rules)
}

fn write_policy_file(path: &Path, rules: &[AppRule]) -> Result<()> {
    write_parent(path)?;
    let contents = toml::to_string_pretty(&PolicyFile {
        rules: rules.to_vec(),
    })
    .map_err(|e| WhsprError::Config(format!("failed to encode app rules: {e}")))?;
    std::fs::write(path, contents).map_err(|e| {
        WhsprError::Config(format!("failed to write app rules {}: {e}", path.display()))
    })?;
    Ok(())
}

fn read_glossary_file(path: &Path) -> Result<Vec<GlossaryEntry>> {
    if !path.exists() {
        return Ok(Vec::new());
    }

    let contents = std::fs::read_to_string(path).map_err(|e| {
        WhsprError::Config(format!("failed to read glossary {}: {e}", path.display()))
    })?;
    if contents.trim().is_empty() {
        return Ok(Vec::new());
    }
    let file: GlossaryFile = toml::from_str(&contents).map_err(|e| {
        WhsprError::Config(format!("failed to parse glossary {}: {e}", path.display()))
    })?;
    Ok(file.entries)
}

fn write_glossary_file(path: &Path, entries: &[GlossaryEntry]) -> Result<()> {
    write_parent(path)?;
    let contents = toml::to_string_pretty(&GlossaryFile {
        entries: entries.to_vec(),
    })
    .map_err(|e| WhsprError::Config(format!("failed to encode glossary: {e}")))?;
    std::fs::write(path, contents).map_err(|e| {
        WhsprError::Config(format!("failed to write glossary {}: {e}", path.display()))
    })?;
    Ok(())
}

fn load_policy_file_for_runtime(path: &Path) -> Vec<AppRule> {
    match read_policy_file(path) {
        Ok(rules) => rules,
        Err(err) => {
            tracing::warn!("{err}; using built-in app rewrite defaults");
            Vec::new()
        }
    }
}

fn load_glossary_file_for_runtime(path: &Path) -> Vec<GlossaryEntry> {
    match read_glossary_file(path) {
        Ok(entries) => entries,
        Err(err) => {
            tracing::warn!("{err}; ignoring runtime glossary");
            Vec::new()
        }
    }
}

fn write_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            WhsprError::Config(format!(
                "failed to create directory {}: {e}",
                parent.display()
            ))
        })?;
    }
    Ok(())
}

fn upsert_app_rule(rules: &mut Vec<AppRule>, rule: AppRule) {
    if let Some(existing) = rules.iter_mut().find(|existing| existing.name == rule.name) {
        *existing = rule;
        return;
    }
    rules.push(rule);
}

fn remove_app_rule_entry(rules: &mut Vec<AppRule>, name: &str) -> bool {
    let before = rules.len();
    rules.retain(|rule| rule.name != name);
    before != rules.len()
}

fn upsert_glossary_entry(entries: &mut Vec<GlossaryEntry>, entry: GlossaryEntry) {
    if let Some(existing) = entries
        .iter_mut()
        .find(|existing| existing.term == entry.term)
    {
        *existing = entry;
        return;
    }
    entries.push(entry);
}

fn remove_glossary_entry_by_term(entries: &mut Vec<GlossaryEntry>, term: &str) -> bool {
    let before = entries.len();
    entries.retain(|entry| entry.term != term);
    before != entries.len()
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
        write_glossary_file(
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
        let rules = read_policy_file(&config.resolved_agentic_policy_path()).expect("rules");
        assert_eq!(rules.len(), 1);

        add_glossary_entry(
            None,
            "serde_json",
            &[String::from("sir dee json")],
            ContextMatcher::default(),
        )
        .expect("add glossary entry");
        let entries =
            read_glossary_file(&config.resolved_agentic_glossary_path()).expect("entries");
        assert_eq!(entries.len(), 1);

        remove_app_rule(None, "zed").expect("remove app rule");
        remove_glossary_entry(None, "serde_json").expect("remove glossary entry");

        let rules = read_policy_file(&config.resolved_agentic_policy_path()).expect("rules");
        let entries =
            read_glossary_file(&config.resolved_agentic_glossary_path()).expect("entries");
        assert!(rules.is_empty());
        assert!(entries.is_empty());
    }
}

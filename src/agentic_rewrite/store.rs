use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::error::{Result, WhsprError};

use super::{AppRule, GlossaryEntry};

const DEFAULT_POLICY_PATH: &str = "~/.local/share/whispers/app-rewrite-policy.toml";
const DEFAULT_GLOSSARY_PATH: &str = "~/.local/share/whispers/technical-glossary.toml";

const POLICY_STARTER: &str = r#"# App-aware rewrite policy for whispers rewrite mode.
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

const GLOSSARY_STARTER: &str = r#"# Technical glossary for whispers rewrite mode.
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

pub(super) fn default_policy_path() -> &'static str {
    DEFAULT_POLICY_PATH
}

pub(super) fn default_glossary_path() -> &'static str {
    DEFAULT_GLOSSARY_PATH
}

pub(super) fn ensure_starter_files(config: &Config) -> Result<Vec<String>> {
    if !config.postprocess.mode.uses_rewrite() {
        return Ok(Vec::new());
    }

    let mut created = Vec::new();
    let policy_path = config.resolved_rewrite_policy_path();
    if ensure_text_file(&policy_path, POLICY_STARTER)? {
        created.push(policy_path.display().to_string());
    }

    let glossary_path = config.resolved_rewrite_glossary_path();
    if ensure_text_file(&glossary_path, GLOSSARY_STARTER)? {
        created.push(glossary_path.display().to_string());
    }

    Ok(created)
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

pub(super) fn read_policy_file(path: &Path) -> Result<Vec<AppRule>> {
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

pub(super) fn write_policy_file(path: &Path, rules: &[AppRule]) -> Result<()> {
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

pub(super) fn read_glossary_file(path: &Path) -> Result<Vec<GlossaryEntry>> {
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

pub(super) fn write_glossary_file(path: &Path, entries: &[GlossaryEntry]) -> Result<()> {
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

pub(super) fn load_policy_file_for_runtime(path: &Path) -> Vec<AppRule> {
    match read_policy_file(path) {
        Ok(rules) => rules,
        Err(err) => {
            tracing::warn!("{err}; using built-in app rewrite defaults");
            Vec::new()
        }
    }
}

pub(super) fn load_glossary_file_for_runtime(path: &Path) -> Vec<GlossaryEntry> {
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

pub(super) fn upsert_app_rule(rules: &mut Vec<AppRule>, rule: AppRule) {
    if let Some(existing) = rules.iter_mut().find(|existing| existing.name == rule.name) {
        *existing = rule;
        return;
    }
    rules.push(rule);
}

pub(super) fn remove_app_rule_entry(rules: &mut Vec<AppRule>, name: &str) -> bool {
    let before = rules.len();
    rules.retain(|rule| rule.name != name);
    before != rules.len()
}

pub(super) fn upsert_glossary_entry(entries: &mut Vec<GlossaryEntry>, entry: GlossaryEntry) {
    if let Some(existing) = entries
        .iter_mut()
        .find(|existing| existing.term == entry.term)
    {
        *existing = entry;
        return;
    }
    entries.push(entry);
}

pub(super) fn remove_glossary_entry_by_term(entries: &mut Vec<GlossaryEntry>, term: &str) -> bool {
    let before = entries.len();
    entries.retain(|entry| entry.term != term);
    before != entries.len()
}

use std::path::Path;

use crate::config::Config;
use crate::error::Result;
use crate::rewrite_protocol::RewriteCorrectionPolicy;

use super::{AppRule, ContextMatcher, GlossaryEntry, store};

pub(super) fn print_app_rule_path(config_override: Option<&Path>) -> Result<()> {
    let config = Config::load(config_override)?;
    println!("{}", config.resolved_agentic_policy_path().display());
    Ok(())
}

pub(super) fn print_glossary_path(config_override: Option<&Path>) -> Result<()> {
    let config = Config::load(config_override)?;
    println!("{}", config.resolved_agentic_glossary_path().display());
    Ok(())
}

pub(super) fn list_app_rules(config_override: Option<&Path>) -> Result<()> {
    let config = Config::load(config_override)?;
    let rules = store::read_policy_file(&config.resolved_agentic_policy_path())?;
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

pub(super) fn add_app_rule(
    config_override: Option<&Path>,
    name: &str,
    instructions: &str,
    matcher: ContextMatcher,
    correction_policy: Option<RewriteCorrectionPolicy>,
) -> Result<()> {
    let config = Config::load(config_override)?;
    let path = config.resolved_agentic_policy_path();
    let mut rules = store::read_policy_file(&path)?;
    store::upsert_app_rule(
        &mut rules,
        AppRule {
            name: name.to_string(),
            matcher,
            instructions: instructions.to_string(),
            correction_policy,
        },
    );
    store::write_policy_file(&path, &rules)?;
    println!("Added app rule: {name}");
    println!("App rules updated: {}", path.display());
    Ok(())
}

pub(super) fn remove_app_rule(config_override: Option<&Path>, name: &str) -> Result<()> {
    let config = Config::load(config_override)?;
    let path = config.resolved_agentic_policy_path();
    let mut rules = store::read_policy_file(&path)?;
    let removed = store::remove_app_rule_entry(&mut rules, name);
    store::write_policy_file(&path, &rules)?;
    if removed {
        println!("Removed app rule: {name}");
    } else {
        println!("No app rule matched: {name}");
    }
    println!("App rules updated: {}", path.display());
    Ok(())
}

pub(super) fn list_glossary(config_override: Option<&Path>) -> Result<()> {
    let config = Config::load(config_override)?;
    let entries = store::read_glossary_file(&config.resolved_agentic_glossary_path())?;
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

pub(super) fn add_glossary_entry(
    config_override: Option<&Path>,
    term: &str,
    aliases: &[String],
    matcher: ContextMatcher,
) -> Result<()> {
    let config = Config::load(config_override)?;
    let path = config.resolved_agentic_glossary_path();
    let mut entries = store::read_glossary_file(&path)?;
    store::upsert_glossary_entry(
        &mut entries,
        GlossaryEntry {
            term: term.to_string(),
            aliases: aliases.to_vec(),
            matcher,
        },
    );
    store::write_glossary_file(&path, &entries)?;
    println!("Added glossary entry: {term}");
    println!("Glossary updated: {}", path.display());
    Ok(())
}

pub(super) fn remove_glossary_entry(config_override: Option<&Path>, term: &str) -> Result<()> {
    let config = Config::load(config_override)?;
    let path = config.resolved_agentic_glossary_path();
    let mut entries = store::read_glossary_file(&path)?;
    let removed = store::remove_glossary_entry_by_term(&mut entries, term);
    store::write_glossary_file(&path, &entries)?;
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

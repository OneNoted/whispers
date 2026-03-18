use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::error::{Result, WhsprError};
use crate::rewrite_protocol::{
    RewriteCandidate, RewriteCandidateKind, RewriteCorrectionPolicy, RewritePolicyContext,
    RewritePolicyGlossaryTerm, RewriteSurfaceKind, RewriteTranscript, RewriteTypingContext,
};

const DEFAULT_POLICY_PATH: &str = "~/.local/share/whispers/app-rewrite-policy.toml";
const DEFAULT_GLOSSARY_PATH: &str = "~/.local/share/whispers/technical-glossary.toml";
const MAX_GLOSSARY_CANDIDATES: usize = 4;

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

#[derive(Debug, Clone)]
struct WordSpan {
    start: usize,
    end: usize,
    normalized: String,
}

impl ContextMatcher {
    fn matches(&self, context: Option<&RewriteTypingContext>) -> bool {
        if self.is_empty() {
            return true;
        }

        let Some(context) = context else {
            return false;
        };

        if let Some(surface_kind) = self.surface_kind
            && context.surface_kind != surface_kind
        {
            return false;
        }

        if let Some(app_id) = self.app_id.as_deref()
            && context.app_id.as_deref() != Some(app_id)
        {
            return false;
        }

        if let Some(needle) = self.window_title_contains.as_deref()
            && !contains_ignore_ascii_case(context.window_title.as_deref(), needle)
        {
            return false;
        }

        if let Some(needle) = self.browser_domain_contains.as_deref()
            && !contains_ignore_ascii_case(context.browser_domain.as_deref(), needle)
        {
            return false;
        }

        true
    }

    fn specificity_rank(&self) -> (u8, u8) {
        let matcher_count = [
            self.surface_kind.is_some(),
            self.app_id.is_some(),
            self.window_title_contains.is_some(),
            self.browser_domain_contains.is_some(),
        ]
        .into_iter()
        .filter(|present| *present)
        .count() as u8;
        let strongest_layer = if self.browser_domain_contains.is_some() {
            4
        } else if self.window_title_contains.is_some() {
            3
        } else if self.app_id.is_some() {
            2
        } else if self.surface_kind.is_some() {
            1
        } else {
            0
        };
        (matcher_count, strongest_layer)
    }

    fn is_empty(&self) -> bool {
        self.surface_kind.is_none()
            && self.app_id.is_none()
            && self.window_title_contains.is_none()
            && self.browser_domain_contains.is_none()
    }
}

impl AppRule {
    fn built_in(
        name: &str,
        matcher: ContextMatcher,
        instructions: &str,
        correction_policy: Option<RewriteCorrectionPolicy>,
    ) -> Self {
        Self {
            name: name.to_string(),
            matcher,
            instructions: instructions.to_string(),
            correction_policy,
        }
    }
}

impl PreparedGlossaryEntry {
    fn new(entry: GlossaryEntry) -> Option<Self> {
        let term = entry.term.trim().to_string();
        if term.is_empty() {
            return None;
        }

        let aliases = entry
            .aliases
            .into_iter()
            .map(|alias| alias.trim().to_string())
            .filter(|alias| !alias.is_empty())
            .collect::<Vec<_>>();
        let normalized_aliases = aliases
            .iter()
            .map(|alias| normalized_words(alias))
            .filter(|words| !words.is_empty())
            .collect::<Vec<_>>();

        Some(Self {
            term,
            aliases,
            matcher: entry.matcher,
            normalized_aliases,
        })
    }
}

pub fn default_policy_path() -> &'static str {
    DEFAULT_POLICY_PATH
}

pub fn default_glossary_path() -> &'static str {
    DEFAULT_GLOSSARY_PATH
}

pub fn apply_runtime_policy(config: &Config, transcript: &mut RewriteTranscript) {
    let policy_rules = load_policy_file_for_runtime(&config.resolved_agentic_policy_path());
    let glossary_entries = load_glossary_file_for_runtime(&config.resolved_agentic_glossary_path());

    let policy_context = resolve_policy_context(
        config.rewrite.default_correction_policy,
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
    promote_policy_preferred_candidate(transcript);
}

fn promote_policy_preferred_candidate(transcript: &mut RewriteTranscript) {
    if matches!(
        transcript.policy_context.correction_policy,
        RewriteCorrectionPolicy::Conservative
    ) {
        return;
    }

    let preferred = transcript
        .policy_context
        .glossary_candidates
        .iter()
        .find(|candidate| {
            let text = candidate.text.trim();
            !text.is_empty() && text != transcript.correction_aware_text.trim()
        })
        .cloned();

    let Some(preferred) = preferred else {
        return;
    };

    if transcript.recommended_candidate.is_some() {
        return;
    }

    transcript.recommended_candidate = Some(preferred.clone());
    tracing::debug!(
        preferred_candidate = preferred.text,
        "promoted glossary-backed rewrite candidate to recommended candidate"
    );

    if let Some(index) = transcript
        .rewrite_candidates
        .iter()
        .position(|candidate| candidate.text == preferred.text)
        && index > 0
    {
        let candidate = transcript.rewrite_candidates.remove(index);
        transcript.rewrite_candidates.insert(0, candidate);
    }
}

pub fn conservative_output_allowed(transcript: &RewriteTranscript, text: &str) -> bool {
    let text = text.trim();
    if text.is_empty() {
        return false;
    }

    transcript
        .rewrite_candidates
        .iter()
        .any(|candidate| candidate_supports_output(&candidate.text, text))
        || transcript
            .policy_context
            .glossary_candidates
            .iter()
            .any(|candidate| candidate_supports_output(&candidate.text, text))
}

fn candidate_supports_output(candidate: &str, output: &str) -> bool {
    if candidate.trim() == output.trim() {
        return true;
    }

    let candidate_words = normalized_words(candidate);
    let output_words = normalized_words(output);
    if candidate_words.is_empty() || output_words.is_empty() {
        return false;
    }

    if candidate_words == output_words {
        return true;
    }

    // Conservative mode should still allow small technical-term normalizations
    // when the output clearly preserves a longer sentence candidate.
    if candidate_words.len() != output_words.len() || candidate_words.len() < 4 {
        return false;
    }

    let differing_pairs = candidate_words
        .iter()
        .zip(&output_words)
        .filter(|(candidate_word, output_word)| candidate_word != output_word)
        .collect::<Vec<_>>();
    if differing_pairs.is_empty() || differing_pairs.len() > 2 {
        return false;
    }

    differing_pairs
        .into_iter()
        .all(|(candidate_word, output_word)| {
            is_minor_term_normalization(candidate_word, output_word)
        })
}

fn is_minor_term_normalization(candidate_word: &str, output_word: &str) -> bool {
    let candidate_len = candidate_word.chars().count();
    let output_len = output_word.chars().count();
    let max_len = candidate_len.max(output_len);
    if max_len < 3 {
        return false;
    }

    let distance = levenshtein_distance(candidate_word, output_word);
    distance > 0 && distance <= 3 && distance * 2 <= max_len + 1
}

fn levenshtein_distance(left: &str, right: &str) -> usize {
    if left == right {
        return 0;
    }

    let right_chars = right.chars().collect::<Vec<_>>();
    let mut previous = (0..=right_chars.len()).collect::<Vec<_>>();
    let mut current = vec![0; right_chars.len() + 1];

    for (left_index, left_char) in left.chars().enumerate() {
        current[0] = left_index + 1;
        for (right_index, right_char) in right_chars.iter().enumerate() {
            let substitution_cost = usize::from(left_char != *right_char);
            current[right_index + 1] = (previous[right_index + 1] + 1)
                .min(current[right_index] + 1)
                .min(previous[right_index] + substitution_cost);
        }
        std::mem::swap(&mut previous, &mut current);
    }

    previous[right_chars.len()]
}

pub fn ensure_starter_files(config: &Config) -> Result<Vec<String>> {
    if !config.postprocess.mode.uses_rewrite() {
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

fn resolve_policy_context(
    default_policy: RewriteCorrectionPolicy,
    context: Option<&RewriteTypingContext>,
    rewrite_candidates: &[RewriteCandidate],
    policy_rules: &[AppRule],
    glossary_entries: &[GlossaryEntry],
) -> RewritePolicyContext {
    let mut matched_rule_names = Vec::new();
    let mut effective_rule_instructions = Vec::new();
    let mut correction_policy = default_policy;

    for rule in built_in_rules(default_policy)
        .into_iter()
        .filter(|rule| rule.matcher.matches(context))
        .chain(matching_rules(policy_rules, context))
    {
        matched_rule_names.push(rule.name.clone());
        if let Some(policy) = rule.correction_policy {
            correction_policy = policy;
        }

        let instructions = rule.instructions.trim();
        if !instructions.is_empty() {
            effective_rule_instructions.push(instructions.to_string());
        }
    }

    let mut active_glossary_entries = built_in_glossary_entries()
        .into_iter()
        .chain(glossary_entries.iter().cloned())
        .enumerate()
        .filter_map(|(index, entry)| PreparedGlossaryEntry::new(entry).map(|entry| (index, entry)))
        .filter(|(_, entry)| entry.matcher.matches(context))
        .collect::<Vec<_>>();
    active_glossary_entries
        .sort_by_key(|(index, entry)| (entry.matcher.specificity_rank(), *index));
    let active_glossary_entries = active_glossary_entries
        .into_iter()
        .map(|(_, entry)| entry)
        .collect::<Vec<_>>();

    RewritePolicyContext {
        correction_policy,
        matched_rule_names,
        effective_rule_instructions,
        active_glossary_terms: collapse_glossary_terms(&active_glossary_entries),
        glossary_candidates: build_glossary_candidates(
            rewrite_candidates,
            &active_glossary_entries,
        ),
    }
}

fn built_in_rules(default_policy: RewriteCorrectionPolicy) -> Vec<AppRule> {
    vec![
        AppRule::built_in(
            "baseline/global-default",
            ContextMatcher::default(),
            "Use the active typing context, recent dictation context, glossary terms, and bounded candidates to resolve technical dictation cleanly while keeping the final-text-only contract. When the utterance clearly points to software, tools, APIs, Linux components, product names, or other technical concepts, prefer the most plausible intended technical term over a phonetically similar common word. If a token is an obvious phonetic near-miss for a technical term and nearby category words make the intended term clear, proactively normalize it to the canonical spelling. Use category cues like window manager, editor, language, library, shell, or package manager to disambiguate nearby technical names. If multiple plausible technical interpretations remain similarly credible, stay close to the transcript.",
            Some(default_policy),
        ),
        AppRule::built_in(
            "baseline/browser",
            ContextMatcher {
                surface_kind: Some(RewriteSurfaceKind::Browser),
                ..ContextMatcher::default()
            },
            "Favor clean prose and natural punctuation for browser text fields, but stay grounded in the listed candidates, glossary evidence, and the utterance's technical topic when it clearly refers to software or documentation. Correct obvious phonetic misses of technical terms or product names when the surrounding sentence makes the intended term clear.",
            Some(RewriteCorrectionPolicy::Balanced),
        ),
        AppRule::built_in(
            "baseline/generic-text",
            ContextMatcher {
                surface_kind: Some(RewriteSurfaceKind::GenericText),
                ..ContextMatcher::default()
            },
            "Favor clean prose and natural punctuation for general text entry while staying grounded in the listed candidates and glossary evidence. If the utterance clearly discusses technical tools or software, prefer the most plausible technical term over a phonetically similar common word and proactively fix obvious phonetic near-misses to the canonical spelling.",
            Some(RewriteCorrectionPolicy::Balanced),
        ),
        AppRule::built_in(
            "baseline/editor",
            ContextMatcher {
                surface_kind: Some(RewriteSurfaceKind::Editor),
                ..ContextMatcher::default()
            },
            "Preserve identifiers, filenames, API names, symbols, and technical casing. Avoid rewriting technical wording into generic prose. Infer likely technical terms and proper names from the utterance when the topic is clearly code, tooling, or software, and proactively normalize obvious phonetic misses to the canonical technical spelling.",
            Some(RewriteCorrectionPolicy::Balanced),
        ),
        AppRule::built_in(
            "baseline/terminal",
            ContextMatcher {
                surface_kind: Some(RewriteSurfaceKind::Terminal),
                ..ContextMatcher::default()
            },
            "Preserve commands, flags, paths, package names, environment variables, and punctuation that changes command meaning. Infer technical commands or package names only when the utterance strongly supports them. If uncertain, prefer the closest listed candidate.",
            Some(RewriteCorrectionPolicy::Conservative),
        ),
    ]
}

fn built_in_glossary_entries() -> Vec<GlossaryEntry> {
    vec![
        GlossaryEntry {
            term: "TypeScript".into(),
            aliases: vec!["type script".into(), "types script".into()],
            matcher: ContextMatcher {
                surface_kind: Some(RewriteSurfaceKind::Editor),
                ..ContextMatcher::default()
            },
        },
        GlossaryEntry {
            term: "Neovim".into(),
            aliases: vec!["neo vim".into(), "neo-vim".into()],
            matcher: ContextMatcher::default(),
        },
        GlossaryEntry {
            term: "Hyprland".into(),
            aliases: vec!["hyperland".into(), "hyper land".into(), "highprland".into()],
            matcher: ContextMatcher::default(),
        },
        GlossaryEntry {
            term: "niri".into(),
            aliases: vec!["neary".into(), "niry".into(), "nearie".into()],
            matcher: ContextMatcher::default(),
        },
        GlossaryEntry {
            term: "Sway".into(),
            aliases: vec!["sui".into(), "swayy".into()],
            matcher: ContextMatcher::default(),
        },
    ]
}

fn matching_rules(rules: &[AppRule], context: Option<&RewriteTypingContext>) -> Vec<AppRule> {
    let mut matches = rules
        .iter()
        .enumerate()
        .filter(|(_, rule)| rule.matcher.matches(context))
        .collect::<Vec<_>>();
    matches.sort_by_key(|(index, rule)| (rule.matcher.specificity_rank(), *index));
    matches.into_iter().map(|(_, rule)| rule.clone()).collect()
}

fn collapse_glossary_terms(entries: &[PreparedGlossaryEntry]) -> Vec<RewritePolicyGlossaryTerm> {
    let mut collapsed = Vec::<RewritePolicyGlossaryTerm>::new();
    for entry in entries {
        if let Some(existing) = collapsed
            .iter_mut()
            .find(|candidate| candidate.term == entry.term)
        {
            for alias in &entry.aliases {
                if !existing
                    .aliases
                    .iter()
                    .any(|existing_alias| existing_alias == alias)
                {
                    existing.aliases.push(alias.clone());
                }
            }
            continue;
        }

        collapsed.push(RewritePolicyGlossaryTerm {
            term: entry.term.clone(),
            aliases: entry.aliases.clone(),
        });
    }
    collapsed
}

fn build_glossary_candidates(
    rewrite_candidates: &[RewriteCandidate],
    glossary_entries: &[PreparedGlossaryEntry],
) -> Vec<RewriteCandidate> {
    let mut generated = Vec::new();
    for candidate in rewrite_candidates {
        if generated.len() >= MAX_GLOSSARY_CANDIDATES {
            break;
        }

        if let Some(text) = apply_glossary_entries(&candidate.text, glossary_entries)
            && text != candidate.text
            && !generated
                .iter()
                .any(|existing: &RewriteCandidate| existing.text == text)
            && !rewrite_candidates
                .iter()
                .any(|existing| existing.text == text)
        {
            generated.push(RewriteCandidate {
                kind: RewriteCandidateKind::GlossaryCorrection,
                text,
            });
        }
    }
    generated
}

fn apply_glossary_entries(text: &str, entries: &[PreparedGlossaryEntry]) -> Option<String> {
    if text.trim().is_empty() || entries.is_empty() {
        return None;
    }

    let mut output = text.to_string();
    let mut applied_any = false;
    for entry in entries {
        let updated = replace_aliases(&output, entry);
        if updated != output {
            applied_any = true;
            output = updated;
        }
    }

    applied_any.then_some(output.trim().to_string())
}

fn replace_aliases(text: &str, entry: &PreparedGlossaryEntry) -> String {
    if entry.normalized_aliases.is_empty() {
        return text.to_string();
    }

    let spans = collect_word_spans(text);
    if spans.is_empty() {
        return text.to_string();
    }

    let mut output = String::new();
    let mut cursor = 0usize;
    let mut index = 0usize;

    while index < spans.len() {
        let Some(alias_len) = best_alias_match(&spans, index, &entry.normalized_aliases) else {
            index += 1;
            continue;
        };

        output.push_str(&text[cursor..spans[index].start]);
        output.push_str(&entry.term);
        cursor = spans[index + alias_len - 1].end;
        index += alias_len;
    }

    output.push_str(&text[cursor..]);
    output
}

fn best_alias_match(spans: &[WordSpan], index: usize, aliases: &[Vec<String>]) -> Option<usize> {
    aliases
        .iter()
        .filter(|alias| matches_words(spans, index, alias))
        .map(Vec::len)
        .max()
}

fn matches_words(spans: &[WordSpan], index: usize, words: &[String]) -> bool {
    if words.is_empty() || index + words.len() > spans.len() {
        return false;
    }

    spans[index..index + words.len()]
        .iter()
        .zip(words)
        .all(|(span, word)| span.normalized == *word)
}

fn collect_word_spans(text: &str) -> Vec<WordSpan> {
    let mut spans = Vec::new();
    let mut current_start = None;

    for (index, ch) in text.char_indices() {
        if is_word_char(ch) {
            current_start.get_or_insert(index);
            continue;
        }

        if let Some(start) = current_start.take() {
            spans.push(WordSpan {
                start,
                end: index,
                normalized: normalize_word(&text[start..index]),
            });
        }
    }

    if let Some(start) = current_start {
        spans.push(WordSpan {
            start,
            end: text.len(),
            normalized: normalize_word(&text[start..]),
        });
    }

    spans
}

fn normalized_words(text: &str) -> Vec<String> {
    collect_word_spans(text)
        .into_iter()
        .map(|span| span.normalized)
        .collect()
}

fn normalize_word(word: &str) -> String {
    word.chars()
        .filter(|ch| is_word_char(*ch))
        .flat_map(|ch| ch.to_lowercase())
        .collect()
}

fn is_word_char(ch: char) -> bool {
    ch.is_alphanumeric() || matches!(ch, '\'' | '-' | '_')
}

fn contains_ignore_ascii_case(haystack: Option<&str>, needle: &str) -> bool {
    let Some(haystack) = haystack else {
        return false;
    };
    haystack
        .to_ascii_lowercase()
        .contains(&needle.to_ascii_lowercase())
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
            edit_context: crate::rewrite_protocol::RewriteEditContext::default(),
            policy_context: RewritePolicyContext::default(),
        }
    }

    #[test]
    fn built_in_terminal_policy_is_conservative() {
        let context = typing_context(RewriteSurfaceKind::Terminal);
        let policy = resolve_policy_context(
            RewriteCorrectionPolicy::Balanced,
            Some(&context),
            &[],
            &[],
            &[],
        );
        assert_eq!(
            policy.correction_policy,
            RewriteCorrectionPolicy::Conservative
        );
        assert!(
            policy
                .matched_rule_names
                .iter()
                .any(|name| name == "baseline/terminal")
        );
    }

    #[test]
    fn built_in_policy_guides_technical_term_inference() {
        let context = typing_context(RewriteSurfaceKind::GenericText);
        let policy = resolve_policy_context(
            RewriteCorrectionPolicy::Balanced,
            Some(&context),
            &[],
            &[],
            &[],
        );
        assert!(
            policy
                .effective_rule_instructions
                .iter()
                .any(|instruction| instruction
                    .contains("proactively fix obvious phonetic near-misses"))
        );
        assert!(
            policy
                .effective_rule_instructions
                .iter()
                .any(|instruction| instruction.contains("obvious phonetic near-miss"))
        );
    }

    #[test]
    fn more_specific_rules_override_less_specific_rules() {
        let rules = vec![
            AppRule {
                name: "surface".into(),
                matcher: ContextMatcher {
                    surface_kind: Some(RewriteSurfaceKind::Editor),
                    ..ContextMatcher::default()
                },
                instructions: "surface".into(),
                correction_policy: Some(RewriteCorrectionPolicy::Balanced),
            },
            AppRule {
                name: "app".into(),
                matcher: ContextMatcher {
                    app_id: Some("dev.zed.Zed".into()),
                    ..ContextMatcher::default()
                },
                instructions: "app".into(),
                correction_policy: Some(RewriteCorrectionPolicy::Aggressive),
            },
        ];
        let context = typing_context(RewriteSurfaceKind::Editor);
        let policy = resolve_policy_context(
            RewriteCorrectionPolicy::Balanced,
            Some(&context),
            &[],
            &rules,
            &[],
        );
        assert_eq!(
            policy.correction_policy,
            RewriteCorrectionPolicy::Aggressive
        );
        assert_eq!(
            policy
                .effective_rule_instructions
                .last()
                .map(String::as_str),
            Some("app")
        );
    }

    #[test]
    fn glossary_candidates_follow_matching_scope() {
        let glossary = vec![
            GlossaryEntry {
                term: "TypeScript".into(),
                aliases: vec!["type script".into()],
                matcher: ContextMatcher {
                    surface_kind: Some(RewriteSurfaceKind::Editor),
                    ..ContextMatcher::default()
                },
            },
            GlossaryEntry {
                term: "serde_json".into(),
                aliases: vec!["sir dee json".into()],
                matcher: ContextMatcher {
                    browser_domain_contains: Some("docs.rs".into()),
                    ..ContextMatcher::default()
                },
            },
        ];
        let policy = resolve_policy_context(
            RewriteCorrectionPolicy::Balanced,
            Some(&typing_context(RewriteSurfaceKind::Editor)),
            &[RewriteCandidate {
                kind: RewriteCandidateKind::Literal,
                text: "type script and sir dee json".into(),
            }],
            &[],
            &glossary,
        );
        assert!(
            policy
                .active_glossary_terms
                .iter()
                .any(|term| term.term == "TypeScript")
        );
        assert!(
            policy
                .active_glossary_terms
                .iter()
                .any(|term| term.term == "serde_json")
        );
        assert_eq!(policy.glossary_candidates.len(), 1);
        assert_eq!(
            policy.glossary_candidates[0].text,
            "TypeScript and serde_json"
        );
    }

    #[test]
    fn conservative_acceptance_requires_explicit_candidate() {
        let mut transcript = transcript_with_candidates(RewriteSurfaceKind::Terminal);
        transcript.policy_context.correction_policy = RewriteCorrectionPolicy::Conservative;
        transcript.policy_context.glossary_candidates = vec![RewriteCandidate {
            kind: RewriteCandidateKind::GlossaryCorrection,
            text: "TypeScript and serde_json".into(),
        }];
        assert!(conservative_output_allowed(
            &transcript,
            "type script and sir dee json"
        ));
        assert!(conservative_output_allowed(
            &transcript,
            "TypeScript and serde_json"
        ));
        assert!(!conservative_output_allowed(
            &transcript,
            "A different rewrite"
        ));
    }

    #[test]
    fn conservative_acceptance_allows_sentence_like_minor_term_normalization() {
        let mut hyperland_transcript = RewriteTranscript {
            raw_text: "I'm currently using the window manager hyperland.".into(),
            correction_aware_text: "I'm currently using the window manager hyperland.".into(),
            aggressive_correction_text: None,
            detected_language: Some("en".into()),
            typing_context: Some(typing_context(RewriteSurfaceKind::Terminal)),
            recent_session_entries: Vec::new(),
            session_backtrack_candidates: Vec::new(),
            recommended_session_candidate: None,
            segments: Vec::new(),
            edit_intents: Vec::new(),
            edit_signals: Vec::new(),
            edit_hypotheses: Vec::new(),
            rewrite_candidates: vec![RewriteCandidate {
                kind: RewriteCandidateKind::ConservativeCorrection,
                text: "I'm currently using the window manager hyperland.".into(),
            }],
            recommended_candidate: None,
            edit_context: crate::rewrite_protocol::RewriteEditContext::default(),
            policy_context: RewritePolicyContext::default(),
        };
        hyperland_transcript.policy_context = resolve_policy_context(
            RewriteCorrectionPolicy::Conservative,
            hyperland_transcript.typing_context.as_ref(),
            &hyperland_transcript.rewrite_candidates,
            &[],
            &[],
        );

        assert!(conservative_output_allowed(
            &hyperland_transcript,
            "I'm currently using the window manager Hyprland."
        ));

        let mut switch_transcript = RewriteTranscript {
            raw_text: "I'm switching from Sui to Hyperland.".into(),
            correction_aware_text: "I'm switching from Sui to Hyperland.".into(),
            aggressive_correction_text: None,
            detected_language: Some("en".into()),
            typing_context: Some(typing_context(RewriteSurfaceKind::Terminal)),
            recent_session_entries: Vec::new(),
            session_backtrack_candidates: Vec::new(),
            recommended_session_candidate: None,
            segments: Vec::new(),
            edit_intents: Vec::new(),
            edit_signals: Vec::new(),
            edit_hypotheses: Vec::new(),
            rewrite_candidates: vec![RewriteCandidate {
                kind: RewriteCandidateKind::ConservativeCorrection,
                text: "I'm switching from Sui to Hyperland.".into(),
            }],
            recommended_candidate: None,
            edit_context: crate::rewrite_protocol::RewriteEditContext::default(),
            policy_context: RewritePolicyContext::default(),
        };
        switch_transcript.policy_context = resolve_policy_context(
            RewriteCorrectionPolicy::Conservative,
            switch_transcript.typing_context.as_ref(),
            &switch_transcript.rewrite_candidates,
            &[],
            &[],
        );

        assert!(conservative_output_allowed(
            &switch_transcript,
            "I'm switching from Sway to Hyprland."
        ));
    }

    #[test]
    fn conservative_acceptance_keeps_short_command_fragments_strict() {
        let mut transcript = RewriteTranscript {
            raw_text: "cargo clipy".into(),
            correction_aware_text: "cargo clipy".into(),
            aggressive_correction_text: None,
            detected_language: Some("en".into()),
            typing_context: Some(typing_context(RewriteSurfaceKind::Terminal)),
            recent_session_entries: Vec::new(),
            session_backtrack_candidates: Vec::new(),
            recommended_session_candidate: None,
            segments: Vec::new(),
            edit_intents: Vec::new(),
            edit_signals: Vec::new(),
            edit_hypotheses: Vec::new(),
            rewrite_candidates: vec![RewriteCandidate {
                kind: RewriteCandidateKind::ConservativeCorrection,
                text: "cargo clipy".into(),
            }],
            recommended_candidate: None,
            edit_context: crate::rewrite_protocol::RewriteEditContext::default(),
            policy_context: RewritePolicyContext::default(),
        };
        transcript.policy_context.correction_policy = RewriteCorrectionPolicy::Conservative;

        assert!(!conservative_output_allowed(&transcript, "cargo clippy"));
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
        assert_eq!(
            transcript
                .recommended_candidate
                .as_ref()
                .map(|candidate| candidate.text.as_str()),
            Some("TypeScript and sir dee json")
        );
        assert_eq!(
            transcript
                .rewrite_candidates
                .first()
                .map(|candidate| candidate.text.as_str()),
            Some("TypeScript and sir dee json")
        );
    }

    #[test]
    fn built_in_glossary_candidates_cover_window_manager_terms() {
        let context = typing_context(RewriteSurfaceKind::GenericText);
        let policy = resolve_policy_context(
            RewriteCorrectionPolicy::Balanced,
            Some(&context),
            &[RewriteCandidate {
                kind: RewriteCandidateKind::Literal,
                text: "I'm switching from sui to hyperland and neary.".into(),
            }],
            &[],
            &[],
        );

        assert!(
            policy
                .glossary_candidates
                .iter()
                .any(|candidate| candidate.text == "I'm switching from Sway to Hyprland and niri.")
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

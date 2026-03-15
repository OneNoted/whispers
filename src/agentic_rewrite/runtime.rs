use super::{AppRule, ContextMatcher, GlossaryEntry, PreparedGlossaryEntry};
use crate::rewrite_protocol::{
    RewriteCandidate, RewriteCandidateKind, RewriteCorrectionPolicy, RewritePolicyContext,
    RewritePolicyGlossaryTerm, RewriteSurfaceKind, RewriteTranscript, RewriteTypingContext,
};

const MAX_GLOSSARY_CANDIDATES: usize = 4;

pub(super) fn resolve_policy_context(
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

    let mut active_glossary_entries = glossary_entries
        .iter()
        .enumerate()
        .filter_map(|(index, entry)| {
            PreparedGlossaryEntry::new(entry.clone()).map(|entry| (index, entry))
        })
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

fn built_in_rules(default_policy: RewriteCorrectionPolicy) -> Vec<AppRule> {
    vec![
        AppRule::built_in(
            "baseline/global-default",
            ContextMatcher::default(),
            "Use the active typing context, recent dictation context, glossary terms, and bounded candidates to resolve technical dictation cleanly while keeping the final-text-only contract. When the utterance clearly points to software, tools, APIs, Linux components, product names, or other technical concepts, prefer the most plausible intended technical term over a phonetically similar common word. Use category cues like window manager, editor, language, library, shell, or package manager to disambiguate nearby technical names. If it remains genuinely ambiguous, stay close to the transcript.",
            Some(default_policy),
        ),
        AppRule::built_in(
            "baseline/browser",
            ContextMatcher {
                surface_kind: Some(RewriteSurfaceKind::Browser),
                ..ContextMatcher::default()
            },
            "Favor clean prose and natural punctuation for browser text fields, but stay grounded in the listed candidates, glossary evidence, and the utterance's technical topic when it clearly refers to software or documentation.",
            Some(RewriteCorrectionPolicy::Balanced),
        ),
        AppRule::built_in(
            "baseline/generic-text",
            ContextMatcher {
                surface_kind: Some(RewriteSurfaceKind::GenericText),
                ..ContextMatcher::default()
            },
            "Favor clean prose and natural punctuation for general text entry while staying grounded in the listed candidates and glossary evidence. If the utterance clearly discusses technical tools or software, prefer the most plausible technical term over a phonetically similar common word.",
            Some(RewriteCorrectionPolicy::Balanced),
        ),
        AppRule::built_in(
            "baseline/editor",
            ContextMatcher {
                surface_kind: Some(RewriteSurfaceKind::Editor),
                ..ContextMatcher::default()
            },
            "Preserve identifiers, filenames, API names, symbols, and technical casing. Avoid rewriting technical wording into generic prose. Infer likely technical terms and proper names from the utterance when the topic is clearly code, tooling, or software.",
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
    ch.is_alphanumeric() || matches!(ch, '\'' | '-' | '_' | '.')
}

fn contains_ignore_ascii_case(haystack: Option<&str>, needle: &str) -> bool {
    let Some(haystack) = haystack else {
        return false;
    };
    haystack
        .to_ascii_lowercase()
        .contains(&needle.to_ascii_lowercase())
}

#[derive(Debug, Clone)]
struct WordSpan {
    start: usize,
    end: usize,
    normalized: String,
}

#[cfg(test)]
mod tests {
    use super::super::{AppRule, ContextMatcher, GlossaryEntry};
    use super::*;
    use crate::rewrite_protocol::{
        RewriteCandidate, RewriteCandidateKind, RewritePolicyContext, RewriteSurfaceKind,
        RewriteTranscript, RewriteTypingContext,
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
                .any(|instruction| instruction.contains("phonetically similar common word"))
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
        assert_eq!(policy.active_glossary_terms.len(), 2);
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
            policy_context: RewritePolicyContext::default(),
        };
        hyperland_transcript.policy_context.correction_policy =
            RewriteCorrectionPolicy::Conservative;

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
            policy_context: RewritePolicyContext::default(),
        };
        switch_transcript.policy_context.correction_policy = RewriteCorrectionPolicy::Conservative;

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
            policy_context: RewritePolicyContext::default(),
        };
        transcript.policy_context.correction_policy = RewriteCorrectionPolicy::Conservative;

        assert!(!conservative_output_allowed(&transcript, "cargo clippy"));
    }
}

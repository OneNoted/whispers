use crate::config::Config;
use crate::error::Result;

mod rewrite;
mod store;

pub use rewrite::build_rewrite_transcript;
pub use store::{
    DictionaryEntry, SnippetEntry, add_dictionary, add_snippet, list_dictionary, list_snippets,
    print_rewrite_instructions_path, remove_dictionary, remove_snippet,
};

#[derive(Debug, Clone, Default)]
pub struct PersonalizationRules {
    dictionary: Vec<PreparedDictionaryEntry>,
    snippets: Vec<PreparedSnippet>,
    snippet_trigger_words: Vec<String>,
    custom_instructions: String,
}

#[derive(Debug, Clone)]
pub(super) struct PreparedDictionaryEntry {
    replace: String,
    words: Vec<String>,
}

#[derive(Debug, Clone)]
pub(super) struct PreparedSnippet {
    text: String,
    words: Vec<String>,
}

#[derive(Debug, Clone)]
pub(super) struct WordSpan {
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) normalized: String,
}

pub fn load_rules(config: &Config) -> Result<PersonalizationRules> {
    let dictionary_entries = store::read_dictionary_file(&config.resolved_dictionary_path())?;
    let snippet_entries = store::read_snippet_file(&config.resolved_snippets_path())?;
    let custom_instructions = store::load_custom_instructions(config)?;

    Ok(PersonalizationRules {
        dictionary: dictionary_entries
            .into_iter()
            .filter_map(|entry| PreparedDictionaryEntry::new(entry).ok())
            .collect(),
        snippets: snippet_entries
            .into_iter()
            .filter_map(|entry| PreparedSnippet::new(entry).ok())
            .collect(),
        snippet_trigger_words: normalized_words(&config.personalization.snippet_trigger),
        custom_instructions,
    })
}

pub fn finalize_text(text: &str, rules: &PersonalizationRules) -> String {
    let corrected = apply_dictionary(text, rules);
    let expanded = expand_snippets(&corrected, rules);
    normalize_numeric_dot_runs(&expanded)
}

pub fn custom_instructions(rules: &PersonalizationRules) -> Option<&str> {
    (!rules.custom_instructions.trim().is_empty()).then_some(rules.custom_instructions.as_str())
}

pub fn transcription_prompt(rules: &PersonalizationRules) -> Option<String> {
    const MAX_TERMS: usize = 24;
    const MAX_PROMPT_LEN: usize = 480;

    let mut terms = Vec::new();
    for entry in &rules.dictionary {
        let replace = entry.replace.trim();
        if replace.is_empty() {
            continue;
        }
        if terms.iter().any(|existing: &String| existing == replace) {
            continue;
        }
        let projected_len = if terms.is_empty() {
            replace.len()
        } else {
            terms.iter().map(String::len).sum::<usize>() + (terms.len() * 2) + replace.len()
        };
        if terms.len() >= MAX_TERMS || projected_len > MAX_PROMPT_LEN {
            break;
        }
        terms.push(replace.to_string());
    }

    if terms.is_empty() {
        return None;
    }

    Some(format!(
        "This is direct dictation. Prefer these exact spellings when heard: {}.",
        terms.join(", ")
    ))
}

pub(super) fn apply_dictionary(text: &str, rules: &PersonalizationRules) -> String {
    apply_replacements(text, &rules.dictionary)
}

fn expand_snippets(text: &str, rules: &PersonalizationRules) -> String {
    if rules.snippets.is_empty() || rules.snippet_trigger_words.is_empty() {
        return text.trim().to_string();
    }

    let spans = collect_word_spans(text);
    if spans.is_empty() {
        return text.trim().to_string();
    }

    let mut output = String::new();
    let mut cursor = 0usize;
    let mut index = 0usize;

    while index < spans.len() {
        let Some(best) =
            best_snippet_match(&spans, index, &rules.snippet_trigger_words, &rules.snippets)
        else {
            index += 1;
            continue;
        };

        output.push_str(&text[cursor..spans[index].start]);
        output.push_str(best.text);
        cursor = spans[index + best.total_words - 1].end;
        index += best.total_words;
    }

    output.push_str(&text[cursor..]);
    output.trim().to_string()
}

fn apply_replacements(text: &str, entries: &[PreparedDictionaryEntry]) -> String {
    if entries.is_empty() {
        return text.trim().to_string();
    }

    let spans = collect_word_spans(text);
    if spans.is_empty() {
        return text.trim().to_string();
    }

    let mut output = String::new();
    let mut cursor = 0usize;
    let mut index = 0usize;

    while index < spans.len() {
        let Some(best) = best_dictionary_match(&spans, index, entries) else {
            index += 1;
            continue;
        };

        output.push_str(&text[cursor..spans[index].start]);
        output.push_str(&best.replace);
        cursor = spans[index + best.words.len() - 1].end;
        index += best.words.len();
    }

    output.push_str(&text[cursor..]);
    output.trim().to_string()
}

fn best_dictionary_match<'a>(
    spans: &[WordSpan],
    index: usize,
    entries: &'a [PreparedDictionaryEntry],
) -> Option<&'a PreparedDictionaryEntry> {
    entries
        .iter()
        .filter(|entry| entry.matches(spans, index))
        .max_by_key(|entry| entry.words.len())
}

fn best_snippet_match<'a>(
    spans: &[WordSpan],
    index: usize,
    trigger_words: &[String],
    snippets: &'a [PreparedSnippet],
) -> Option<SnippetMatch<'a>> {
    if !matches_words(spans, index, trigger_words) {
        return None;
    }

    let snippet_index = index + trigger_words.len();
    snippets
        .iter()
        .filter(|snippet| snippet.matches(spans, snippet_index))
        .max_by_key(|snippet| snippet.words.len())
        .map(|snippet| SnippetMatch {
            text: snippet.text.as_str(),
            total_words: trigger_words.len() + snippet.words.len(),
        })
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

pub(super) fn collect_word_spans(text: &str) -> Vec<WordSpan> {
    let mut spans = Vec::new();
    let mut current_start = None;

    for (idx, ch) in text.char_indices() {
        if is_word_char(ch) {
            current_start.get_or_insert(idx);
            continue;
        }

        if let Some(start) = current_start.take() {
            spans.push(WordSpan {
                start,
                end: idx,
                normalized: normalize_word(&text[start..idx]),
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

fn normalize_word(word: &str) -> String {
    word.chars()
        .filter(|ch| is_word_char(*ch))
        .flat_map(|ch| ch.to_lowercase())
        .collect()
}

pub(super) fn normalized_words(text: &str) -> Vec<String> {
    collect_word_spans(text)
        .into_iter()
        .map(|span| span.normalized)
        .collect()
}

fn is_word_char(ch: char) -> bool {
    ch.is_alphanumeric() || matches!(ch, '\'' | '-')
}

impl PreparedDictionaryEntry {
    pub(super) fn new(entry: DictionaryEntry) -> std::result::Result<Self, DictionaryEntry> {
        let words = normalized_words(&entry.phrase);
        if words.is_empty() {
            return Err(entry);
        }

        Ok(Self {
            replace: entry.replace,
            words,
        })
    }

    fn matches(&self, spans: &[WordSpan], index: usize) -> bool {
        matches_words(spans, index, &self.words)
    }
}

impl PreparedSnippet {
    pub(super) fn new(entry: SnippetEntry) -> std::result::Result<Self, SnippetEntry> {
        let words = normalized_words(&entry.name);
        if words.is_empty() {
            return Err(entry);
        }

        Ok(Self {
            text: entry.text,
            words,
        })
    }

    fn matches(&self, spans: &[WordSpan], index: usize) -> bool {
        matches_words(spans, index, &self.words)
    }
}

struct SnippetMatch<'a> {
    text: &'a str,
    total_words: usize,
}

fn normalize_numeric_dot_runs(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut output = String::with_capacity(text.len());
    let mut index = 0usize;

    while index < chars.len() {
        let ch = chars[index];

        if ch == ' '
            && previous_non_space_char(&output).is_some_and(|previous| previous.is_ascii_digit())
        {
            let mut lookahead = index;
            while lookahead < chars.len() && chars[lookahead] == ' ' {
                lookahead += 1;
            }

            if lookahead < chars.len()
                && chars[lookahead] == '.'
                && dot_has_numeric_suffix(&chars, lookahead)
            {
                index = lookahead;
                continue;
            }
        }

        output.push(ch);

        if ch == '.'
            && previous_non_space_char(&output[..output.len().saturating_sub(1)])
                .is_some_and(|previous| previous.is_ascii_digit())
        {
            let mut lookahead = index + 1;
            while lookahead < chars.len() && chars[lookahead] == ' ' {
                lookahead += 1;
            }

            if lookahead > index + 1 && lookahead < chars.len() && chars[lookahead].is_ascii_digit()
            {
                index = lookahead;
                continue;
            }
        }

        index += 1;
    }

    output
}

fn previous_non_space_char(text: &str) -> Option<char> {
    text.chars().rev().find(|ch| !ch.is_whitespace())
}

fn dot_has_numeric_suffix(chars: &[char], dot_index: usize) -> bool {
    let mut lookahead = dot_index + 1;
    while lookahead < chars.len() && chars[lookahead] == ' ' {
        lookahead += 1;
    }

    lookahead < chars.len() && chars[lookahead].is_ascii_digit()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, PostprocessMode};
    use crate::rewrite_profile::RewriteProfile;

    fn rules() -> PersonalizationRules {
        PersonalizationRules {
            dictionary: vec![
                PreparedDictionaryEntry::new(DictionaryEntry {
                    phrase: "wisper flow".into(),
                    replace: "Wispr Flow".into(),
                })
                .expect("dictionary"),
                PreparedDictionaryEntry::new(DictionaryEntry {
                    phrase: "open ai".into(),
                    replace: "OpenAI".into(),
                })
                .expect("dictionary"),
            ],
            snippets: vec![
                PreparedSnippet::new(SnippetEntry {
                    name: "signature".into(),
                    text: "Best regards,\nNotes".into(),
                })
                .expect("snippet"),
                PreparedSnippet::new(SnippetEntry {
                    name: "meeting follow up".into(),
                    text: "Thanks for the meeting.".into(),
                })
                .expect("snippet"),
            ],
            snippet_trigger_words: normalized_words("insert"),
            custom_instructions: "Keep brand names exact.".into(),
        }
    }

    #[test]
    fn dictionary_applies_exact_normalized_replacements() {
        let applied = apply_dictionary("I use wisper flow with open, ai.", &rules());
        assert_eq!(applied, "I use Wispr Flow with OpenAI.");
    }

    #[test]
    fn dictionary_prefers_longest_match() {
        let rules = PersonalizationRules {
            dictionary: vec![
                PreparedDictionaryEntry::new(DictionaryEntry {
                    phrase: "open".into(),
                    replace: "X".into(),
                })
                .expect("dictionary"),
                PreparedDictionaryEntry::new(DictionaryEntry {
                    phrase: "open ai".into(),
                    replace: "OpenAI".into(),
                })
                .expect("dictionary"),
            ],
            ..PersonalizationRules::default()
        };
        let applied = apply_dictionary("open ai works", &rules);
        assert_eq!(applied, "OpenAI works");
    }

    #[test]
    fn snippets_expand_after_trigger() {
        let expanded = expand_snippets("please insert signature now", &rules());
        assert_eq!(expanded, "please Best regards,\nNotes now");
    }

    #[test]
    fn unmatched_snippet_leaves_text_unchanged() {
        let expanded = expand_snippets("please insert unknown now", &rules());
        assert_eq!(expanded, "please insert unknown now");
    }

    #[test]
    fn finalize_text_applies_dictionary_then_snippets() {
        let finalized = finalize_text("insert meeting follow up about wisper flow", &rules());
        assert_eq!(finalized, "Thanks for the meeting. about Wispr Flow");
    }

    #[test]
    fn finalize_text_collapses_spaced_numeric_dot_runs() {
        let finalized = finalize_text("MPL 2. 0 and TLS 1 . 3 are common references", &rules());
        assert_eq!(finalized, "MPL 2.0 and TLS 1.3 are common references");
    }

    #[test]
    fn finalize_text_preserves_sentence_period_before_words() {
        let finalized = finalize_text("Section 2. Next step", &rules());
        assert_eq!(finalized, "Section 2. Next step");
    }

    #[test]
    fn transcription_prompt_includes_dictionary_targets() {
        let prompt = transcription_prompt(&rules()).expect("prompt");
        assert!(prompt.contains("Wispr Flow"));
        assert!(prompt.contains("OpenAI"));
    }

    #[test]
    fn default_config_paths_support_personalization_files() {
        let config = Config::default();
        assert_eq!(config.postprocess.mode, PostprocessMode::Raw);
        assert_eq!(config.rewrite.profile, RewriteProfile::Auto);
        assert_eq!(config.personalization.snippet_trigger, "insert");
    }
}

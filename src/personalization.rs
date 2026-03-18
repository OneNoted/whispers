use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::cleanup;
use crate::config::Config;
use crate::error::{Result, WhsprError};
use crate::rewrite_protocol::{
    RewriteCandidate, RewriteCandidateKind, RewriteEditAction, RewriteEditContext,
    RewriteEditHypothesis, RewriteEditHypothesisMatchSource, RewriteEditIntent, RewriteEditSignal,
    RewriteEditSignalKind, RewriteEditSignalScope, RewriteEditSignalStrength,
    RewriteIntentConfidence, RewritePolicyContext, RewriteReplacementScope, RewriteTailShape,
    RewriteTranscript, RewriteTranscriptSegment,
};
use crate::transcribe::Transcript;

#[derive(Debug, Clone, Default)]
pub struct PersonalizationRules {
    dictionary: Vec<PreparedDictionaryEntry>,
    snippets: Vec<PreparedSnippet>,
    snippet_trigger_words: Vec<String>,
    custom_instructions: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct DictionaryEntry {
    pub phrase: String,
    pub replace: String,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
struct DictionaryFile {
    entries: Vec<DictionaryEntry>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct SnippetEntry {
    pub name: String,
    pub text: String,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
struct SnippetFile {
    snippets: Vec<SnippetEntry>,
}

#[derive(Debug, Clone)]
struct PreparedDictionaryEntry {
    replace: String,
    words: Vec<String>,
}

#[derive(Debug, Clone)]
struct PreparedSnippet {
    text: String,
    words: Vec<String>,
}

#[derive(Debug, Clone)]
struct WordSpan {
    start: usize,
    end: usize,
    normalized: String,
}

#[derive(Debug, Clone, Default)]
struct TailParts {
    normalized_phrase_tail: Option<String>,
    span_replacement_tail: Option<String>,
}

pub fn load_rules(config: &Config) -> Result<PersonalizationRules> {
    let dictionary_entries = read_dictionary_file(&config.resolved_dictionary_path())?;
    let snippet_entries = read_snippet_file(&config.resolved_snippets_path())?;
    let custom_instructions = load_custom_instructions(config)?;

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

pub fn build_rewrite_transcript(
    transcript: &Transcript,
    rules: &PersonalizationRules,
) -> RewriteTranscript {
    let analysis = cleanup::correction_analysis(transcript);
    let raw_text = apply_dictionary(&transcript.raw_text, rules);
    let correction_aware_text = apply_dictionary(&analysis.text, rules);
    let aggressive_correction_text = analysis
        .aggressive_text
        .as_deref()
        .map(|text| apply_dictionary(text, rules))
        .filter(|text| !text.trim().is_empty());
    let edit_hypotheses: Vec<RewriteEditHypothesis> = analysis
        .edit_hypotheses
        .iter()
        .map(|hypothesis| RewriteEditHypothesis {
            cue_family: hypothesis.cue_family.to_string(),
            matched_text: apply_dictionary(&hypothesis.matched_text, rules),
            match_source: match hypothesis.match_source {
                cleanup::EditHypothesisMatchSource::Exact => {
                    RewriteEditHypothesisMatchSource::Exact
                }
                cleanup::EditHypothesisMatchSource::Alias => {
                    RewriteEditHypothesisMatchSource::Alias
                }
                cleanup::EditHypothesisMatchSource::NearMiss => {
                    RewriteEditHypothesisMatchSource::NearMiss
                }
            },
            kind: match hypothesis.kind {
                cleanup::EditSignalKind::Cancel => RewriteEditSignalKind::Cancel,
                cleanup::EditSignalKind::Replace => RewriteEditSignalKind::Replace,
                cleanup::EditSignalKind::Restatement => RewriteEditSignalKind::Restatement,
            },
            scope_hint: match hypothesis.scope_hint {
                cleanup::EditSignalScope::Phrase => RewriteEditSignalScope::Phrase,
                cleanup::EditSignalScope::Clause => RewriteEditSignalScope::Clause,
                cleanup::EditSignalScope::Sentence => RewriteEditSignalScope::Sentence,
                cleanup::EditSignalScope::Unknown => RewriteEditSignalScope::Unknown,
            },
            replacement_scope: match hypothesis.replacement_scope {
                cleanup::ReplacementScope::Span => RewriteReplacementScope::Span,
                cleanup::ReplacementScope::Clause => RewriteReplacementScope::Clause,
                cleanup::ReplacementScope::Sentence => RewriteReplacementScope::Sentence,
            },
            tail_shape: match hypothesis.tail_shape {
                cleanup::TailShape::Empty => RewriteTailShape::Empty,
                cleanup::TailShape::Phrase => RewriteTailShape::Phrase,
                cleanup::TailShape::Clause => RewriteTailShape::Clause,
            },
            strength: match hypothesis.strength {
                cleanup::EditSignalStrength::Possible => RewriteEditSignalStrength::Possible,
                cleanup::EditSignalStrength::Strong => RewriteEditSignalStrength::Strong,
            },
        })
        .collect();
    let rewrite_candidates = build_rewrite_candidates(
        &transcript.raw_text,
        &analysis.text,
        analysis.aggressive_text.as_deref(),
        &analysis.edit_hypotheses,
        rules,
    );
    let edit_context = derive_edit_context(&transcript.raw_text, &analysis.edit_hypotheses);
    let recommended_candidate = recommended_candidate(
        &rewrite_candidates,
        &analysis.edit_hypotheses,
        &edit_context,
    );

    RewriteTranscript {
        raw_text,
        correction_aware_text,
        aggressive_correction_text,
        detected_language: transcript.detected_language.clone(),
        typing_context: None,
        recent_session_entries: Vec::new(),
        session_backtrack_candidates: Vec::new(),
        recommended_session_candidate: None,
        segments: transcript
            .segments
            .iter()
            .map(|segment| RewriteTranscriptSegment {
                text: apply_dictionary(&segment.text, rules),
                start_ms: segment.start_ms,
                end_ms: segment.end_ms,
            })
            .collect(),
        edit_intents: analysis
            .edit_intents
            .into_iter()
            .map(|intent| RewriteEditIntent {
                action: match intent.action {
                    cleanup::EditIntentAction::ReplacePreviousPhrase => {
                        RewriteEditAction::ReplacePreviousPhrase
                    }
                    cleanup::EditIntentAction::ReplacePreviousClause => {
                        RewriteEditAction::ReplacePreviousClause
                    }
                    cleanup::EditIntentAction::ReplacePreviousSentence => {
                        RewriteEditAction::ReplacePreviousSentence
                    }
                    cleanup::EditIntentAction::DropEditCue => RewriteEditAction::DropEditCue,
                },
                trigger: intent.trigger.to_string(),
                confidence: match intent.confidence {
                    cleanup::EditIntentConfidence::High => RewriteIntentConfidence::High,
                },
            })
            .collect(),
        edit_signals: analysis
            .edit_signals
            .into_iter()
            .map(|signal| RewriteEditSignal {
                trigger: signal.trigger.to_string(),
                kind: match signal.kind {
                    cleanup::EditSignalKind::Cancel => RewriteEditSignalKind::Cancel,
                    cleanup::EditSignalKind::Replace => RewriteEditSignalKind::Replace,
                    cleanup::EditSignalKind::Restatement => RewriteEditSignalKind::Restatement,
                },
                scope_hint: match signal.scope_hint {
                    cleanup::EditSignalScope::Phrase => RewriteEditSignalScope::Phrase,
                    cleanup::EditSignalScope::Clause => RewriteEditSignalScope::Clause,
                    cleanup::EditSignalScope::Sentence => RewriteEditSignalScope::Sentence,
                    cleanup::EditSignalScope::Unknown => RewriteEditSignalScope::Unknown,
                },
                strength: match signal.strength {
                    cleanup::EditSignalStrength::Possible => RewriteEditSignalStrength::Possible,
                    cleanup::EditSignalStrength::Strong => RewriteEditSignalStrength::Strong,
                },
            })
            .collect(),
        edit_hypotheses,
        rewrite_candidates,
        recommended_candidate,
        edit_context,
        policy_context: RewritePolicyContext::default(),
    }
}

fn build_rewrite_candidates(
    raw_text: &str,
    correction_aware_text: &str,
    aggressive_correction_text: Option<&str>,
    edit_hypotheses: &[cleanup::EditHypothesis],
    rules: &PersonalizationRules,
) -> Vec<RewriteCandidate> {
    let mut candidates = Vec::new();
    push_rewrite_candidate(
        &mut candidates,
        RewriteCandidateKind::Literal,
        raw_text,
        rules,
    );
    push_rewrite_candidate(
        &mut candidates,
        RewriteCandidateKind::ConservativeCorrection,
        correction_aware_text,
        rules,
    );
    if let Some(text) = aggressive_correction_text {
        push_rewrite_candidate(
            &mut candidates,
            RewriteCandidateKind::AggressiveCorrection,
            text,
            rules,
        );
    }

    let spans = collect_word_spans(raw_text);
    for hypothesis in edit_hypotheses {
        if candidates.len() >= 5 {
            break;
        }

        let tail = tail_text_after_hypothesis(raw_text, &spans, hypothesis);
        let tail_parts = tail.as_deref().map(split_tail_parts).unwrap_or_default();
        let normalized_phrase_tail = tail_parts.normalized_phrase_tail.as_deref();
        let span_replacement_tail = tail_parts.span_replacement_tail.as_deref();

        let mut added_scoped_candidate = false;
        match hypothesis.replacement_scope {
            cleanup::ReplacementScope::Span => {
                if let Some(text) = span_replacement_candidate(
                    raw_text,
                    &spans,
                    hypothesis,
                    span_replacement_tail
                        .or(normalized_phrase_tail)
                        .or(tail.as_deref()),
                ) {
                    push_rewrite_candidate(
                        &mut candidates,
                        RewriteCandidateKind::SpanReplacement,
                        &text,
                        rules,
                    );
                    added_scoped_candidate = true;
                }
                if let Some(text) =
                    clause_replacement_candidate(raw_text, &spans, hypothesis, tail.as_deref())
                {
                    push_rewrite_candidate(
                        &mut candidates,
                        RewriteCandidateKind::ClauseReplacement,
                        &text,
                        rules,
                    );
                }
            }
            cleanup::ReplacementScope::Clause => {
                if let Some(text) =
                    span_replacement_candidate(raw_text, &spans, hypothesis, span_replacement_tail)
                {
                    push_rewrite_candidate(
                        &mut candidates,
                        RewriteCandidateKind::SpanReplacement,
                        &text,
                        rules,
                    );
                    added_scoped_candidate = true;
                }
                if let Some(text) =
                    clause_replacement_candidate(raw_text, &spans, hypothesis, tail.as_deref())
                {
                    push_rewrite_candidate(
                        &mut candidates,
                        RewriteCandidateKind::ClauseReplacement,
                        &text,
                        rules,
                    );
                    added_scoped_candidate = true;
                }
                if let Some(text) = sentence_replacement_candidate(
                    hypothesis,
                    tail.as_deref(),
                    normalized_phrase_tail,
                ) {
                    push_rewrite_candidate(
                        &mut candidates,
                        RewriteCandidateKind::SentenceReplacement,
                        &text,
                        rules,
                    );
                }
            }
            cleanup::ReplacementScope::Sentence => {
                if let Some(text) =
                    span_replacement_candidate(raw_text, &spans, hypothesis, span_replacement_tail)
                {
                    push_rewrite_candidate(
                        &mut candidates,
                        RewriteCandidateKind::SpanReplacement,
                        &text,
                        rules,
                    );
                    added_scoped_candidate = true;
                }
                if let Some(text) = sentence_replacement_candidate(
                    hypothesis,
                    tail.as_deref(),
                    normalized_phrase_tail,
                ) {
                    push_rewrite_candidate(
                        &mut candidates,
                        RewriteCandidateKind::SentenceReplacement,
                        &text,
                        rules,
                    );
                    added_scoped_candidate = true;
                }
                if let Some(text) =
                    clause_replacement_candidate(raw_text, &spans, hypothesis, tail.as_deref())
                {
                    push_rewrite_candidate(
                        &mut candidates,
                        RewriteCandidateKind::ClauseReplacement,
                        &text,
                        rules,
                    );
                }
            }
        }

        if candidates.len() >= 5 {
            break;
        }

        if !added_scoped_candidate
            && let Some(text) =
                remove_word_range(raw_text, &spans, hypothesis.word_start, hypothesis.word_end)
        {
            push_rewrite_candidate(
                &mut candidates,
                RewriteCandidateKind::DropCueOnly,
                &text,
                rules,
            );
        }
    }

    let edit_context = derive_edit_context(raw_text, edit_hypotheses);
    if has_strong_explicit_hypothesis(edit_hypotheses)
        && !(edit_context.cue_is_utterance_initial && edit_context.courtesy_prefix_detected)
    {
        candidates.sort_by_key(|candidate| candidate_priority(candidate.kind));
    }

    candidates.truncate(5);
    candidates
}

fn push_rewrite_candidate(
    candidates: &mut Vec<RewriteCandidate>,
    kind: RewriteCandidateKind,
    text: &str,
    rules: &PersonalizationRules,
) {
    let text = apply_dictionary(text, rules);
    let text = text.trim();
    if text.is_empty() || candidates.iter().any(|candidate| candidate.text == text) {
        return;
    }

    candidates.push(RewriteCandidate {
        kind,
        text: text.to_string(),
    });
}

fn remove_word_range(
    raw_text: &str,
    spans: &[WordSpan],
    word_start: usize,
    word_end: usize,
) -> Option<String> {
    if word_start >= word_end || word_end > spans.len() {
        return None;
    }

    let mut output = String::new();
    output.push_str(&raw_text[..spans[word_start].start]);
    output.push_str(&raw_text[spans[word_end - 1].end..]);
    normalize_candidate_spacing(&output)
}

fn tail_text_after_hypothesis(
    raw_text: &str,
    spans: &[WordSpan],
    hypothesis: &cleanup::EditHypothesis,
) -> Option<String> {
    if hypothesis.word_end == 0 || hypothesis.word_end > spans.len() {
        return None;
    }

    let trailing = raw_text[spans[hypothesis.word_end - 1].end..]
        .trim_start_matches(|ch: char| {
            ch.is_whitespace() || matches!(ch, ',' | ':' | ';' | '.' | '!' | '?')
        })
        .trim();
    normalize_candidate_spacing(trailing)
}

fn split_tail_parts(text: &str) -> TailParts {
    let normalized_phrase_tail = normalize_candidate_spacing(&normalize_phrase_tail(text));
    let spans = collect_word_spans(text);
    let Some(boundary) = phrase_clause_boundary(&spans) else {
        return TailParts {
            normalized_phrase_tail,
            span_replacement_tail: None,
        };
    };

    let phrase_tail = text[..spans[boundary].start].trim();
    let remainder_tail = text[spans[boundary].start..].trim();
    let normalized_phrase_tail = normalize_candidate_spacing(&normalize_phrase_tail(phrase_tail));
    let remainder_tail = normalize_candidate_spacing(remainder_tail);
    let span_replacement_tail = match (normalized_phrase_tail.as_deref(), remainder_tail.as_deref())
    {
        (Some(phrase), Some(remainder)) => {
            normalize_candidate_spacing(&format!("{phrase} {remainder}"))
        }
        _ => None,
    };

    TailParts {
        normalized_phrase_tail,
        span_replacement_tail,
    }
}

fn sentence_replacement_candidate(
    hypothesis: &cleanup::EditHypothesis,
    tail: Option<&str>,
    normalized_phrase_tail: Option<&str>,
) -> Option<String> {
    match hypothesis.tail_shape {
        cleanup::TailShape::Empty => None,
        cleanup::TailShape::Phrase => normalized_phrase_tail
            .filter(|text| !text.is_empty())
            .map(str::to_string)
            .or_else(|| tail.map(str::to_string)),
        cleanup::TailShape::Clause => tail.map(str::to_string),
    }
}

fn span_replacement_candidate(
    raw_text: &str,
    spans: &[WordSpan],
    hypothesis: &cleanup::EditHypothesis,
    replacement_tail: Option<&str>,
) -> Option<String> {
    let replacement_tail = replacement_tail?;
    let anchor = span_anchor_index(raw_text, spans, hypothesis.word_start);
    scoped_replacement_candidate(raw_text, spans, anchor, replacement_tail)
}

fn clause_replacement_candidate(
    raw_text: &str,
    spans: &[WordSpan],
    hypothesis: &cleanup::EditHypothesis,
    tail: Option<&str>,
) -> Option<String> {
    let trailing = tail?;
    let anchor = contextual_anchor_index(raw_text, spans, hypothesis.word_start);
    scoped_replacement_candidate(raw_text, spans, anchor, trailing)
}

fn phrase_clause_boundary(spans: &[WordSpan]) -> Option<usize> {
    const SUBJECT_WORDS: &[&str] = &[
        "i", "it", "he", "she", "they", "we", "you", "this", "that", "there", "my", "our", "their",
        "your",
    ];

    if spans.len() < 3 {
        return None;
    }

    (1..spans.len() - 1).find(|&index| {
        let word = spans[index].normalized.as_str();
        let next = spans[index + 1].normalized.as_str();
        SUBJECT_WORDS.contains(&word) && !matches!(next, "and" | "or" | "but")
    })
}

fn scoped_replacement_candidate(
    raw_text: &str,
    spans: &[WordSpan],
    anchor: usize,
    replacement_tail: &str,
) -> Option<String> {
    if anchor >= spans.len() {
        return None;
    }

    let mut output = String::new();
    output.push_str(&raw_text[..spans[anchor].start]);
    output.push_str(replacement_tail);
    normalize_candidate_spacing(&output)
}

fn span_anchor_index(raw_text: &str, spans: &[WordSpan], mut anchor: usize) -> usize {
    let mut walked_words = 0usize;

    while anchor > 0 {
        let previous = anchor - 1;
        let gap = &raw_text[spans[previous].end..spans[anchor].start];
        if gap
            .chars()
            .any(|ch| matches!(ch, '.' | '?' | '!' | ':' | ';'))
        {
            break;
        }

        let previous_word = spans[previous].normalized.as_str();
        if matches!(
            previous_word,
            "is" | "are"
                | "was"
                | "were"
                | "be"
                | "been"
                | "being"
                | "called"
                | "named"
                | "unlike"
                | "like"
                | "with"
                | "without"
                | "between"
                | "versus"
                | "vs"
                | "than"
                | "from"
                | "for"
        ) {
            return anchor;
        }

        anchor = previous;
        walked_words += 1;
        if walked_words >= 6 {
            break;
        }
    }

    anchor
}

fn contextual_anchor_index(raw_text: &str, spans: &[WordSpan], mut anchor: usize) -> usize {
    let mut walked_words = 0usize;

    while anchor > 0 {
        let previous = anchor - 1;
        let gap = &raw_text[spans[previous].end..spans[anchor].start];
        if gap
            .chars()
            .any(|ch| matches!(ch, '.' | '?' | '!' | ':' | ';'))
        {
            break;
        }

        let previous_word = spans[previous].normalized.as_str();
        if matches!(
            previous_word,
            "unlike"
                | "like"
                | "with"
                | "without"
                | "between"
                | "versus"
                | "vs"
                | "than"
                | "from"
                | "for"
        ) {
            return anchor;
        }

        anchor = previous;
        walked_words += 1;
        if walked_words >= 6 {
            break;
        }
    }

    anchor
}

fn normalize_phrase_tail(text: &str) -> String {
    let mut words = collect_word_spans(text)
        .into_iter()
        .map(|span| text[span.start..span.end].to_string())
        .collect::<Vec<_>>();

    while matches!(
        words.first().map(|word| word.to_ascii_lowercase()),
        Some(prefix) if matches!(
            prefix.as_str(),
            "just" | "only" | "simply" | "rather" | "instead" | "actually"
        )
    ) {
        words.remove(0);
    }

    words.join(" ")
}

fn normalize_candidate_spacing(text: &str) -> Option<String> {
    let mut normalized = text
        .replace(" ,", ",")
        .replace(" .", ".")
        .replace(" !", "!")
        .replace(" ?", "?")
        .replace(" :", ":")
        .replace(" ;", ";");

    while normalized.contains("  ") {
        normalized = normalized.replace("  ", " ");
    }

    let normalized = normalized.trim().to_string();
    (!normalized.is_empty()).then_some(normalized)
}

fn recommended_candidate(
    rewrite_candidates: &[RewriteCandidate],
    edit_hypotheses: &[cleanup::EditHypothesis],
    edit_context: &RewriteEditContext,
) -> Option<RewriteCandidate> {
    if edit_context.cue_is_utterance_initial && edit_context.courtesy_prefix_detected {
        return None;
    }

    has_strong_explicit_hypothesis(edit_hypotheses)
        .then(|| rewrite_candidates.first().cloned())
        .flatten()
}

fn derive_edit_context(
    raw_text: &str,
    edit_hypotheses: &[cleanup::EditHypothesis],
) -> RewriteEditContext {
    let spans = collect_word_spans(raw_text);
    let Some(hypothesis) = earliest_strong_explicit_hypothesis(edit_hypotheses) else {
        return RewriteEditContext::default();
    };

    let prefix_words = spans
        .get(..hypothesis.word_start)
        .unwrap_or(&[])
        .iter()
        .map(|span| span.normalized.as_str())
        .collect::<Vec<_>>();
    let courtesy_prefix_word_count = courtesy_prefix_word_count(&prefix_words);
    let preceding_content_word_count = prefix_words
        .len()
        .saturating_sub(courtesy_prefix_word_count);

    RewriteEditContext {
        cue_is_utterance_initial: prefix_words.is_empty() || preceding_content_word_count == 0,
        preceding_content_word_count,
        courtesy_prefix_detected: courtesy_prefix_word_count > 0,
        has_recent_same_focus_entry: false,
        recommended_session_action_is_replace: false,
    }
}

fn earliest_strong_explicit_hypothesis(
    edit_hypotheses: &[cleanup::EditHypothesis],
) -> Option<&cleanup::EditHypothesis> {
    edit_hypotheses
        .iter()
        .filter(|hypothesis| {
            hypothesis.strength == cleanup::EditSignalStrength::Strong
                && matches!(
                    hypothesis.match_source,
                    cleanup::EditHypothesisMatchSource::Exact
                        | cleanup::EditHypothesisMatchSource::Alias
                )
        })
        .min_by_key(|hypothesis| hypothesis.word_start)
}

fn courtesy_prefix_word_count(words: &[&str]) -> usize {
    match words {
        ["my", "apologies"] => 2,
        ["apologies"] | ["sorry"] => 1,
        _ => 0,
    }
}

fn has_strong_explicit_hypothesis(edit_hypotheses: &[cleanup::EditHypothesis]) -> bool {
    edit_hypotheses.iter().any(|hypothesis| {
        hypothesis.strength == cleanup::EditSignalStrength::Strong
            && matches!(
                hypothesis.match_source,
                cleanup::EditHypothesisMatchSource::Exact
                    | cleanup::EditHypothesisMatchSource::Alias
            )
    })
}

fn candidate_priority(kind: RewriteCandidateKind) -> u8 {
    match kind {
        RewriteCandidateKind::SpanReplacement => 0,
        RewriteCandidateKind::ClauseReplacement => 1,
        RewriteCandidateKind::SentenceReplacement => 2,
        RewriteCandidateKind::ContextualReplacement => 3,
        RewriteCandidateKind::AggressiveCorrection => 4,
        RewriteCandidateKind::CancelPreviousSentence => 5,
        RewriteCandidateKind::CancelPreviousClause => 6,
        RewriteCandidateKind::FollowingReplacement => 7,
        RewriteCandidateKind::GlossaryCorrection => 8,
        RewriteCandidateKind::ConservativeCorrection => 9,
        RewriteCandidateKind::DropCueOnly => 10,
        RewriteCandidateKind::Literal => 11,
    }
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

pub fn list_dictionary(config_override: Option<&Path>) -> Result<()> {
    let config = Config::load(config_override)?;
    let entries = read_dictionary_file(&config.resolved_dictionary_path())?;
    if entries.is_empty() {
        println!("No dictionary entries configured.");
        return Ok(());
    }

    for entry in entries {
        println!("{} -> {}", entry.phrase, entry.replace);
    }

    Ok(())
}

pub fn add_dictionary(config_override: Option<&Path>, phrase: &str, replace: &str) -> Result<()> {
    let config = Config::load(config_override)?;
    let path = config.resolved_dictionary_path();
    let mut entries = read_dictionary_file(&path)?;
    upsert_dictionary_entry(&mut entries, phrase, replace);
    write_dictionary_file(&path, &entries)?;
    println!("Added dictionary entry: {} -> {}", phrase, replace);
    println!("Dictionary updated: {}", path.display());
    Ok(())
}

pub fn remove_dictionary(config_override: Option<&Path>, phrase: &str) -> Result<()> {
    let config = Config::load(config_override)?;
    let path = config.resolved_dictionary_path();
    let mut entries = read_dictionary_file(&path)?;
    let removed = remove_dictionary_entry(&mut entries, phrase);
    write_dictionary_file(&path, &entries)?;
    if removed {
        println!("Removed dictionary entry: {}", phrase);
    } else {
        println!("No dictionary entry matched: {}", phrase);
    }
    println!("Dictionary updated: {}", path.display());
    Ok(())
}

pub fn list_snippets(config_override: Option<&Path>) -> Result<()> {
    let config = Config::load(config_override)?;
    let snippets = read_snippet_file(&config.resolved_snippets_path())?;
    if snippets.is_empty() {
        println!("No snippets configured.");
        return Ok(());
    }

    for snippet in snippets {
        println!("{} -> {}", snippet.name, snippet.text.replace('\n', "\\n"));
    }

    Ok(())
}

pub fn add_snippet(config_override: Option<&Path>, name: &str, text: &str) -> Result<()> {
    let config = Config::load(config_override)?;
    let path = config.resolved_snippets_path();
    let mut snippets = read_snippet_file(&path)?;
    upsert_snippet(&mut snippets, name, text);
    write_snippet_file(&path, &snippets)?;
    println!("Added snippet: {}", name);
    println!("Snippets updated: {}", path.display());
    Ok(())
}

pub fn remove_snippet(config_override: Option<&Path>, name: &str) -> Result<()> {
    let config = Config::load(config_override)?;
    let path = config.resolved_snippets_path();
    let mut snippets = read_snippet_file(&path)?;
    let removed = remove_snippet_entry(&mut snippets, name);
    write_snippet_file(&path, &snippets)?;
    if removed {
        println!("Removed snippet: {}", name);
    } else {
        println!("No snippet matched: {}", name);
    }
    println!("Snippets updated: {}", path.display());
    Ok(())
}

pub fn print_rewrite_instructions_path(config_override: Option<&Path>) -> Result<()> {
    let config = Config::load(config_override)?;
    match config.resolved_rewrite_instructions_path() {
        Some(path) => println!("{}", path.display()),
        None => println!("No rewrite instructions path configured."),
    }
    Ok(())
}

fn load_custom_instructions(config: &Config) -> Result<String> {
    let Some(path) = config.resolved_rewrite_instructions_path() else {
        return Ok(String::new());
    };

    match std::fs::read_to_string(&path) {
        Ok(contents) => Ok(contents.trim().to_string()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(String::new()),
        Err(err) => Err(WhsprError::Config(format!(
            "failed to read rewrite instructions {}: {err}",
            path.display()
        ))),
    }
}

fn apply_dictionary(text: &str, rules: &PersonalizationRules) -> String {
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

fn collect_word_spans(text: &str) -> Vec<WordSpan> {
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

fn normalized_words(text: &str) -> Vec<String> {
    collect_word_spans(text)
        .into_iter()
        .map(|span| span.normalized)
        .collect()
}

fn is_word_char(ch: char) -> bool {
    ch.is_alphanumeric() || matches!(ch, '\'' | '-')
}

fn read_dictionary_file(path: &Path) -> Result<Vec<DictionaryEntry>> {
    if !path.exists() {
        return Ok(Vec::new());
    }

    let contents = std::fs::read_to_string(path).map_err(|e| {
        WhsprError::Config(format!("failed to read dictionary {}: {e}", path.display()))
    })?;
    let file: DictionaryFile = toml::from_str(&contents).map_err(|e| {
        WhsprError::Config(format!(
            "failed to parse dictionary {}: {e}",
            path.display()
        ))
    })?;
    Ok(file.entries)
}

fn write_dictionary_file(path: &Path, entries: &[DictionaryEntry]) -> Result<()> {
    write_parent(path)?;
    let file = DictionaryFile {
        entries: entries.to_vec(),
    };
    let contents = toml::to_string_pretty(&file)
        .map_err(|e| WhsprError::Config(format!("failed to encode dictionary: {e}")))?;
    std::fs::write(path, contents).map_err(|e| {
        WhsprError::Config(format!(
            "failed to write dictionary {}: {e}",
            path.display()
        ))
    })?;
    Ok(())
}

fn read_snippet_file(path: &Path) -> Result<Vec<SnippetEntry>> {
    if !path.exists() {
        return Ok(Vec::new());
    }

    let contents = std::fs::read_to_string(path).map_err(|e| {
        WhsprError::Config(format!("failed to read snippets {}: {e}", path.display()))
    })?;
    let file: SnippetFile = toml::from_str(&contents).map_err(|e| {
        WhsprError::Config(format!("failed to parse snippets {}: {e}", path.display()))
    })?;
    Ok(file.snippets)
}

fn write_snippet_file(path: &Path, snippets: &[SnippetEntry]) -> Result<()> {
    write_parent(path)?;
    let file = SnippetFile {
        snippets: snippets.to_vec(),
    };
    let contents = toml::to_string_pretty(&file)
        .map_err(|e| WhsprError::Config(format!("failed to encode snippets: {e}")))?;
    std::fs::write(path, contents).map_err(|e| {
        WhsprError::Config(format!("failed to write snippets {}: {e}", path.display()))
    })?;
    Ok(())
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

fn upsert_dictionary_entry(entries: &mut Vec<DictionaryEntry>, phrase: &str, replace: &str) {
    let target = normalized_words(phrase);
    if let Some(existing) = entries
        .iter_mut()
        .find(|entry| normalized_words(&entry.phrase) == target)
    {
        existing.phrase = phrase.to_string();
        existing.replace = replace.to_string();
        return;
    }

    entries.push(DictionaryEntry {
        phrase: phrase.to_string(),
        replace: replace.to_string(),
    });
}

fn remove_dictionary_entry(entries: &mut Vec<DictionaryEntry>, phrase: &str) -> bool {
    let target = normalized_words(phrase);
    let before = entries.len();
    entries.retain(|entry| normalized_words(&entry.phrase) != target);
    before != entries.len()
}

fn upsert_snippet(snippets: &mut Vec<SnippetEntry>, name: &str, text: &str) {
    let target = normalized_words(name);
    if let Some(existing) = snippets
        .iter_mut()
        .find(|entry| normalized_words(&entry.name) == target)
    {
        existing.name = name.to_string();
        existing.text = text.to_string();
        return;
    }

    snippets.push(SnippetEntry {
        name: name.to_string(),
        text: text.to_string(),
    });
}

fn remove_snippet_entry(snippets: &mut Vec<SnippetEntry>, name: &str) -> bool {
    let target = normalized_words(name);
    let before = snippets.len();
    snippets.retain(|entry| normalized_words(&entry.name) != target);
    before != snippets.len()
}

impl PreparedDictionaryEntry {
    fn new(entry: DictionaryEntry) -> std::result::Result<Self, DictionaryEntry> {
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
    fn new(entry: SnippetEntry) -> std::result::Result<Self, SnippetEntry> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, PostprocessMode};
    use crate::rewrite_profile::RewriteProfile;
    use crate::rewrite_protocol::{
        RewriteCandidateKind, RewriteEditHypothesisMatchSource, RewriteEditSignalKind,
        RewriteEditSignalScope, RewriteEditSignalStrength,
    };

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
    fn build_rewrite_transcript_applies_dictionary_before_rewrite() {
        let transcript = Transcript {
            raw_text: "wisper flow is useful".into(),
            detected_language: Some("en".into()),
            segments: vec![crate::transcribe::TranscriptSegment {
                text: "wisper flow".into(),
                start_ms: 0,
                end_ms: 100,
            }],
        };
        let rewrite = build_rewrite_transcript(&transcript, &rules());
        assert_eq!(rewrite.raw_text, "Wispr Flow is useful");
        assert_eq!(rewrite.segments[0].text, "Wispr Flow");
        assert!(rewrite.edit_intents.is_empty());
        assert!(rewrite.edit_signals.is_empty());
        assert!(rewrite.edit_hypotheses.is_empty());
        assert_eq!(rewrite.rewrite_candidates.len(), 1);
        assert_eq!(
            rewrite.rewrite_candidates[0].kind,
            RewriteCandidateKind::Literal
        );
        assert!(rewrite.recommended_candidate.is_none());
    }

    #[test]
    fn build_rewrite_transcript_carries_edit_intents() {
        let transcript = Transcript {
            raw_text: "hello there never mind".into(),
            detected_language: Some("en".into()),
            segments: Vec::new(),
        };

        let rewrite = build_rewrite_transcript(&transcript, &rules());
        assert_eq!(rewrite.correction_aware_text, "");
        assert_eq!(rewrite.edit_intents.len(), 1);
        assert_eq!(
            rewrite.edit_intents[0].action,
            RewriteEditAction::ReplacePreviousClause
        );
        assert_eq!(rewrite.edit_intents[0].trigger, "never mind");
        assert_eq!(rewrite.edit_signals.len(), 1);
        assert_eq!(rewrite.edit_signals[0].kind, RewriteEditSignalKind::Cancel);
        assert_eq!(
            rewrite.edit_signals[0].scope_hint,
            RewriteEditSignalScope::Clause
        );
        assert_eq!(
            rewrite.edit_signals[0].strength,
            RewriteEditSignalStrength::Strong
        );
        assert_eq!(rewrite.edit_hypotheses.len(), 1);
        assert_eq!(rewrite.edit_hypotheses[0].cue_family, "never_mind");
        assert_eq!(
            rewrite.edit_hypotheses[0].match_source,
            RewriteEditHypothesisMatchSource::Exact
        );
        assert_eq!(
            rewrite.edit_hypotheses[0].replacement_scope,
            crate::rewrite_protocol::RewriteReplacementScope::Clause
        );
        assert_eq!(
            rewrite.edit_hypotheses[0].tail_shape,
            crate::rewrite_protocol::RewriteTailShape::Empty
        );
        assert_eq!(
            rewrite
                .recommended_candidate
                .as_ref()
                .map(|candidate| candidate.kind),
            Some(RewriteCandidateKind::DropCueOnly)
        );
    }

    #[test]
    fn build_rewrite_transcript_carries_aggressive_candidate() {
        let transcript = Transcript {
            raw_text: "my name is notes, scratch that my name is jonatan".into(),
            detected_language: Some("en".into()),
            segments: Vec::new(),
        };

        let rewrite = build_rewrite_transcript(&transcript, &rules());
        assert_eq!(rewrite.correction_aware_text, "My my name is jonatan");
        assert_eq!(
            rewrite.aggressive_correction_text.as_deref(),
            Some("My name is jonatan")
        );
        assert_eq!(rewrite.rewrite_candidates.len(), 4);
        assert_eq!(
            rewrite.rewrite_candidates[0].kind,
            RewriteCandidateKind::ClauseReplacement
        );
        assert_eq!(
            rewrite
                .rewrite_candidates
                .last()
                .map(|candidate| candidate.kind),
            Some(RewriteCandidateKind::Literal)
        );
        assert_eq!(
            rewrite.rewrite_candidates[1].kind,
            RewriteCandidateKind::AggressiveCorrection
        );
        assert_eq!(
            rewrite
                .recommended_candidate
                .as_ref()
                .map(|candidate| candidate.kind),
            Some(RewriteCandidateKind::ClauseReplacement)
        );
        assert!(rewrite.rewrite_candidates.iter().any(|candidate| {
            candidate.kind == RewriteCandidateKind::ClauseReplacement
                && candidate.text.contains("jonatan")
        }));
    }

    #[test]
    fn build_rewrite_transcript_adds_candidate_from_hypothesis_suffix() {
        let transcript = Transcript {
            raw_text: "hello there scratch that hi".into(),
            detected_language: Some("en".into()),
            segments: Vec::new(),
        };

        let rewrite = build_rewrite_transcript(&transcript, &rules());
        assert!(rewrite.rewrite_candidates.iter().any(|candidate| matches!(
            candidate.kind,
            RewriteCandidateKind::SpanReplacement
                | RewriteCandidateKind::ClauseReplacement
                | RewriteCandidateKind::SentenceReplacement
        ) && candidate.text == "hi"));
        assert!(rewrite.recommended_candidate.is_some());
    }

    #[test]
    fn contextual_replacement_can_preserve_unlike_prefix() {
        let transcript = Transcript {
            raw_text:
                "unlike mobile apps or sms codes scratch that sms codes it requires a physical touch"
                    .into(),
            detected_language: Some("en".into()),
            segments: Vec::new(),
        };

        let rewrite = build_rewrite_transcript(&transcript, &rules());
        assert!(rewrite.rewrite_candidates.iter().any(|candidate| {
            candidate.kind == RewriteCandidateKind::SpanReplacement
                && candidate.text.starts_with("unlike sms codes")
        }));
    }

    #[test]
    fn strong_exact_cues_prioritize_non_literal_candidates() {
        let transcript = Transcript {
            raw_text:
                "unlike mobile apps or sms codes scratch that sms codes it requires a physical touch"
                    .into(),
            detected_language: Some("en".into()),
            segments: Vec::new(),
        };

        let rewrite = build_rewrite_transcript(&transcript, &rules());
        assert_eq!(
            rewrite
                .rewrite_candidates
                .first()
                .map(|candidate| candidate.kind),
            Some(RewriteCandidateKind::SpanReplacement)
        );
        assert_eq!(
            rewrite
                .rewrite_candidates
                .last()
                .map(|candidate| candidate.kind),
            Some(RewriteCandidateKind::Literal)
        );
        assert_eq!(
            rewrite
                .recommended_candidate
                .as_ref()
                .map(|candidate| candidate.kind),
            Some(RewriteCandidateKind::SpanReplacement)
        );
    }

    #[test]
    fn courtesy_prefixed_opening_does_not_force_non_literal_recommendation() {
        let transcript = Transcript {
            raw_text: "my apologies i meant jonatan".into(),
            detected_language: Some("en".into()),
            segments: Vec::new(),
        };

        let rewrite = build_rewrite_transcript(&transcript, &rules());
        assert!(rewrite.edit_context.cue_is_utterance_initial);
        assert!(rewrite.edit_context.courtesy_prefix_detected);
        assert_eq!(rewrite.edit_context.preceding_content_word_count, 0);
        assert_eq!(
            rewrite
                .rewrite_candidates
                .first()
                .map(|candidate| candidate.kind),
            Some(RewriteCandidateKind::Literal)
        );
        assert!(rewrite.recommended_candidate.is_none());
    }

    #[test]
    fn load_custom_instructions_tolerates_missing_file() {
        let mut config = Config::default();
        config.rewrite.instructions_path = "/tmp/whispers-missing-instructions.txt".into();
        let loaded = load_custom_instructions(&config).expect("load");
        assert!(loaded.is_empty());
    }

    #[test]
    fn transcription_prompt_includes_dictionary_targets() {
        let prompt = transcription_prompt(&rules()).expect("prompt");
        assert!(prompt.contains("Wispr Flow"));
        assert!(prompt.contains("OpenAI"));
    }

    #[test]
    fn add_and_remove_dictionary_entries_roundtrip() {
        let _env_lock = crate::test_support::env_lock();
        let _guard = crate::test_support::EnvVarGuard::capture(&[
            "HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
        ]);
        let home = crate::test_support::unique_temp_dir("personalization-dict-home");
        crate::test_support::set_env("HOME", &home.to_string_lossy());
        crate::test_support::remove_env("XDG_CONFIG_HOME");
        crate::test_support::remove_env("XDG_DATA_HOME");

        add_dictionary(None, "wisper flow", "Wispr Flow").expect("add dictionary");
        let config = Config::load(None).expect("config");
        let entries = read_dictionary_file(&config.resolved_dictionary_path()).expect("read");
        assert_eq!(entries.len(), 1);

        remove_dictionary(None, "wisper flow").expect("remove dictionary");
        let entries = read_dictionary_file(&config.resolved_dictionary_path()).expect("read");
        assert!(entries.is_empty());
    }

    #[test]
    fn add_and_remove_snippets_roundtrip() {
        let _env_lock = crate::test_support::env_lock();
        let _guard = crate::test_support::EnvVarGuard::capture(&[
            "HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
        ]);
        let home = crate::test_support::unique_temp_dir("personalization-snippet-home");
        crate::test_support::set_env("HOME", &home.to_string_lossy());
        crate::test_support::remove_env("XDG_CONFIG_HOME");
        crate::test_support::remove_env("XDG_DATA_HOME");

        add_snippet(None, "signature", "Best regards,\nNotes").expect("add snippet");
        let config = Config::load(None).expect("config");
        let entries = read_snippet_file(&config.resolved_snippets_path()).expect("read");
        assert_eq!(entries.len(), 1);

        remove_snippet(None, "signature").expect("remove snippet");
        let entries = read_snippet_file(&config.resolved_snippets_path()).expect("read");
        assert!(entries.is_empty());
    }

    #[test]
    fn default_config_paths_support_personalization_files() {
        let config = Config::default();
        assert_eq!(config.postprocess.mode, PostprocessMode::Raw);
        assert_eq!(config.rewrite.profile, RewriteProfile::Auto);
        assert_eq!(config.personalization.snippet_trigger, "insert");
    }
}

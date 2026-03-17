use crate::cleanup;
use crate::rewrite_protocol::{
    RewriteCandidate, RewriteCandidateKind, RewriteEditAction, RewriteEditHypothesis,
    RewriteEditHypothesisMatchSource, RewriteEditIntent, RewriteEditSignal, RewriteEditSignalKind,
    RewriteEditSignalScope, RewriteEditSignalStrength, RewriteIntentConfidence,
    RewritePolicyContext, RewriteReplacementScope, RewriteTailShape, RewriteTranscript,
    RewriteTranscriptSegment,
};
use crate::transcribe::Transcript;

use super::{PersonalizationRules, WordSpan, apply_dictionary, collect_word_spans};

#[derive(Debug, Clone, Default)]
struct TailParts {
    normalized_phrase_tail: Option<String>,
    span_replacement_tail: Option<String>,
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
    let recommended_candidate =
        recommended_candidate(&rewrite_candidates, &analysis.edit_hypotheses);

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

    if has_strong_explicit_hypothesis(edit_hypotheses) {
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
) -> Option<RewriteCandidate> {
    has_strong_explicit_hypothesis(edit_hypotheses)
        .then(|| rewrite_candidates.first().cloned())
        .flatten()
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

#[cfg(test)]
mod tests {
    use super::build_rewrite_transcript;
    use crate::rewrite_protocol::{
        RewriteCandidateKind, RewriteEditHypothesisMatchSource, RewriteEditSignalKind,
        RewriteEditSignalScope, RewriteEditSignalStrength,
    };
    use crate::transcribe::Transcript;

    use super::super::store::{DictionaryEntry, SnippetEntry};
    use super::super::{
        PersonalizationRules, PreparedDictionaryEntry, PreparedSnippet, normalized_words,
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
        assert_eq!(
            rewrite.edit_intents[0].action,
            crate::rewrite_protocol::RewriteEditAction::ReplacePreviousClause
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
}

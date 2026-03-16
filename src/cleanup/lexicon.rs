use super::{
    CueFamilySpec, EditHypothesis, EditHypothesisMatchSource, EditSignal, EditSignalKind,
    EditSignalScope, EditSignalStrength, ObservedWord, Piece, ReplacementScope, TailShape,
    match_words_with_soft_punctuation, normalized_word_str, skip_correction_gap, tokenize,
};

pub(super) fn explicit_followup_cue_lookahead(pieces: &[Piece]) -> Option<usize> {
    for spec in cue_family_specs() {
        if !matches!(spec.kind, EditSignalKind::Cancel) {
            continue;
        }

        if let Some(end) = match_words_with_soft_punctuation(pieces, 0, spec.canonical) {
            return Some(skip_correction_gap(pieces, end));
        }

        for alias in spec.session_followup_aliases {
            if let Some(end) = match_words_with_soft_punctuation(pieces, 0, alias) {
                return Some(skip_correction_gap(pieces, end));
            }
        }
    }

    None
}

pub(super) fn cue_family_specs() -> &'static [CueFamilySpec] {
    const SCRATCH_THAT_ALIASES: &[&[&str]] = &[&["scratchthat"]];
    const SCRATCH_THAT_SESSION_ALIASES: &[&[&str]] = &[
        &["scratchthat"],
        &["scratchvat"],
        &["scratchfat"],
        &["scratchfart"],
        &["scratchfarts"],
        &["scratchfot"],
        &["scratchbot"],
        &["scratchvatnotes"],
        &["rajvat"],
        &["srajvat"],
        &["srajvatnotes"],
    ];
    const NEVER_MIND_ALIASES: &[&[&str]] = &[&["nevermind"]];
    const NEVER_MIND_SESSION_ALIASES: &[&[&str]] = &[&["nevermind"], &["nevarmind"]];
    const WAIT_NO_ALIASES: &[&[&str]] = &[&["waitno"]];
    const I_MEANT_ALIASES: &[&[&str]] = &[&["imeant"]];
    const I_MEAN_ALIASES: &[&[&str]] = &[&["imean"]];
    const OR_RATHER_ALIASES: &[&[&str]] = &[&["orrather"]];
    const SPECS: &[CueFamilySpec] = &[
        CueFamilySpec {
            cue_family: "scratch_that",
            kind: EditSignalKind::Cancel,
            scope_hint: EditSignalScope::Sentence,
            canonical: &["scratch", "that"],
            aliases: SCRATCH_THAT_ALIASES,
            session_followup_aliases: SCRATCH_THAT_SESSION_ALIASES,
        },
        CueFamilySpec {
            cue_family: "never_mind",
            kind: EditSignalKind::Cancel,
            scope_hint: EditSignalScope::Sentence,
            canonical: &["never", "mind"],
            aliases: NEVER_MIND_ALIASES,
            session_followup_aliases: NEVER_MIND_SESSION_ALIASES,
        },
        CueFamilySpec {
            cue_family: "wait_no",
            kind: EditSignalKind::Replace,
            scope_hint: EditSignalScope::Sentence,
            canonical: &["wait", "no"],
            aliases: WAIT_NO_ALIASES,
            session_followup_aliases: &[],
        },
        CueFamilySpec {
            cue_family: "i_meant",
            kind: EditSignalKind::Replace,
            scope_hint: EditSignalScope::Phrase,
            canonical: &["i", "meant"],
            aliases: I_MEANT_ALIASES,
            session_followup_aliases: &[],
        },
        CueFamilySpec {
            cue_family: "i_mean",
            kind: EditSignalKind::Replace,
            scope_hint: EditSignalScope::Phrase,
            canonical: &["i", "mean"],
            aliases: I_MEAN_ALIASES,
            session_followup_aliases: &[],
        },
        CueFamilySpec {
            cue_family: "or_rather",
            kind: EditSignalKind::Restatement,
            scope_hint: EditSignalScope::Phrase,
            canonical: &["or", "rather"],
            aliases: OR_RATHER_ALIASES,
            session_followup_aliases: &[],
        },
    ];

    SPECS
}

pub(super) fn collect_edit_hypotheses(
    raw: &str,
    edit_signals: &[EditSignal],
) -> Vec<EditHypothesis> {
    let observed_words = collect_observed_words(raw);
    if observed_words.is_empty() {
        return Vec::new();
    }

    let mut hypotheses = Vec::new();

    for spec in cue_family_specs() {
        let mut index = 0usize;
        while index < observed_words.len() {
            let Some((match_source, matched_len)) =
                match_cue_family_at(&observed_words, index, spec)
            else {
                index += 1;
                continue;
            };

            let signal = signal_for_cue_family(edit_signals, spec.cue_family);
            let strength = signal
                .map(|signal| signal.strength)
                .unwrap_or(match match_source {
                    EditHypothesisMatchSource::NearMiss => EditSignalStrength::Possible,
                    EditHypothesisMatchSource::Exact | EditHypothesisMatchSource::Alias => {
                        EditSignalStrength::Strong
                    }
                });
            let scope_hint = signal
                .map(|signal| signal.scope_hint)
                .unwrap_or(spec.scope_hint);
            let kind = signal.map(|signal| signal.kind).unwrap_or(spec.kind);
            let (replacement_scope, tail_shape) =
                classify_replacement_scope(&observed_words, index + matched_len, scope_hint);

            hypotheses.push(EditHypothesis {
                cue_family: spec.cue_family,
                matched_text: observed_words[index..index + matched_len]
                    .iter()
                    .map(|word| word.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" "),
                match_source,
                kind,
                scope_hint,
                replacement_scope,
                tail_shape,
                strength,
                word_start: index,
                word_end: index + matched_len,
            });

            index += matched_len.max(1);
        }
    }

    hypotheses.sort_by_key(|hypothesis| (hypothesis.word_start, hypothesis.word_end));
    hypotheses.dedup_by(|right, left| {
        right.cue_family == left.cue_family
            && right.word_start == left.word_start
            && right.word_end == left.word_end
    });
    hypotheses
}

fn collect_observed_words(raw: &str) -> Vec<ObservedWord> {
    tokenize(raw)
        .into_iter()
        .filter_map(|piece| match piece {
            Piece::Word(text) => {
                let normalized = normalized_word_str(&text);
                (!normalized.is_empty()).then_some(ObservedWord { text, normalized })
            }
            Piece::Punctuation(_) | Piece::Break(_) => None,
        })
        .collect()
}

fn match_cue_family_at(
    observed_words: &[ObservedWord],
    start: usize,
    spec: &CueFamilySpec,
) -> Option<(EditHypothesisMatchSource, usize)> {
    if matches_observed_words(observed_words, start, spec.canonical) {
        return Some((EditHypothesisMatchSource::Exact, spec.canonical.len()));
    }

    for alias in spec.aliases {
        if matches_observed_words(observed_words, start, alias) {
            return Some((EditHypothesisMatchSource::Alias, alias.len()));
        }
    }

    let mut candidate_lengths = vec![1usize, spec.canonical.len()];
    candidate_lengths.extend(spec.aliases.iter().map(|alias| alias.len()));
    candidate_lengths.sort_unstable();
    candidate_lengths.dedup();

    for len in candidate_lengths {
        if start + len > observed_words.len() {
            continue;
        }

        let observed = compact_observed_words(&observed_words[start..start + len]);
        if observed.is_empty() {
            continue;
        }

        if is_limited_near_miss(&observed, &compact_phrase(spec.canonical))
            || spec
                .aliases
                .iter()
                .any(|alias| is_limited_near_miss(&observed, &compact_phrase(alias)))
        {
            return Some((EditHypothesisMatchSource::NearMiss, len));
        }
    }

    None
}

fn matches_observed_words(
    observed_words: &[ObservedWord],
    start: usize,
    expected: &[&str],
) -> bool {
    expected.iter().enumerate().all(|(offset, expected_word)| {
        observed_words
            .get(start + offset)
            .map(|word| word.normalized.as_str())
            == Some(*expected_word)
    })
}

fn compact_observed_words(observed_words: &[ObservedWord]) -> String {
    observed_words
        .iter()
        .map(|word| word.normalized.as_str())
        .collect::<Vec<_>>()
        .join("")
}

fn compact_phrase(words: &[&str]) -> String {
    words.join("")
}

fn signal_for_cue_family<'a>(
    edit_signals: &'a [EditSignal],
    cue_family: &str,
) -> Option<&'a EditSignal> {
    edit_signals
        .iter()
        .find(|signal| cue_family_for_phrase(signal.trigger) == Some(cue_family))
}

fn cue_family_for_phrase(phrase: &str) -> Option<&'static str> {
    match phrase {
        "scratch that" | "actually scratch that" => Some("scratch_that"),
        "never mind"
        | "nevermind"
        | "actually never mind"
        | "actually nevermind"
        | "oh wait never mind"
        | "oh wait nevermind"
        | "forget that" => Some("never_mind"),
        "wait no" | "actually wait no" => Some("wait_no"),
        "i meant" | "actually i meant" => Some("i_meant"),
        "i mean" | "actually i mean" => Some("i_mean"),
        "or rather" => Some("or_rather"),
        _ => None,
    }
}

fn classify_replacement_scope(
    observed_words: &[ObservedWord],
    tail_start: usize,
    scope_hint: EditSignalScope,
) -> (ReplacementScope, TailShape) {
    let tail_words = observed_words
        .get(tail_start..)
        .unwrap_or(&[])
        .iter()
        .map(|word| word.normalized.as_str())
        .take(8)
        .collect::<Vec<_>>();

    if tail_words.is_empty() {
        let replacement_scope = match scope_hint {
            EditSignalScope::Sentence => ReplacementScope::Sentence,
            EditSignalScope::Clause => ReplacementScope::Clause,
            EditSignalScope::Phrase | EditSignalScope::Unknown => ReplacementScope::Span,
        };
        return (replacement_scope, TailShape::Empty);
    }

    let tail_shape = if looks_like_clause_tail(&tail_words) {
        TailShape::Clause
    } else {
        TailShape::Phrase
    };

    let replacement_scope = match scope_hint {
        EditSignalScope::Sentence => {
            if matches!(tail_shape, TailShape::Phrase) && tail_words.len() > 3 {
                ReplacementScope::Clause
            } else {
                ReplacementScope::Sentence
            }
        }
        EditSignalScope::Clause => {
            if matches!(tail_shape, TailShape::Phrase) {
                ReplacementScope::Span
            } else {
                ReplacementScope::Clause
            }
        }
        EditSignalScope::Phrase => ReplacementScope::Span,
        EditSignalScope::Unknown => {
            if matches!(tail_shape, TailShape::Clause) {
                ReplacementScope::Clause
            } else {
                ReplacementScope::Span
            }
        }
    };

    (replacement_scope, tail_shape)
}

fn looks_like_clause_tail(tail_words: &[&str]) -> bool {
    const CLAUSE_WORDS: &[&str] = &[
        "am", "are", "be", "been", "being", "can", "could", "did", "do", "does", "had", "has",
        "have", "is", "must", "need", "needs", "required", "requires", "should", "was", "were",
        "will", "would",
    ];
    const SUBJECT_WORDS: &[&str] = &[
        "i", "it", "he", "she", "they", "we", "you", "this", "that", "there", "my", "our", "their",
        "your",
    ];

    tail_words.iter().any(|word| CLAUSE_WORDS.contains(word))
        || (tail_words.len() >= 2
            && SUBJECT_WORDS.contains(&tail_words[0])
            && !matches!(tail_words[1], "and" | "or" | "but"))
}

fn is_limited_near_miss(observed: &str, target: &str) -> bool {
    if observed.is_empty() || target.is_empty() || observed == target {
        return false;
    }

    let common_prefix = observed
        .chars()
        .zip(target.chars())
        .take_while(|(left, right)| left == right)
        .count();
    if common_prefix < observed.len().min(target.len()).min(4) {
        return false;
    }

    let observed_prefix = if observed.chars().count() > target.chars().count() {
        observed
            .chars()
            .take(target.chars().count())
            .collect::<String>()
    } else {
        observed.to_string()
    };
    let distance = bounded_levenshtein(&observed_prefix, target, 3);
    distance <= 3
}

fn bounded_levenshtein(left: &str, right: &str, max_distance: usize) -> usize {
    let left_chars: Vec<char> = left.chars().collect();
    let right_chars: Vec<char> = right.chars().collect();

    if left_chars.is_empty() {
        return right_chars.len();
    }
    if right_chars.is_empty() {
        return left_chars.len();
    }
    if left_chars.len().abs_diff(right_chars.len()) > max_distance {
        return max_distance + 1;
    }

    let mut prev: Vec<usize> = (0..=right_chars.len()).collect();
    let mut curr = vec![0usize; right_chars.len() + 1];

    for (i, left_char) in left_chars.iter().enumerate() {
        curr[0] = i + 1;
        let mut row_min = curr[0];
        for (j, right_char) in right_chars.iter().enumerate() {
            let cost = usize::from(left_char != right_char);
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
            row_min = row_min.min(curr[j + 1]);
        }
        if row_min > max_distance {
            return max_distance + 1;
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[right_chars.len()]
}

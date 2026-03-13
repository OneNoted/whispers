use crate::config::{CleanupConfig, CleanupProfile};
use crate::transcribe::Transcript;

#[derive(Debug, Clone, PartialEq, Eq)]
enum Piece {
    Word(String),
    Punctuation(char),
    Break(BreakKind),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BreakKind {
    Line,
    Paragraph,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CorrectionKind {
    Phrase,
    Sentence,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorrectionAnalysis {
    pub text: String,
    pub aggressive_text: Option<String>,
    pub edit_intents: Vec<EditIntent>,
    pub edit_signals: Vec<EditSignal>,
    pub edit_hypotheses: Vec<EditHypothesis>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditIntent {
    pub action: EditIntentAction,
    pub trigger: &'static str,
    pub confidence: EditIntentConfidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditIntentAction {
    ReplacePreviousPhrase,
    ReplacePreviousClause,
    ReplacePreviousSentence,
    DropEditCue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditIntentConfidence {
    High,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditSignal {
    pub trigger: &'static str,
    pub kind: EditSignalKind,
    pub scope_hint: EditSignalScope,
    pub strength: EditSignalStrength,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditHypothesis {
    pub cue_family: &'static str,
    pub matched_text: String,
    pub match_source: EditHypothesisMatchSource,
    pub kind: EditSignalKind,
    pub scope_hint: EditSignalScope,
    pub replacement_scope: ReplacementScope,
    pub tail_shape: TailShape,
    pub strength: EditSignalStrength,
    pub word_start: usize,
    pub word_end: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditSignalKind {
    Cancel,
    Replace,
    Restatement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditSignalScope {
    Phrase,
    Clause,
    Sentence,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditSignalStrength {
    Possible,
    Strong,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditHypothesisMatchSource {
    Exact,
    Alias,
    NearMiss,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplacementScope {
    Span,
    Clause,
    Sentence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TailShape {
    Empty,
    Phrase,
    Clause,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CorrectionTrigger {
    kind: CorrectionKind,
    signal_kind: EditSignalKind,
    trigger_end: usize,
    min_context_words: usize,
    phrase: &'static str,
    allow_terminal_cancel: bool,
    drop_cue_without_context: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BacktrackOutcome {
    pieces: Vec<Piece>,
    edit_intents: Vec<EditIntent>,
    edit_signals: Vec<EditSignal>,
}

#[derive(Debug, Clone)]
struct ObservedWord {
    text: String,
    normalized: String,
}

#[derive(Debug, Clone, Copy)]
struct CueFamilySpec {
    cue_family: &'static str,
    kind: EditSignalKind,
    scope_hint: EditSignalScope,
    canonical: &'static [&'static str],
    aliases: &'static [&'static [&'static str]],
    session_followup_aliases: &'static [&'static [&'static str]],
}

pub fn clean_transcript(transcript: &Transcript, config: &CleanupConfig) -> String {
    let raw = transcript.raw_text.trim();
    if raw.is_empty()
        || !config.enabled
        || !supports_cleanup_language(transcript.detected_language.as_deref())
    {
        return raw.to_string();
    }

    let mut pieces = tokenize(raw);
    if config.spoken_formatting {
        pieces = apply_spoken_formatting(pieces);
    }
    if config.backtrack {
        pieces = apply_backtrack(pieces, config.profile).pieces;
    }
    if config.remove_fillers {
        pieces = remove_fillers(pieces);
    }

    render_pieces(&pieces)
}

pub fn correction_analysis(transcript: &Transcript) -> CorrectionAnalysis {
    let raw = transcript.raw_text.trim();
    if raw.is_empty() || !supports_cleanup_language(transcript.detected_language.as_deref()) {
        return CorrectionAnalysis {
            text: raw.to_string(),
            aggressive_text: None,
            edit_intents: Vec::new(),
            edit_signals: Vec::new(),
            edit_hypotheses: Vec::new(),
        };
    }

    let mut pieces = tokenize(raw);
    pieces = apply_spoken_formatting(pieces);
    let aggressive_outcome = apply_backtrack(pieces.clone(), CleanupProfile::Aggressive);
    let outcome = apply_backtrack(pieces, CleanupProfile::Basic);
    let text = render_pieces(&outcome.pieces);
    let aggressive_text = render_pieces(&aggressive_outcome.pieces);
    let edit_hypotheses = collect_edit_hypotheses(raw, &outcome.edit_signals);

    CorrectionAnalysis {
        aggressive_text: (!outcome.edit_signals.is_empty()
            && aggressive_text != text
            && !aggressive_text.is_empty())
        .then_some(aggressive_text),
        text,
        edit_intents: outcome.edit_intents,
        edit_signals: outcome.edit_signals,
        edit_hypotheses,
    }
}

pub fn correction_aware_text(transcript: &Transcript) -> String {
    correction_analysis(transcript).text
}

pub fn explicit_followup_replacement(raw: &str) -> Option<String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }

    let pieces = apply_spoken_formatting(tokenize(raw));
    let lookahead = if let Some(trigger) = match_correction_trigger(&pieces, 0) {
        if !matches!(trigger.kind, CorrectionKind::Sentence)
            || !matches!(trigger.signal_kind, EditSignalKind::Cancel)
            || !trigger.drop_cue_without_context
        {
            return None;
        }
        skip_correction_gap(&pieces, trigger.trigger_end)
    } else {
        explicit_followup_cue_lookahead(&pieces)?
    };
    if count_upcoming_words(&pieces, lookahead) == 0 {
        return None;
    }

    let rendered = render_pieces(&pieces[lookahead..]);
    (!rendered.is_empty()).then_some(rendered)
}

fn explicit_followup_cue_lookahead(pieces: &[Piece]) -> Option<usize> {
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

fn cue_family_specs() -> &'static [CueFamilySpec] {
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

fn collect_edit_hypotheses(raw: &str, edit_signals: &[EditSignal]) -> Vec<EditHypothesis> {
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

fn supports_cleanup_language(language: Option<&str>) -> bool {
    matches!(language, Some("en"))
}

fn tokenize(input: &str) -> Vec<Piece> {
    let mut pieces = Vec::new();
    let mut word = String::new();
    let mut newline_streak = 0u8;

    for ch in input.chars() {
        if is_word_char(ch) {
            flush_newlines(&mut pieces, &mut newline_streak);
            word.push(ch);
            continue;
        }

        flush_word(&mut pieces, &mut word);

        if ch == '\n' {
            newline_streak = newline_streak.saturating_add(1);
            continue;
        }

        flush_newlines(&mut pieces, &mut newline_streak);

        if matches!(ch, '.' | ',' | '?' | '!' | ':' | ';') {
            pieces.push(Piece::Punctuation(ch));
        }
    }

    flush_word(&mut pieces, &mut word);
    flush_newlines(&mut pieces, &mut newline_streak);

    pieces
}

fn is_word_char(ch: char) -> bool {
    ch.is_alphanumeric() || matches!(ch, '\'' | '-')
}

fn flush_word(pieces: &mut Vec<Piece>, word: &mut String) {
    if word.is_empty() {
        return;
    }
    pieces.push(Piece::Word(std::mem::take(word)));
}

fn flush_newlines(pieces: &mut Vec<Piece>, streak: &mut u8) {
    if *streak == 0 {
        return;
    }

    let break_kind = if *streak >= 2 {
        BreakKind::Paragraph
    } else {
        BreakKind::Line
    };

    if !matches!(pieces.last(), Some(Piece::Break(_))) {
        pieces.push(Piece::Break(break_kind));
    }
    *streak = 0;
}

fn apply_spoken_formatting(pieces: Vec<Piece>) -> Vec<Piece> {
    let mut out = Vec::with_capacity(pieces.len());
    let mut i = 0;

    while i < pieces.len() {
        if matches_words(&pieces, i, &["new", "paragraph"]) {
            out.push(Piece::Break(BreakKind::Paragraph));
            i += 2;
            continue;
        }
        if matches_words(&pieces, i, &["new", "line"]) {
            out.push(Piece::Break(BreakKind::Line));
            i += 2;
            continue;
        }
        if matches_words(&pieces, i, &["question", "mark"]) {
            out.push(Piece::Punctuation('?'));
            i += 2;
            continue;
        }
        if matches_words(&pieces, i, &["exclamation", "point"])
            || matches_words(&pieces, i, &["exclamation", "mark"])
        {
            out.push(Piece::Punctuation('!'));
            i += 2;
            continue;
        }
        if matches_words(&pieces, i, &["full", "stop"]) {
            out.push(Piece::Punctuation('.'));
            i += 2;
            continue;
        }

        match normalized_word(pieces.get(i)).as_deref() {
            Some("comma") => {
                out.push(Piece::Punctuation(','));
                i += 1;
            }
            Some("period") => {
                out.push(Piece::Punctuation('.'));
                i += 1;
            }
            Some("colon") => {
                out.push(Piece::Punctuation(':'));
                i += 1;
            }
            Some("semicolon") => {
                out.push(Piece::Punctuation(';'));
                i += 1;
            }
            _ => {
                out.push(pieces[i].clone());
                i += 1;
            }
        }
    }

    out
}

fn apply_backtrack(pieces: Vec<Piece>, profile: CleanupProfile) -> BacktrackOutcome {
    let mut out = Vec::with_capacity(pieces.len());
    let mut edit_intents = Vec::new();
    let mut edit_signals = Vec::new();
    let mut i = 0;

    while i < pieces.len() {
        let Some(trigger) = match_correction_trigger(&pieces, i) else {
            out.push(pieces[i].clone());
            i += 1;
            continue;
        };

        let lookahead = skip_correction_gap(&pieces, trigger.trigger_end);
        let replacement_words = count_upcoming_words(&pieces, lookahead);
        let prior_context_words = output_word_count(&out);
        let signal_strength = if prior_context_words >= trigger.min_context_words
            || trigger.drop_cue_without_context
            || (replacement_words == 0 && trigger.allow_terminal_cancel)
        {
            EditSignalStrength::Strong
        } else {
            EditSignalStrength::Possible
        };
        let default_scope = match trigger.kind {
            CorrectionKind::Phrase => EditSignalScope::Phrase,
            CorrectionKind::Sentence => EditSignalScope::Sentence,
        };

        if replacement_words == 0 {
            if prior_context_words >= trigger.min_context_words && trigger.allow_terminal_cancel {
                let action = trim_terminal_cancel_scope(&mut out, trigger.kind);
                edit_signals.push(EditSignal {
                    trigger: trigger.phrase,
                    kind: trigger.signal_kind,
                    scope_hint: scope_hint_for_action(action),
                    strength: signal_strength,
                });
                edit_intents.push(EditIntent {
                    action,
                    trigger: trigger.phrase,
                    confidence: EditIntentConfidence::High,
                });
                i = lookahead;
                continue;
            }
            if trigger.drop_cue_without_context {
                edit_signals.push(EditSignal {
                    trigger: trigger.phrase,
                    kind: trigger.signal_kind,
                    scope_hint: EditSignalScope::Unknown,
                    strength: EditSignalStrength::Strong,
                });
                edit_intents.push(EditIntent {
                    action: EditIntentAction::DropEditCue,
                    trigger: trigger.phrase,
                    confidence: EditIntentConfidence::High,
                });
                i = lookahead;
                continue;
            }
            edit_signals.push(EditSignal {
                trigger: trigger.phrase,
                kind: trigger.signal_kind,
                scope_hint: default_scope,
                strength: signal_strength,
            });
            out.push(pieces[i].clone());
            i += 1;
            continue;
        }

        if prior_context_words < trigger.min_context_words {
            if trigger.drop_cue_without_context {
                edit_signals.push(EditSignal {
                    trigger: trigger.phrase,
                    kind: trigger.signal_kind,
                    scope_hint: EditSignalScope::Unknown,
                    strength: EditSignalStrength::Strong,
                });
                edit_intents.push(EditIntent {
                    action: EditIntentAction::DropEditCue,
                    trigger: trigger.phrase,
                    confidence: EditIntentConfidence::High,
                });
                i = lookahead;
                continue;
            }
            edit_signals.push(EditSignal {
                trigger: trigger.phrase,
                kind: trigger.signal_kind,
                scope_hint: default_scope,
                strength: signal_strength,
            });
            out.push(pieces[i].clone());
            i += 1;
            continue;
        }

        let action = match trigger.kind {
            CorrectionKind::Phrase => {
                trim_recent_phrase(&mut out, profile, replacement_words);
                EditIntentAction::ReplacePreviousPhrase
            }
            CorrectionKind::Sentence => {
                if ends_with_sentence_boundary(&out) {
                    trim_last_sentence(&mut out);
                    EditIntentAction::ReplacePreviousSentence
                } else {
                    trim_recent_phrase(&mut out, profile, replacement_words);
                    EditIntentAction::ReplacePreviousClause
                }
            }
        };
        edit_signals.push(EditSignal {
            trigger: trigger.phrase,
            kind: trigger.signal_kind,
            scope_hint: scope_hint_for_action(action),
            strength: signal_strength,
        });
        edit_intents.push(EditIntent {
            action,
            trigger: trigger.phrase,
            confidence: EditIntentConfidence::High,
        });
        i = lookahead;
    }

    BacktrackOutcome {
        pieces: out,
        edit_intents,
        edit_signals,
    }
}

fn match_correction_trigger(pieces: &[Piece], i: usize) -> Option<CorrectionTrigger> {
    if let Some(end) =
        match_words_with_soft_punctuation(pieces, i, &["oh", "wait", "never", "mind"])
    {
        return Some(cancel_sentence_trigger(
            "oh wait never mind",
            end,
            true,
            true,
        ));
    }
    if let Some(end) = match_words_with_soft_punctuation(pieces, i, &["oh", "wait", "nevermind"]) {
        return Some(cancel_sentence_trigger(
            "oh wait nevermind",
            end,
            true,
            true,
        ));
    }
    if let Some(end) = match_words_with_soft_punctuation(pieces, i, &["forget", "that"]) {
        return Some(cancel_sentence_trigger("forget that", end, true, true));
    }
    if let Some(end) = match_words_with_soft_punctuation(pieces, i, &["wait", "no"]) {
        return Some(replace_sentence_trigger("wait no", end, true, false));
    }
    if let Some(end) = match_words_with_soft_punctuation(pieces, i, &["scratch", "that"]) {
        return Some(cancel_sentence_trigger("scratch that", end, true, true));
    }
    if let Some(end) = match_words_with_soft_punctuation(pieces, i, &["never", "mind"]) {
        return Some(cancel_sentence_trigger("never mind", end, true, true));
    }
    if let Some(end) = match_words_with_soft_punctuation(pieces, i, &["nevermind"]) {
        return Some(cancel_sentence_trigger("nevermind", end, true, true));
    }
    if let Some(end) = match_words_with_soft_punctuation(pieces, i, &["actually", "wait", "no"]) {
        return Some(replace_sentence_trigger(
            "actually wait no",
            end,
            true,
            false,
        ));
    }
    if let Some(end) =
        match_words_with_soft_punctuation(pieces, i, &["actually", "scratch", "that"])
    {
        return Some(cancel_sentence_trigger(
            "actually scratch that",
            end,
            true,
            true,
        ));
    }
    if let Some(end) = match_words_with_soft_punctuation(pieces, i, &["actually", "never", "mind"])
    {
        return Some(cancel_sentence_trigger(
            "actually never mind",
            end,
            true,
            true,
        ));
    }
    if let Some(end) = match_words_with_soft_punctuation(pieces, i, &["actually", "nevermind"]) {
        return Some(cancel_sentence_trigger(
            "actually nevermind",
            end,
            true,
            true,
        ));
    }
    if let Some(end) = match_words_with_soft_punctuation(pieces, i, &["actually", "no"]) {
        return Some(replace_phrase_trigger("actually no", end));
    }
    if let Some(end) = match_words_with_soft_punctuation(pieces, i, &["actually", "i", "meant"]) {
        return Some(replace_phrase_trigger("actually i meant", end));
    }
    if let Some(end) = match_words_with_soft_punctuation(pieces, i, &["actually", "i", "mean"]) {
        return Some(restatement_phrase_trigger("actually i mean", end));
    }
    if let Some(end) = match_words_with_soft_punctuation(pieces, i, &["i", "meant"]) {
        return Some(replace_phrase_trigger("i meant", end));
    }
    if let Some(end) = match_words_with_soft_punctuation(pieces, i, &["i", "mean"]) {
        return Some(restatement_phrase_trigger("i mean", end));
    }
    if let Some(end) = match_words_with_soft_punctuation(pieces, i, &["or", "rather"]) {
        return Some(restatement_phrase_trigger("or rather", end));
    }
    if let Some(end) = match_words_with_soft_punctuation(pieces, i, &["no"]) {
        let previous_word = previous_word_before(pieces, i);
        if matches!(pieces.get(end), Some(Piece::Punctuation(',')))
            && !matches!(previous_word.as_deref(), Some("wait" | "actually"))
        {
            return Some(CorrectionTrigger {
                kind: CorrectionKind::Phrase,
                signal_kind: EditSignalKind::Replace,
                trigger_end: end + 1,
                min_context_words: 2,
                phrase: "no",
                allow_terminal_cancel: false,
                drop_cue_without_context: false,
            });
        }
    }
    None
}

fn cancel_sentence_trigger(
    phrase: &'static str,
    trigger_end: usize,
    allow_terminal_cancel: bool,
    drop_cue_without_context: bool,
) -> CorrectionTrigger {
    CorrectionTrigger {
        kind: CorrectionKind::Sentence,
        signal_kind: EditSignalKind::Cancel,
        trigger_end,
        min_context_words: 1,
        phrase,
        allow_terminal_cancel,
        drop_cue_without_context,
    }
}

fn replace_sentence_trigger(
    phrase: &'static str,
    trigger_end: usize,
    allow_terminal_cancel: bool,
    drop_cue_without_context: bool,
) -> CorrectionTrigger {
    CorrectionTrigger {
        kind: CorrectionKind::Sentence,
        signal_kind: EditSignalKind::Replace,
        trigger_end,
        min_context_words: 1,
        phrase,
        allow_terminal_cancel,
        drop_cue_without_context,
    }
}

fn replace_phrase_trigger(phrase: &'static str, trigger_end: usize) -> CorrectionTrigger {
    CorrectionTrigger {
        kind: CorrectionKind::Phrase,
        signal_kind: EditSignalKind::Replace,
        trigger_end,
        min_context_words: 1,
        phrase,
        allow_terminal_cancel: false,
        drop_cue_without_context: false,
    }
}

fn restatement_phrase_trigger(phrase: &'static str, trigger_end: usize) -> CorrectionTrigger {
    CorrectionTrigger {
        kind: CorrectionKind::Phrase,
        signal_kind: EditSignalKind::Restatement,
        trigger_end,
        min_context_words: 1,
        phrase,
        allow_terminal_cancel: false,
        drop_cue_without_context: false,
    }
}

fn scope_hint_for_action(action: EditIntentAction) -> EditSignalScope {
    match action {
        EditIntentAction::ReplacePreviousPhrase => EditSignalScope::Phrase,
        EditIntentAction::ReplacePreviousClause => EditSignalScope::Clause,
        EditIntentAction::ReplacePreviousSentence => EditSignalScope::Sentence,
        EditIntentAction::DropEditCue => EditSignalScope::Unknown,
    }
}

fn trim_terminal_cancel_scope(out: &mut Vec<Piece>, kind: CorrectionKind) -> EditIntentAction {
    match kind {
        CorrectionKind::Phrase => {
            trim_last_clause(out);
            EditIntentAction::ReplacePreviousClause
        }
        CorrectionKind::Sentence => {
            if ends_with_sentence_boundary(out) {
                trim_last_sentence(out);
                EditIntentAction::ReplacePreviousSentence
            } else {
                trim_last_clause(out);
                EditIntentAction::ReplacePreviousClause
            }
        }
    }
}

fn count_upcoming_words(pieces: &[Piece], start: usize) -> usize {
    let mut count = 0;
    for piece in pieces.iter().skip(start) {
        match piece {
            Piece::Word(_) => count += 1,
            Piece::Break(_) => break,
            Piece::Punctuation(ch) if is_strong_boundary(*ch) => break,
            _ => {}
        }
    }
    count
}

fn trim_recent_phrase(out: &mut Vec<Piece>, profile: CleanupProfile, replacement_words: usize) {
    trim_soft_suffix(out);

    let max_words = match profile {
        CleanupProfile::Basic => replacement_words.clamp(1, 3),
        CleanupProfile::Aggressive => replacement_words.clamp(2, 6),
    };

    let mut removed_words = 0usize;
    while removed_words < max_words {
        match out.pop() {
            Some(Piece::Word(_)) => removed_words += 1,
            Some(Piece::Punctuation(ch)) if !is_strong_boundary(ch) => {}
            Some(piece @ Piece::Punctuation(_)) | Some(piece @ Piece::Break(_)) => {
                out.push(piece);
                break;
            }
            None => break,
        }
    }

    if profile == CleanupProfile::Aggressive {
        while let Some(piece) = out.pop() {
            match piece {
                Piece::Word(_) => continue,
                Piece::Punctuation(ch) if !is_strong_boundary(ch) => continue,
                other => {
                    out.push(other);
                    break;
                }
            }
        }
    }

    trim_soft_suffix(out);
}

fn trim_last_sentence(out: &mut Vec<Piece>) {
    trim_trailing_boundaries(out);

    let mut removed_word = false;
    while let Some(piece) = out.pop() {
        match piece {
            Piece::Word(_) => removed_word = true,
            Piece::Punctuation(ch) if is_strong_boundary(ch) => {
                if removed_word {
                    break;
                }
            }
            Piece::Break(_) => {
                if removed_word {
                    break;
                }
            }
            Piece::Punctuation(_) => {}
        }
    }

    trim_soft_suffix(out);
}

fn trim_last_clause(out: &mut Vec<Piece>) {
    trim_trailing_boundaries(out);

    let mut removed_word = false;
    while let Some(piece) = out.pop() {
        match piece {
            Piece::Word(_) => removed_word = true,
            Piece::Punctuation(ch) if is_clause_boundary(ch) => {
                if removed_word {
                    break;
                }
            }
            Piece::Break(_) => {
                if removed_word {
                    break;
                }
            }
            Piece::Punctuation(_) => {}
        }
    }

    trim_soft_suffix(out);
}

fn trim_soft_suffix(out: &mut Vec<Piece>) {
    while let Some(last) = out.last() {
        match last {
            Piece::Punctuation(ch) if !is_strong_boundary(*ch) => {
                out.pop();
            }
            _ => break,
        }
    }
}

fn output_word_count(pieces: &[Piece]) -> usize {
    pieces
        .iter()
        .filter(|piece| matches!(piece, Piece::Word(_)))
        .count()
}

fn skip_soft_punctuation(pieces: &[Piece], mut index: usize) -> usize {
    while matches!(pieces.get(index), Some(Piece::Punctuation(',' | ':' | ';'))) {
        index += 1;
    }
    index
}

fn skip_correction_gap(pieces: &[Piece], mut index: usize) -> usize {
    loop {
        match pieces.get(index) {
            Some(Piece::Punctuation(_)) | Some(Piece::Break(_)) => index += 1,
            _ => return index,
        }
    }
}

fn trim_trailing_boundaries(out: &mut Vec<Piece>) {
    while let Some(last) = out.last() {
        match last {
            Piece::Punctuation(_) | Piece::Break(_) => {
                out.pop();
            }
            Piece::Word(_) => break,
        }
    }
}

fn ends_with_sentence_boundary(out: &[Piece]) -> bool {
    out.iter().rev().find_map(|piece| match piece {
        Piece::Punctuation(ch) => Some(is_strong_boundary(*ch)),
        Piece::Break(_) => Some(true),
        Piece::Word(_) => None,
    }) == Some(true)
}

fn remove_fillers(pieces: Vec<Piece>) -> Vec<Piece> {
    let mut out = Vec::with_capacity(pieces.len());
    for piece in pieces {
        match piece {
            Piece::Word(word) if is_filler(&word) => continue,
            other => out.push(other),
        }
    }
    out
}

fn is_filler(word: &str) -> bool {
    matches!(
        normalized_word_str(word).as_str(),
        "um" | "umm" | "uh" | "uhh" | "er" | "erm" | "ah"
    )
}

fn render_pieces(pieces: &[Piece]) -> String {
    let mut rendered = String::new();
    let mut capitalize_next = true;

    for piece in pieces {
        match piece {
            Piece::Word(word) => {
                if !rendered.is_empty() && !rendered.ends_with([' ', '\n']) {
                    rendered.push(' ');
                }
                if capitalize_next {
                    rendered.push_str(&capitalize_first(word));
                } else {
                    rendered.push_str(word);
                }
                capitalize_next = false;
            }
            Piece::Punctuation(ch) => {
                trim_trailing_spaces(&mut rendered);
                rendered.push(*ch);
                capitalize_next = matches!(ch, '.' | '?' | '!');
            }
            Piece::Break(BreakKind::Line) => {
                trim_trailing_spaces(&mut rendered);
                if !rendered.is_empty() {
                    if !rendered.ends_with('\n') {
                        rendered.push('\n');
                    }
                    capitalize_next = true;
                }
            }
            Piece::Break(BreakKind::Paragraph) => {
                trim_trailing_spaces(&mut rendered);
                if !rendered.is_empty() {
                    while rendered.ends_with('\n') {
                        rendered.pop();
                    }
                    rendered.push('\n');
                    rendered.push('\n');
                    capitalize_next = true;
                }
            }
        }
    }

    trim_trailing_spaces(&mut rendered);
    rendered
}

fn trim_trailing_spaces(text: &mut String) {
    while text.ends_with(' ') {
        text.pop();
    }
}

fn capitalize_first(word: &str) -> String {
    let mut chars = word.chars();
    let Some(first) = chars.next() else {
        return String::new();
    };

    let mut result = String::new();
    result.extend(first.to_uppercase());
    result.extend(chars);
    result
}

fn matches_words(pieces: &[Piece], start: usize, words: &[&str]) -> bool {
    words.iter().enumerate().all(|(offset, expected)| {
        normalized_word(pieces.get(start + offset)).as_deref() == Some(*expected)
    })
}

fn match_words_with_soft_punctuation(
    pieces: &[Piece],
    start: usize,
    words: &[&str],
) -> Option<usize> {
    let mut index = start;
    for (offset, expected) in words.iter().enumerate() {
        if offset > 0 {
            index = skip_soft_punctuation(pieces, index);
        }

        if normalized_word(pieces.get(index)).as_deref() != Some(*expected) {
            return None;
        }
        index += 1;
    }

    Some(index)
}

fn normalized_word(piece: Option<&Piece>) -> Option<String> {
    match piece {
        Some(Piece::Word(word)) => Some(normalized_word_str(word)),
        _ => None,
    }
}

fn previous_word_before(pieces: &[Piece], mut index: usize) -> Option<String> {
    while index > 0 {
        index -= 1;
        if let Some(word) = normalized_word(pieces.get(index)) {
            return Some(word);
        }
    }

    None
}

fn normalized_word_str(word: &str) -> String {
    word.trim_matches(|ch: char| !is_word_char(ch))
        .trim_matches('\'')
        .trim_matches('-')
        .to_ascii_lowercase()
}

fn is_strong_boundary(ch: char) -> bool {
    matches!(ch, '.' | '?' | '!')
}

fn is_clause_boundary(ch: char) -> bool {
    is_strong_boundary(ch) || matches!(ch, ',' | ':' | ';')
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CleanupConfig, CleanupProfile};
    use crate::transcribe::Transcript;

    fn transcript(text: &str) -> Transcript {
        Transcript {
            raw_text: text.to_string(),
            detected_language: Some("en".to_string()),
            segments: Vec::new(),
        }
    }

    #[test]
    fn removes_common_fillers() {
        let cleaned = clean_transcript(
            &transcript("um i think we should go"),
            &CleanupConfig::default(),
        );
        assert_eq!(cleaned, "I think we should go");
    }

    #[test]
    fn preserves_non_filler_words() {
        let cleaned = clean_transcript(&transcript("i like apples"), &CleanupConfig::default());
        assert_eq!(cleaned, "I like apples");
    }

    #[test]
    fn converts_spoken_punctuation_commands() {
        let cleaned = clean_transcript(
            &transcript("hello comma world question mark"),
            &CleanupConfig::default(),
        );
        assert_eq!(cleaned, "Hello, world?");
    }

    #[test]
    fn converts_spoken_line_and_paragraph_commands() {
        let cleaned = clean_transcript(
            &transcript("first line new line second line new paragraph third line"),
            &CleanupConfig::default(),
        );
        assert_eq!(cleaned, "First line\nSecond line\n\nThird line");
    }

    #[test]
    fn basic_backtrack_replaces_recent_phrase() {
        let cleaned = clean_transcript(
            &transcript("let's meet at 4 actually no 3"),
            &CleanupConfig::default(),
        );
        assert_eq!(cleaned, "Let's meet at 3");
    }

    #[test]
    fn standalone_actually_is_preserved() {
        let cleaned = correction_aware_text(&transcript("it actually works"));
        assert_eq!(cleaned, "It actually works");
    }

    #[test]
    fn rather_is_preserved_in_normal_phrasing() {
        let cleaned = correction_aware_text(&transcript("i would rather stay home"));
        assert_eq!(cleaned, "I would rather stay home");
    }

    #[test]
    fn actually_rather_is_preserved_in_normal_phrasing() {
        let cleaned = correction_aware_text(&transcript("i would actually rather stay home"));
        assert_eq!(cleaned, "I would actually rather stay home");
    }

    #[test]
    fn punctuated_wait_no_replaces_last_sentence() {
        let cleaned = clean_transcript(
            &transcript("hi there, this is a test of whisper osd. wait, no. hi there."),
            &CleanupConfig::default(),
        );
        assert_eq!(cleaned, "Hi there.");
    }

    #[test]
    fn punctuated_wait_no_still_replaces_inline_phrase() {
        let cleaned = clean_transcript(
            &transcript("let's meet at 4 wait, no, 3"),
            &CleanupConfig::default(),
        );
        assert_eq!(cleaned, "Let's meet at 3");
    }

    #[test]
    fn scratch_that_replaces_recent_word() {
        let cleaned = clean_transcript(
            &transcript("i'll bring cookies scratch that brownies"),
            &CleanupConfig::default(),
        );
        assert_eq!(cleaned, "I'll bring brownies");
    }

    #[test]
    fn correction_aware_text_drops_previous_sentence_for_scratch_that() {
        let cleaned = correction_aware_text(&transcript(
            "hello there, this is a test of whisper rs. scratch that. hi.",
        ));
        assert_eq!(cleaned, "Hi.");
    }

    #[test]
    fn terminal_never_mind_cancels_previous_clause() {
        let cleaned = correction_aware_text(&transcript("hello there oh wait never mind"));
        assert_eq!(cleaned, "");
    }

    #[test]
    fn utterance_initial_never_mind_is_dropped_when_content_follows() {
        let cleaned = correction_aware_text(&transcript("never mind. hi, how are you today?"));
        assert_eq!(cleaned, "Hi, how are you today?");
    }

    #[test]
    fn explicit_followup_replacement_handles_session_aliases() {
        assert_eq!(
            explicit_followup_replacement("srajvat, hi").as_deref(),
            Some("Hi")
        );
        assert_eq!(
            explicit_followup_replacement("scratchfarts, hi").as_deref(),
            Some("Hi")
        );
    }

    #[test]
    fn correction_analysis_reports_terminal_cancel_intent() {
        let analysis = correction_analysis(&transcript("hello there never mind"));
        assert_eq!(analysis.text, "");
        assert_eq!(analysis.edit_intents.len(), 1);
        assert_eq!(
            analysis.edit_intents[0].action,
            EditIntentAction::ReplacePreviousClause
        );
        assert_eq!(analysis.edit_intents[0].trigger, "never mind");
        assert_eq!(analysis.edit_signals.len(), 1);
        assert_eq!(analysis.edit_signals[0].kind, EditSignalKind::Cancel);
        assert_eq!(analysis.edit_signals[0].scope_hint, EditSignalScope::Clause);
        assert_eq!(
            analysis.edit_signals[0].strength,
            EditSignalStrength::Strong
        );
        assert_eq!(analysis.edit_hypotheses.len(), 1);
        assert_eq!(analysis.edit_hypotheses[0].cue_family, "never_mind");
        assert_eq!(
            analysis.edit_hypotheses[0].match_source,
            EditHypothesisMatchSource::Exact
        );
    }

    #[test]
    fn utterance_initial_wait_no_is_not_treated_as_backtrack() {
        let analysis = correction_analysis(&transcript("wait, no, it actually works"));
        assert_eq!(analysis.text, "Wait, no, it actually works");
        assert_eq!(analysis.edit_intents.len(), 0);
        assert_eq!(analysis.edit_signals.len(), 1);
        assert_eq!(analysis.edit_signals[0].kind, EditSignalKind::Replace);
        assert_eq!(
            analysis.edit_signals[0].scope_hint,
            EditSignalScope::Sentence
        );
        assert_eq!(
            analysis.edit_signals[0].strength,
            EditSignalStrength::Possible
        );
        assert!(analysis.edit_hypotheses.iter().any(|hypothesis| {
            hypothesis.cue_family == "wait_no"
                && hypothesis.match_source == EditHypothesisMatchSource::Exact
        }));
    }

    #[test]
    fn correction_analysis_reports_restatement_signal_for_or_rather() {
        let analysis = correction_analysis(&transcript("let's meet tomorrow or rather friday"));
        assert_eq!(analysis.text, "Let's meet friday");
        assert_eq!(analysis.edit_signals.len(), 1);
        assert_eq!(analysis.edit_signals[0].kind, EditSignalKind::Restatement);
        assert_eq!(analysis.edit_signals[0].scope_hint, EditSignalScope::Phrase);
    }

    #[test]
    fn correction_analysis_exposes_aggressive_candidate_for_ambiguous_replacement() {
        let analysis = correction_analysis(&transcript(
            "my name is notes, scratch that my name is jonatan",
        ));
        assert_eq!(analysis.text, "My my name is jonatan");
        assert_eq!(
            analysis.aggressive_text.as_deref(),
            Some("My name is jonatan")
        );
        assert_eq!(
            analysis.edit_hypotheses[0].replacement_scope,
            ReplacementScope::Clause
        );
        assert_eq!(analysis.edit_hypotheses[0].tail_shape, TailShape::Clause);
    }

    #[test]
    fn correction_analysis_collects_near_miss_hypothesis_for_scratch_that_family() {
        let analysis = correction_analysis(&transcript("hello there scratch vat hi"));
        assert!(analysis.edit_hypotheses.iter().any(|hypothesis| {
            hypothesis.cue_family == "scratch_that"
                && hypothesis.match_source == EditHypothesisMatchSource::NearMiss
                && hypothesis.matched_text == "scratch vat"
        }));
    }

    #[test]
    fn correction_analysis_marks_phrase_tail_as_span_scope() {
        let analysis = correction_analysis(&transcript(
            "mobile apps or sms codes scratch that just sms codes",
        ));
        assert!(analysis.edit_hypotheses.iter().any(|hypothesis| {
            hypothesis.cue_family == "scratch_that"
                && hypothesis.replacement_scope == ReplacementScope::Span
                && hypothesis.tail_shape == TailShape::Phrase
        }));
    }

    #[test]
    fn aggressive_profile_trims_more_context() {
        let config = CleanupConfig {
            profile: CleanupProfile::Aggressive,
            ..CleanupConfig::default()
        };
        let cleaned =
            clean_transcript(&transcript("alpha beta gamma delta wait no omega"), &config);
        assert_eq!(cleaned, "Omega");
    }

    #[test]
    fn skips_advanced_cleanup_for_non_english_transcripts() {
        let transcript = Transcript {
            raw_text: "um hola comma mundo".to_string(),
            detected_language: Some("es".to_string()),
            segments: Vec::new(),
        };
        let cleaned = clean_transcript(&transcript, &CleanupConfig::default());
        assert_eq!(cleaned, "um hola comma mundo");
    }
}

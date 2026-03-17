use crate::config::{CleanupConfig, CleanupProfile};
use crate::transcribe::Transcript;

mod analysis;
mod lexicon;
mod render;
#[cfg(test)]
mod tests;

pub use analysis::{
    clean_transcript, correction_analysis, correction_aware_text, explicit_followup_replacement,
};

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

fn matches_words(pieces: &[Piece], start: usize, words: &[&str]) -> bool {
    words.iter().enumerate().all(|(offset, expected)| {
        normalized_word(pieces.get(start + offset)).as_deref() == Some(*expected)
    })
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

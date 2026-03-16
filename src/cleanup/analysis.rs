use super::lexicon::{collect_edit_hypotheses, explicit_followup_cue_lookahead};
use super::render::render_pieces;
use super::{
    BacktrackOutcome, CleanupConfig, CleanupProfile, CorrectionAnalysis, CorrectionKind,
    CorrectionTrigger, EditIntent, EditIntentAction, EditIntentConfidence, EditSignal,
    EditSignalKind, EditSignalScope, EditSignalStrength, Piece, Transcript, is_clause_boundary,
    is_strong_boundary, match_words_with_soft_punctuation, matches_words, previous_word_before,
    skip_correction_gap, supports_cleanup_language,
};

pub fn clean_transcript(transcript: &Transcript, config: &CleanupConfig) -> String {
    let raw = transcript.raw_text.trim();
    if raw.is_empty()
        || !config.enabled
        || !supports_cleanup_language(transcript.detected_language.as_deref())
    {
        return raw.to_string();
    }

    let mut pieces = super::tokenize(raw);
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

    let mut pieces = super::tokenize(raw);
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

    let pieces = apply_spoken_formatting(super::tokenize(raw));
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

fn apply_spoken_formatting(pieces: Vec<Piece>) -> Vec<Piece> {
    let mut out = Vec::with_capacity(pieces.len());
    let mut i = 0;

    while i < pieces.len() {
        if matches_words(&pieces, i, &["new", "paragraph"]) {
            out.push(Piece::Break(super::BreakKind::Paragraph));
            i += 2;
            continue;
        }
        if matches_words(&pieces, i, &["new", "line"]) {
            out.push(Piece::Break(super::BreakKind::Line));
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

        match super::normalized_word(pieces.get(i)).as_deref() {
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
        super::normalized_word_str(word).as_str(),
        "um" | "umm" | "uh" | "uhh" | "er" | "erm" | "ah"
    )
}

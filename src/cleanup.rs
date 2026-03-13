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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CorrectionTrigger {
    kind: CorrectionKind,
    trigger_end: usize,
    min_context_words: usize,
}

pub fn correction_aware_text(transcript: &Transcript) -> String {
    let raw = transcript.raw_text.trim();
    if raw.is_empty() || !supports_cleanup_language(transcript.detected_language.as_deref()) {
        return raw.to_string();
    }

    let mut pieces = tokenize(raw);
    pieces = apply_spoken_formatting(pieces);
    pieces = apply_backtrack(pieces);

    render_pieces(&pieces)
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

fn apply_backtrack(pieces: Vec<Piece>) -> Vec<Piece> {
    let mut out = Vec::with_capacity(pieces.len());
    let mut i = 0;

    while i < pieces.len() {
        let Some(trigger) = match_correction_trigger(&pieces, i) else {
            out.push(pieces[i].clone());
            i += 1;
            continue;
        };

        let lookahead = skip_correction_gap(&pieces, trigger.trigger_end);
        let replacement_words = count_upcoming_words(&pieces, lookahead);
        if replacement_words == 0 || output_word_count(&out) < trigger.min_context_words {
            out.push(pieces[i].clone());
            i += 1;
            continue;
        }

        match trigger.kind {
            CorrectionKind::Phrase => trim_recent_phrase(&mut out, replacement_words),
            CorrectionKind::Sentence => {
                if ends_with_sentence_boundary(&out) {
                    trim_last_sentence(&mut out);
                } else {
                    trim_recent_phrase(&mut out, replacement_words);
                }
            }
        }
        i = lookahead;
    }

    out
}

fn match_correction_trigger(pieces: &[Piece], i: usize) -> Option<CorrectionTrigger> {
    if let Some(end) = match_words_with_soft_punctuation(pieces, i, &["wait", "no"]) {
        return Some(CorrectionTrigger {
            kind: CorrectionKind::Sentence,
            trigger_end: end,
            min_context_words: 1,
        });
    }
    if let Some(end) = match_words_with_soft_punctuation(pieces, i, &["scratch", "that"]) {
        return Some(CorrectionTrigger {
            kind: CorrectionKind::Sentence,
            trigger_end: end,
            min_context_words: 1,
        });
    }
    if let Some(end) = match_words_with_soft_punctuation(pieces, i, &["never", "mind"]) {
        return Some(CorrectionTrigger {
            kind: CorrectionKind::Sentence,
            trigger_end: end,
            min_context_words: 1,
        });
    }
    if let Some(end) = match_words_with_soft_punctuation(pieces, i, &["nevermind"]) {
        return Some(CorrectionTrigger {
            kind: CorrectionKind::Sentence,
            trigger_end: end,
            min_context_words: 1,
        });
    }
    if let Some(end) = match_words_with_soft_punctuation(pieces, i, &["actually", "wait", "no"]) {
        return Some(CorrectionTrigger {
            kind: CorrectionKind::Sentence,
            trigger_end: end,
            min_context_words: 1,
        });
    }
    if let Some(end) =
        match_words_with_soft_punctuation(pieces, i, &["actually", "scratch", "that"])
    {
        return Some(CorrectionTrigger {
            kind: CorrectionKind::Sentence,
            trigger_end: end,
            min_context_words: 1,
        });
    }
    if let Some(end) = match_words_with_soft_punctuation(pieces, i, &["actually", "never", "mind"])
    {
        return Some(CorrectionTrigger {
            kind: CorrectionKind::Sentence,
            trigger_end: end,
            min_context_words: 1,
        });
    }
    if let Some(end) = match_words_with_soft_punctuation(pieces, i, &["actually", "nevermind"]) {
        return Some(CorrectionTrigger {
            kind: CorrectionKind::Sentence,
            trigger_end: end,
            min_context_words: 1,
        });
    }
    if let Some(end) = match_words_with_soft_punctuation(pieces, i, &["actually", "no"]) {
        return Some(CorrectionTrigger {
            kind: CorrectionKind::Phrase,
            trigger_end: end,
            min_context_words: 1,
        });
    }
    if let Some(end) = match_words_with_soft_punctuation(pieces, i, &["actually", "i", "meant"]) {
        return Some(CorrectionTrigger {
            kind: CorrectionKind::Phrase,
            trigger_end: end,
            min_context_words: 1,
        });
    }
    if let Some(end) = match_words_with_soft_punctuation(pieces, i, &["actually", "i", "mean"]) {
        return Some(CorrectionTrigger {
            kind: CorrectionKind::Phrase,
            trigger_end: end,
            min_context_words: 1,
        });
    }
    if let Some(end) = match_words_with_soft_punctuation(pieces, i, &["i", "meant"]) {
        return Some(CorrectionTrigger {
            kind: CorrectionKind::Phrase,
            trigger_end: end,
            min_context_words: 1,
        });
    }
    if let Some(end) = match_words_with_soft_punctuation(pieces, i, &["i", "mean"]) {
        return Some(CorrectionTrigger {
            kind: CorrectionKind::Phrase,
            trigger_end: end,
            min_context_words: 1,
        });
    }
    if let Some(end) = match_words_with_soft_punctuation(pieces, i, &["no"]) {
        if matches!(pieces.get(end), Some(Piece::Punctuation(','))) {
            return Some(CorrectionTrigger {
                kind: CorrectionKind::Phrase,
                trigger_end: end + 1,
                min_context_words: 2,
            });
        }
    }
    None
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

fn trim_recent_phrase(out: &mut Vec<Piece>, replacement_words: usize) {
    trim_soft_suffix(out);

    let max_words = replacement_words.clamp(1, 3);

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

fn normalized_word_str(word: &str) -> String {
    word.trim_matches(|ch: char| !is_word_char(ch))
        .trim_matches('\'')
        .trim_matches('-')
        .to_ascii_lowercase()
}

fn is_strong_boundary(ch: char) -> bool {
    matches!(ch, '.' | '?' | '!')
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcribe::Transcript;

    fn transcript(text: &str) -> Transcript {
        Transcript {
            raw_text: text.to_string(),
            detected_language: Some("en".to_string()),
            segments: Vec::new(),
        }
    }

    #[test]
    fn preserves_non_filler_words() {
        let cleaned = correction_aware_text(&transcript("i like apples"));
        assert_eq!(cleaned, "I like apples");
    }

    #[test]
    fn converts_spoken_punctuation_commands() {
        let cleaned = correction_aware_text(&transcript("hello comma world question mark"));
        assert_eq!(cleaned, "Hello, world?");
    }

    #[test]
    fn converts_spoken_line_and_paragraph_commands() {
        let cleaned = correction_aware_text(&transcript(
            "first line new line second line new paragraph third line",
        ));
        assert_eq!(cleaned, "First line\nSecond line\n\nThird line");
    }

    #[test]
    fn basic_backtrack_replaces_recent_phrase() {
        let cleaned = correction_aware_text(&transcript("let's meet at 4 actually no 3"));
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
        let cleaned = correction_aware_text(&transcript(
            "hi there, this is a test of whispers osd. wait, no. hi there.",
        ));
        assert_eq!(cleaned, "Hi there.");
    }

    #[test]
    fn punctuated_wait_no_still_replaces_inline_phrase() {
        let cleaned = correction_aware_text(&transcript("let's meet at 4 wait, no, 3"));
        assert_eq!(cleaned, "Let's meet at 3");
    }

    #[test]
    fn scratch_that_replaces_recent_word() {
        let cleaned =
            correction_aware_text(&transcript("i'll bring cookies scratch that brownies"));
        assert_eq!(cleaned, "I'll bring brownies");
    }

    #[test]
    fn correction_aware_text_drops_previous_sentence_for_scratch_that() {
        let cleaned = correction_aware_text(&transcript(
            "hello there, this is a test of whispers. scratch that. hi.",
        ));
        assert_eq!(cleaned, "Hi.");
    }

    #[test]
    fn utterance_initial_wait_no_is_not_treated_as_backtrack() {
        let cleaned = correction_aware_text(&transcript("wait, no, it actually works"));
        assert_eq!(cleaned, "Wait, no, it actually works");
    }

    #[test]
    fn skips_cleanup_for_non_english_transcripts() {
        let transcript = Transcript {
            raw_text: "um hola comma mundo".to_string(),
            detected_language: Some("es".to_string()),
            segments: Vec::new(),
        };
        let cleaned = correction_aware_text(&transcript);
        assert_eq!(cleaned, "um hola comma mundo");
    }
}

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
    let cleaned = clean_transcript(&transcript("alpha beta gamma delta wait no omega"), &config);
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

use super::RewritePrompt;
use super::local::{build_oaicompat_messages_json, effective_max_tokens};
use super::prompt::{build_system_instructions, build_user_message, rewrite_instructions};
use super::routing::{RewriteRoute, rewrite_route};
use crate::rewrite_profile::ResolvedRewriteProfile;
use crate::rewrite_protocol::{
    RewriteCandidate, RewriteCandidateKind, RewriteCorrectionPolicy, RewriteEditAction,
    RewriteEditHypothesis, RewriteEditHypothesisMatchSource, RewriteEditIntent, RewriteEditSignal,
    RewriteEditSignalKind, RewriteEditSignalScope, RewriteEditSignalStrength,
    RewriteIntentConfidence, RewritePolicyContext, RewriteReplacementScope,
    RewriteSessionBacktrackCandidate, RewriteSessionBacktrackCandidateKind, RewriteSessionEntry,
    RewriteSurfaceKind, RewriteTailShape, RewriteTranscript, RewriteTranscriptSegment,
    RewriteTypingContext,
};

fn correction_transcript() -> RewriteTranscript {
    RewriteTranscript {
        raw_text: "Hi there, this is a test. Wait, no. Hi there.".into(),
        correction_aware_text: "Hi there.".into(),
        aggressive_correction_text: None,
        detected_language: Some("en".into()),
        typing_context: None,
        recent_session_entries: Vec::new(),
        session_backtrack_candidates: Vec::new(),
        recommended_session_candidate: None,
        segments: vec![
            RewriteTranscriptSegment {
                text: "Hi there, this is a test.".into(),
                start_ms: 0,
                end_ms: 1200,
            },
            RewriteTranscriptSegment {
                text: "Wait, no. Hi there.".into(),
                start_ms: 1200,
                end_ms: 2200,
            },
        ],
        edit_intents: vec![RewriteEditIntent {
            action: RewriteEditAction::ReplacePreviousSentence,
            trigger: "wait no".into(),
            confidence: RewriteIntentConfidence::High,
        }],
        edit_signals: vec![RewriteEditSignal {
            trigger: "wait no".into(),
            kind: RewriteEditSignalKind::Replace,
            scope_hint: RewriteEditSignalScope::Sentence,
            strength: RewriteEditSignalStrength::Strong,
        }],
        edit_hypotheses: vec![RewriteEditHypothesis {
            cue_family: "wait_no".into(),
            matched_text: "wait no".into(),
            match_source: RewriteEditHypothesisMatchSource::Exact,
            kind: RewriteEditSignalKind::Replace,
            scope_hint: RewriteEditSignalScope::Sentence,
            replacement_scope: RewriteReplacementScope::Sentence,
            tail_shape: RewriteTailShape::Phrase,
            strength: RewriteEditSignalStrength::Strong,
        }],
        rewrite_candidates: vec![
            RewriteCandidate {
                kind: RewriteCandidateKind::Literal,
                text: "Hi there, this is a test. Wait, no. Hi there.".into(),
            },
            RewriteCandidate {
                kind: RewriteCandidateKind::ConservativeCorrection,
                text: "Hi there.".into(),
            },
        ],
        recommended_candidate: Some(RewriteCandidate {
            kind: RewriteCandidateKind::Literal,
            text: "Hi there, this is a test. Wait, no. Hi there.".into(),
        }),
        policy_context: RewritePolicyContext::default(),
    }
}

fn candidate_only_transcript() -> RewriteTranscript {
    RewriteTranscript {
        raw_text: "Hi there, this is a test. Scratch that. Hi there.".into(),
        correction_aware_text: "Hi there.".into(),
        aggressive_correction_text: None,
        detected_language: Some("en".into()),
        typing_context: None,
        recent_session_entries: Vec::new(),
        session_backtrack_candidates: Vec::new(),
        recommended_session_candidate: None,
        segments: Vec::new(),
        edit_intents: vec![RewriteEditIntent {
            action: RewriteEditAction::ReplacePreviousSentence,
            trigger: "scratch that".into(),
            confidence: RewriteIntentConfidence::High,
        }],
        edit_signals: Vec::new(),
        edit_hypotheses: Vec::new(),
        rewrite_candidates: vec![
            RewriteCandidate {
                kind: RewriteCandidateKind::Literal,
                text: "Hi there, this is a test. Scratch that. Hi there.".into(),
            },
            RewriteCandidate {
                kind: RewriteCandidateKind::ConservativeCorrection,
                text: "Hi there.".into(),
            },
        ],
        recommended_candidate: None,
        policy_context: RewritePolicyContext::default(),
    }
}

fn fast_agentic_transcript() -> RewriteTranscript {
    RewriteTranscript {
        raw_text: "I'm currently using the window manager hyperland.".into(),
        correction_aware_text: "I'm currently using the window manager hyperland.".into(),
        aggressive_correction_text: None,
        detected_language: Some("en".into()),
        typing_context: Some(RewriteTypingContext {
            focus_fingerprint: "focus".into(),
            app_id: Some("browser".into()),
            window_title: Some("Matrix".into()),
            surface_kind: RewriteSurfaceKind::GenericText,
            browser_domain: None,
            captured_at_ms: 42,
        }),
        recent_session_entries: Vec::new(),
        session_backtrack_candidates: Vec::new(),
        recommended_session_candidate: None,
        segments: Vec::new(),
        edit_intents: Vec::new(),
        edit_signals: Vec::new(),
        edit_hypotheses: Vec::new(),
        rewrite_candidates: vec![
            RewriteCandidate {
                kind: RewriteCandidateKind::Literal,
                text: "I'm currently using the window manager hyperland.".into(),
            },
            RewriteCandidate {
                kind: RewriteCandidateKind::ConservativeCorrection,
                text: "I'm currently using the window manager hyperland.".into(),
            },
        ],
        recommended_candidate: None,
        policy_context: RewritePolicyContext {
            correction_policy: RewriteCorrectionPolicy::Balanced,
            matched_rule_names: vec!["baseline/global-default".into()],
            effective_rule_instructions: vec![
                "Use category cues like window manager to disambiguate nearby technical names."
                    .into(),
            ],
            active_glossary_terms: Vec::new(),
            glossary_candidates: Vec::new(),
        },
    }
}

#[test]
fn instructions_cover_self_correction_examples() {
    let instructions = rewrite_instructions(ResolvedRewriteProfile::LlamaCompat);
    assert!(instructions.contains("Return only the finished text"));
    assert!(instructions.contains("Never reintroduce text"));
    assert!(instructions.contains("scratch that, brownies"));
    assert!(instructions.contains("window manager Hyperland"));
    assert!(instructions.contains("switching from Sui to Hyperland"));
}

#[test]
fn qwen_instructions_forbid_reasoning_tags() {
    let instructions = rewrite_instructions(ResolvedRewriteProfile::Qwen);
    assert!(instructions.contains("Do not emit reasoning"));
    assert!(instructions.contains("phonetically similar common word"));
}

#[test]
fn base_instructions_allow_technical_term_inference() {
    let instructions = rewrite_instructions(ResolvedRewriteProfile::LlamaCompat);
    assert!(instructions.contains("technical concepts"));
    assert!(instructions.contains("phonetically similar common word"));
}

#[test]
fn custom_instructions_append_to_system_prompt() {
    let instructions = build_system_instructions(
        &correction_transcript(),
        ResolvedRewriteProfile::Qwen,
        Some("Keep product names exact."),
    );
    assert!(instructions.contains("Return only the finished text"));
    assert!(instructions.contains("Keep product names exact."));
}

#[test]
fn oaicompat_messages_json_contains_system_and_user_roles() {
    let prompt = RewritePrompt {
        system: "system instructions".into(),
        user: "user input".into(),
    };

    let messages_json = build_oaicompat_messages_json(&prompt).expect("encode oaicompat messages");
    let messages: serde_json::Value =
        serde_json::from_str(&messages_json).expect("parse oaicompat messages");
    let messages = messages.as_array().expect("messages array");

    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[0]["content"], "system instructions");
    assert_eq!(messages[1]["role"], "user");
    assert_eq!(messages[1]["content"], "user input");
}

#[test]
fn agentic_system_prompt_relaxes_candidate_restrictions() {
    let instructions = build_system_instructions(
        &fast_agentic_transcript(),
        ResolvedRewriteProfile::Qwen,
        None,
    );
    assert!(instructions.contains("Agentic latitude contract"));
    assert!(instructions.contains(
        "do not keep an obviously wrong technical spelling just because it appears in the candidate list"
    ));
    assert!(instructions.contains(
        "even when the literal transcript spelling is noisy or the exact canonical form is not already present in the candidate list"
    ));
}

#[test]
fn fast_route_prompt_allows_agentic_technical_normalization() {
    let transcript = fast_agentic_transcript();
    assert!(matches!(rewrite_route(&transcript), RewriteRoute::Fast));
    let prompt = build_user_message(&transcript);
    assert!(prompt.contains(
        "you may normalize likely technical terms or proper names when category cues in the utterance make the intended technical meaning clearly better than the literal transcript"
    ));
    assert!(
        prompt.contains("Available rewrite candidates (advisory, not exhaustive in agentic mode)")
    );
}

#[test]
fn cue_prompt_includes_raw_candidate_and_signals() {
    let prompt = build_user_message(&correction_transcript());
    assert!(matches!(
        rewrite_route(&correction_transcript()),
        RewriteRoute::CandidateAdjudication
    ));
    assert!(prompt.contains("Structured edit hypotheses"));
    assert!(prompt.contains("cue_family: wait_no"));
    assert!(prompt.contains("replacement_scope: sentence"));
    assert!(prompt.contains("tail_shape: phrase"));
    assert!(prompt.contains("Candidate interpretations"));
    assert!(prompt.contains("A strong explicit spoken edit cue was detected"));
    assert!(
        prompt.contains(
            "The candidate list is ordered from most likely to least likely by heuristics."
        )
    );
    assert!(prompt.contains("the first candidate is the heuristic best guess"));
    assert!(prompt.contains("Recommended interpretation:"));
    assert!(prompt.contains(
        "Use this as the default final text unless another candidate is clearly better."
    ));
    assert!(
        prompt.contains("Prefer the smallest replacement scope that yields a coherent result.")
    );
    assert!(prompt.contains("- preferred_candidate"));
    assert!(prompt.contains(
        "- preferred_candidate literal (keep only if the cue was not actually an edit): Hi there, this is a test. Wait, no. Hi there."
    ));
    assert!(prompt.contains("Structured edit signals"));
    assert!(prompt.contains("trigger: \"wait no\""));
    assert!(prompt.contains("Structured edit intents"));
    assert!(prompt.contains("replace_previous_sentence"));
    assert!(prompt.contains("Choose the best candidate interpretation"));
    assert!(prompt.contains("Candidate interpretations:\n"));
    assert!(prompt.contains("Correction candidate:\nHi there."));
    assert!(prompt.contains("Raw transcript:\nHi there, this is a test. Wait, no. Hi there."));
    assert!(prompt.contains("Recent segments"));
}

#[test]
fn cue_prompt_includes_aggressive_candidate_when_available() {
    let mut transcript = correction_transcript();
    transcript.aggressive_correction_text = Some("Hi there.".into());

    let prompt = build_user_message(&transcript);
    assert!(prompt.contains("Aggressive correction candidate"));
}

#[test]
fn user_message_prefers_correction_candidate_without_signals() {
    let prompt = build_user_message(&candidate_only_transcript());
    assert!(matches!(
        rewrite_route(&candidate_only_transcript()),
        RewriteRoute::ResolvedCorrection
    ));
    assert!(!prompt.contains("Recommended interpretation:"));
    assert!(prompt.contains("Structured edit signals"));
    assert!(prompt.contains("Structured edit intents"));
    assert!(prompt.contains("Self-corrections were already resolved"));
    assert!(prompt.contains("Do not restore any canceled wording"));
    assert!(!prompt.contains("Recent segments"));
    assert!(!prompt.contains("Raw transcript"));
}

#[test]
fn user_message_includes_recent_segments_when_correction_matches_raw() {
    let transcript = RewriteTranscript {
        raw_text: "Hi there.".into(),
        correction_aware_text: "Hi there.".into(),
        aggressive_correction_text: None,
        detected_language: Some("en".into()),
        typing_context: None,
        recent_session_entries: Vec::new(),
        session_backtrack_candidates: Vec::new(),
        recommended_session_candidate: None,
        segments: vec![RewriteTranscriptSegment {
            text: "Hi there.".into(),
            start_ms: 0,
            end_ms: 1200,
        }],
        edit_intents: Vec::new(),
        edit_signals: Vec::new(),
        edit_hypotheses: Vec::new(),
        rewrite_candidates: vec![RewriteCandidate {
            kind: RewriteCandidateKind::Literal,
            text: "Hi there.".into(),
        }],
        recommended_candidate: None,
        policy_context: RewritePolicyContext::default(),
    };

    let prompt = build_user_message(&transcript);
    assert!(matches!(rewrite_route(&transcript), RewriteRoute::Fast));
    assert!(prompt.contains("Correction-aware transcript"));
    assert!(prompt.contains("Structured edit signals"));
    assert!(prompt.contains("Recent segments"));
    assert!(prompt.contains("0-1200 ms"));
    assert!(prompt.contains("Hi there."));
}

#[test]
fn effective_max_tokens_scales_with_transcript_length() {
    let short = RewriteTranscript {
        raw_text: "hi there".into(),
        correction_aware_text: "hi there".into(),
        aggressive_correction_text: None,
        detected_language: Some("en".into()),
        typing_context: None,
        recent_session_entries: Vec::new(),
        session_backtrack_candidates: Vec::new(),
        recommended_session_candidate: None,
        segments: Vec::new(),
        edit_intents: Vec::new(),
        edit_signals: Vec::new(),
        edit_hypotheses: Vec::new(),
        rewrite_candidates: vec![RewriteCandidate {
            kind: RewriteCandidateKind::Literal,
            text: "hi there".into(),
        }],
        recommended_candidate: None,
        policy_context: RewritePolicyContext::default(),
    };
    assert_eq!(effective_max_tokens(256, &short), 48);

    let long = RewriteTranscript {
        raw_text: "word ".repeat(80),
        correction_aware_text: "word ".repeat(80),
        aggressive_correction_text: None,
        detected_language: Some("en".into()),
        typing_context: None,
        recent_session_entries: Vec::new(),
        session_backtrack_candidates: Vec::new(),
        recommended_session_candidate: None,
        segments: Vec::new(),
        edit_intents: Vec::new(),
        edit_signals: Vec::new(),
        edit_hypotheses: Vec::new(),
        rewrite_candidates: vec![RewriteCandidate {
            kind: RewriteCandidateKind::Literal,
            text: "word ".repeat(80),
        }],
        recommended_candidate: None,
        policy_context: RewritePolicyContext::default(),
    };
    assert_eq!(effective_max_tokens(256, &long), 184);
}

#[test]
fn effective_max_tokens_gives_edit_heavy_prompts_more_budget() {
    let transcript = correction_transcript();
    assert_eq!(effective_max_tokens(256, &transcript), 64);
}

#[test]
fn session_prompt_includes_recent_entry_and_context() {
    let transcript = RewriteTranscript {
        raw_text: "scratch that hi".into(),
        correction_aware_text: "Hi".into(),
        aggressive_correction_text: None,
        detected_language: Some("en".into()),
        typing_context: Some(RewriteTypingContext {
            focus_fingerprint: "hyprland:0x123".into(),
            app_id: Some("firefox".into()),
            window_title: Some("Example".into()),
            surface_kind: RewriteSurfaceKind::Browser,
            browser_domain: None,
            captured_at_ms: 10,
        }),
        recent_session_entries: vec![RewriteSessionEntry {
            id: 7,
            final_text: "Hello there".into(),
            grapheme_len: 11,
            focus_fingerprint: "hyprland:0x123".into(),
            surface_kind: RewriteSurfaceKind::Browser,
            app_id: Some("firefox".into()),
            window_title: Some("Example".into()),
        }],
        session_backtrack_candidates: vec![
            RewriteSessionBacktrackCandidate {
                kind: RewriteSessionBacktrackCandidateKind::ReplaceLastEntry,
                entry_id: Some(7),
                delete_graphemes: 11,
                text: "Hi".into(),
            },
            RewriteSessionBacktrackCandidate {
                kind: RewriteSessionBacktrackCandidateKind::AppendCurrent,
                entry_id: None,
                delete_graphemes: 0,
                text: "Hi".into(),
            },
        ],
        recommended_session_candidate: Some(RewriteSessionBacktrackCandidate {
            kind: RewriteSessionBacktrackCandidateKind::ReplaceLastEntry,
            entry_id: Some(7),
            delete_graphemes: 11,
            text: "Hi".into(),
        }),
        segments: Vec::new(),
        edit_intents: Vec::new(),
        edit_signals: Vec::new(),
        edit_hypotheses: Vec::new(),
        rewrite_candidates: vec![RewriteCandidate {
            kind: RewriteCandidateKind::SentenceReplacement,
            text: "Hi".into(),
        }],
        recommended_candidate: Some(RewriteCandidate {
            kind: RewriteCandidateKind::SentenceReplacement,
            text: "Hi".into(),
        }),
        policy_context: RewritePolicyContext::default(),
    };

    let prompt = build_user_message(&transcript);
    assert!(matches!(
        rewrite_route(&transcript),
        RewriteRoute::SessionCandidateAdjudication
    ));
    assert!(prompt.contains("Active typing context"));
    assert!(prompt.contains("Recent dictation session"));
    assert!(prompt.contains("replace_last_entry"));
    assert!(prompt.contains("treat your final text as the replacement text"));
}

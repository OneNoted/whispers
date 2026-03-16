use super::RewritePrompt;
use super::routing::{
    RewriteRoute, has_policy_context, has_strong_explicit_edit_cue,
    requires_candidate_adjudication, rewrite_route,
};
use crate::rewrite_profile::{ResolvedRewriteProfile, RewriteProfile};
use crate::rewrite_protocol::RewriteTranscript;

pub fn build_prompt(
    transcript: &RewriteTranscript,
    profile: ResolvedRewriteProfile,
    custom_instructions: Option<&str>,
) -> std::result::Result<RewritePrompt, String> {
    Ok(RewritePrompt {
        system: build_system_instructions(transcript, profile, custom_instructions),
        user: build_user_message(transcript),
    })
}

pub fn resolved_profile_for_cloud(profile: RewriteProfile) -> ResolvedRewriteProfile {
    match profile {
        RewriteProfile::Auto => ResolvedRewriteProfile::Generic,
        RewriteProfile::Generic => ResolvedRewriteProfile::Generic,
        RewriteProfile::Qwen => ResolvedRewriteProfile::Qwen,
        RewriteProfile::LlamaCompat => ResolvedRewriteProfile::LlamaCompat,
    }
}

pub fn build_oaicompat_messages_json(
    prompt: &RewritePrompt,
) -> std::result::Result<String, String> {
    serde_json::to_string(&[
        serde_json::json!({
            "role": "system",
            "content": prompt.system,
        }),
        serde_json::json!({
            "role": "user",
            "content": prompt.user,
        }),
    ])
    .map_err(|e| format!("failed to encode rewrite chat messages: {e}"))
}

pub fn effective_max_tokens(max_tokens: usize, transcript: &RewriteTranscript) -> usize {
    let word_count = transcript
        .correction_aware_text
        .split_whitespace()
        .filter(|word| !word.is_empty())
        .count();
    let extra_budget = if requires_candidate_adjudication(transcript) {
        24
    } else {
        0
    };
    let minimum = if requires_candidate_adjudication(transcript) {
        64
    } else {
        48
    };
    let derived = word_count
        .saturating_mul(2)
        .saturating_add(24)
        .saturating_add(extra_budget);
    derived.clamp(minimum, max_tokens)
}

pub(crate) fn build_system_instructions(
    transcript: &RewriteTranscript,
    profile: ResolvedRewriteProfile,
    custom_instructions: Option<&str>,
) -> String {
    let mut instructions = rewrite_instructions(profile).to_string();
    if has_policy_context(transcript) {
        let policy_context = &transcript.policy_context;
        instructions.push_str("\n\nCorrection policy contract:\n");
        instructions.push_str(correction_policy_contract(policy_context.correction_policy));
        instructions.push_str("\n\nAgentic latitude contract:\n");
        instructions.push_str(agentic_latitude_contract(policy_context.correction_policy));
        if !policy_context.effective_rule_instructions.is_empty() {
            instructions.push_str("\n\nMatched app rewrite policy instructions:\n");
            for instruction in &policy_context.effective_rule_instructions {
                instructions.push_str("- ");
                instructions.push_str(instruction.trim());
                instructions.push('\n');
            }
        }
    }
    if let Some(custom) = custom_instructions
        .map(str::trim)
        .filter(|text| !text.is_empty())
    {
        instructions.push_str("\n\nAdditional user rewrite instructions:\n");
        instructions.push_str(custom);
    }
    instructions
}

fn correction_policy_contract(
    policy: crate::rewrite_protocol::RewriteCorrectionPolicy,
) -> &'static str {
    match policy {
        crate::rewrite_protocol::RewriteCorrectionPolicy::Conservative => {
            "Conservative: stay close to explicit rewrite candidates and glossary evidence. If uncertain, prefer candidate-preserving output over freer rewriting."
        }
        crate::rewrite_protocol::RewriteCorrectionPolicy::Balanced => {
            "Balanced: allow stronger technical correction when the glossary, app context, or utterance semantics support it. Prefer candidate-backed output when it is competitive, but do not keep an obviously wrong technical spelling just because it appears in the candidate list."
        }
        crate::rewrite_protocol::RewriteCorrectionPolicy::Aggressive => {
            "Aggressive: allow freer technical correction and contextual cleanup when the utterance strongly points to a technical term or proper name. Candidates are useful evidence, not hard limits, as long as you still return only final text within the provided bounds."
        }
    }
}

fn agentic_latitude_contract(
    policy: crate::rewrite_protocol::RewriteCorrectionPolicy,
) -> &'static str {
    match policy {
        crate::rewrite_protocol::RewriteCorrectionPolicy::Conservative => {
            "In conservative mode, treat the candidate list and glossary as the main evidence. Only make a freer technical normalization when the utterance itself makes the intended term unusually clear."
        }
        crate::rewrite_protocol::RewriteCorrectionPolicy::Balanced => {
            "In balanced mode, you may normalize likely technical terms, product names, commands, libraries, languages, editors, or Linux components even when the literal transcript spelling is noisy or the exact canonical form is not already present in the candidate list, as long as the utterance strongly supports that normalization."
        }
        crate::rewrite_protocol::RewriteCorrectionPolicy::Aggressive => {
            "In aggressive mode, you may confidently rewrite phonetically similar words into the most plausible technical term or proper name when the utterance semantics, app context, or nearby category cues make that interpretation clearly better than the literal transcript."
        }
    }
}

pub(crate) fn rewrite_instructions(profile: ResolvedRewriteProfile) -> &'static str {
    let base = "You clean up dictated speech into the final text the user meant to type. \
Return only the finished text. Do not explain anything. Remove obvious disfluencies when natural. \
Use the correction-aware transcript as the primary source of truth unless structured edit signals say the \
utterance may still be ambiguous. The raw transcript may still contain spoken editing phrases or canceled wording. \
Never reintroduce text that was removed by an explicit spoken correction cue. Respect any structured edit intents \
provided alongside the transcript. If structured edit signals or edit hypotheses are present, use the candidate \
interpretations as bounded options, choose the best interpretation, and lightly refine it only when needed for natural \
final text. Prefer transcript spellings for names, brands, and uncommon proper nouns unless a user dictionary or \
explicit correction says otherwise. Do not normalize names into more common spellings just because they look familiar. \
When the utterance clearly refers to software, tools, APIs, libraries, Linux components, product names, or other \
technical concepts, prefer the most plausible intended technical term or proper name over a phonetically similar common \
word. Use nearby category words like window manager, editor, language, library, package manager, shell, or terminal \
tool to disambiguate technical names. If the utterance remains genuinely ambiguous, stay close to the transcript rather \
than inventing a niche term. \
If an edit intent says to replace or cancel previous wording, preserve that edit and do not keep the spoken correction \
phrase itself unless the transcript clearly still intends it. Examples:\n\
- raw: Hello there. Scratch that. Hi.\n  correction-aware: Hi.\n  final: Hi.\n\
- raw: I'll bring cookies, scratch that, brownies.\n  correction-aware: I'll bring brownies.\n  final: I'll bring brownies.\n\
- raw: My name is Notes, scratch that my name is Jonatan.\n  correction-aware: My my name is Jonatan.\n  aggressive correction-aware: My name is Jonatan.\n  final: My name is Jonatan.\n\
- raw: Never mind. Hi, how are you today?\n  correction-aware: Hi, how are you today?\n  final: Hi, how are you today?\n\
- raw: Wait, no, it actually works.\n  correction-aware: Wait, no, it actually works.\n  final: Wait, no, it actually works.\n\
- raw: Let's meet tomorrow, or rather Friday.\n  correction-aware: Let's meet Friday.\n  final: Let's meet Friday.\n\
- raw: I'm currently using the window manager Hyperland.\n  correction-aware: I'm currently using the window manager Hyperland.\n  final: I'm currently using the window manager Hyprland.\n\
- raw: I'm switching from Sui to Hyperland.\n  correction-aware: I'm switching from Sui to Hyperland.\n  final: I'm switching from Sway to Hyprland.\n\
- raw: I use type script for backend tooling.\n  correction-aware: I use type script for backend tooling.\n  final: I use TypeScript for backend tooling.\n\
- raw: I edit the config in neo vim.\n  correction-aware: I edit the config in neo vim.\n  final: I edit the config in Neovim.";

    match profile {
        ResolvedRewriteProfile::Qwen => {
            "You clean up dictated speech into the final text the user meant to type. \
Return only the finished text. Do not explain anything. Do not emit reasoning, think tags, or XML wrappers. \
Remove obvious disfluencies when natural. Use the correction-aware transcript as the primary source of truth unless \
structured edit signals say the utterance may still be ambiguous. The raw transcript may still contain spoken editing \
phrases or canceled wording. Never reintroduce text that was removed by an explicit spoken correction cue. Respect \
any structured edit intents provided alongside the transcript. If structured edit signals or edit hypotheses are \
present, use the candidate interpretations as bounded options, choose the best interpretation, and lightly refine it \
only when needed for natural final text. Prefer transcript spellings for names, brands, and uncommon proper nouns \
unless a user dictionary or explicit correction says otherwise. Do not normalize names into more common spellings just \
because they look familiar. When the utterance clearly refers to software, tools, APIs, libraries, Linux components, \
product names, or other technical concepts, prefer the most plausible intended technical term or proper name over a \
phonetically similar common word. Use nearby category words like window manager, editor, language, library, package \
manager, shell, or terminal tool to disambiguate technical names. If the utterance remains genuinely ambiguous, stay \
close to the transcript rather than inventing a niche term. If an edit intent says to replace or cancel previous wording, preserve that edit and do \
not keep the spoken correction phrase itself unless the transcript clearly still intends it. Examples:\n\
- raw: Hello there. Scratch that. Hi.\n  correction-aware: Hi.\n  final: Hi.\n\
- raw: I'll bring cookies, scratch that, brownies.\n  correction-aware: I'll bring brownies.\n  final: I'll bring brownies.\n\
- raw: My name is Notes, scratch that my name is Jonatan.\n  correction-aware: My my name is Jonatan.\n  aggressive correction-aware: My name is Jonatan.\n  final: My name is Jonatan.\n\
- raw: Never mind. Hi, how are you today?\n  correction-aware: Hi, how are you today?\n  final: Hi, how are you today?\n\
- raw: Wait, no, it actually works.\n  correction-aware: Wait, no, it actually works.\n  final: Wait, no, it actually works.\n\
- raw: Let's meet tomorrow, or rather Friday.\n  correction-aware: Let's meet Friday.\n  final: Let's meet Friday.\n\
- raw: I'm currently using the window manager Hyperland.\n  correction-aware: I'm currently using the window manager Hyperland.\n  final: I'm currently using the window manager Hyprland.\n\
- raw: I'm switching from Sui to Hyperland.\n  correction-aware: I'm switching from Sui to Hyperland.\n  final: I'm switching from Sway to Hyprland.\n\
- raw: I use type script for backend tooling.\n  correction-aware: I use type script for backend tooling.\n  final: I use TypeScript for backend tooling.\n\
- raw: I edit the config in neo vim.\n  correction-aware: I edit the config in neo vim.\n  final: I edit the config in Neovim."
        }
        ResolvedRewriteProfile::Generic | ResolvedRewriteProfile::LlamaCompat => base,
    }
}

pub(crate) fn build_user_message(transcript: &RewriteTranscript) -> String {
    let language = transcript.detected_language.as_deref().unwrap_or("unknown");
    let correction_aware = transcript.correction_aware_text.trim();
    let raw = transcript.raw_text.trim();
    let edit_intents = render_edit_intents(transcript);
    let edit_signals = render_edit_signals(transcript);
    let agentic_context = render_agentic_context(transcript);
    let route = rewrite_route(transcript);
    tracing::debug!(
        route = ?route,
        edit_signals = transcript.edit_signals.len(),
        edit_hypotheses = transcript.edit_hypotheses.len(),
        rewrite_candidates = transcript.rewrite_candidates.len(),
        "rewrite prompt route selected"
    );

    match route {
        RewriteRoute::SessionCandidateAdjudication => {
            let typing_context = render_typing_context(transcript);
            let recent_session_entries = render_recent_session_entries(transcript);
            let agentic_policy_context = render_agentic_policy_context(transcript);
            let session_candidates = render_session_backtrack_candidates(transcript);
            let recommended_session_candidate = render_recommended_session_candidate(transcript);
            let rewrite_candidates = render_rewrite_candidates(transcript);
            let surface_guidance = transcript
                .typing_context
                .as_ref()
                .filter(|context| {
                    matches!(
                        context.surface_kind,
                        crate::rewrite_protocol::RewriteSurfaceKind::Terminal
                    )
                })
                .map(|_| {
                    "The active surface looks like a terminal. Stay conservative unless an explicit correction cue clearly indicates replacing the most recent prior dictation.\n"
                })
                .unwrap_or("");
            format!(
                "Language: {language}\n\
Active typing context:\n\
{typing_context}\
Recent dictation session:\n\
{recent_session_entries}\
{agentic_policy_context}\
Session backtrack candidates:\n\
{session_candidates}\
{recommended_session_candidate}\
The user may be correcting the most recent prior dictation entry rather than appending new text.\n\
If the recommended session candidate says replace_last_entry, treat your final text as the replacement text for that previous dictation entry, not as newly appended text.\n\
Prefer the recommended session candidate unless another listed session candidate is clearly better.\n\
{surface_guidance}\
Current utterance correction candidate:\n\
{correction_aware}\n\
Raw current utterance:\n\
{raw}\n\
Current utterance bounded candidates:\n\
{rewrite_candidates}\
Final text:"
            )
        }
        RewriteRoute::CandidateAdjudication => {
            let edit_hypotheses = render_edit_hypotheses(transcript);
            let rewrite_candidates = render_rewrite_candidates(transcript);
            let recommended_candidate = render_recommended_candidate(transcript);
            let recent_segments = render_recent_segments(transcript, 4);
            let aggressive_candidate = render_aggressive_candidate(transcript);
            let exact_cue_guidance = if has_strong_explicit_edit_cue(transcript) {
                "A strong explicit spoken edit cue was detected. The literal raw transcript probably contains canceled wording. \
Prefer a candidate interpretation that removes the cue and canceled wording unless doing so would clearly lose intended meaning. \
If the cue is an exact strong match for phrases like scratch that, never mind, or wait no, do not keep the literal cue text in the final output.\n"
            } else {
                ""
            };
            tracing::trace!("rewrite hypotheses:\n{edit_hypotheses}");
            tracing::trace!("rewrite candidates:\n{rewrite_candidates}");
            format!(
                "Language: {language}\n\
{agentic_context}\
Structured edit hypotheses:\n\
{edit_hypotheses}\
Structured edit signals:\n\
{edit_signals}\
Structured edit intents:\n\
{edit_intents}\
This utterance likely contains spoken self-corrections or restatements.\n\
Choose the best candidate interpretation and lightly refine it only when needed.\n\
{exact_cue_guidance}\
When an exact strong edit cue is present, treat the non-literal candidates as more trustworthy than the literal transcript.\n\
The candidate list is ordered from most likely to least likely by heuristics.\n\
For exact strong edit cues, the first candidate is the heuristic best guess and should usually win unless another candidate is clearly better.\n\
Prefer the smallest replacement scope that yields a coherent result.\n\
Use span-level replacements when only a key phrase was corrected, clause-level replacements when the correction replaces the surrounding thought, and sentence-level replacements only when the whole sentence was canceled.\n\
Preserve literal wording when the cue is not actually an edit.\n\
Do not over-normalize names or brands.\n\
Do not keep spoken edit cues in the final text when they act as edits.\n\
{recommended_candidate}\
Candidate interpretations:\n\
{rewrite_candidates}\
Correction candidate:\n\
{correction_aware}\n\
{aggressive_candidate}\
Raw transcript:\n\
{raw}\n\
Recent segments:\n\
{recent_segments}\n\
Final text:"
            )
        }
        RewriteRoute::ResolvedCorrection => format!(
            "Language: {language}\n\
{agentic_context}\
Structured edit signals:\n\
{edit_signals}\
Structured edit intents:\n\
{edit_intents}\
Self-corrections were already resolved before rewriting.\n\
Use this correction-aware transcript as the main source text. In agentic mode, you may still normalize likely \
technical terms or proper names when the utterance strongly supports them, even if the exact canonical spelling is not \
already present in the candidate list:\n\
{correction_aware}\n\
{agentic_candidates}\
Do not restore any canceled wording from earlier in the utterance.\n\
Final text:",
            agentic_candidates = render_agentic_candidates(transcript),
        ),
        RewriteRoute::Fast => {
            let recent_segments = render_recent_segments(transcript, 4);
            format!(
                "Language: {language}\n\
{agentic_context}\
Structured edit signals:\n\
{edit_signals}\
Structured edit intents:\n\
{edit_intents}\
Correction-aware transcript:\n\
{correction_aware}\n\
Treat the correction-aware transcript as authoritative for explicit spoken edits and overall meaning, but in agentic \
mode you may normalize likely technical terms or proper names when category cues in the utterance make the intended \
technical meaning clearly better than the literal transcript.\n\
{agentic_candidates}\
\
Recent segments:\n\
{recent_segments}\n\
Final text:",
                agentic_candidates = render_agentic_candidates(transcript),
            )
        }
    }
}

fn render_agentic_context(transcript: &RewriteTranscript) -> String {
    if !has_policy_context(transcript) {
        return String::new();
    }
    format!(
        "{}{}",
        render_agentic_runtime_context(transcript),
        render_agentic_policy_context(transcript)
    )
}

fn render_agentic_policy_context(transcript: &RewriteTranscript) -> String {
    if !has_policy_context(transcript) {
        return String::new();
    }
    let policy_context = &transcript.policy_context;

    format!(
        "Agentic correction policy:\n\
- mode: {}\n\
Matched app rewrite rules:\n\
{matched_rules}\
Matched app policy instructions:\n\
{effective_instructions}\
Active glossary terms:\n\
{glossary_terms}\
",
        policy_context.correction_policy.as_str(),
        matched_rules = render_matched_rule_names(transcript),
        effective_instructions = render_effective_rule_instructions(transcript),
        glossary_terms = render_active_glossary_terms(transcript),
    )
}

fn render_agentic_runtime_context(transcript: &RewriteTranscript) -> String {
    has_policy_context(transcript)
        .then(|| {
            format!(
                "Active typing context:\n\
{}\
Recent dictation session:\n\
{}",
                render_typing_context(transcript),
                render_recent_session_entries(transcript),
            )
        })
        .unwrap_or_default()
}

fn render_agentic_candidates(transcript: &RewriteTranscript) -> String {
    has_policy_context(transcript)
        .then(|| {
            format!(
                "Available rewrite candidates (advisory, not exhaustive in agentic mode):\n\
{}\
Glossary-backed candidates:\n\
{}",
                render_rewrite_candidates(transcript),
                render_glossary_candidates(transcript)
            )
        })
        .unwrap_or_default()
}

fn render_edit_intents(transcript: &RewriteTranscript) -> String {
    if transcript.edit_intents.is_empty() {
        return "- none detected\n".to_string();
    }

    let mut rendered = String::new();
    for intent in &transcript.edit_intents {
        let action = match intent.action {
            crate::rewrite_protocol::RewriteEditAction::ReplacePreviousPhrase => {
                "replace_previous_phrase"
            }
            crate::rewrite_protocol::RewriteEditAction::ReplacePreviousClause => {
                "replace_previous_clause"
            }
            crate::rewrite_protocol::RewriteEditAction::ReplacePreviousSentence => {
                "replace_previous_sentence"
            }
            crate::rewrite_protocol::RewriteEditAction::DropEditCue => "drop_edit_cue",
        };
        let confidence = match intent.confidence {
            crate::rewrite_protocol::RewriteIntentConfidence::High => "high",
        };
        rendered.push_str(&format!(
            "- action: {action}, trigger: \"{}\", confidence: {confidence}\n",
            intent.trigger
        ));
    }

    rendered
}

fn render_edit_signals(transcript: &RewriteTranscript) -> String {
    if transcript.edit_signals.is_empty() {
        return "- none detected\n".to_string();
    }

    let mut rendered = String::new();
    for signal in &transcript.edit_signals {
        let kind = match signal.kind {
            crate::rewrite_protocol::RewriteEditSignalKind::Cancel => "cancel",
            crate::rewrite_protocol::RewriteEditSignalKind::Replace => "replace",
            crate::rewrite_protocol::RewriteEditSignalKind::Restatement => "restatement",
        };
        let scope_hint = match signal.scope_hint {
            crate::rewrite_protocol::RewriteEditSignalScope::Phrase => "phrase",
            crate::rewrite_protocol::RewriteEditSignalScope::Clause => "clause",
            crate::rewrite_protocol::RewriteEditSignalScope::Sentence => "sentence",
            crate::rewrite_protocol::RewriteEditSignalScope::Unknown => "unknown",
        };
        let strength = match signal.strength {
            crate::rewrite_protocol::RewriteEditSignalStrength::Possible => "possible",
            crate::rewrite_protocol::RewriteEditSignalStrength::Strong => "strong",
        };
        rendered.push_str(&format!(
            "- trigger: \"{}\", kind: {kind}, scope_hint: {scope_hint}, strength: {strength}\n",
            signal.trigger
        ));
    }

    rendered
}

fn render_edit_hypotheses(transcript: &RewriteTranscript) -> String {
    if transcript.edit_hypotheses.is_empty() {
        return "- none detected\n".to_string();
    }

    let mut rendered = String::new();
    for hypothesis in &transcript.edit_hypotheses {
        let match_source = match hypothesis.match_source {
            crate::rewrite_protocol::RewriteEditHypothesisMatchSource::Exact => "exact",
            crate::rewrite_protocol::RewriteEditHypothesisMatchSource::Alias => "alias",
            crate::rewrite_protocol::RewriteEditHypothesisMatchSource::NearMiss => "near_miss",
        };
        let kind = match hypothesis.kind {
            crate::rewrite_protocol::RewriteEditSignalKind::Cancel => "cancel",
            crate::rewrite_protocol::RewriteEditSignalKind::Replace => "replace",
            crate::rewrite_protocol::RewriteEditSignalKind::Restatement => "restatement",
        };
        let scope_hint = match hypothesis.scope_hint {
            crate::rewrite_protocol::RewriteEditSignalScope::Phrase => "phrase",
            crate::rewrite_protocol::RewriteEditSignalScope::Clause => "clause",
            crate::rewrite_protocol::RewriteEditSignalScope::Sentence => "sentence",
            crate::rewrite_protocol::RewriteEditSignalScope::Unknown => "unknown",
        };
        let strength = match hypothesis.strength {
            crate::rewrite_protocol::RewriteEditSignalStrength::Possible => "possible",
            crate::rewrite_protocol::RewriteEditSignalStrength::Strong => "strong",
        };
        let replacement_scope = match hypothesis.replacement_scope {
            crate::rewrite_protocol::RewriteReplacementScope::Span => "span",
            crate::rewrite_protocol::RewriteReplacementScope::Clause => "clause",
            crate::rewrite_protocol::RewriteReplacementScope::Sentence => "sentence",
        };
        let tail_shape = match hypothesis.tail_shape {
            crate::rewrite_protocol::RewriteTailShape::Empty => "empty",
            crate::rewrite_protocol::RewriteTailShape::Phrase => "phrase",
            crate::rewrite_protocol::RewriteTailShape::Clause => "clause",
        };
        rendered.push_str(&format!(
            "- cue_family: {}, matched_text: \"{}\", match_source: {match_source}, kind: {kind}, scope_hint: {scope_hint}, replacement_scope: {replacement_scope}, tail_shape: {tail_shape}, strength: {strength}\n",
            hypothesis.cue_family, hypothesis.matched_text
        ));
    }

    rendered
}

fn render_rewrite_candidates(transcript: &RewriteTranscript) -> String {
    if transcript.rewrite_candidates.is_empty() {
        return "- no candidates available\n".to_string();
    }

    let mut rendered = String::new();
    let highlight_first = has_strong_explicit_edit_cue(transcript);
    for (index, candidate) in transcript.rewrite_candidates.iter().enumerate() {
        let prefix = if highlight_first && index == 0 {
            "- preferred_candidate"
        } else {
            "-"
        };
        let kind = match candidate.kind {
            crate::rewrite_protocol::RewriteCandidateKind::Literal => {
                "literal (keep only if the cue was not actually an edit)"
            }
            crate::rewrite_protocol::RewriteCandidateKind::ConservativeCorrection => {
                "conservative_correction (balanced cleanup)"
            }
            crate::rewrite_protocol::RewriteCandidateKind::AggressiveCorrection => {
                "aggressive_correction (use when canceled wording should be removed more fully)"
            }
            crate::rewrite_protocol::RewriteCandidateKind::GlossaryCorrection => {
                "glossary_correction (supported by active glossary aliases)"
            }
            crate::rewrite_protocol::RewriteCandidateKind::SpanReplacement => {
                "span_replacement (replace only the corrected phrase)"
            }
            crate::rewrite_protocol::RewriteCandidateKind::ClauseReplacement => {
                "clause_replacement (replace the corrected clause while keeping surrounding context)"
            }
            crate::rewrite_protocol::RewriteCandidateKind::SentenceReplacement => {
                "sentence_replacement (replace the whole corrected sentence)"
            }
            crate::rewrite_protocol::RewriteCandidateKind::ContextualReplacement => {
                "contextual_replacement (replace the corrected span while keeping earlier context)"
            }
            crate::rewrite_protocol::RewriteCandidateKind::DropCueOnly => {
                "drop_cue_only (remove just the spoken edit cue)"
            }
            crate::rewrite_protocol::RewriteCandidateKind::FollowingReplacement => {
                "following_replacement (keep only the wording after the cue)"
            }
            crate::rewrite_protocol::RewriteCandidateKind::CancelPreviousClause => {
                "cancel_previous_clause (treat the cue as canceling the prior clause)"
            }
            crate::rewrite_protocol::RewriteCandidateKind::CancelPreviousSentence => {
                "cancel_previous_sentence (treat the cue as canceling the prior sentence)"
            }
        };
        rendered.push_str(&format!("{prefix} {kind}: {}\n", candidate.text));
    }

    rendered
}

fn render_recommended_candidate(transcript: &RewriteTranscript) -> String {
    transcript
        .recommended_candidate
        .as_ref()
        .map(|candidate| {
            format!(
                "Recommended interpretation:\n{}\nUse this as the default final text unless another candidate is clearly better.\n",
                candidate.text
            )
        })
        .unwrap_or_default()
}

fn render_typing_context(transcript: &RewriteTranscript) -> String {
    transcript
        .typing_context
        .as_ref()
        .map(|context| {
            format!(
                "- focus_fingerprint: {}\n- app_id: {}\n- window_title: {}\n- surface_kind: {}\n- browser_domain: {}\n",
                context.focus_fingerprint,
                context.app_id.as_deref().unwrap_or("unknown"),
                context.window_title.as_deref().unwrap_or("unknown"),
                context.surface_kind.as_str(),
                context.browser_domain.as_deref().unwrap_or("unknown"),
            )
        })
        .unwrap_or_else(|| "- none available\n".to_string())
}

fn render_recent_session_entries(transcript: &RewriteTranscript) -> String {
    if transcript.recent_session_entries.is_empty() {
        return "- none available\n".to_string();
    }

    let mut rendered = String::new();
    for entry in &transcript.recent_session_entries {
        rendered.push_str(&format!(
            "- id: {}, text: {}, grapheme_len: {}, surface_kind: {}\n",
            entry.id,
            entry.final_text,
            entry.grapheme_len,
            entry.surface_kind.as_str()
        ));
    }
    rendered
}

fn render_session_backtrack_candidates(transcript: &RewriteTranscript) -> String {
    if transcript.session_backtrack_candidates.is_empty() {
        return "- no session backtrack candidates\n".to_string();
    }

    let mut rendered = String::new();
    for candidate in &transcript.session_backtrack_candidates {
        let kind = match candidate.kind {
            crate::rewrite_protocol::RewriteSessionBacktrackCandidateKind::AppendCurrent => {
                "append_current"
            }
            crate::rewrite_protocol::RewriteSessionBacktrackCandidateKind::ReplaceLastEntry => {
                "replace_last_entry"
            }
        };
        rendered.push_str(&format!(
            "- kind: {kind}, entry_id: {}, delete_graphemes: {}, text: {}\n",
            candidate
                .entry_id
                .map(|entry_id| entry_id.to_string())
                .unwrap_or_else(|| "none".to_string()),
            candidate.delete_graphemes,
            candidate.text
        ));
    }
    rendered
}

fn render_recommended_session_candidate(transcript: &RewriteTranscript) -> String {
    transcript
        .recommended_session_candidate
        .as_ref()
        .map(|candidate| {
            let mode = match candidate.kind {
                crate::rewrite_protocol::RewriteSessionBacktrackCandidateKind::AppendCurrent => {
                    "append_current"
                }
                crate::rewrite_protocol::RewriteSessionBacktrackCandidateKind::ReplaceLastEntry => {
                    "replace_last_entry"
                }
            };
            format!(
                "Recommended session action:\nmode: {mode}\nentry_id: {}\ndelete_graphemes: {}\ntext: {}\n",
                candidate
                    .entry_id
                    .map(|entry_id| entry_id.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                candidate.delete_graphemes,
                candidate.text
            )
        })
        .unwrap_or_default()
}

fn render_recent_segments(transcript: &RewriteTranscript, limit: usize) -> String {
    let total_segments = transcript.segments.len();
    let start = total_segments.saturating_sub(limit);
    let mut rendered = String::new();

    for segment in &transcript.segments[start..] {
        let line = format!(
            "- {}-{} ms: {}\n",
            segment.start_ms, segment.end_ms, segment.text
        );
        rendered.push_str(&line);
    }

    if rendered.is_empty() {
        rendered.push_str("- no segments available\n");
    }

    rendered
}

fn render_aggressive_candidate(transcript: &RewriteTranscript) -> String {
    transcript
        .aggressive_correction_text
        .as_deref()
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(|text| format!("Aggressive correction candidate:\n{text}\n"))
        .unwrap_or_default()
}

fn render_matched_rule_names(transcript: &RewriteTranscript) -> String {
    if !has_policy_context(transcript) {
        return "- none\n".to_string();
    }
    let policy_context = &transcript.policy_context;
    if policy_context.matched_rule_names.is_empty() {
        return "- none\n".to_string();
    }
    policy_context
        .matched_rule_names
        .iter()
        .map(|name| format!("- {name}\n"))
        .collect()
}

fn render_effective_rule_instructions(transcript: &RewriteTranscript) -> String {
    if !has_policy_context(transcript) {
        return "- none\n".to_string();
    }
    let policy_context = &transcript.policy_context;
    if policy_context.effective_rule_instructions.is_empty() {
        return "- none\n".to_string();
    }
    policy_context
        .effective_rule_instructions
        .iter()
        .map(|instruction| format!("- {}\n", instruction.trim()))
        .collect()
}

fn render_active_glossary_terms(transcript: &RewriteTranscript) -> String {
    if !has_policy_context(transcript) {
        return "- none\n".to_string();
    }
    let policy_context = &transcript.policy_context;
    if policy_context.active_glossary_terms.is_empty() {
        return "- none\n".to_string();
    }
    policy_context
        .active_glossary_terms
        .iter()
        .map(|entry| format!("- {} <- [{}]\n", entry.term, entry.aliases.join(", ")))
        .collect()
}

fn render_glossary_candidates(transcript: &RewriteTranscript) -> String {
    if !has_policy_context(transcript) {
        return "- none\n".to_string();
    }
    let policy_context = &transcript.policy_context;
    if policy_context.glossary_candidates.is_empty() {
        return "- none\n".to_string();
    }
    policy_context
        .glossary_candidates
        .iter()
        .map(|candidate| format!("- {}\n", candidate.text))
        .collect()
}

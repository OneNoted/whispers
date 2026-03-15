mod prompt;
mod routing;

#[cfg(feature = "local-rewrite")]
use std::num::NonZeroU32;
#[cfg(feature = "local-rewrite")]
use std::path::Path;
#[cfg(feature = "local-rewrite")]
use std::sync::OnceLock;

#[cfg(feature = "local-rewrite")]
use encoding_rs::UTF_8;
#[cfg(feature = "local-rewrite")]
use llama_cpp_2::context::params::LlamaContextParams;
#[cfg(feature = "local-rewrite")]
use llama_cpp_2::llama_backend::LlamaBackend;
#[cfg(feature = "local-rewrite")]
use llama_cpp_2::llama_batch::LlamaBatch;
#[cfg(feature = "local-rewrite")]
use llama_cpp_2::model::params::LlamaModelParams;
#[cfg(feature = "local-rewrite")]
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
#[cfg(feature = "local-rewrite")]
use llama_cpp_2::openai::OpenAIChatTemplateParams;
#[cfg(feature = "local-rewrite")]
use llama_cpp_2::sampling::LlamaSampler;
#[cfg(any(test, feature = "local-rewrite"))]
use serde_json::json;

#[cfg(feature = "local-rewrite")]
use crate::rewrite_profile::ResolvedRewriteProfile;
use crate::rewrite_protocol::RewriteTranscript;
pub use prompt::{build_prompt, resolved_profile_for_cloud};
#[cfg(test)]
use prompt::{build_system_instructions, build_user_message, rewrite_instructions};
use routing::requires_candidate_adjudication;
#[cfg(test)]
use routing::{RewriteRoute, rewrite_route};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewritePrompt {
    pub system: String,
    pub user: String,
}

#[cfg(feature = "local-rewrite")]
pub struct LocalRewriter {
    model: LlamaModel,
    chat_template: LlamaChatTemplate,
    profile: ResolvedRewriteProfile,
    max_tokens: usize,
    max_output_chars: usize,
}

#[cfg(feature = "local-rewrite")]
static LLAMA_BACKEND: OnceLock<&'static LlamaBackend> = OnceLock::new();
#[cfg(feature = "local-rewrite")]
static EXTERNAL_LLAMA_BACKEND: LlamaBackend = LlamaBackend {};

#[cfg(feature = "local-rewrite")]
impl LocalRewriter {
    #[allow(dead_code)]
    pub fn new(
        model_path: &Path,
        profile: ResolvedRewriteProfile,
        max_tokens: usize,
        max_output_chars: usize,
    ) -> std::result::Result<Self, String> {
        if !model_path.exists() {
            return Err(format!(
                "rewrite model file not found: {}",
                model_path.display()
            ));
        }

        let backend = llama_backend()?;

        let mut model_params = LlamaModelParams::default();
        if cfg!(feature = "cuda") {
            model_params = model_params.with_n_gpu_layers(1000);
        }

        let model = LlamaModel::load_from_file(backend, model_path, &model_params)
            .map_err(|e| format!("failed to load rewrite model: {e}"))?;
        let chat_template = model
            .chat_template(None)
            .map_err(|e| format!("rewrite model does not expose a usable chat template: {e}"))?;

        Ok(Self {
            model,
            chat_template,
            profile,
            max_tokens,
            max_output_chars,
        })
    }

    #[allow(dead_code)]
    pub fn rewrite_with_instructions(
        &self,
        transcript: &RewriteTranscript,
        custom_instructions: Option<&str>,
    ) -> std::result::Result<String, String> {
        if transcript.raw_text.trim().is_empty() {
            return Ok(String::new());
        }

        let prompt = build_rewrite_prompt(
            &self.model,
            &self.chat_template,
            transcript,
            self.profile,
            custom_instructions,
        )?;
        let effective_max_tokens = effective_max_tokens(self.max_tokens, transcript);
        let prompt_tokens = self
            .model
            .str_to_token(&prompt, AddBos::Never)
            .map_err(|e| format!("failed to tokenize rewrite prompt: {e}"))?;
        let behavior = rewrite_behavior(self.profile);

        let n_ctx_tokens = prompt_tokens
            .len()
            .saturating_add(effective_max_tokens)
            .saturating_add(64)
            .max(2048)
            .min(u32::MAX as usize) as u32;
        let n_batch = prompt_tokens
            .len()
            .max(512)
            .min(n_ctx_tokens as usize)
            .min(u32::MAX as usize) as u32;
        let threads = std::thread::available_parallelism()
            .map(|threads| threads.get())
            .unwrap_or(4)
            .clamp(1, i32::MAX as usize) as i32;

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx_tokens))
            .with_n_batch(n_batch)
            .with_n_ubatch(n_batch)
            .with_n_threads(threads)
            .with_n_threads_batch(threads);
        let backend = llama_backend()?;
        let mut ctx = self
            .model
            .new_context(backend, ctx_params)
            .map_err(|e| format!("failed to create rewrite context: {e}"))?;

        let mut prompt_batch = LlamaBatch::new(prompt_tokens.len(), 1);
        prompt_batch
            .add_sequence(&prompt_tokens, 0, false)
            .map_err(|e| format!("failed to enqueue rewrite prompt: {e}"))?;
        ctx.decode(&mut prompt_batch)
            .map_err(|e| format!("failed to decode rewrite prompt: {e}"))?;

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::top_k(behavior.top_k),
            LlamaSampler::top_p(behavior.top_p, 1),
            LlamaSampler::temp(behavior.temperature),
            LlamaSampler::greedy(),
        ]);
        sampler.accept_many(prompt_tokens.iter());

        let mut decoder = UTF_8.new_decoder_without_bom_handling();
        let mut output = String::new();
        let start_pos = i32::try_from(prompt_tokens.len()).unwrap_or(i32::MAX);

        for i in 0..effective_max_tokens {
            let mut candidates = ctx.token_data_array();
            candidates.apply_sampler(&sampler);
            let token = candidates
                .selected_token()
                .ok_or_else(|| "rewrite sampler did not select a token".to_string())?;

            if token == self.model.token_eos() {
                break;
            }

            sampler.accept(token);

            let piece = self
                .model
                .token_to_piece(token, &mut decoder, true, None)
                .map_err(|e| format!("failed to decode rewrite token: {e}"))?;
            output.push_str(&piece);

            if output.contains("</output>") || output.len() >= self.max_output_chars {
                break;
            }

            let mut batch = LlamaBatch::new(1, 1);
            batch
                .add(
                    token,
                    start_pos.saturating_add(i32::try_from(i).unwrap_or(i32::MAX)),
                    &[0],
                    true,
                )
                .map_err(|e| format!("failed to enqueue rewrite token: {e}"))?;
            ctx.decode(&mut batch)
                .map_err(|e| format!("failed to decode rewrite token: {e}"))?;
        }

        let rewritten = sanitize_rewrite_output(&output);
        if rewritten.is_empty() {
            return Err("rewrite model returned empty output".into());
        }

        Ok(rewritten)
    }
}

#[cfg(feature = "local-rewrite")]
fn llama_backend() -> std::result::Result<&'static LlamaBackend, String> {
    if let Some(backend) = LLAMA_BACKEND.get().copied() {
        return Ok(backend);
    }

    match LlamaBackend::init() {
        Ok(backend) => {
            let backend = Box::leak(Box::new(backend));
            let _ = LLAMA_BACKEND.set(backend);
            Ok(LLAMA_BACKEND
                .get()
                .copied()
                .expect("llama backend initialized"))
        }
        // Use a static non-dropping token when another part of the process already
        // owns llama.cpp global initialization. The worker never drops this token.
        Err(llama_cpp_2::LlamaCppError::BackendAlreadyInitialized) => {
            let _ = LLAMA_BACKEND.set(&EXTERNAL_LLAMA_BACKEND);
            Ok(LLAMA_BACKEND
                .get()
                .copied()
                .expect("external llama backend cached"))
        }
        Err(err) => Err(format!("failed to initialize llama backend: {err}")),
    }
}

#[cfg(feature = "local-rewrite")]
fn build_rewrite_prompt(
    model: &LlamaModel,
    chat_template: &LlamaChatTemplate,
    transcript: &RewriteTranscript,
    profile: ResolvedRewriteProfile,
    custom_instructions: Option<&str>,
) -> std::result::Result<String, String> {
    let prompt = build_prompt(transcript, profile, custom_instructions)?;
    if matches!(profile, ResolvedRewriteProfile::Qwen) {
        return build_qwen_rewrite_prompt(model, chat_template, &prompt);
    }

    let messages = vec![
        LlamaChatMessage::new("system".into(), prompt.system)
            .map_err(|e| format!("failed to build rewrite system message: {e}"))?,
        LlamaChatMessage::new("user".into(), prompt.user)
            .map_err(|e| format!("failed to build rewrite user message: {e}"))?,
    ];

    model
        .apply_chat_template(chat_template, &messages, true)
        .map_err(|e| format!("failed to apply rewrite chat template: {e}"))
}

#[cfg(feature = "local-rewrite")]
fn build_qwen_rewrite_prompt(
    model: &LlamaModel,
    chat_template: &LlamaChatTemplate,
    prompt: &RewritePrompt,
) -> std::result::Result<String, String> {
    let messages_json = build_oaicompat_messages_json(prompt)?;
    let result = model
        .apply_chat_template_oaicompat(
            chat_template,
            &OpenAIChatTemplateParams {
                messages_json: &messages_json,
                tools_json: None,
                tool_choice: None,
                json_schema: None,
                grammar: None,
                reasoning_format: None,
                chat_template_kwargs: None,
                add_generation_prompt: true,
                use_jinja: true,
                parallel_tool_calls: false,
                enable_thinking: false,
                add_bos: false,
                add_eos: false,
                parse_tool_calls: false,
            },
        )
        .map_err(|e| format!("failed to apply Qwen rewrite chat template: {e}"))?;
    Ok(result.prompt)
}

#[cfg(any(test, feature = "local-rewrite"))]
fn build_oaicompat_messages_json(prompt: &RewritePrompt) -> std::result::Result<String, String> {
    serde_json::to_string(&vec![
        json!({
            "role": "system",
            "content": prompt.system,
        }),
        json!({
            "role": "user",
            "content": prompt.user,
        }),
    ])
    .map_err(|e| format!("failed to encode rewrite chat messages: {e}"))
}

#[cfg(feature = "local-rewrite")]
pub const fn local_rewrite_available() -> bool {
    true
}

#[cfg(not(feature = "local-rewrite"))]
pub const fn local_rewrite_available() -> bool {
    false
}

#[cfg(feature = "local-rewrite")]
struct RewriteBehavior {
    top_k: i32,
    top_p: f32,
    temperature: f32,
}

#[cfg(feature = "local-rewrite")]
fn rewrite_behavior(profile: ResolvedRewriteProfile) -> RewriteBehavior {
    match profile {
        ResolvedRewriteProfile::Qwen => RewriteBehavior {
            top_k: 24,
            top_p: 0.9,
            temperature: 0.1,
        },
        ResolvedRewriteProfile::Generic => RewriteBehavior {
            top_k: 32,
            top_p: 0.92,
            temperature: 0.12,
        },
        ResolvedRewriteProfile::LlamaCompat => RewriteBehavior {
            top_k: 40,
            top_p: 0.95,
            temperature: 0.15,
        },
    }
}

#[allow(dead_code)]
fn effective_max_tokens(max_tokens: usize, transcript: &RewriteTranscript) -> usize {
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

pub(crate) fn sanitize_rewrite_output(raw: &str) -> String {
    let mut text = raw.replace("\r\n", "\n");

    for stop in ["<|eot_id|>", "<|end_of_text|>", "</s>"] {
        if let Some(index) = text.find(stop) {
            text.truncate(index);
        }
    }

    if let Some(index) = text.find("</output>") {
        text.truncate(index);
    }

    text = strip_tagged_section(&text, "<think>", "</think>");

    let mut text = text.trim().to_string();

    if let Some(stripped) = text.strip_prefix("<output>") {
        text = stripped.trim().to_string();
    }

    for prefix in ["Final text:", "Output:", "Rewritten text:"] {
        if text
            .get(..prefix.len())
            .map(|candidate| candidate.eq_ignore_ascii_case(prefix))
            .unwrap_or(false)
        {
            text = text[prefix.len()..].trim().to_string();
            break;
        }
    }

    if text.starts_with('"') && text.ends_with('"') && text.len() >= 2 {
        text = text[1..text.len() - 1].trim().to_string();
    }

    text
}

fn strip_tagged_section(input: &str, open: &str, close: &str) -> String {
    let mut output = input.to_string();

    while let Some(start) = output.find(open) {
        let close_start = match output[start + open.len()..].find(close) {
            Some(offset) => start + open.len() + offset,
            None => {
                output.truncate(start);
                break;
            }
        };
        output.replace_range(start..close_start + close.len(), "");
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rewrite_profile::ResolvedRewriteProfile;
    use crate::rewrite_protocol::{
        RewriteCandidate, RewriteCandidateKind, RewriteCorrectionPolicy, RewriteEditAction,
        RewriteEditHypothesis, RewriteEditHypothesisMatchSource, RewriteEditIntent,
        RewriteEditSignal, RewriteEditSignalKind, RewriteEditSignalScope,
        RewriteEditSignalStrength, RewriteIntentConfidence, RewritePolicyContext,
        RewriteReplacementScope, RewriteSessionBacktrackCandidate,
        RewriteSessionBacktrackCandidateKind, RewriteSessionEntry, RewriteSurfaceKind,
        RewriteTailShape, RewriteTranscript, RewriteTranscriptSegment, RewriteTypingContext,
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

        let messages_json =
            build_oaicompat_messages_json(&prompt).expect("encode oaicompat messages");
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
            prompt.contains(
                "Available rewrite candidates (advisory, not exhaustive in agentic mode)"
            )
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
        assert!(prompt.contains(
            "The candidate list is ordered from most likely to least likely by heuristics."
        ));
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

    #[test]
    fn sanitize_rewrite_output_strips_wrapper_and_label() {
        let cleaned = sanitize_rewrite_output("<output>\nFinal text: Hi there.\n</output>");
        assert_eq!(cleaned, "Hi there.");
    }

    #[test]
    fn sanitize_rewrite_output_strips_llama_stop_tokens() {
        let cleaned = sanitize_rewrite_output("Hi there.<|eot_id|>ignored");
        assert_eq!(cleaned, "Hi there.");
    }

    #[test]
    fn sanitize_rewrite_output_strips_think_blocks() {
        let cleaned = sanitize_rewrite_output("<think>reasoning</think>\nHi there.");
        assert_eq!(cleaned, "Hi there.");
    }
}

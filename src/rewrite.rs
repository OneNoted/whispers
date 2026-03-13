use std::num::NonZeroU32;
use std::path::Path;
use std::sync::OnceLock;

use encoding_rs::UTF_8;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::openai::OpenAIChatTemplateParams;
use llama_cpp_2::sampling::LlamaSampler;
use serde_json::json;

use crate::rewrite_profile::ResolvedRewriteProfile;
use crate::rewrite_profile::RewriteProfile;
use crate::rewrite_protocol::RewriteTranscript;

#[allow(dead_code)]
pub struct LocalRewriter {
    model: LlamaModel,
    chat_template: LlamaChatTemplate,
    profile: ResolvedRewriteProfile,
    max_tokens: usize,
    max_output_chars: usize,
}

#[allow(dead_code)]
static LLAMA_BACKEND: OnceLock<&'static LlamaBackend> = OnceLock::new();
#[allow(dead_code)]
static EXTERNAL_LLAMA_BACKEND: LlamaBackend = LlamaBackend {};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewritePrompt {
    pub system: String,
    pub user: String,
}

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

#[allow(dead_code)]
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

#[allow(dead_code)]
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

#[allow(dead_code)]
pub fn resolved_profile_for_cloud(profile: RewriteProfile) -> ResolvedRewriteProfile {
    match profile {
        RewriteProfile::Auto => ResolvedRewriteProfile::Generic,
        RewriteProfile::Generic => ResolvedRewriteProfile::Generic,
        RewriteProfile::Qwen => ResolvedRewriteProfile::Qwen,
        RewriteProfile::LlamaCompat => ResolvedRewriteProfile::LlamaCompat,
    }
}

fn build_system_instructions(
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

#[allow(dead_code)]
struct RewriteBehavior {
    top_k: i32,
    top_p: f32,
    temperature: f32,
}

#[allow(dead_code)]
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

fn rewrite_instructions(profile: ResolvedRewriteProfile) -> &'static str {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RewriteRoute {
    Fast,
    ResolvedCorrection,
    SessionCandidateAdjudication,
    CandidateAdjudication,
}

fn build_user_message(transcript: &RewriteTranscript) -> String {
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

fn rewrite_route(transcript: &RewriteTranscript) -> RewriteRoute {
    if has_session_backtrack_candidate(transcript) {
        RewriteRoute::SessionCandidateAdjudication
    } else if requires_candidate_adjudication(transcript) {
        RewriteRoute::CandidateAdjudication
    } else if transcript.correction_aware_text.trim() != transcript.raw_text.trim() {
        RewriteRoute::ResolvedCorrection
    } else {
        RewriteRoute::Fast
    }
}

fn requires_candidate_adjudication(transcript: &RewriteTranscript) -> bool {
    !transcript.edit_signals.is_empty() || !transcript.edit_hypotheses.is_empty()
}

fn has_strong_explicit_edit_cue(transcript: &RewriteTranscript) -> bool {
    transcript.edit_hypotheses.iter().any(|hypothesis| {
        hypothesis.strength == crate::rewrite_protocol::RewriteEditSignalStrength::Strong
            && matches!(
                hypothesis.match_source,
                crate::rewrite_protocol::RewriteEditHypothesisMatchSource::Exact
                    | crate::rewrite_protocol::RewriteEditHypothesisMatchSource::Alias
            )
    })
}

fn has_session_backtrack_candidate(transcript: &RewriteTranscript) -> bool {
    transcript.recommended_session_candidate.is_some()
        || !transcript.session_backtrack_candidates.is_empty()
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

fn has_policy_context(transcript: &RewriteTranscript) -> bool {
    transcript.policy_context.is_active()
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

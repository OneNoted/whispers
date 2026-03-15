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

#[cfg(any(test, feature = "local-rewrite"))]
use super::RewritePrompt;
#[cfg(feature = "local-rewrite")]
use super::prompt::build_prompt;
use super::routing::requires_candidate_adjudication;

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

        let rewritten = super::sanitize_rewrite_output(&output);
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
pub(super) fn build_oaicompat_messages_json(
    prompt: &RewritePrompt,
) -> std::result::Result<String, String> {
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
pub(super) fn effective_max_tokens(max_tokens: usize, transcript: &RewriteTranscript) -> usize {
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

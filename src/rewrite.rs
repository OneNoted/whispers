use std::num::NonZeroU32;
use std::path::Path;
use std::sync::OnceLock;

use encoding_rs::UTF_8;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

use crate::rewrite_protocol::RewriteTranscript;

pub struct LocalRewriter {
    model: LlamaModel,
    chat_template: LlamaChatTemplate,
    max_tokens: usize,
    max_output_chars: usize,
}

static LLAMA_BACKEND: OnceLock<&'static LlamaBackend> = OnceLock::new();
static EXTERNAL_LLAMA_BACKEND: LlamaBackend = LlamaBackend {};

impl LocalRewriter {
    pub fn new(
        model_path: &Path,
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
            max_tokens,
            max_output_chars,
        })
    }

    pub fn rewrite(&self, transcript: &RewriteTranscript) -> std::result::Result<String, String> {
        if transcript.raw_text.trim().is_empty() {
            return Ok(String::new());
        }

        let prompt = build_rewrite_prompt(&self.model, &self.chat_template, transcript)?;
        let prompt_tokens = self
            .model
            .str_to_token(&prompt, AddBos::Never)
            .map_err(|e| format!("failed to tokenize rewrite prompt: {e}"))?;

        let n_ctx_tokens = prompt_tokens
            .len()
            .saturating_add(self.max_tokens)
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
            LlamaSampler::top_k(40),
            LlamaSampler::top_p(0.95, 1),
            LlamaSampler::temp(0.15),
            LlamaSampler::greedy(),
        ]);
        sampler.accept_many(prompt_tokens.iter());

        let mut decoder = UTF_8.new_decoder_without_bom_handling();
        let mut output = String::new();
        let start_pos = i32::try_from(prompt_tokens.len()).unwrap_or(i32::MAX);

        for i in 0..self.max_tokens {
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

fn build_rewrite_prompt(
    model: &LlamaModel,
    chat_template: &LlamaChatTemplate,
    transcript: &RewriteTranscript,
) -> std::result::Result<String, String> {
    let messages = vec![
        LlamaChatMessage::new("system".into(), rewrite_instructions().to_string())
            .map_err(|e| format!("failed to build rewrite system message: {e}"))?,
        LlamaChatMessage::new("user".into(), build_user_message(transcript))
            .map_err(|e| format!("failed to build rewrite user message: {e}"))?,
    ];

    model
        .apply_chat_template(chat_template, &messages, true)
        .map_err(|e| format!("failed to apply rewrite chat template: {e}"))
}

fn rewrite_instructions() -> &'static str {
    "You clean up dictated speech into the final text the user meant to type. \
Return only the finished text. Do not explain anything. Remove obvious disfluencies when natural. \
Use the correction-aware transcript as the primary source of truth. The raw transcript may still contain \
spoken editing phrases or canceled wording. Never reintroduce text that was removed by an explicit spoken \
correction cue. Examples:\n\
- raw: Hello there. Scratch that. Hi.\n  correction-aware: Hi.\n  final: Hi.\n\
- raw: I'll bring cookies, scratch that, brownies.\n  correction-aware: I'll bring brownies.\n  final: I'll bring brownies."
}

fn build_user_message(transcript: &RewriteTranscript) -> String {
    let language = transcript.detected_language.as_deref().unwrap_or("unknown");
    let correction_aware = transcript.correction_aware_text.trim();
    let raw = transcript.raw_text.trim();

    if correction_aware != raw {
        return format!(
            "Language: {language}\n\
Self-corrections were already resolved before rewriting.\n\
Use only this correction-aware transcript as the source text:\n\
{correction_aware}\n\
Do not restore any canceled wording from earlier in the utterance.\n\
Final text:"
        );
    }

    let recent_segments = render_recent_segments(transcript);

    format!(
        "Language: {language}\n\
Correction-aware transcript:\n\
{correction_aware}\n\
Treat the correction-aware transcript as authoritative for explicit spoken edits.\n\
Use the raw transcript only to recover unclear words, never to restore canceled text.\n\
\
Recent segments:\n\
{recent_segments}\n\
Raw transcript:\n\
{raw}\n\
Final text:",
    )
}

fn render_recent_segments(transcript: &RewriteTranscript) -> String {
    let total_segments = transcript.segments.len();
    let start = total_segments.saturating_sub(8);
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

fn sanitize_rewrite_output(raw: &str) -> String {
    let mut text = raw.replace("\r\n", "\n");

    for stop in ["<|eot_id|>", "<|end_of_text|>", "</s>"] {
        if let Some(index) = text.find(stop) {
            text.truncate(index);
        }
    }

    if let Some(index) = text.find("</output>") {
        text.truncate(index);
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rewrite_protocol::{RewriteTranscript, RewriteTranscriptSegment};

    fn correction_transcript() -> RewriteTranscript {
        RewriteTranscript {
            raw_text: "Hi there, this is a test. Wait, no. Hi there.".into(),
            correction_aware_text: "Hi there.".into(),
            detected_language: Some("en".into()),
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
        }
    }

    #[test]
    fn instructions_cover_self_correction_examples() {
        let instructions = rewrite_instructions();
        assert!(instructions.contains("Return only the finished text"));
        assert!(instructions.contains("Never reintroduce text"));
        assert!(instructions.contains("scratch that, brownies"));
    }

    #[test]
    fn user_message_prefers_correction_aware_transcript_when_it_differs() {
        let prompt = build_user_message(&correction_transcript());
        assert!(prompt.contains("correction-aware transcript"));
        assert!(prompt.contains("Self-corrections were already resolved"));
        assert!(prompt.contains("Do not restore any canceled wording"));
        assert!(!prompt.contains("Recent segments"));
        assert!(!prompt.contains("Raw transcript"));
    }

    #[test]
    fn user_message_includes_recent_segments_when_correction_matches_raw() {
        let mut transcript = correction_transcript();
        transcript.correction_aware_text = transcript.raw_text.clone();

        let prompt = build_user_message(&transcript);
        assert!(prompt.contains("Correction-aware transcript"));
        assert!(prompt.contains("Recent segments"));
        assert!(prompt.contains("0-1200 ms"));
        assert!(prompt.contains("Raw transcript"));
        assert!(prompt.contains("Wait, no. Hi there."));
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
}

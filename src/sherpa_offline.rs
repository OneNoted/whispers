use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::Path;
use std::thread;

use libloading::Library;

use crate::config::Config;
use crate::{Error, Result, paths};

#[repr(C)]
struct FeatureConfig {
    sample_rate: c_int,
    feature_dim: c_int,
}

#[repr(C)]
struct HomophoneReplacerConfig {
    dict_dir: *const c_char,
    lexicon: *const c_char,
    rule_fsts: *const c_char,
}

#[repr(C)]
struct TransducerModelConfig {
    encoder: *const c_char,
    decoder: *const c_char,
    joiner: *const c_char,
}

#[repr(C)]
struct SingleModelConfig {
    model: *const c_char,
}

#[repr(C)]
struct WhisperModelConfig {
    encoder: *const c_char,
    decoder: *const c_char,
    language: *const c_char,
    task: *const c_char,
    tail_paddings: c_int,
    enable_token_timestamps: c_int,
    enable_segment_timestamps: c_int,
}

#[repr(C)]
struct CanaryModelConfig {
    encoder: *const c_char,
    decoder: *const c_char,
    src_lang: *const c_char,
    tgt_lang: *const c_char,
    use_pnc: c_int,
}

#[repr(C)]
struct CohereModelConfig {
    encoder: *const c_char,
    decoder: *const c_char,
    language: *const c_char,
    use_punct: c_int,
    use_itn: c_int,
}

#[repr(C)]
struct EncoderDecoderModelConfig {
    encoder: *const c_char,
    decoder: *const c_char,
}

#[repr(C)]
struct MoonshineModelConfig {
    preprocessor: *const c_char,
    encoder: *const c_char,
    uncached_decoder: *const c_char,
    cached_decoder: *const c_char,
    merged_decoder: *const c_char,
}

#[repr(C)]
struct LanguageModelConfig {
    model: *const c_char,
    scale: f32,
}

#[repr(C)]
struct SenseVoiceModelConfig {
    model: *const c_char,
    language: *const c_char,
    use_itn: c_int,
}

#[repr(C)]
struct FunAsrNanoModelConfig {
    encoder_adaptor: *const c_char,
    llm: *const c_char,
    embedding: *const c_char,
    tokenizer: *const c_char,
    system_prompt: *const c_char,
    user_prompt: *const c_char,
    max_new_tokens: c_int,
    temperature: f32,
    top_p: f32,
    seed: c_int,
    language: *const c_char,
    itn: c_int,
    hotwords: *const c_char,
}

#[repr(C)]
struct Qwen3AsrModelConfig {
    conv_frontend: *const c_char,
    encoder: *const c_char,
    decoder: *const c_char,
    tokenizer: *const c_char,
    max_total_len: c_int,
    max_new_tokens: c_int,
    temperature: f32,
    top_p: f32,
    seed: c_int,
    hotwords: *const c_char,
}

// This mirrors the pinned sherpa-onnx v1.13.4 C ABI. Unused model-family
// fields remain zeroed when the Parakeet recognizer is created.
#[repr(C)]
struct OfflineModelConfig {
    transducer: TransducerModelConfig,
    paraformer: SingleModelConfig,
    nemo_ctc: SingleModelConfig,
    whisper: WhisperModelConfig,
    tdnn: SingleModelConfig,
    tokens: *const c_char,
    num_threads: c_int,
    debug: c_int,
    provider: *const c_char,
    model_type: *const c_char,
    modeling_unit: *const c_char,
    bpe_vocab: *const c_char,
    telespeech_ctc: *const c_char,
    sense_voice: SenseVoiceModelConfig,
    moonshine: MoonshineModelConfig,
    fire_red_asr: EncoderDecoderModelConfig,
    dolphin: SingleModelConfig,
    zipformer_ctc: SingleModelConfig,
    canary: CanaryModelConfig,
    wenet_ctc: SingleModelConfig,
    omnilingual: SingleModelConfig,
    medasr: SingleModelConfig,
    funasr_nano: FunAsrNanoModelConfig,
    fire_red_asr_ctc: SingleModelConfig,
    qwen3_asr: Qwen3AsrModelConfig,
    cohere_transcribe: CohereModelConfig,
}

#[repr(C)]
struct OfflineRecognizerConfig {
    feat_config: FeatureConfig,
    model_config: OfflineModelConfig,
    lm_config: LanguageModelConfig,
    decoding_method: *const c_char,
    max_active_paths: c_int,
    hotwords_file: *const c_char,
    hotwords_score: f32,
    rule_fsts: *const c_char,
    rule_fars: *const c_char,
    blank_penalty: f32,
    hr: HomophoneReplacerConfig,
}

type CreateRecognizer = unsafe extern "C" fn(*const OfflineRecognizerConfig) -> *const c_void;
type DestroyRecognizer = unsafe extern "C" fn(*const c_void);
type CreateStream = unsafe extern "C" fn(*const c_void) -> *const c_void;
type DestroyStream = unsafe extern "C" fn(*const c_void);
type AcceptWaveform = unsafe extern "C" fn(*const c_void, c_int, *const f32, c_int);
type Decode = unsafe extern "C" fn(*const c_void, *const c_void);
type GetResultJson = unsafe extern "C" fn(*const c_void) -> *const c_char;
type DestroyResultJson = unsafe extern "C" fn(*const c_char);

struct SherpaApi {
    _library: Library,
    create_recognizer: CreateRecognizer,
    destroy_recognizer: DestroyRecognizer,
    create_stream: CreateStream,
    destroy_stream: DestroyStream,
    accept_waveform: AcceptWaveform,
    decode: Decode,
    get_result_json: GetResultJson,
    destroy_result_json: DestroyResultJson,
}

impl SherpaApi {
    unsafe fn load(path: &Path) -> Result<Self> {
        // SAFETY: The runtime pack is version-pinned and its symbols are checked below.
        let library = unsafe { Library::new(path) }.map_err(|error| {
            Error::message(format!("could not load {}: {error}", path.display()))
        })?;
        macro_rules! symbol {
            ($name:literal, $ty:ty) => {{
                // SAFETY: Symbol names and signatures match the pinned sherpa-onnx C API.
                let symbol = unsafe { library.get::<$ty>(concat!($name, "\0").as_bytes()) }
                    .map_err(|error| {
                        Error::message(format!("missing sherpa symbol {}: {error}", $name))
                    })?;
                *symbol
            }};
        }
        Ok(Self {
            create_recognizer: symbol!("SherpaOnnxCreateOfflineRecognizer", CreateRecognizer),
            destroy_recognizer: symbol!("SherpaOnnxDestroyOfflineRecognizer", DestroyRecognizer),
            create_stream: symbol!("SherpaOnnxCreateOfflineStream", CreateStream),
            destroy_stream: symbol!("SherpaOnnxDestroyOfflineStream", DestroyStream),
            accept_waveform: symbol!("SherpaOnnxAcceptWaveformOffline", AcceptWaveform),
            decode: symbol!("SherpaOnnxDecodeOfflineStream", Decode),
            get_result_json: symbol!("SherpaOnnxGetOfflineStreamResultAsJson", GetResultJson),
            destroy_result_json: symbol!(
                "SherpaOnnxDestroyOfflineStreamResultJson",
                DestroyResultJson
            ),
            _library: library,
        })
    }
}

pub(crate) struct OfflineRecognizer {
    api: SherpaApi,
    recognizer: *const c_void,
}

impl OfflineRecognizer {
    pub(crate) fn load(config: &Config) -> Result<Self> {
        let model_dir = paths::recognition_model_dir(&config.recognition.final_model)?;
        let encoder = c_path(&model_dir.join("encoder.int8.onnx"))?;
        let decoder = c_path(&model_dir.join("decoder.int8.onnx"))?;
        let joiner = c_path(&model_dir.join("joiner.int8.onnx"))?;
        let tokens = c_path(&model_dir.join("tokens.txt"))?;
        let provider = CString::new("cpu").map_err(|_| Error::message("invalid provider"))?;
        let model_type =
            CString::new("nemo_transducer").map_err(|_| Error::message("invalid model type"))?;
        let method =
            CString::new("greedy_search").map_err(|_| Error::message("invalid decoder"))?;

        // SAFETY: Every member of this C configuration is a pointer or numeric scalar;
        // zero is the documented default for unused model families and options.
        let mut native = unsafe { std::mem::zeroed::<OfflineRecognizerConfig>() };
        native.feat_config = FeatureConfig {
            sample_rate: 16_000,
            feature_dim: 80,
        };
        native.model_config.transducer = TransducerModelConfig {
            encoder: encoder.as_ptr(),
            decoder: decoder.as_ptr(),
            joiner: joiner.as_ptr(),
        };
        native.model_config.tokens = tokens.as_ptr();
        native.model_config.num_threads = c_int::try_from(
            thread::available_parallelism()
                .map_or(4, std::num::NonZeroUsize::get)
                .min(8),
        )
        .unwrap_or(4);
        native.model_config.provider = provider.as_ptr();
        native.model_config.model_type = model_type.as_ptr();
        native.decoding_method = method.as_ptr();
        native.max_active_paths = 4;

        // SAFETY: The pinned runtime ABI and all config pointers are valid for this call.
        let api = unsafe { SherpaApi::load(&paths::sherpa_library()?) }?;
        // SAFETY: `native` mirrors the v1.13.4 C structure and its strings live for the call.
        let recognizer = unsafe { (api.create_recognizer)(&raw const native) };
        if recognizer.is_null() {
            return Err(Error::message(
                "sherpa-onnx rejected the offline recognizer configuration",
            ));
        }
        Ok(Self { api, recognizer })
    }

    pub(crate) fn transcribe(&self, samples: &[f32]) -> Result<String> {
        let count = c_int::try_from(samples.len())
            .map_err(|_| Error::message("dictation contains too many samples"))?;
        // SAFETY: Recognizer was created by this API and remains alive.
        let stream = unsafe { (self.api.create_stream)(self.recognizer) };
        if stream.is_null() {
            return Err(Error::message(
                "sherpa-onnx could not create an offline stream",
            ));
        }
        // SAFETY: Stream is alive and samples points to `count` contiguous f32 values.
        unsafe { (self.api.accept_waveform)(stream, 16_000, samples.as_ptr(), count) };
        // SAFETY: Recognizer and stream are alive and belong to the same API.
        unsafe { (self.api.decode)(self.recognizer, stream) };
        // SAFETY: The decoded stream remains alive for this result call.
        let pointer = unsafe { (self.api.get_result_json)(stream) };
        let result = if pointer.is_null() {
            Err(Error::message("sherpa-onnx returned no offline result"))
        } else {
            // SAFETY: API returned a NUL-terminated JSON string valid until destroyed.
            let bytes = unsafe { CStr::from_ptr(pointer) }.to_bytes().to_vec();
            // SAFETY: Pointer was allocated by this API for this result call.
            unsafe { (self.api.destroy_result_json)(pointer) };
            match serde_json::from_slice::<serde_json::Value>(&bytes) {
                Ok(value) => {
                    let text = value
                        .get("text")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .trim()
                        .to_string();
                    if text.is_empty() {
                        Err(Error::message("offline recognition returned no text"))
                    } else {
                        Ok(text)
                    }
                }
                Err(error) => Err(error.into()),
            }
        };
        // SAFETY: Stream was created by this API and will not be used again.
        unsafe { (self.api.destroy_stream)(stream) };
        result
    }
}

impl Drop for OfflineRecognizer {
    fn drop(&mut self) {
        // SAFETY: Recognizer was created by this API and is destroyed exactly once.
        unsafe { (self.api.destroy_recognizer)(self.recognizer) };
    }
}

fn c_path(path: &Path) -> Result<CString> {
    CString::new(path.as_os_str().as_encoded_bytes())
        .map_err(|_| Error::message(format!("path contains a NUL byte: {}", path.display())))
}

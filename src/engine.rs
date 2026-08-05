use std::ffi::{CStr, CString, c_char, c_float, c_int, c_void};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Duration;

use libloading::Library;

use crate::config::Config;
use crate::desktop;
use crate::protocol::{
    ENGINE_PROTOCOL_VERSION, EngineRequest, EngineResponse, FrameKind, read_frame, read_json_frame,
    write_frame, write_json_frame,
};
use crate::sherpa_offline::OfflineRecognizer;
use crate::{Error, Result, paths};

const FINAL_TAIL_SAMPLE_COUNT: c_int = 4_800;

pub struct EngineClient {
    child: Child,
    input: BufWriter<ChildStdin>,
    responses: Receiver<EngineResponse>,
}

impl EngineClient {
    pub fn spawn(runtime: &str) -> Result<Self> {
        let executable = std::env::var_os("WHISPERS_ENGINE_HOST")
            .map(PathBuf::from)
            .map_or_else(std::env::current_exe, Ok)?;
        let mut child = Command::new(executable)
            .args(["__engine-host", runtime])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        let Some(input) = child.stdin.take() else {
            terminate_child(&mut child);
            return Err(Error::message("engine host stdin is unavailable"));
        };
        let Some(output) = child.stdout.take() else {
            terminate_child(&mut child);
            return Err(Error::message("engine host stdout is unavailable"));
        };
        let input = BufWriter::new(input);
        let mut output = BufReader::new(output);
        let (sender, responses) = mpsc::channel();
        thread::spawn(move || {
            while let Ok(response) = read_json_frame::<_, EngineResponse>(&mut output) {
                if let EngineResponse::Partial { text, .. } = &response {
                    desktop::show_preview(text);
                }
                if sender.send(response).is_err() {
                    break;
                }
            }
        });
        let mut client = Self {
            child,
            input,
            responses,
        };
        write_json_frame(
            &mut client.input,
            &EngineRequest::Hello {
                protocol_version: ENGINE_PROTOCOL_VERSION,
            },
        )?;
        let hello = client
            .responses
            .recv_timeout(Duration::from_secs(30))
            .map_err(|error| Error::message(format!("engine handshake failed: {error}")))?;
        match hello {
            EngineResponse::Hello {
                protocol_version, ..
            } if protocol_version == ENGINE_PROTOCOL_VERSION => {}
            EngineResponse::Hello {
                protocol_version, ..
            } => {
                return Err(Error::ProtocolVersion {
                    expected: ENGINE_PROTOCOL_VERSION,
                    actual: protocol_version,
                });
            }
            response => {
                return Err(Error::message(format!(
                    "engine host returned an invalid greeting: {response:?}"
                )));
            }
        }
        Ok(client)
    }

    pub fn start(&mut self, session_id: u64, language: &str) -> Result<()> {
        while self.responses.try_recv().is_ok() {}
        write_json_frame(
            &mut self.input,
            &EngineRequest::Start {
                session_id,
                sample_rate: 16_000,
                language: language.into(),
            },
        )
    }

    pub fn push_audio(&mut self, samples: &[f32]) -> Result<()> {
        let mut bytes = Vec::with_capacity(samples.len() * 4);
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        write_frame(&mut self.input, FrameKind::Audio, &bytes)
    }

    pub fn finish(&mut self) -> Result<String> {
        write_json_frame(&mut self.input, &EngineRequest::Finish)?;
        loop {
            match self.responses.recv_timeout(Duration::from_secs(30)) {
                Ok(EngineResponse::Final { text, .. }) => return Ok(text),
                Ok(EngineResponse::Error { message, .. }) => return Err(Error::message(message)),
                Ok(_) => {}
                Err(error) => {
                    return Err(Error::message(format!("engine response failed: {error}")));
                }
            }
        }
    }

    pub fn cancel(&mut self) -> Result<()> {
        write_json_frame(&mut self.input, &EngineRequest::Cancel)
    }

    pub fn is_healthy(&mut self) -> bool {
        self.child.try_wait().is_ok_and(|status| status.is_none())
    }
}

impl Drop for EngineClient {
    fn drop(&mut self) {
        terminate_child(&mut self.child);
    }
}

fn terminate_child(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

pub fn run_host(runtime: &str) -> Result<()> {
    if runtime != "sherpa-onnx" {
        return Err(Error::message(format!(
            "unsupported engine runtime: {runtime}"
        )));
    }
    let config = Config::load(None)?;
    let mut recognizer = SherpaRecognizer::load(&config)?;
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut input = stdin.lock();
    let mut output = stdout.lock();
    loop {
        let (kind, payload) = read_frame(&mut input)?;
        match kind {
            FrameKind::Audio => {
                if payload.len() % std::mem::size_of::<f32>() != 0 {
                    return Err(Error::message("audio frame is not aligned to f32 samples"));
                }
                recognizer.accept_audio(&payload, &mut output)?;
            }
            FrameKind::Control => {
                let request: EngineRequest = serde_json::from_slice(&payload)?;
                match request {
                    EngineRequest::Hello { protocol_version } => {
                        if protocol_version != ENGINE_PROTOCOL_VERSION {
                            return Err(Error::ProtocolVersion {
                                expected: ENGINE_PROTOCOL_VERSION,
                                actual: protocol_version,
                            });
                        }
                        write_json_frame(
                            &mut output,
                            &EngineResponse::Hello {
                                protocol_version: ENGINE_PROTOCOL_VERSION,
                                engine: "sherpa-onnx/nemotron-preview+parakeet-tdt-v3-final".into(),
                                capabilities: vec![
                                    "partials".into(),
                                    "punctuation".into(),
                                    "language_detection".into(),
                                ],
                            },
                        )?;
                    }
                    EngineRequest::Start { language, .. } => {
                        recognizer.start(&language)?;
                    }
                    EngineRequest::Finish => {
                        let text = recognizer.finish()?;
                        write_json_frame(
                            &mut output,
                            &EngineResponse::Final {
                                text,
                                language: None,
                                confidence: None,
                            },
                        )?;
                    }
                    EngineRequest::Cancel => recognizer.cancel(),
                }
            }
        }
    }
}

#[repr(C)]
struct OnlineTransducerModelConfig {
    encoder: *const c_char,
    decoder: *const c_char,
    joiner: *const c_char,
}

#[repr(C)]
struct OnlineParaformerModelConfig {
    encoder: *const c_char,
    decoder: *const c_char,
}

#[repr(C)]
struct OnlineCtcModelConfig {
    model: *const c_char,
}

#[repr(C)]
struct OnlineModelConfig {
    transducer: OnlineTransducerModelConfig,
    paraformer: OnlineParaformerModelConfig,
    zipformer2_ctc: OnlineCtcModelConfig,
    tokens: *const c_char,
    num_threads: c_int,
    provider: *const c_char,
    debug: c_int,
    model_type: *const c_char,
    modeling_unit: *const c_char,
    bpe_vocab: *const c_char,
    tokens_buf: *const c_char,
    tokens_buf_size: c_int,
    nemo_ctc: OnlineCtcModelConfig,
    t_one_ctc: OnlineCtcModelConfig,
}

#[repr(C)]
struct FeatureConfig {
    sample_rate: c_int,
    feature_dim: c_int,
}

#[repr(C)]
struct CtcFstDecoderConfig {
    graph: *const c_char,
    max_active: c_int,
}

#[repr(C)]
struct HomophoneReplacerConfig {
    dict_dir: *const c_char,
    lexicon: *const c_char,
    rule_fsts: *const c_char,
}

#[repr(C)]
struct OnlineRecognizerConfig {
    feat_config: FeatureConfig,
    model_config: OnlineModelConfig,
    decoding_method: *const c_char,
    max_active_paths: c_int,
    enable_endpoint: c_int,
    rule1_min_trailing_silence: c_float,
    rule2_min_trailing_silence: c_float,
    rule3_min_utterance_length: c_float,
    hotwords_file: *const c_char,
    hotwords_score: c_float,
    ctc_fst_decoder_config: CtcFstDecoderConfig,
    rule_fsts: *const c_char,
    rule_fars: *const c_char,
    blank_penalty: c_float,
    hotwords_buf: *const c_char,
    hotwords_buf_size: c_int,
    hr: HomophoneReplacerConfig,
}

type CreateRecognizer = unsafe extern "C" fn(*const OnlineRecognizerConfig) -> *const c_void;
type DestroyRecognizer = unsafe extern "C" fn(*const c_void);
type CreateStream = unsafe extern "C" fn(*const c_void) -> *const c_void;
type DestroyStream = unsafe extern "C" fn(*const c_void);
type SetStreamOption = unsafe extern "C" fn(*const c_void, *const c_char, *const c_char);
type AcceptWaveform = unsafe extern "C" fn(*const c_void, c_int, *const f32, c_int);
type IsReady = unsafe extern "C" fn(*const c_void, *const c_void) -> c_int;
type Decode = unsafe extern "C" fn(*const c_void, *const c_void);
type GetResultJson = unsafe extern "C" fn(*const c_void, *const c_void) -> *const c_char;
type DestroyResultJson = unsafe extern "C" fn(*const c_char);
type InputFinished = unsafe extern "C" fn(*const c_void);

struct SherpaApi {
    _library: Library,
    create_recognizer: CreateRecognizer,
    destroy_recognizer: DestroyRecognizer,
    create_stream: CreateStream,
    destroy_stream: DestroyStream,
    set_stream_option: SetStreamOption,
    accept_waveform: AcceptWaveform,
    is_ready: IsReady,
    decode: Decode,
    get_result_json: GetResultJson,
    destroy_result_json: DestroyResultJson,
    input_finished: InputFinished,
}

impl SherpaApi {
    unsafe fn load(path: &Path) -> Result<Self> {
        // SAFETY: The runtime pack is version-pinned and its symbols are validated below.
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
            create_recognizer: symbol!("SherpaOnnxCreateOnlineRecognizer", CreateRecognizer),
            destroy_recognizer: symbol!("SherpaOnnxDestroyOnlineRecognizer", DestroyRecognizer),
            create_stream: symbol!("SherpaOnnxCreateOnlineStream", CreateStream),
            destroy_stream: symbol!("SherpaOnnxDestroyOnlineStream", DestroyStream),
            set_stream_option: symbol!("SherpaOnnxOnlineStreamSetOption", SetStreamOption),
            accept_waveform: symbol!("SherpaOnnxOnlineStreamAcceptWaveform", AcceptWaveform),
            is_ready: symbol!("SherpaOnnxIsOnlineStreamReady", IsReady),
            decode: symbol!("SherpaOnnxDecodeOnlineStream", Decode),
            get_result_json: symbol!("SherpaOnnxGetOnlineStreamResultAsJson", GetResultJson),
            destroy_result_json: symbol!(
                "SherpaOnnxDestroyOnlineStreamResultJson",
                DestroyResultJson
            ),
            input_finished: symbol!("SherpaOnnxOnlineStreamInputFinished", InputFinished),
            _library: library,
        })
    }
}

struct SherpaRecognizer {
    api: SherpaApi,
    recognizer: *const c_void,
    final_recognizer: OfflineRecognizer,
    stream: Option<*const c_void>,
    audio: Vec<f32>,
    last_partial: String,
    revision: u64,
}

impl SherpaRecognizer {
    fn load(config: &Config) -> Result<Self> {
        let final_recognizer = OfflineRecognizer::load(config)?;
        let library_path = paths::sherpa_library()?;
        let model_dir = paths::recognition_model_dir(&config.recognition.model)?;
        let encoder = c_path(&model_dir.join("encoder.int8.onnx"))?;
        let decoder = c_path(&model_dir.join("decoder.int8.onnx"))?;
        let joiner = c_path(&model_dir.join("joiner.int8.onnx"))?;
        let tokens = c_path(&model_dir.join("tokens.txt"))?;
        let provider = CString::new("cpu").map_err(|_| Error::message("invalid provider"))?;
        let method =
            CString::new("greedy_search").map_err(|_| Error::message("invalid decoder"))?;
        let null = std::ptr::null();
        let threads = c_int::try_from(
            thread::available_parallelism()
                .map_or(2, std::num::NonZeroUsize::get)
                .min(4),
        )
        .unwrap_or(2);
        let native = OnlineRecognizerConfig {
            feat_config: FeatureConfig {
                sample_rate: 16_000,
                feature_dim: 80,
            },
            model_config: OnlineModelConfig {
                transducer: OnlineTransducerModelConfig {
                    encoder: encoder.as_ptr(),
                    decoder: decoder.as_ptr(),
                    joiner: joiner.as_ptr(),
                },
                paraformer: OnlineParaformerModelConfig {
                    encoder: null,
                    decoder: null,
                },
                zipformer2_ctc: OnlineCtcModelConfig { model: null },
                tokens: tokens.as_ptr(),
                num_threads: threads,
                provider: provider.as_ptr(),
                debug: 0,
                model_type: null,
                modeling_unit: null,
                bpe_vocab: null,
                tokens_buf: null,
                tokens_buf_size: 0,
                nemo_ctc: OnlineCtcModelConfig { model: null },
                t_one_ctc: OnlineCtcModelConfig { model: null },
            },
            decoding_method: method.as_ptr(),
            max_active_paths: 4,
            enable_endpoint: 0,
            rule1_min_trailing_silence: 0.0,
            rule2_min_trailing_silence: 0.0,
            rule3_min_utterance_length: 0.0,
            hotwords_file: null,
            hotwords_score: 1.5,
            ctc_fst_decoder_config: CtcFstDecoderConfig {
                graph: null,
                max_active: 0,
            },
            rule_fsts: null,
            rule_fars: null,
            blank_penalty: 0.0,
            hotwords_buf: null,
            hotwords_buf_size: 0,
            hr: HomophoneReplacerConfig {
                dict_dir: null,
                lexicon: null,
                rule_fsts: null,
            },
        };
        // SAFETY: The pinned runtime ABI and all config pointers are valid for this call.
        let api = unsafe { SherpaApi::load(&library_path) }?;
        // SAFETY: `native` is a complete C-compatible config whose strings remain alive for the call.
        let recognizer = unsafe { (api.create_recognizer)(&raw const native) };
        if recognizer.is_null() {
            return Err(Error::message(
                "sherpa-onnx rejected the recognizer configuration",
            ));
        }
        Ok(Self {
            api,
            recognizer,
            final_recognizer,
            stream: None,
            audio: Vec::new(),
            last_partial: String::new(),
            revision: 0,
        })
    }

    fn start(&mut self, language: &str) -> Result<()> {
        self.cancel();
        // SAFETY: Recognizer was created by this API and is alive.
        let stream = unsafe { (self.api.create_stream)(self.recognizer) };
        if stream.is_null() {
            return Err(Error::message("sherpa-onnx could not create a stream"));
        }
        let key = CString::new("language").map_err(|_| Error::message("invalid option name"))?;
        let language = CString::new(language.trim())
            .map_err(|_| Error::message("language contains an invalid null byte"))?;
        // SAFETY: The stream is alive and the runtime copies the option strings.
        unsafe { (self.api.set_stream_option)(stream, key.as_ptr(), language.as_ptr()) };
        self.stream = Some(stream);
        self.audio.clear();
        self.last_partial.clear();
        self.revision = 0;
        Ok(())
    }

    fn accept_audio(&mut self, payload: &[u8], output: &mut impl std::io::Write) -> Result<()> {
        let Some(stream) = self.stream else {
            return Err(Error::message(
                "received audio before starting an engine stream",
            ));
        };
        let samples = payload
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect::<Vec<_>>();
        self.audio.extend_from_slice(&samples);
        let count = c_int::try_from(samples.len())
            .map_err(|_| Error::message("audio frame contains too many samples"))?;
        // SAFETY: Stream is alive and samples points to `count` contiguous f32 values.
        unsafe { (self.api.accept_waveform)(stream, 16_000, samples.as_ptr(), count) };
        self.decode_ready(stream);
        let text = self.result(stream)?;
        if text != self.last_partial {
            self.last_partial.clone_from(&text);
            self.revision += 1;
            write_json_frame(
                output,
                &EngineResponse::Partial {
                    revision: self.revision,
                    text,
                },
            )?;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<String> {
        let stream = self
            .stream
            .take()
            .ok_or_else(|| Error::message("engine stream is not active"))?;
        let final_result = self.final_recognizer.transcribe(&self.audio);
        self.audio.clear();
        match final_result {
            Ok(text) => {
                // SAFETY: Stream was created by this API and will not be used again.
                unsafe { (self.api.destroy_stream)(stream) };
                return Ok(text);
            }
            Err(error) => {
                eprintln!("whispers: offline ASR failed, using streaming result: {error}");
            }
        }
        // Nemotron's streaming export requires virtual trailing silence to
        // flush its final chunk. This is decoded immediately; it is not a
        // user-visible recording delay.
        let tail = vec![0.0_f32; 4_800];
        // SAFETY: Stream is alive and `tail` contains 4800 contiguous f32 samples.
        unsafe {
            (self.api.accept_waveform)(stream, 16_000, tail.as_ptr(), FINAL_TAIL_SAMPLE_COUNT);
        }
        // SAFETY: Stream is alive and owned by this recognizer.
        unsafe { (self.api.input_finished)(stream) };
        self.decode_ready(stream);
        let result = self.result(stream);
        // SAFETY: Stream was created by this API and will not be used again.
        unsafe { (self.api.destroy_stream)(stream) };
        result
    }

    fn cancel(&mut self) {
        self.audio.clear();
        if let Some(stream) = self.stream.take() {
            // SAFETY: Stream was created by this API and will not be used again.
            unsafe { (self.api.destroy_stream)(stream) };
        }
    }

    fn decode_ready(&self, stream: *const c_void) {
        // SAFETY: Recognizer and stream are alive and belong to the same API.
        while unsafe { (self.api.is_ready)(self.recognizer, stream) } != 0 {
            // SAFETY: `is_ready` confirmed this stream can be decoded.
            unsafe { (self.api.decode)(self.recognizer, stream) };
        }
    }

    fn result(&self, stream: *const c_void) -> Result<String> {
        // SAFETY: Recognizer and stream are alive.
        let pointer = unsafe { (self.api.get_result_json)(self.recognizer, stream) };
        if pointer.is_null() {
            return Err(Error::message("sherpa-onnx returned no recognition result"));
        }
        // SAFETY: API returned a valid NUL-terminated JSON string until it is destroyed.
        let bytes = unsafe { CStr::from_ptr(pointer) }.to_bytes().to_vec();
        // SAFETY: Pointer was allocated by this API for this result call.
        unsafe { (self.api.destroy_result_json)(pointer) };
        let value: serde_json::Value = serde_json::from_slice(&bytes)?;
        Ok(value
            .get("text")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string())
    }
}

impl Drop for SherpaRecognizer {
    fn drop(&mut self) {
        self.cancel();
        // SAFETY: Recognizer was created by this API and is destroyed exactly once.
        unsafe { (self.api.destroy_recognizer)(self.recognizer) };
    }
}

fn c_path(path: &Path) -> Result<CString> {
    CString::new(path.as_os_str().as_encoded_bytes())
        .map_err(|_| Error::message(format!("path contains a NUL byte: {}", path.display())))
}

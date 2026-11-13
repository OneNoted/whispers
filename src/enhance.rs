use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::config::EnhancementConfig;
use crate::{Error, Result, paths};

const SYSTEM_PROMPT: &str = "Convert this speech transcript into the text the speaker intended to type. Correct obvious recognition mistakes, punctuation, and light grammar. Remove obvious verbal fillers such as um and uh, accidental repetitions, and abandoned false starts. Remove phrases such as like and you know only when they are clearly fillers. Do not paraphrase, replace valid word choices for style, or flatten deliberate informality. Preserve meaning, names, numbers, code identifiers, symbols, URLs, and line breaks. If a correction is uncertain, keep the original wording. Return only the corrected text without quotes, commentary, or markdown.";

pub struct Enhancer {
    config: EnhancementConfig,
    server: Mutex<Option<LlamaServer>>,
}

impl Enhancer {
    pub fn new(config: EnhancementConfig) -> Arc<Self> {
        Arc::new(Self {
            config,
            server: Mutex::new(None),
        })
    }

    pub fn prewarm(self: &Arc<Self>) {
        if !self.config.enabled {
            return;
        }
        let enhancer = Arc::clone(self);
        thread::spawn(move || {
            // Load the model and prime llama.cpp's prompt cache before the first
            // dictation reaches enhancement.
            let _ = enhancer.enhance("Ready.");
        });
    }

    pub fn enhance(&self, input: &str) -> Result<String> {
        if !self.config.enabled || input.trim().is_empty() {
            return Ok(input.to_string());
        }
        let request = CompletionRequest {
            model: "local",
            messages: [
                Message {
                    role: "system",
                    content: SYSTEM_PROMPT,
                },
                Message {
                    role: "user",
                    content: input,
                },
            ],
            temperature: 0.0,
            max_tokens: output_token_budget(input),
            stream: false,
        };
        let body = serde_json::to_vec(&request)?;
        let mut server = self.lock_server()?;
        if server.as_mut().is_none_or(|server| !server.is_healthy()) {
            *server = Some(LlamaServer::start(&self.config)?);
        }
        let timeout = Duration::from_millis(self.config.timeout_ms.max(100));
        let response = server
            .as_mut()
            .ok_or_else(|| Error::message("enhancement server was not started"))?
            .post("/v1/chat/completions", &body, timeout);
        let body = match response {
            Ok(body) => body,
            Err(error) => {
                *server = None;
                return Err(error);
            }
        };
        let completion: CompletionResponse = serde_json::from_slice(&body)?;
        let output = completion
            .choices
            .first()
            .map(|choice| choice.message.content.trim())
            .filter(|output| !output.is_empty())
            .ok_or_else(|| Error::message("enhancement returned no text"))?;
        validate_output(input, output)
    }

    fn lock_server(&self) -> Result<std::sync::MutexGuard<'_, Option<LlamaServer>>> {
        self.server
            .lock()
            .map_err(|_| Error::message("enhancement server lock is poisoned"))
    }
}

struct LlamaServer {
    child: Child,
    port: u16,
}

impl LlamaServer {
    fn start(config: &EnhancementConfig) -> Result<Self> {
        let model = paths::enhancement_model(&config.model)?;
        if !model.is_file() {
            return Err(Error::message(format!(
                "enhancement model is missing: {}",
                model.display()
            )));
        }
        let command_path = if config.command.trim().is_empty() || config.command == "managed" {
            paths::llama_server()?
        } else {
            PathBuf::from(&config.command)
        };
        let port = reserve_port()?;
        let threads = thread::available_parallelism()
            .map_or(2, std::num::NonZeroUsize::get)
            .clamp(2, 8);
        let mut command = Command::new(&command_path);
        command
            .arg("--model")
            .arg(&model)
            .args(["--host", "127.0.0.1", "--port"])
            .arg(port.to_string())
            .args(["--ctx-size", "2048", "--parallel", "1"])
            .args(["--threads", &threads.to_string()])
            .args(["--reasoning", "off", "--no-webui", "--log-disable"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        if command_path == paths::llama_server()? {
            if let Some(parent) = command_path.parent() {
                command.env("LD_LIBRARY_PATH", parent);
            }
        }
        let child = command.spawn().map_err(|error| {
            Error::message(format!(
                "could not start enhancement runtime {}: {error}",
                command_path.display()
            ))
        })?;
        let mut server = Self { child, port };
        let deadline = Instant::now() + Duration::from_secs(60);
        while Instant::now() < deadline {
            if server.child.try_wait()?.is_some() {
                return Err(Error::message("enhancement runtime exited during startup"));
            }
            if server.health_request() {
                return Ok(server);
            }
            thread::sleep(Duration::from_millis(50));
        }
        Err(Error::message("enhancement runtime did not become ready"))
    }

    fn is_healthy(&mut self) -> bool {
        self.child.try_wait().is_ok_and(|status| status.is_none()) && self.health_request()
    }

    fn health_request(&self) -> bool {
        http_request(self.port, "GET", "/health", &[], Duration::from_millis(100)).is_ok()
    }

    fn post(&self, path: &str, body: &[u8], timeout: Duration) -> Result<Vec<u8>> {
        http_request(self.port, "POST", path, body, timeout)
    }
}

impl Drop for LlamaServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[derive(Serialize)]
struct CompletionRequest<'a> {
    model: &'static str,
    messages: [Message<'a>; 2],
    temperature: f32,
    max_tokens: usize,
    stream: bool,
}

#[derive(Serialize)]
struct Message<'a> {
    role: &'static str,
    content: &'a str,
}

#[derive(Deserialize)]
struct CompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: CompletionMessage,
}

#[derive(Deserialize)]
struct CompletionMessage {
    content: String,
}

fn reserve_port() -> Result<u16> {
    let listener = TcpListener::bind(("127.0.0.1", 0))?;
    Ok(listener.local_addr()?.port())
}

fn http_request(
    port: u16,
    method: &str,
    path: &str,
    body: &[u8],
    timeout: Duration,
) -> Result<Vec<u8>> {
    let address = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let mut stream = TcpStream::connect_timeout(&address, timeout)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)?;
    stream.flush()?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response)?;
    parse_http_response(&response)
}

fn parse_http_response(response: &[u8]) -> Result<Vec<u8>> {
    let separator = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| Error::message("enhancement runtime returned an invalid HTTP response"))?;
    let headers = String::from_utf8_lossy(&response[..separator]);
    let status = headers
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|status| status.parse::<u16>().ok())
        .ok_or_else(|| Error::message("enhancement runtime omitted the HTTP status"))?;
    let body = &response[separator + 4..];
    if !(200..300).contains(&status) {
        return Err(Error::message(format!(
            "enhancement runtime returned HTTP {status}: {}",
            String::from_utf8_lossy(body).trim()
        )));
    }
    if headers.lines().any(|line| {
        line.split_once(':').is_some_and(|(name, value)| {
            name.eq_ignore_ascii_case("transfer-encoding")
                && value.trim().eq_ignore_ascii_case("chunked")
        })
    }) {
        decode_chunked(body)
    } else {
        Ok(body.to_vec())
    }
}

fn decode_chunked(mut input: &[u8]) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    loop {
        let line_end = input
            .windows(2)
            .position(|window| window == b"\r\n")
            .ok_or_else(|| Error::message("invalid chunked HTTP response"))?;
        let size = usize::from_str_radix(
            String::from_utf8_lossy(&input[..line_end])
                .split(';')
                .next()
                .unwrap_or_default()
                .trim(),
            16,
        )
        .map_err(|_| Error::message("invalid HTTP chunk size"))?;
        input = &input[line_end + 2..];
        if size == 0 {
            return Ok(output);
        }
        if input.len() < size + 2 || &input[size..size + 2] != b"\r\n" {
            return Err(Error::message("truncated chunked HTTP response"));
        }
        output.extend_from_slice(&input[..size]);
        input = &input[size + 2..];
    }
}

fn validate_output(input: &str, output: &str) -> Result<String> {
    let output = strip_outer_quotes(output.trim());
    if output.is_empty() || output.contains('\0') {
        return Err(Error::message("enhancement returned invalid text"));
    }
    let maximum = input.chars().count().saturating_mul(2).saturating_add(128);
    if output.chars().count() > maximum {
        return Err(Error::message(
            "enhancement expanded the transcript too much",
        ));
    }
    for symbol in ['/', '\\', '@', '#', '|', '_', '&', '*', '+', '=', '-'] {
        if input.matches(symbol).count() > output.matches(symbol).count() {
            return Err(Error::message(format!(
                "enhancement removed protected symbol {symbol}"
            )));
        }
    }
    if input.matches('\n').count() != output.matches('\n').count() {
        return Err(Error::message(
            "enhancement changed protected line structure",
        ));
    }
    let (changes, input_words) = lexical_edit_distance(input, output);
    let change_budget = if input_words <= 1 {
        0
    } else {
        input_words.div_ceil(4)
    };
    if changes > change_budget {
        return Err(Error::message(format!(
            "enhancement changed too many words ({changes}, limit {change_budget})"
        )));
    }
    for number in digit_runs(input) {
        if !output.contains(number) {
            return Err(Error::message(format!(
                "enhancement changed protected number {number}"
            )));
        }
    }
    Ok(output.to_string())
}

fn lexical_edit_distance(input: &str, output: &str) -> (usize, usize) {
    let input = lexical_words(input);
    let output = lexical_words(output);
    let mut previous = (0..=output.len()).collect::<Vec<_>>();
    let mut current = vec![0; output.len() + 1];
    for (input_index, input_word) in input.iter().enumerate() {
        current[0] = input_index + 1;
        for (output_index, output_word) in output.iter().enumerate() {
            let substitution = usize::from(input_word != output_word);
            current[output_index + 1] = (previous[output_index + 1] + 1)
                .min(current[output_index] + 1)
                .min(previous[output_index] + substitution);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    (previous[output.len()], input.len())
}

fn lexical_words(input: &str) -> Vec<String> {
    input
        .split_whitespace()
        .map(|word| word.trim_matches(|character: char| !character.is_alphanumeric()))
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn strip_outer_quotes(input: &str) -> &str {
    if input.len() >= 2
        && ((input.starts_with('"') && input.ends_with('"'))
            || (input.starts_with('\'') && input.ends_with('\'')))
    {
        &input[1..input.len() - 1]
    } else {
        input
    }
}

fn digit_runs(input: &str) -> impl Iterator<Item = &str> {
    input
        .split(|character: char| !character.is_ascii_digit() && character != '.')
        .filter(|value| value.chars().any(|character| character.is_ascii_digit()))
}

fn output_token_budget(input: &str) -> usize {
    input
        .split_whitespace()
        .count()
        .saturating_mul(3)
        .clamp(32, 512)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_conservative_rewrites() {
        assert_eq!(
            validate_output(
                "Use this/that in serde_json 3.2",
                "Use this/that in serde_json 3.2."
            )
            .ok()
            .as_deref(),
            Some("Use this/that in serde_json 3.2.")
        );
        assert!(validate_output("Use this/that", "Use this or that").is_err());
        assert!(validate_output("version 3.2", "version 4.0").is_err());
        assert!(validate_output("x+y", "x plus y").is_err());
        assert!(validate_output("first\nsecond", "first second").is_err());
        assert!(validate_output("accurate", "Accura").is_err());
        assert_eq!(
            validate_output("I agre", "I agree").ok().as_deref(),
            Some("I agree")
        );
        assert_eq!(
            validate_output(
                "Okay uh I think we should like probably ship this now",
                "Okay, I think we should probably ship this now."
            )
            .ok()
            .as_deref(),
            Some("Okay, I think we should probably ship this now.")
        );
        assert_eq!(
            validate_output("that result is accurate", "That result is accurate.")
                .ok()
                .as_deref(),
            Some("That result is accurate.")
        );
        assert!(
            validate_output(
                "keep every ordinary word in this sentence exactly as spoken",
                "replace most of the sentence with a different polished thought"
            )
            .is_err()
        );
    }

    #[test]
    fn counts_only_lexical_changes() {
        assert_eq!(
            lexical_edit_distance(
                "For example it said Accura instead of accurate",
                "For example, it said ‘Accura’ instead of ‘accurate’."
            ),
            (0, 8)
        );
        assert_eq!(lexical_edit_distance("one two three", "one four"), (2, 3));
    }

    #[test]
    fn parses_fixed_and_chunked_http() {
        assert_eq!(
            parse_http_response(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}").ok(),
            Some(b"{}".to_vec())
        );
        assert_eq!(
            parse_http_response(
                b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\n{}\r\n0\r\n\r\n"
            )
            .ok(),
            Some(b"{}".to_vec())
        );
    }
}

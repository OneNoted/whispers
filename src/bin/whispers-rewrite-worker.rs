#[path = "../branding.rs"]
mod branding;
#[path = "../rewrite.rs"]
mod rewrite;
#[path = "../rewrite_protocol.rs"]
mod rewrite_protocol;

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use clap::Parser;

use crate::rewrite::LocalRewriter;
use crate::rewrite_protocol::{WorkerRequest, WorkerResponse};

#[derive(Parser, Debug)]
#[command(name = branding::REWRITE_WORKER_BINARY)]
struct Cli {
    #[arg(long)]
    model_path: PathBuf,

    #[arg(long, default_value_t = 256)]
    max_tokens: usize,

    #[arg(long, default_value_t = 1200)]
    max_output_chars: usize,
}

fn main() {
    let cli = Cli::parse();

    let rewriter = match LocalRewriter::new(&cli.model_path, cli.max_tokens, cli.max_output_chars) {
        Ok(rewriter) => rewriter,
        Err(message) => {
            let _ = emit(&WorkerResponse::Error { message });
            std::process::exit(1);
        }
    };

    if emit(&WorkerResponse::Ready).is_err() {
        std::process::exit(1);
    }

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(_) => break,
        };

        let request = match serde_json::from_str::<WorkerRequest>(&line) {
            Ok(request) => request,
            Err(err) => {
                let _ = emit(&WorkerResponse::Error {
                    message: format!("invalid rewrite request: {err}"),
                });
                continue;
            }
        };

        let response = match request {
            WorkerRequest::Rewrite { transcript } => match rewriter.rewrite(&transcript) {
                Ok(text) => WorkerResponse::Result { text },
                Err(message) => WorkerResponse::Error { message },
            },
        };

        if emit(&response).is_err() {
            break;
        }
    }
}

fn emit(response: &WorkerResponse) -> io::Result<()> {
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    serde_json::to_writer(&mut handle, response)?;
    handle.write_all(b"\n")?;
    handle.flush()
}

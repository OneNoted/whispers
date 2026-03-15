use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use whispers::branding;
use whispers::rewrite::LocalRewriter;
use whispers::rewrite_profile::ResolvedRewriteProfile;
use whispers::rewrite_protocol::{WorkerRequest, WorkerResponse};

#[derive(Parser, Debug)]
#[command(name = branding::REWRITE_WORKER_BINARY)]
struct Cli {
    #[arg(long)]
    model_path: PathBuf,

    #[arg(long)]
    socket_path: PathBuf,

    #[arg(long, value_enum, default_value_t = ResolvedRewriteProfile::Generic)]
    profile: ResolvedRewriteProfile,

    #[arg(long, default_value_t = 256)]
    max_tokens: usize,

    #[arg(long, default_value_t = 1200)]
    max_output_chars: usize,

    #[arg(long, default_value_t = 120000)]
    idle_timeout_ms: u64,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let rewriter = match LocalRewriter::new(
        &cli.model_path,
        cli.profile,
        cli.max_tokens,
        cli.max_output_chars,
    ) {
        Ok(rewriter) => rewriter,
        Err(_) => std::process::exit(1),
    };

    if let Some(parent) = cli.socket_path.parent() {
        if tokio::fs::create_dir_all(parent).await.is_err() {
            std::process::exit(1);
        }
    }

    if cli.socket_path.exists() {
        let _ = tokio::fs::remove_file(&cli.socket_path).await;
    }

    let listener = match UnixListener::bind(&cli.socket_path) {
        Ok(listener) => listener,
        Err(_) => std::process::exit(1),
    };

    loop {
        let accepted = if cli.idle_timeout_ms == 0 {
            listener.accept().await
        } else {
            match tokio::time::timeout(
                Duration::from_millis(cli.idle_timeout_ms),
                listener.accept(),
            )
            .await
            {
                Ok(result) => result,
                Err(_) => break,
            }
        };
        let (stream, _) = match accepted {
            Ok(connection) => connection,
            Err(_) => break,
        };

        if handle_connection(stream, &rewriter).await.is_err() {
            continue;
        }
    }

    let _ = tokio::fs::remove_file(&cli.socket_path).await;
}

async fn handle_connection(stream: UnixStream, rewriter: &LocalRewriter) -> Result<(), ()> {
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    let mut line = String::new();
    if reader.read_line(&mut line).await.map_err(|_| ())? == 0 {
        return Err(());
    }

    let request = serde_json::from_str::<WorkerRequest>(&line).map_err(|_| ())?;
    let response = match request {
        WorkerRequest::Rewrite {
            transcript,
            custom_instructions,
        } => {
            match rewriter.rewrite_with_instructions(&transcript, custom_instructions.as_deref()) {
                Ok(text) => WorkerResponse::Result { text },
                Err(message) => WorkerResponse::Error { message },
            }
        }
    };

    let mut payload = serde_json::to_vec(&response).map_err(|_| ())?;
    payload.push(b'\n');
    writer.write_all(&payload).await.map_err(|_| ())?;
    writer.flush().await.map_err(|_| ())?;
    Ok(())
}

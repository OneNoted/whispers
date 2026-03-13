use std::path::{Path, PathBuf};

use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use tokio::io::AsyncWriteExt;

use crate::config::{self, data_dir, resolve_config_path, update_config_model_path};
use crate::error::{Result, WhsprError};

const MODEL_BASE_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

pub struct ModelInfo {
    pub name: &'static str,
    pub filename: &'static str,
    pub size: &'static str,
    pub description: &'static str,
}

pub const MODELS: &[ModelInfo] = &[
    ModelInfo {
        name: "large-v3-turbo",
        filename: "ggml-large-v3-turbo.bin",
        size: "1.6 GB",
        description: "Best balance of speed and accuracy (recommended)",
    },
    ModelInfo {
        name: "large-v3-turbo-q5_0",
        filename: "ggml-large-v3-turbo-q5_0.bin",
        size: "574 MB",
        description: "Quantized turbo, smaller and slightly less accurate",
    },
    ModelInfo {
        name: "large-v3",
        filename: "ggml-large-v3.bin",
        size: "3.1 GB",
        description: "Most accurate, significantly slower",
    },
    ModelInfo {
        name: "large-v3-q5_0",
        filename: "ggml-large-v3-q5_0.bin",
        size: "1.1 GB",
        description: "Quantized large, good accuracy/size tradeoff",
    },
    ModelInfo {
        name: "medium",
        filename: "ggml-medium.bin",
        size: "1.5 GB",
        description: "Medium model",
    },
    ModelInfo {
        name: "medium.en",
        filename: "ggml-medium.en.bin",
        size: "1.5 GB",
        description: "Medium model, English only",
    },
    ModelInfo {
        name: "small",
        filename: "ggml-small.bin",
        size: "488 MB",
        description: "Small model, fast",
    },
    ModelInfo {
        name: "small.en",
        filename: "ggml-small.en.bin",
        size: "488 MB",
        description: "Small model, English only",
    },
    ModelInfo {
        name: "base",
        filename: "ggml-base.bin",
        size: "148 MB",
        description: "Base model, very fast",
    },
    ModelInfo {
        name: "base.en",
        filename: "ggml-base.en.bin",
        size: "148 MB",
        description: "Base model, English only",
    },
    ModelInfo {
        name: "tiny",
        filename: "ggml-tiny.bin",
        size: "78 MB",
        description: "Tiny model, fastest, least accurate",
    },
    ModelInfo {
        name: "tiny.en",
        filename: "ggml-tiny.en.bin",
        size: "78 MB",
        description: "Tiny model, English only",
    },
];

pub fn find_model(name: &str) -> Option<&'static ModelInfo> {
    MODELS.iter().find(|m| m.name == name)
}

fn model_path(filename: &str) -> PathBuf {
    data_dir().join(filename)
}

fn path_for_config(path: &std::path::Path, home: Option<&std::path::Path>) -> String {
    if let Some(home_path) = home {
        if let Ok(stripped) = path.strip_prefix(home_path) {
            return format!("~/{}", stripped.display());
        }
    }
    path.display().to_string()
}

pub fn model_path_for_config(filename: &str) -> String {
    let path = model_path(filename);
    let home = std::env::var("HOME").ok().map(PathBuf::from);
    path_for_config(&path, home.as_deref())
}

fn active_model_path(config_path_override: Option<&Path>) -> Option<String> {
    let config_path = resolve_config_path(config_path_override);
    if !config_path.exists() {
        return None;
    }
    let contents = std::fs::read_to_string(&config_path).ok()?;
    let config: config::Config = toml::from_str(&contents).ok()?;
    Some(config.whisper.model_path)
}

fn model_status(info: &ModelInfo, active_resolved: Option<&std::path::Path>) -> &'static str {
    let path = model_path(info.filename);
    let is_active = active_resolved == Some(path.as_path());
    let is_local = path.exists();

    match (is_active, is_local) {
        (true, _) => "active",
        (_, true) => "local",
        _ => "remote",
    }
}

pub fn list_models(config_path_override: Option<&Path>) {
    tracing::debug!("listing models with config override: {config_path_override:?}");
    let active_resolved = active_model_path(config_path_override)
        .map(|p| std::path::PathBuf::from(config::expand_tilde(&p)));
    println!(
        "{:<22} {:>8}  {:<8}  DESCRIPTION",
        "MODEL", "SIZE", "STATUS"
    );
    println!("{}", "-".repeat(80));
    for m in MODELS {
        let status = model_status(m, active_resolved.as_deref());
        let marker = match status {
            "active" => "* ",
            _ => "  ",
        };
        println!(
            "{}{:<20} {:>8}  {:<8}  {}",
            marker, m.name, m.size, status, m.description
        );
    }
}

fn validated_existing_len(existing_len: u64, status: reqwest::StatusCode) -> Result<u64> {
    if existing_len > 0 {
        match status {
            reqwest::StatusCode::PARTIAL_CONTENT => Ok(existing_len),
            reqwest::StatusCode::OK => Ok(0),
            _ => Err(WhsprError::Download(format!(
                "download failed with HTTP {}",
                status
            ))),
        }
    } else if status.is_success() {
        Ok(0)
    } else {
        Err(WhsprError::Download(format!(
            "download failed with HTTP {}",
            status
        )))
    }
}

pub async fn download_model(name: &str) -> Result<PathBuf> {
    download_model_from_base(name, MODEL_BASE_URL).await
}

pub(crate) async fn download_model_from_base(name: &str, base_url: &str) -> Result<PathBuf> {
    let info = find_model(name).ok_or_else(|| {
        let available: Vec<&str> = MODELS.iter().map(|m| m.name).collect();
        WhsprError::Download(format!(
            "unknown model '{}'. Available: {}",
            name,
            available.join(", ")
        ))
    })?;

    let dest = model_path(info.filename);
    let part_path = dest.with_extension("bin.part");

    if dest.exists() {
        tracing::info!("model '{name}' already downloaded at {}", dest.display());
        println!("Model '{}' already downloaded at {}", name, dest.display());
        return Ok(dest);
    }

    // Ensure data directory exists
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| WhsprError::Download(format!("failed to create data directory: {e}")))?;
    }

    let url = format!("{}/{}", base_url.trim_end_matches('/'), info.filename);
    tracing::info!("downloading model '{}' from {}", info.name, url);

    println!("Downloading {} ({})...", info.name, info.size);

    let client = reqwest::Client::new();

    // Check for partial download to support resume
    let mut existing_len = if part_path.exists() {
        std::fs::metadata(&part_path).map(|m| m.len()).unwrap_or(0)
    } else {
        0
    };

    let mut request = client.get(&url);
    if existing_len > 0 {
        tracing::info!("resuming model download from {existing_len} bytes");
        println!("Resuming from {} bytes...", existing_len);
        request = request.header("Range", format!("bytes={}-", existing_len));
    }

    let response = request
        .send()
        .await
        .map_err(|e| WhsprError::Download(format!("failed to start download: {e}")))?;

    let original_len = existing_len;
    existing_len = validated_existing_len(existing_len, response.status())?;
    if original_len > 0 && existing_len == 0 {
        tracing::warn!("server ignored range request, restarting model download from zero");
        println!("Server ignored range request, restarting download from zero");
    }

    let total_size = if existing_len > 0 {
        // For range requests, content-length is remaining bytes
        response
            .content_length()
            .map(|cl| cl + existing_len)
            .unwrap_or(0)
    } else {
        response.content_length().unwrap_or(0)
    };

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );
    pb.set_position(existing_len);

    let mut open_opts = tokio::fs::OpenOptions::new();
    open_opts.create(true);
    if existing_len > 0 {
        open_opts.append(true);
    } else {
        open_opts.write(true).truncate(true);
    }
    let mut file = open_opts
        .open(&part_path)
        .await
        .map_err(|e| WhsprError::Download(format!("failed to open file: {e}")))?;

    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk =
            chunk.map_err(|e| WhsprError::Download(format!("download interrupted: {e}")))?;
        file.write_all(&chunk)
            .await
            .map_err(|e| WhsprError::Download(format!("failed to write: {e}")))?;
        pb.inc(chunk.len() as u64);
    }

    file.flush()
        .await
        .map_err(|e| WhsprError::Download(format!("failed to flush: {e}")))?;
    drop(file);

    pb.finish_with_message("done");

    // Atomic rename
    std::fs::rename(&part_path, &dest)
        .map_err(|e| WhsprError::Download(format!("failed to finalize download: {e}")))?;

    tracing::info!("model '{}' saved to {}", info.name, dest.display());
    println!("Saved to {}", dest.display());
    Ok(dest)
}

pub fn select_model(name: &str, config_path_override: Option<&Path>) -> Result<()> {
    let info =
        find_model(name).ok_or_else(|| WhsprError::Download(format!("unknown model '{name}'")))?;

    let dest = model_path(info.filename);
    if !dest.exists() {
        return Err(WhsprError::Download(format!(
            "model '{}' is not downloaded yet. Run: whispers model download {}",
            name, name
        )));
    }

    let config_path = resolve_config_path(config_path_override);
    let model_path_str = model_path_for_config(info.filename);

    if config_path.exists() {
        tracing::info!(
            "updating model selection in config {} to {}",
            config_path.display(),
            model_path_str
        );
        update_config_model_path(&config_path, &model_path_str)?;
    } else {
        tracing::info!(
            "writing new config {} with selected model {}",
            config_path.display(),
            model_path_str
        );
        config::write_default_config(&config_path, &model_path_str)?;
    }

    println!("Selected model '{}' as active.", name);
    println!("Config updated: {}", config_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::error::WhsprError;
    use httpmock::prelude::*;

    #[test]
    fn path_for_config_uses_tilde_when_under_home() {
        let home = PathBuf::from("/home/alice");
        let path = PathBuf::from("/home/alice/.local/share/whispers/ggml.bin");
        assert_eq!(
            path_for_config(&path, Some(&home)),
            "~/.local/share/whispers/ggml.bin"
        );
    }

    #[test]
    fn path_for_config_keeps_absolute_when_outside_home() {
        let home = PathBuf::from("/home/alice");
        let path = PathBuf::from("/var/lib/whispers/ggml.bin");
        assert_eq!(
            path_for_config(&path, Some(&home)),
            "/var/lib/whispers/ggml.bin"
        );
    }

    #[test]
    fn validated_existing_len_accepts_partial_content_resume() {
        let len = validated_existing_len(100, reqwest::StatusCode::PARTIAL_CONTENT).unwrap();
        assert_eq!(len, 100);
    }

    #[test]
    fn validated_existing_len_restarts_on_ok_resume_response() {
        let len = validated_existing_len(100, reqwest::StatusCode::OK).unwrap();
        assert_eq!(len, 0);
    }

    #[test]
    fn validated_existing_len_rejects_resume_on_error_status() {
        let err = validated_existing_len(100, reqwest::StatusCode::RANGE_NOT_SATISFIABLE);
        assert!(err.is_err());
    }

    #[test]
    fn active_model_path_uses_override_config() {
        let config_path = crate::test_support::unique_temp_path("active-model-config", "toml");
        crate::config::write_default_config(&config_path, "~/override-model.bin")
            .expect("write config");
        let active = active_model_path(Some(&config_path));
        assert_eq!(active.as_deref(), Some("~/override-model.bin"));
    }

    #[test]
    fn model_status_distinguishes_remote_local_and_active() {
        let _env_lock = crate::test_support::env_lock();
        let _guard = crate::test_support::EnvVarGuard::capture(&["HOME", "XDG_DATA_HOME"]);
        let home = crate::test_support::unique_temp_dir("model-status-home");
        crate::test_support::set_env("HOME", &home.to_string_lossy());
        crate::test_support::remove_env("XDG_DATA_HOME");

        let info = find_model("tiny").expect("tiny model should exist");
        let path = model_path(info.filename);
        if path.exists() {
            let _ = std::fs::remove_file(&path);
        }
        assert_eq!(model_status(info, None), "remote");

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create model directory");
        }
        std::fs::write(&path, b"stub").expect("write model file");
        assert_eq!(model_status(info, None), "local");
        assert_eq!(model_status(info, Some(path.as_path())), "active");
    }

    #[test]
    fn select_model_rejects_missing_model_file() {
        let _env_lock = crate::test_support::env_lock();
        let _guard = crate::test_support::EnvVarGuard::capture(&["HOME", "XDG_DATA_HOME"]);
        let home = crate::test_support::unique_temp_dir("select-missing-home");
        crate::test_support::set_env("HOME", &home.to_string_lossy());
        crate::test_support::remove_env("XDG_DATA_HOME");

        let config_path = crate::test_support::unique_temp_path("select-missing-config", "toml");
        let err = select_model("tiny", Some(&config_path)).expect_err("should fail");
        match err {
            WhsprError::Download(msg) => {
                assert!(
                    msg.contains("not downloaded"),
                    "unexpected error message: {msg}"
                );
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn select_model_updates_custom_config_path() {
        let _env_lock = crate::test_support::env_lock();
        let _guard = crate::test_support::EnvVarGuard::capture(&["HOME", "XDG_DATA_HOME"]);
        let home = crate::test_support::unique_temp_dir("select-custom-home");
        crate::test_support::set_env("HOME", &home.to_string_lossy());
        crate::test_support::remove_env("XDG_DATA_HOME");

        let tiny = find_model("tiny").expect("tiny model should exist");
        let model_file = home
            .join(".local")
            .join("share")
            .join("whispers")
            .join(tiny.filename);
        std::fs::create_dir_all(model_file.parent().expect("model parent path"))
            .expect("create model parent");
        std::fs::write(&model_file, b"stub model").expect("write model");

        let config_path = crate::test_support::unique_temp_path("select-custom-config", "toml");
        select_model("tiny", Some(&config_path)).expect("select model");

        let loaded = Config::load(Some(&config_path)).expect("load selected config");
        assert_eq!(
            loaded.whisper.model_path,
            "~/.local/share/whispers/ggml-tiny.bin"
        );
    }

    #[test]
    fn download_model_from_base_resumes_partial_download() {
        let _env_lock = crate::test_support::env_lock();
        let _guard = crate::test_support::EnvVarGuard::capture(&["HOME", "XDG_DATA_HOME"]);
        let home = crate::test_support::unique_temp_dir("download-resume-home");
        crate::test_support::set_env("HOME", &home.to_string_lossy());
        crate::test_support::remove_env("XDG_DATA_HOME");

        let tiny = find_model("tiny").expect("tiny model should exist");
        let dest = model_path(tiny.filename);
        let part_path = dest.with_extension("bin.part");
        std::fs::create_dir_all(dest.parent().expect("model parent")).expect("create model dir");
        std::fs::write(&part_path, b"abc").expect("write partial model");

        let server = MockServer::start();
        let resumed = server.mock(|when, then| {
            when.method(GET)
                .path(format!("/{}", tiny.filename))
                .header("range", "bytes=3-");
            then.status(206).header("content-length", "3").body("def");
        });

        let runtime = tokio::runtime::Runtime::new().expect("runtime");
        let result = runtime
            .block_on(download_model_from_base("tiny", &server.base_url()))
            .expect("download should succeed");
        resumed.assert();
        assert_eq!(result, dest);
        assert_eq!(
            std::fs::read_to_string(&dest).expect("read final model"),
            "abcdef"
        );
        assert!(!part_path.exists(), "part file should be renamed away");
    }

    #[test]
    fn download_model_from_base_restarts_when_server_ignores_range() {
        let _env_lock = crate::test_support::env_lock();
        let _guard = crate::test_support::EnvVarGuard::capture(&["HOME", "XDG_DATA_HOME"]);
        let home = crate::test_support::unique_temp_dir("download-restart-home");
        crate::test_support::set_env("HOME", &home.to_string_lossy());
        crate::test_support::remove_env("XDG_DATA_HOME");

        let tiny = find_model("tiny").expect("tiny model should exist");
        let dest = model_path(tiny.filename);
        let part_path = dest.with_extension("bin.part");
        std::fs::create_dir_all(dest.parent().expect("model parent")).expect("create model dir");
        std::fs::write(&part_path, b"stale").expect("write stale partial");

        let server = MockServer::start();
        let restarted = server.mock(|when, then| {
            when.method(GET).path(format!("/{}", tiny.filename));
            then.status(200).header("content-length", "3").body("new");
        });

        let runtime = tokio::runtime::Runtime::new().expect("runtime");
        let result = runtime
            .block_on(download_model_from_base("tiny", &server.base_url()))
            .expect("download should succeed");
        restarted.assert();
        assert_eq!(result, dest);
        assert_eq!(
            std::fs::read_to_string(&dest).expect("read final model"),
            "new"
        );
        assert!(!part_path.exists(), "part file should be renamed away");
    }
}

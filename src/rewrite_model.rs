use std::path::{Path, PathBuf};

use futures_util::StreamExt;
use tokio::io::AsyncWriteExt;

use crate::config::{
    self, RewriteBackend, data_dir, resolve_config_path, update_config_rewrite_selection,
};
use crate::error::{Result, WhsprError};
use crate::rewrite_profile::RewriteProfile;

pub struct RewriteModelInfo {
    pub name: &'static str,
    pub filename: &'static str,
    pub size: &'static str,
    pub description: &'static str,
    pub url: &'static str,
    pub profile: RewriteProfile,
}

pub const REWRITE_MODELS: &[RewriteModelInfo] = &[
    RewriteModelInfo {
        name: "qwen-3.5-2b-q4_k_m",
        filename: "Qwen_Qwen3.5-2B-Q4_K_M.gguf",
        size: "~1.3 GB",
        description: "Fallback for weaker hardware",
        url: "https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF/resolve/main/Qwen_Qwen3.5-2B-Q4_K_M.gguf",
        profile: RewriteProfile::Qwen,
    },
    RewriteModelInfo {
        name: "qwen-3.5-4b-q4_k_m",
        filename: "Qwen_Qwen3.5-4B-Q4_K_M.gguf",
        size: "~2.9 GB",
        description: "Recommended balance for rewrite mode",
        url: "https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF/resolve/main/Qwen_Qwen3.5-4B-Q4_K_M.gguf",
        profile: RewriteProfile::Qwen,
    },
    RewriteModelInfo {
        name: "qwen-3.5-9b-q4_k_m",
        filename: "Qwen_Qwen3.5-9B-Q4_K_M.gguf",
        size: "~5.9 GB",
        description: "High-end optional model",
        url: "https://huggingface.co/bartowski/Qwen_Qwen3.5-9B-GGUF/resolve/main/Qwen_Qwen3.5-9B-Q4_K_M.gguf",
        profile: RewriteProfile::Qwen,
    },
];

pub fn find_model(name: &str) -> Option<&'static RewriteModelInfo> {
    REWRITE_MODELS.iter().find(|m| m.name == name)
}

pub fn setup_label(info: &RewriteModelInfo) -> String {
    format!(
        "{}  {}  {}",
        crate::ui::value(format!("{:<24}", info.name)),
        crate::ui::size_token(format!("{:>9}", info.size)),
        crate::ui::description_token(info.description)
    )
}

pub fn selected_model_path(name: &str) -> Option<PathBuf> {
    find_model(name).map(|info| model_path(info.filename))
}

fn model_path(filename: &str) -> PathBuf {
    data_dir().join("rewrite").join(filename)
}

pub fn managed_profile(name: &str) -> Option<RewriteProfile> {
    find_model(name).map(|info| info.profile)
}

fn active_model_name(config_path_override: Option<&Path>) -> Option<String> {
    let config_path = resolve_config_path(config_path_override);
    if !config_path.exists() {
        return None;
    }
    let config = config::Config::load(Some(&config_path)).ok()?;
    Some(config.rewrite.selected_model)
}

fn model_status(info: &RewriteModelInfo, active_name: Option<&str>) -> &'static str {
    let path = model_path(info.filename);
    let is_active = active_name == Some(info.name);
    let is_local = path.exists();

    match (is_active, is_local) {
        (true, true) => "active",
        (true, false) => "active (missing)",
        (_, true) => "local",
        _ => "remote",
    }
}

pub fn list_models(config_path_override: Option<&Path>) {
    tracing::debug!("listing rewrite models with config override: {config_path_override:?}");
    let active_name = active_model_name(config_path_override);
    println!(
        "{:<24} {:>9}  {:<8}  DESCRIPTION",
        "MODEL", "SIZE", "STATUS"
    );
    println!("{}", "-".repeat(88));
    for model in REWRITE_MODELS {
        let status = model_status(model, active_name.as_deref());
        let marker = if active_name.as_deref() == Some(model.name) {
            "* "
        } else {
            "  "
        };
        println!(
            "{}{:<22} {:>9}  {:<8}  {}",
            marker,
            model.name,
            crate::ui::size_token(format!("{:>9}", model.size)),
            crate::ui::status_token(format!("{:<8}", status)),
            model.description
        );
    }
}

pub async fn download_model(name: &str) -> Result<PathBuf> {
    let info = find_model(name).ok_or_else(|| {
        let available: Vec<&str> = REWRITE_MODELS.iter().map(|m| m.name).collect();
        WhsprError::Rewrite(format!(
            "unknown rewrite model '{}'. Available: {}",
            name,
            available.join(", ")
        ))
    })?;

    download_model_with_url(info, info.url).await
}

pub async fn download_model_with_url(info: &RewriteModelInfo, url: &str) -> Result<PathBuf> {
    let dest = model_path(info.filename);
    let part_path = dest.with_extension("gguf.part");

    if dest.exists() {
        tracing::info!(
            "rewrite model '{}' already downloaded at {}",
            info.name,
            dest.display()
        );
        println!("{}", crate::ui::ready_message("Rewrite", info.name));
        return Ok(dest);
    }

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| WhsprError::Download(format!("failed to create data directory: {e}")))?;
    }

    tracing::info!("downloading rewrite model '{}' from {}", info.name, url);
    println!(
        "{} Downloading rewrite model {} ({})...",
        crate::ui::info_label(),
        crate::ui::value(info.name),
        info.size
    );

    let client = reqwest::Client::new();
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| WhsprError::Download(format!("failed to start download: {e}")))?;

    if !response.status().is_success() {
        return Err(WhsprError::Download(format!(
            "download failed with HTTP {}",
            response.status()
        )));
    }

    let total_size = response.content_length().unwrap_or(0);
    let pb = crate::ui::progress_bar(total_size);

    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
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

    std::fs::rename(&part_path, &dest)
        .map_err(|e| WhsprError::Download(format!("failed to finalize download: {e}")))?;

    tracing::info!("rewrite model '{}' saved to {}", info.name, dest.display());
    println!("{}", crate::ui::ready_message("Rewrite", info.name));
    Ok(dest)
}

pub fn select_model(name: &str, config_path_override: Option<&Path>) -> Result<()> {
    let info = find_model(name)
        .ok_or_else(|| WhsprError::Rewrite(format!("unknown rewrite model '{name}'")))?;

    let dest = model_path(info.filename);
    if !dest.exists() {
        return Err(WhsprError::Rewrite(format!(
            "rewrite model '{}' is not downloaded yet. Run: whispers rewrite-model download {}",
            name, name
        )));
    }

    let config_path = resolve_config_path(config_path_override);
    if !config_path.exists() {
        let whisper_model = config::Config::default().transcription.model_path;
        config::write_default_config(&config_path, &whisper_model)?;
    }

    update_config_rewrite_selection(&config_path, info.name)?;
    if config::Config::load(Some(&config_path))
        .map(|config| config.rewrite.backend == RewriteBackend::Cloud)
        .unwrap_or(false)
    {
        println!(
            "{} Kept cloud rewrite enabled and updated the local fallback model.",
            crate::ui::info_label()
        );
    }

    println!(
        "{} Active rewrite model: {}",
        crate::ui::ok_label(),
        crate::ui::value(name)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, PostprocessMode};
    use httpmock::prelude::*;

    #[test]
    fn selected_model_path_uses_known_catalog_entry() {
        let path = selected_model_path("qwen-3.5-4b-q4_k_m").expect("catalog entry");
        assert!(path.ends_with("Qwen_Qwen3.5-4B-Q4_K_M.gguf"));
    }

    #[test]
    fn model_status_distinguishes_missing_active_model() {
        let _env_lock = crate::test_support::env_lock();
        let _guard = crate::test_support::EnvVarGuard::capture(&["HOME", "XDG_DATA_HOME"]);
        let home = crate::test_support::unique_temp_dir("rewrite-status-home");
        crate::test_support::set_env("HOME", &home.to_string_lossy());
        crate::test_support::remove_env("XDG_DATA_HOME");

        let info = find_model("qwen-3.5-4b-q4_k_m").expect("model exists");
        assert_eq!(model_status(info, Some(info.name)), "active (missing)");
    }

    #[test]
    fn select_model_updates_custom_config_path() {
        let _env_lock = crate::test_support::env_lock();
        let _guard = crate::test_support::EnvVarGuard::capture(&["HOME", "XDG_DATA_HOME"]);
        let home = crate::test_support::unique_temp_dir("rewrite-select-home");
        crate::test_support::set_env("HOME", &home.to_string_lossy());
        crate::test_support::remove_env("XDG_DATA_HOME");

        let info = find_model("qwen-3.5-2b-q4_k_m").expect("model exists");
        let model_file = home
            .join(".local")
            .join("share")
            .join("whispers")
            .join("rewrite")
            .join(info.filename);
        std::fs::create_dir_all(model_file.parent().expect("parent")).expect("create model dir");
        std::fs::write(&model_file, b"stub model").expect("write model");

        let config_path = crate::test_support::unique_temp_path("rewrite-select", "toml");
        crate::config::write_default_config(&config_path, "~/.local/share/whispers/ggml.bin")
            .expect("write config");

        select_model("qwen-3.5-2b-q4_k_m", Some(&config_path)).expect("select model");
        let loaded = Config::load(Some(&config_path)).expect("load config");
        assert_eq!(loaded.postprocess.mode, PostprocessMode::Rewrite);
        assert_eq!(loaded.rewrite.selected_model, "qwen-3.5-2b-q4_k_m");
    }

    #[test]
    fn select_model_uses_rewrite_error_for_missing_model() {
        let _env_lock = crate::test_support::env_lock();
        let _guard = crate::test_support::EnvVarGuard::capture(&["HOME", "XDG_DATA_HOME"]);
        let home = crate::test_support::unique_temp_dir("rewrite-select-missing-home");
        crate::test_support::set_env("HOME", &home.to_string_lossy());
        crate::test_support::remove_env("XDG_DATA_HOME");

        let config_path = crate::test_support::unique_temp_path("rewrite-select-missing", "toml");
        crate::config::write_default_config(&config_path, "~/.local/share/whispers/ggml.bin")
            .expect("write config");

        let err = select_model("qwen-3.5-2b-q4_k_m", Some(&config_path))
            .expect_err("missing rewrite model should error");
        match err {
            WhsprError::Rewrite(message) => {
                assert!(message.contains("is not downloaded yet"));
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn download_model_with_url_saves_file() {
        let _env_lock = crate::test_support::env_lock();
        let _guard = crate::test_support::EnvVarGuard::capture(&["HOME", "XDG_DATA_HOME"]);
        let home = crate::test_support::unique_temp_dir("rewrite-download-home");
        crate::test_support::set_env("HOME", &home.to_string_lossy());
        crate::test_support::remove_env("XDG_DATA_HOME");

        let info = find_model("qwen-3.5-2b-q4_k_m").expect("model exists");
        let server = MockServer::start();
        let download = server.mock(|when, then| {
            when.method(GET).path("/rewrite.gguf");
            then.status(200)
                .header("content-length", "7")
                .body("gguf123");
        });

        let runtime = tokio::runtime::Runtime::new().expect("runtime");
        let result = runtime
            .block_on(download_model_with_url(
                info,
                &format!("{}/rewrite.gguf", server.base_url()),
            ))
            .expect("download should succeed");

        download.assert();
        assert_eq!(
            std::fs::read_to_string(&result).expect("read final model"),
            "gguf123"
        );
    }

    #[test]
    fn managed_profile_uses_catalog_profile() {
        assert_eq!(
            managed_profile("qwen-3.5-4b-q4_k_m"),
            Some(RewriteProfile::Qwen)
        );
    }
}

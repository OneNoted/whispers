use std::path::{Path, PathBuf};

use futures_util::StreamExt;
use tokio::io::AsyncWriteExt;

use crate::config::{self, Config};
use crate::error::{Result, WhsprError};

pub(crate) struct DownloadSpec<'a> {
    pub tracing_label: &'a str,
    pub user_label: &'a str,
    pub ready_kind: &'a str,
    pub item_name: &'a str,
    pub size: &'a str,
    pub url: &'a str,
    pub dest: PathBuf,
    pub part_path: PathBuf,
    pub resume_partial: bool,
}

pub(crate) fn path_for_config(path: &Path, home: Option<&Path>) -> String {
    if let Some(home_path) = home {
        if let Ok(stripped) = path.strip_prefix(home_path) {
            return format!("~/{}", stripped.display());
        }
    }
    path.display().to_string()
}

pub(crate) fn path_for_current_home(path: &Path) -> String {
    let home = std::env::var("HOME").ok().map(PathBuf::from);
    path_for_config(path, home.as_deref())
}

pub(crate) fn load_config_if_exists(config_path_override: Option<&Path>) -> Option<Config> {
    let config_path = config::resolve_config_path(config_path_override);
    load_config_at_if_exists(&config_path)
}

pub(crate) fn load_config_at_if_exists(config_path: &Path) -> Option<Config> {
    config_path
        .exists()
        .then(|| Config::load(Some(config_path)).ok())?
}

pub(crate) fn ensure_default_config(
    config_path_override: Option<&Path>,
    default_model_path: &str,
) -> Result<(PathBuf, bool)> {
    let config_path = config::resolve_config_path(config_path_override);
    let created = !config_path.exists();
    if created {
        config::write_default_config(&config_path, default_model_path)?;
    }
    Ok((config_path, created))
}

pub(crate) fn configured_file_status(is_active: bool, is_local: bool) -> &'static str {
    match (is_active, is_local) {
        (true, _) => "active",
        (_, true) => "local",
        _ => "remote",
    }
}

pub(crate) fn managed_download_status(is_active: bool, is_local: bool) -> &'static str {
    match (is_active, is_local) {
        (true, true) => "active",
        (true, false) => "active (missing)",
        (_, true) => "local",
        _ => "remote",
    }
}

pub(crate) async fn download_to_path(spec: DownloadSpec<'_>) -> Result<PathBuf> {
    if spec.dest.exists() {
        tracing::info!(
            "{} '{}' already downloaded at {}",
            spec.tracing_label,
            spec.item_name,
            spec.dest.display()
        );
        println!(
            "{}",
            crate::ui::ready_message(spec.ready_kind, spec.item_name)
        );
        return Ok(spec.dest);
    }

    if let Some(parent) = spec.dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| WhsprError::Download(format!("failed to create data directory: {e}")))?;
    }

    tracing::info!(
        "downloading {} '{}' from {}",
        spec.tracing_label,
        spec.item_name,
        spec.url
    );
    println!(
        "{} Downloading {} {} ({})...",
        crate::ui::info_label(),
        spec.user_label,
        crate::ui::value(spec.item_name),
        spec.size
    );

    let client = reqwest::Client::new();
    let mut existing_len = if spec.resume_partial && spec.part_path.exists() {
        std::fs::metadata(&spec.part_path)
            .map(|m| m.len())
            .unwrap_or(0)
    } else {
        0
    };

    let mut request = client.get(spec.url);
    if existing_len > 0 {
        tracing::info!(
            "resuming {} download from {existing_len} bytes",
            spec.tracing_label
        );
        if crate::ui::is_verbose() {
            println!("Resuming from {} bytes...", existing_len);
        }
        request = request.header("Range", format!("bytes={}-", existing_len));
    }

    let response = request
        .send()
        .await
        .map_err(|e| WhsprError::Download(format!("failed to start download: {e}")))?;

    let original_len = existing_len;
    existing_len = validated_existing_len(existing_len, response.status())?;
    if original_len > 0 && existing_len == 0 {
        tracing::warn!(
            "server ignored range request, restarting {} download from zero",
            spec.tracing_label
        );
        if crate::ui::is_verbose() {
            println!("Server ignored range request, restarting download from zero");
        }
    }

    let total_size = if existing_len > 0 {
        response
            .content_length()
            .map(|content_length| content_length + existing_len)
            .unwrap_or(0)
    } else {
        response.content_length().unwrap_or(0)
    };

    let pb = crate::ui::progress_bar(total_size);
    if existing_len > 0 {
        pb.set_position(existing_len);
    }

    let mut open_opts = tokio::fs::OpenOptions::new();
    open_opts.create(true);
    if existing_len > 0 {
        open_opts.append(true);
    } else {
        open_opts.write(true).truncate(true);
    }

    let mut file = open_opts
        .open(&spec.part_path)
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

    std::fs::rename(&spec.part_path, &spec.dest)
        .map_err(|e| WhsprError::Download(format!("failed to finalize download: {e}")))?;

    tracing::info!(
        "{} '{}' saved to {}",
        spec.tracing_label,
        spec.item_name,
        spec.dest.display()
    );
    println!(
        "{}",
        crate::ui::ready_message(spec.ready_kind, spec.item_name)
    );
    Ok(spec.dest)
}

pub(crate) fn validated_existing_len(
    existing_len: u64,
    status: reqwest::StatusCode,
) -> Result<u64> {
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn configured_file_status_prioritizes_active() {
        assert_eq!(configured_file_status(true, false), "active");
        assert_eq!(configured_file_status(false, true), "local");
        assert_eq!(configured_file_status(false, false), "remote");
    }

    #[test]
    fn managed_download_status_reports_missing_active_assets() {
        assert_eq!(managed_download_status(true, true), "active");
        assert_eq!(managed_download_status(true, false), "active (missing)");
        assert_eq!(managed_download_status(false, true), "local");
        assert_eq!(managed_download_status(false, false), "remote");
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
}

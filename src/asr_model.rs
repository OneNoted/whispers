use std::path::{Path, PathBuf};

use crate::config::{
    self, TranscriptionBackend, resolve_config_path, update_config_transcription_selection,
};
use crate::error::{Result, WhsprError};
use crate::{faster_whisper, model, nemo_asr};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanguageScope {
    Multilingual,
    EnglishOnly,
}

impl LanguageScope {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Multilingual => "multi",
            Self::EnglishOnly => "en-only",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportTier {
    Recommended,
    Optional,
    Experimental,
}

impl SupportTier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Recommended => "recommended",
            Self::Optional => "optional",
            Self::Experimental => "experimental",
        }
    }

    pub fn setup_badge(self) -> &'static str {
        match self {
            Self::Recommended => "Recommended",
            Self::Optional => "Optional",
            Self::Experimental => "Experimental",
        }
    }
}

pub struct AsrModelInfo {
    pub name: &'static str,
    pub backend: TranscriptionBackend,
    pub size: &'static str,
    pub description: &'static str,
    pub language_scope: LanguageScope,
    pub support_tier: SupportTier,
    pub setup_note: Option<&'static str>,
}

pub const ASR_MODELS: &[AsrModelInfo] = &[
    AsrModelInfo {
        name: "large-v3-turbo",
        backend: TranscriptionBackend::WhisperCpp,
        size: "1.6 GB",
        description: "Default multilingual balance",
        language_scope: LanguageScope::Multilingual,
        support_tier: SupportTier::Recommended,
        setup_note: Some("Best default local choice for most users."),
    },
    AsrModelInfo {
        name: "large-v3",
        backend: TranscriptionBackend::WhisperCpp,
        size: "3.1 GB",
        description: "Most accurate multilingual Whisper model",
        language_scope: LanguageScope::Multilingual,
        support_tier: SupportTier::Optional,
        setup_note: Some("Higher-quality Whisper option with noticeably more latency."),
    },
    AsrModelInfo {
        name: "medium",
        backend: TranscriptionBackend::WhisperCpp,
        size: "1.5 GB",
        description: "Mid-size multilingual Whisper model",
        language_scope: LanguageScope::Multilingual,
        support_tier: SupportTier::Optional,
        setup_note: None,
    },
    AsrModelInfo {
        name: "medium.en",
        backend: TranscriptionBackend::WhisperCpp,
        size: "1.5 GB",
        description: "Mid-size English Whisper model",
        language_scope: LanguageScope::EnglishOnly,
        support_tier: SupportTier::Optional,
        setup_note: None,
    },
    AsrModelInfo {
        name: "small",
        backend: TranscriptionBackend::WhisperCpp,
        size: "488 MB",
        description: "Smaller multilingual Whisper model",
        language_scope: LanguageScope::Multilingual,
        support_tier: SupportTier::Optional,
        setup_note: None,
    },
    AsrModelInfo {
        name: "small.en",
        backend: TranscriptionBackend::WhisperCpp,
        size: "488 MB",
        description: "Smaller English Whisper model",
        language_scope: LanguageScope::EnglishOnly,
        support_tier: SupportTier::Optional,
        setup_note: None,
    },
    AsrModelInfo {
        name: "base",
        backend: TranscriptionBackend::WhisperCpp,
        size: "148 MB",
        description: "Compact multilingual Whisper model",
        language_scope: LanguageScope::Multilingual,
        support_tier: SupportTier::Optional,
        setup_note: None,
    },
    AsrModelInfo {
        name: "base.en",
        backend: TranscriptionBackend::WhisperCpp,
        size: "148 MB",
        description: "Compact English Whisper model",
        language_scope: LanguageScope::EnglishOnly,
        support_tier: SupportTier::Optional,
        setup_note: None,
    },
    AsrModelInfo {
        name: "tiny",
        backend: TranscriptionBackend::WhisperCpp,
        size: "78 MB",
        description: "Fastest multilingual Whisper model",
        language_scope: LanguageScope::Multilingual,
        support_tier: SupportTier::Optional,
        setup_note: Some("Fastest local Whisper option, with the sharpest quality tradeoff."),
    },
    AsrModelInfo {
        name: "tiny.en",
        backend: TranscriptionBackend::WhisperCpp,
        size: "78 MB",
        description: "Fastest English Whisper model",
        language_scope: LanguageScope::EnglishOnly,
        support_tier: SupportTier::Optional,
        setup_note: Some(
            "Fastest English-only Whisper option, with the sharpest quality tradeoff.",
        ),
    },
    AsrModelInfo {
        name: "distil-large-v3.5",
        backend: TranscriptionBackend::FasterWhisper,
        size: "756M params",
        description: "Fast English distil model on faster-whisper",
        language_scope: LanguageScope::EnglishOnly,
        support_tier: SupportTier::Recommended,
        setup_note: Some(
            "Fastest managed English-first path; best when multilingual support is not required.",
        ),
    },
    AsrModelInfo {
        name: "parakeet-tdt_ctc-1.1b",
        backend: TranscriptionBackend::Nemo,
        size: "1.1B params",
        description: "Experimental English NeMo ASR benchmark path",
        language_scope: LanguageScope::EnglishOnly,
        support_tier: SupportTier::Experimental,
        setup_note: Some(
            "Experimental backend. The first warm-up can be much slower than steady-state dictation, and behavior may vary by machine.",
        ),
    },
    AsrModelInfo {
        name: "canary-qwen-2.5b",
        backend: TranscriptionBackend::Nemo,
        size: "2.5B params",
        description: "Experimental English NeMo ASR/LLM hybrid",
        language_scope: LanguageScope::EnglishOnly,
        support_tier: SupportTier::Experimental,
        setup_note: Some(
            "Experimental backend under evaluation. The managed path is currently blocked by an upstream NeMo/PEFT initialization issue.",
        ),
    },
];

pub fn find_model(name: &str) -> Option<&'static AsrModelInfo> {
    ASR_MODELS.iter().find(|info| info.name == name)
}

pub fn setup_label(info: &AsrModelInfo) -> String {
    format!(
        "[{}] {}  {}  {}  {}",
        crate::ui::tier_token(format!("{:<12}", info.support_tier.setup_badge())),
        crate::ui::value(info.name),
        crate::ui::backend_token(info.backend.as_str()),
        crate::ui::scope_token(info.language_scope.as_str()),
        info.description
    )
}

pub fn experimental_warning(info: &AsrModelInfo) -> Option<&'static str> {
    (info.support_tier == SupportTier::Experimental)
        .then_some(info.setup_note.unwrap_or("Experimental backend."))
}

pub fn experimental_notice_facts(info: &AsrModelInfo) -> &'static [(&'static str, &'static str)] {
    match info.name {
        "parakeet-tdt_ctc-1.1b" => &[
            (
                "Warm-up",
                "First load can be much slower than steady-state dictation.",
            ),
            (
                "Use for",
                "Benchmarking and experimentation, not the default recommendation.",
            ),
        ],
        "canary-qwen-2.5b" => &[
            (
                "Status",
                "Currently blocked by an upstream NeMo/PEFT initialization issue.",
            ),
            (
                "Use for",
                "Experimental evaluation only when the backend is available.",
            ),
        ],
        _ => &[("Use for", "Experimental evaluation only.")],
    }
}

pub fn is_model_available(name: &str) -> bool {
    availability_issue(name).is_none()
}

pub fn availability_issue(name: &str) -> Option<&'static str> {
    match name {
        "canary-qwen-2.5b" => Some(
            "temporarily unavailable: upstream NeMo/PEFT runtime currently fails to initialize the model",
        ),
        _ => None,
    }
}

pub fn selected_model_path(name: &str) -> Option<PathBuf> {
    let info = find_model(name)?;
    match info.backend {
        TranscriptionBackend::WhisperCpp => model::selected_model_local_path(info.name),
        TranscriptionBackend::FasterWhisper => {
            Some(faster_whisper::managed_model_local_path(info.name))
        }
        TranscriptionBackend::Nemo => Some(nemo_asr::managed_model_local_path(info.name)),
        TranscriptionBackend::Cloud => None,
    }
}

fn active_model_name(
    config_path_override: Option<&Path>,
) -> Option<(TranscriptionBackend, String)> {
    let config_path = resolve_config_path(config_path_override);
    if !config_path.exists() {
        return None;
    }
    let config = config::Config::load(Some(&config_path)).ok()?;
    Some((
        config.transcription.resolved_local_backend(),
        config.transcription.selected_model,
    ))
}

fn model_status(info: &AsrModelInfo, active: Option<(TranscriptionBackend, &str)>) -> &'static str {
    if availability_issue(info.name).is_some() {
        return "blocked";
    }
    let path = selected_model_path(info.name);
    let is_active = active == Some((info.backend, info.name)) || active == Some((info.backend, ""));
    let is_local = path
        .as_ref()
        .map(|path| match info.backend {
            TranscriptionBackend::WhisperCpp => path.exists(),
            TranscriptionBackend::FasterWhisper => faster_whisper::model_dir_is_ready(path),
            TranscriptionBackend::Nemo => nemo_asr::model_dir_is_ready(path),
            TranscriptionBackend::Cloud => false,
        })
        .unwrap_or(false);

    match (is_active, is_local) {
        (true, true) => "active",
        (true, false) => "active (missing)",
        (_, true) => "local",
        _ => "remote",
    }
}

pub fn list_models(config_path_override: Option<&Path>) {
    let active_binding = active_model_name(config_path_override);
    let active = active_binding
        .as_ref()
        .map(|(backend, name)| (*backend, name.as_str()));
    println!(
        "{:<24} {:<15} {:<12} {:<8} {:<13} {:<10}  DESCRIPTION",
        "MODEL", "BACKEND", "SIZE", "SCOPE", "TIER", "STATUS"
    );
    println!("{}", "-".repeat(126));
    for info in ASR_MODELS {
        let status = model_status(info, active);
        let marker = if active == Some((info.backend, info.name)) {
            "* "
        } else {
            "  "
        };
        let description = availability_issue(info.name)
            .map(|issue| format!("{} [{}]", info.description, issue))
            .unwrap_or_else(|| info.description.to_string());
        println!(
            "{}{:<22} {:<15} {:<12} {:<8} {:<13} {:<10}  {}",
            marker,
            info.name,
            crate::ui::backend_token(format!("{:<15}", info.backend.as_str())),
            info.size,
            crate::ui::scope_token(format!("{:<8}", info.language_scope.as_str())),
            crate::ui::tier_token(format!("{:<13}", info.support_tier.as_str())),
            crate::ui::status_token(format!("{:<10}", status)),
            description
        );
    }
}

pub async fn download_model(name: &str) -> Result<PathBuf> {
    let info = find_model(name).ok_or_else(|| {
        let available: Vec<&str> = ASR_MODELS.iter().map(|info| info.name).collect();
        WhsprError::Download(format!(
            "unknown ASR model '{}'. Available: {}",
            name,
            available.join(", ")
        ))
    })?;
    if let Some(issue) = availability_issue(name) {
        return Err(WhsprError::Download(format!(
            "ASR model '{}' is {}",
            name, issue
        )));
    }

    match info.backend {
        TranscriptionBackend::WhisperCpp => model::download_model(name).await,
        TranscriptionBackend::FasterWhisper => {
            faster_whisper::download_managed_model(name).await?;
            Ok(faster_whisper::managed_model_local_path(name))
        }
        TranscriptionBackend::Nemo => {
            nemo_asr::download_managed_model(name).await?;
            Ok(nemo_asr::managed_model_local_path(name))
        }
        TranscriptionBackend::Cloud => Err(WhsprError::Download(
            "cloud ASR models are configured through [cloud], not downloaded locally".into(),
        )),
    }
}

pub fn select_model(name: &str, config_path_override: Option<&Path>) -> Result<()> {
    let info = find_model(name)
        .ok_or_else(|| WhsprError::Config(format!("unknown ASR model '{name}'")))?;
    if let Some(issue) = availability_issue(name) {
        return Err(WhsprError::Config(format!(
            "ASR model '{}' is {}",
            name, issue
        )));
    }
    let model_path = selected_model_path(name).ok_or_else(|| {
        WhsprError::Config(format!(
            "failed to resolve local path for ASR model '{name}'"
        ))
    })?;

    let is_ready = match info.backend {
        TranscriptionBackend::WhisperCpp => model_path.exists(),
        TranscriptionBackend::FasterWhisper => faster_whisper::model_dir_is_ready(&model_path),
        TranscriptionBackend::Nemo => nemo_asr::model_dir_is_ready(&model_path),
        TranscriptionBackend::Cloud => false,
    };
    if !is_ready {
        return Err(WhsprError::Config(format!(
            "ASR model '{}' is not downloaded yet or is incomplete. Run: whispers asr-model download {}",
            name, name
        )));
    }

    let config_path = resolve_config_path(config_path_override);
    if !config_path.exists() {
        config::write_default_config(
            &config_path,
            &model::model_path_for_config("ggml-large-v3-turbo.bin"),
        )?;
    }

    let config_model_path = match info.backend {
        TranscriptionBackend::WhisperCpp => model::model_path_for_config(
            model::find_model(name)
                .expect("whisper model info exists")
                .filename,
        ),
        TranscriptionBackend::FasterWhisper => model_path.display().to_string(),
        TranscriptionBackend::Nemo => model_path.display().to_string(),
        TranscriptionBackend::Cloud => String::new(),
    };
    update_config_transcription_selection(
        &config_path,
        info.backend,
        info.name,
        &config_model_path,
        config::Config::load(Some(&config_path))
            .map(|config| config.transcription.backend != TranscriptionBackend::Cloud)
            .unwrap_or(true),
    )?;

    println!(
        "{} Active ASR model: {} ({})",
        crate::ui::ok_label(),
        crate::ui::value(info.name),
        info.backend.as_str()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_model_path_resolves_faster_whisper_dir() {
        let path = selected_model_path("distil-large-v3.5").expect("managed path");
        assert!(path.ends_with("distil-large-v3.5"));
    }

    #[test]
    fn selected_model_path_resolves_whisper_cpp_file() {
        let path = selected_model_path("large-v3-turbo").expect("managed path");
        assert!(path.ends_with("ggml-large-v3-turbo.bin"));
    }

    #[test]
    fn selected_model_path_resolves_nemo_dir() {
        let path = selected_model_path("parakeet-tdt_ctc-1.1b").expect("managed path");
        assert!(path.ends_with("parakeet-tdt_ctc-1.1b"));
    }

    #[test]
    fn canary_qwen_is_reported_unavailable() {
        let issue = availability_issue("canary-qwen-2.5b").expect("availability issue");
        assert!(issue.contains("temporarily unavailable"));
        assert!(!is_model_available("canary-qwen-2.5b"));
    }

    #[test]
    fn parakeet_is_marked_experimental() {
        let info = find_model("parakeet-tdt_ctc-1.1b").expect("parakeet info");
        assert_eq!(info.support_tier, SupportTier::Experimental);
        assert!(
            experimental_warning(info)
                .expect("warning")
                .contains("first warm-up can be much slower")
        );
    }

    #[test]
    fn large_v3_turbo_is_marked_recommended() {
        let info = find_model("large-v3-turbo").expect("turbo info");
        assert_eq!(info.support_tier, SupportTier::Recommended);
        assert!(setup_label(info).contains("[Recommended ]"));
    }
}

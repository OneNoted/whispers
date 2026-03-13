use std::path::Path;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum RewriteProfile {
    #[default]
    Auto,
    Generic,
    Qwen,
    LlamaCompat,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum ResolvedRewriteProfile {
    Generic,
    Qwen,
    LlamaCompat,
}

impl RewriteProfile {
    #[allow(dead_code)]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Generic => "generic",
            Self::Qwen => "qwen",
            Self::LlamaCompat => "llama_compat",
        }
    }

    #[allow(dead_code)]
    pub fn resolve(self, managed_model: Option<&str>, model_path: &Path) -> ResolvedRewriteProfile {
        match self {
            Self::Auto => managed_model
                .and_then(resolve_identifier)
                .or_else(|| {
                    model_path
                        .file_name()
                        .and_then(|name| resolve_identifier(&name.to_string_lossy()))
                })
                .unwrap_or(ResolvedRewriteProfile::Generic),
            Self::Generic => ResolvedRewriteProfile::Generic,
            Self::Qwen => ResolvedRewriteProfile::Qwen,
            Self::LlamaCompat => ResolvedRewriteProfile::LlamaCompat,
        }
    }
}

impl ResolvedRewriteProfile {
    #[allow(dead_code)]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Generic => "generic",
            Self::Qwen => "qwen",
            Self::LlamaCompat => "llama_compat",
        }
    }
}

#[allow(dead_code)]
fn resolve_identifier(identifier: &str) -> Option<ResolvedRewriteProfile> {
    let normalized = identifier.to_ascii_lowercase();
    if normalized.contains("qwen") {
        return Some(ResolvedRewriteProfile::Qwen);
    }

    if normalized.contains("llama") {
        return Some(ResolvedRewriteProfile::LlamaCompat);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_profile_prefers_managed_model_name() {
        let resolved = RewriteProfile::Auto.resolve(
            Some("qwen-3.5-4b-q4_k_m"),
            Path::new("/tmp/Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
        );
        assert_eq!(resolved, ResolvedRewriteProfile::Qwen);
    }

    #[test]
    fn auto_profile_falls_back_to_model_filename() {
        let resolved =
            RewriteProfile::Auto.resolve(None, Path::new("/tmp/Llama-3.2-3B-Instruct-Q4_K_M.gguf"));
        assert_eq!(resolved, ResolvedRewriteProfile::LlamaCompat);
    }

    #[test]
    fn auto_profile_uses_generic_for_unknown_models() {
        let resolved =
            RewriteProfile::Auto.resolve(None, Path::new("/tmp/CustomDictationModel.gguf"));
        assert_eq!(resolved, ResolvedRewriteProfile::Generic);
    }
}

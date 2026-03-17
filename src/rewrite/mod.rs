mod output;
mod prompt;
mod routing;

#[cfg(test)]
mod tests;

pub use output::sanitize_rewrite_output;
pub use prompt::{
    build_oaicompat_messages_json, build_prompt, effective_max_tokens, resolved_profile_for_cloud,
};

#[cfg(feature = "local-rewrite")]
pub const fn local_rewrite_available() -> bool {
    true
}

#[cfg(not(feature = "local-rewrite"))]
pub const fn local_rewrite_available() -> bool {
    false
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewritePrompt {
    pub system: String,
    pub user: String,
}

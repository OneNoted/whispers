mod local;
mod output;
mod prompt;
mod routing;

#[cfg(test)]
mod tests;

#[cfg(feature = "local-rewrite")]
pub use local::LocalRewriter;
pub use local::local_rewrite_available;
pub(crate) use output::sanitize_rewrite_output;
pub use prompt::{build_prompt, resolved_profile_for_cloud};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewritePrompt {
    pub system: String,
    pub user: String,
}

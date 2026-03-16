use serde::{Deserialize, Serialize};

use crate::context::SurfaceKind;
use crate::rewrite_protocol::{RewriteSessionBacktrackCandidate, RewriteSessionEntry};

mod persistence;
mod planning;

pub use persistence::{load_recent_entry, record_append, record_replace};
pub use planning::{build_backtrack_plan, to_rewrite_typing_context};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionEntry {
    pub id: u64,
    pub final_text: String,
    pub grapheme_len: usize,
    pub injected_at_ms: u64,
    pub focus_fingerprint: String,
    pub surface_kind: SurfaceKind,
    pub app_id: Option<String>,
    pub window_title: Option<String>,
    pub rewrite_summary: SessionRewriteSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionRewriteSummary {
    pub had_edit_cues: bool,
    pub rewrite_used: bool,
    pub recommended_candidate: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EligibleSessionEntry {
    pub entry: SessionEntry,
    pub delete_graphemes: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionBacktrackPlan {
    pub recent_entries: Vec<RewriteSessionEntry>,
    pub candidates: Vec<RewriteSessionBacktrackCandidate>,
    pub recommended: Option<RewriteSessionBacktrackCandidate>,
    pub deterministic_replacement_text: Option<String>,
}

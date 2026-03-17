use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RewriteTranscript {
    pub raw_text: String,
    pub correction_aware_text: String,
    pub aggressive_correction_text: Option<String>,
    pub detected_language: Option<String>,
    pub typing_context: Option<RewriteTypingContext>,
    pub recent_session_entries: Vec<RewriteSessionEntry>,
    pub session_backtrack_candidates: Vec<RewriteSessionBacktrackCandidate>,
    pub recommended_session_candidate: Option<RewriteSessionBacktrackCandidate>,
    pub segments: Vec<RewriteTranscriptSegment>,
    pub edit_intents: Vec<RewriteEditIntent>,
    pub edit_signals: Vec<RewriteEditSignal>,
    pub edit_hypotheses: Vec<RewriteEditHypothesis>,
    pub rewrite_candidates: Vec<RewriteCandidate>,
    pub recommended_candidate: Option<RewriteCandidate>,
    #[serde(default)]
    pub policy_context: RewritePolicyContext,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RewriteTranscriptSegment {
    pub text: String,
    pub start_ms: u32,
    pub end_ms: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RewriteTypingContext {
    pub focus_fingerprint: String,
    pub app_id: Option<String>,
    pub window_title: Option<String>,
    pub surface_kind: RewriteSurfaceKind,
    pub browser_domain: Option<String>,
    pub captured_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RewriteSessionEntry {
    pub id: u64,
    pub final_text: String,
    pub grapheme_len: usize,
    pub focus_fingerprint: String,
    pub surface_kind: RewriteSurfaceKind,
    pub app_id: Option<String>,
    pub window_title: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RewriteSessionBacktrackCandidate {
    pub kind: RewriteSessionBacktrackCandidateKind,
    pub entry_id: Option<u64>,
    pub delete_graphemes: usize,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RewriteEditIntent {
    pub action: RewriteEditAction,
    pub trigger: String,
    pub confidence: RewriteIntentConfidence,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RewriteEditSignal {
    pub trigger: String,
    pub kind: RewriteEditSignalKind,
    pub scope_hint: RewriteEditSignalScope,
    pub strength: RewriteEditSignalStrength,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RewriteEditHypothesis {
    pub cue_family: String,
    pub matched_text: String,
    pub match_source: RewriteEditHypothesisMatchSource,
    pub kind: RewriteEditSignalKind,
    pub scope_hint: RewriteEditSignalScope,
    pub replacement_scope: RewriteReplacementScope,
    pub tail_shape: RewriteTailShape,
    pub strength: RewriteEditSignalStrength,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RewriteCandidate {
    pub kind: RewriteCandidateKind,
    pub text: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct RewritePolicyContext {
    pub correction_policy: RewriteCorrectionPolicy,
    pub matched_rule_names: Vec<String>,
    pub effective_rule_instructions: Vec<String>,
    pub active_glossary_terms: Vec<RewritePolicyGlossaryTerm>,
    pub glossary_candidates: Vec<RewriteCandidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RewritePolicyGlossaryTerm {
    pub term: String,
    pub aliases: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RewriteEditAction {
    ReplacePreviousPhrase,
    ReplacePreviousClause,
    ReplacePreviousSentence,
    DropEditCue,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RewriteIntentConfidence {
    High,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RewriteEditSignalKind {
    Cancel,
    Replace,
    Restatement,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RewriteEditSignalScope {
    Phrase,
    Clause,
    Sentence,
    Unknown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RewriteEditSignalStrength {
    Possible,
    Strong,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RewriteEditHypothesisMatchSource {
    Exact,
    Alias,
    NearMiss,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RewriteReplacementScope {
    Span,
    Clause,
    Sentence,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RewriteTailShape {
    Empty,
    Phrase,
    Clause,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum RewriteCorrectionPolicy {
    Conservative,
    #[default]
    Balanced,
    Aggressive,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RewriteCandidateKind {
    Literal,
    ConservativeCorrection,
    AggressiveCorrection,
    GlossaryCorrection,
    SpanReplacement,
    ClauseReplacement,
    SentenceReplacement,
    ContextualReplacement,
    DropCueOnly,
    FollowingReplacement,
    CancelPreviousClause,
    CancelPreviousSentence,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum RewriteSurfaceKind {
    Browser,
    Terminal,
    Editor,
    GenericText,
    Unknown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RewriteSessionBacktrackCandidateKind {
    AppendCurrent,
    ReplaceLastEntry,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkerRequest {
    Rewrite {
        transcript: RewriteTranscript,
        custom_instructions: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkerResponse {
    Result { text: String },
    Error { message: String },
}

impl RewriteCorrectionPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Conservative => "conservative",
            Self::Balanced => "balanced",
            Self::Aggressive => "aggressive",
        }
    }
}

impl RewriteSurfaceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Browser => "browser",
            Self::Terminal => "terminal",
            Self::Editor => "editor",
            Self::GenericText => "generic_text",
            Self::Unknown => "unknown",
        }
    }
}

impl RewritePolicyContext {
    pub fn is_active(&self) -> bool {
        !self.matched_rule_names.is_empty()
            || !self.effective_rule_instructions.is_empty()
            || !self.active_glossary_terms.is_empty()
            || !self.glossary_candidates.is_empty()
    }
}

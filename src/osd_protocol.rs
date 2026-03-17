use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VoiceOsdStatus {
    #[default]
    Listening,
    Transcribing,
    Rewriting,
    Finalizing,
    Frozen,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct VoiceOsdUpdate {
    pub status: VoiceOsdStatus,
    pub stable_text: String,
    pub unstable_text: String,
    pub rewrite_preview: Option<String>,
    pub live_inject: bool,
    pub frozen: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OsdEvent {
    VoiceUpdate(VoiceOsdUpdate),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_update_roundtrips_as_json() {
        let event = OsdEvent::VoiceUpdate(VoiceOsdUpdate {
            status: VoiceOsdStatus::Rewriting,
            stable_text: "hello".into(),
            unstable_text: "world".into(),
            rewrite_preview: Some("Hello world.".into()),
            live_inject: true,
            frozen: false,
        });

        let json = serde_json::to_string(&event).expect("serialize event");
        let parsed: OsdEvent = serde_json::from_str(&json).expect("deserialize event");
        assert_eq!(parsed, event);
    }
}

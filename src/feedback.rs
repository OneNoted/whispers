use std::io::Cursor;
use std::sync::mpsc;
use std::time::Duration;

use rodio::{Decoder, OutputStreamBuilder, Sink};

use crate::error::{Result, WhsprError};

// Bundled sounds (embedded at compile time)
const START_SOUND: &[u8] = include_bytes!("../sounds/start.wav");
const STOP_SOUND: &[u8] = include_bytes!("../sounds/stop.wav");

enum SoundCommand {
    Play {
        custom_path: Option<String>,
        bundled: &'static [u8],
        /// If set, signal completion so the caller can block until playback finishes.
        done: Option<mpsc::SyncSender<()>>,
    },
}

/// Plays feedback sounds through a persistent audio output stream.
///
/// The output stream is opened once on a dedicated thread and kept alive for the
/// lifetime of this struct. This avoids the PulseAudio/PipeWire pop that occurs
/// when a client repeatedly connects and disconnects.
///
/// On drop, the channel is closed and the background thread is joined so the
/// `OutputStream` is torn down gracefully before the process exits.
pub struct FeedbackPlayer {
    enabled: bool,
    start_sound_path: Option<String>,
    stop_sound_path: Option<String>,
    sender: Option<mpsc::Sender<SoundCommand>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl FeedbackPlayer {
    pub fn new(enabled: bool, start_sound: &str, stop_sound: &str) -> Self {
        let start_sound_path = if start_sound.is_empty() {
            None
        } else {
            Some(start_sound.to_string())
        };
        let stop_sound_path = if stop_sound.is_empty() {
            None
        } else {
            Some(stop_sound.to_string())
        };

        let (sender, thread) = if enabled {
            let (sender, receiver) = mpsc::channel::<SoundCommand>();
            let thread = std::thread::spawn(move || {
                // Lazily open the output stream so transient startup failures can recover.
                let mut stream: Option<rodio::OutputStream> = None;

                while let Ok(cmd) = receiver.recv() {
                    match cmd {
                        SoundCommand::Play {
                            custom_path,
                            bundled,
                            done,
                        } => {
                            if stream.is_none() {
                                match OutputStreamBuilder::open_default_stream() {
                                    Ok(s) => stream = Some(s),
                                    Err(e) => {
                                        tracing::warn!(
                                            "failed to open audio output for feedback: {e}"
                                        );
                                        if let Some(done) = done {
                                            let _ = done.send(());
                                        }
                                        continue;
                                    }
                                }
                            }

                            if let Err(e) = play_on_stream(
                                stream.as_ref().expect("stream set"),
                                custom_path.as_deref(),
                                bundled,
                            ) {
                                tracing::warn!("failed to play feedback sound: {e}");
                            }
                            if let Some(done) = done {
                                let _ = done.send(());
                            }
                        }
                    }
                }

                // Leak the OutputStream — cpal's ALSA backend calls snd_pcm_close()
                // on drop without draining first, which causes an audible click on
                // PipeWire.  The OS reclaims file descriptors on process exit.
                if let Some(stream) = stream {
                    std::mem::forget(stream);
                }
            });
            (Some(sender), Some(thread))
        } else {
            (None, None)
        };

        Self {
            enabled,
            start_sound_path,
            stop_sound_path,
            sender,
            thread,
        }
    }

    /// Blocks until the start sound has finished playing.
    ///
    /// This ensures the sound completes before the mic goes live, preventing
    /// the start chime from leaking into the recording.
    pub fn play_start(&self) {
        if !self.enabled {
            return;
        }
        let sender = match self.sender.as_ref() {
            Some(s) => s,
            None => return,
        };
        let (tx, rx) = mpsc::sync_channel(1);
        if sender
            .send(SoundCommand::Play {
                custom_path: self.start_sound_path.clone(),
                bundled: START_SOUND,
                done: Some(tx),
            })
            .is_err()
        {
            tracing::warn!("feedback thread unavailable, skipping start sound");
            return;
        }
        if rx.recv_timeout(Duration::from_secs(2)).is_err() {
            tracing::warn!("timed out waiting for start sound playback");
        }
    }

    /// Blocks until the stop sound has finished playing.
    ///
    /// This prevents the process from exiting (and tearing down the audio
    /// stream) while the sound is still in-flight.
    pub fn play_stop(&self) {
        if !self.enabled {
            return;
        }
        let sender = match self.sender.as_ref() {
            Some(s) => s,
            None => return,
        };
        let (tx, rx) = mpsc::sync_channel(1);
        if sender
            .send(SoundCommand::Play {
                custom_path: self.stop_sound_path.clone(),
                bundled: STOP_SOUND,
                done: Some(tx),
            })
            .is_err()
        {
            tracing::warn!("feedback thread unavailable, skipping stop sound");
            return;
        }
        if rx.recv_timeout(Duration::from_secs(2)).is_err() {
            tracing::warn!("timed out waiting for stop sound playback");
        }
    }
}

impl Drop for FeedbackPlayer {
    fn drop(&mut self) {
        // Close the channel so the background thread's recv loop exits.
        self.sender.take();
        // Join the thread to ensure orderly shutdown before the process exits.
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn play_on_stream(
    stream: &rodio::OutputStream,
    custom_path: Option<&str>,
    bundled: &'static [u8],
) -> Result<()> {
    let sink = Sink::connect_new(stream.mixer());

    if let Some(path) = custom_path {
        let file = std::fs::File::open(path)
            .map_err(|e| WhsprError::Feedback(format!("failed to open sound file: {e}")))?;
        let reader = std::io::BufReader::new(file);
        let source = Decoder::new(reader)
            .map_err(|e| WhsprError::Feedback(format!("failed to decode sound file: {e}")))?;
        sink.append(source);
    } else {
        let cursor = Cursor::new(bundled);
        let source = Decoder::new(cursor)
            .map_err(|e| WhsprError::Feedback(format!("failed to decode bundled sound: {e}")))?;
        sink.append(source);
    }

    sink.sleep_until_end();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_feedback_is_noop() {
        let player = FeedbackPlayer::new(false, "", "");
        player.play_start();
        player.play_stop();
    }

    #[test]
    fn dropping_feedback_player_does_not_panic() {
        let player = FeedbackPlayer::new(true, "", "");
        drop(player);
    }
}

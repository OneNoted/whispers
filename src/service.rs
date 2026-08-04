use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::protocol::{ControlRequest, ControlResponse};
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PublicState {
    Idle,
    Listening,
    Finalizing,
    Delivering,
}

impl PublicState {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Listening => "listening",
            Self::Finalizing => "finalizing",
            Self::Delivering => "delivering",
        }
    }
}

pub trait ActiveDictation: Send {
    fn finish(self: Box<Self>, cancelled: &AtomicBool) -> Result<CompletedDictation>;
    fn cancel(self: Box<Self>) -> Result<()>;
}

pub trait Pipeline: Send + Sync + 'static {
    fn start(&self) -> Result<Box<dyn ActiveDictation>>;
    fn deliver(&self, dictation: &CompletedDictation) -> Result<()>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompletedDictation {
    pub text: String,
    pub target: Option<String>,
}

enum State {
    Idle,
    Listening(Box<dyn ActiveDictation>),
    Finalizing(Arc<AtomicBool>),
    Delivering,
}

impl State {
    fn public(&self) -> PublicState {
        match self {
            Self::Idle => PublicState::Idle,
            Self::Listening(_) => PublicState::Listening,
            Self::Finalizing(_) => PublicState::Finalizing,
            Self::Delivering => PublicState::Delivering,
        }
    }
}

pub struct Service<P: Pipeline> {
    pipeline: Arc<P>,
    state: Arc<Mutex<State>>,
}

impl<P: Pipeline> Clone for Service<P> {
    fn clone(&self) -> Self {
        Self {
            pipeline: Arc::clone(&self.pipeline),
            state: Arc::clone(&self.state),
        }
    }
}

impl<P: Pipeline> Service<P> {
    pub fn new(pipeline: P) -> Self {
        Self {
            pipeline: Arc::new(pipeline),
            state: Arc::new(Mutex::new(State::Idle)),
        }
    }

    pub fn handle(&self, request: &ControlRequest) -> ControlResponse {
        match self.try_handle(request) {
            Ok(message) => self.response(true, message),
            Err(error) => self.response(false, Some(error.to_string())),
        }
    }

    fn try_handle(&self, request: &ControlRequest) -> Result<Option<String>> {
        match request {
            ControlRequest::Status => Ok(None),
            ControlRequest::Shutdown => Err(Error::message("shutdown is handled by the daemon")),
            ControlRequest::Toggle => self.toggle(),
            ControlRequest::Cancel => self.cancel(),
        }
    }

    fn toggle(&self) -> Result<Option<String>> {
        let mut state = self.lock_state()?;
        match &*state {
            State::Idle => {
                *state = State::Listening(self.pipeline.start()?);
                Ok(Some("recording started".into()))
            }
            State::Listening(_) => {
                let State::Listening(active) = std::mem::replace(&mut *state, State::Idle) else {
                    return Err(Error::message("dictation state changed unexpectedly"));
                };
                let cancelled = Arc::new(AtomicBool::new(false));
                *state = State::Finalizing(Arc::clone(&cancelled));
                drop(state);
                self.spawn_finalizer(active, cancelled);
                Ok(Some("recording stopped".into()))
            }
            State::Finalizing(cancelled) => {
                cancelled.store(true, Ordering::Release);
                Ok(Some("pending delivery cancelled".into()))
            }
            State::Delivering => Err(Error::message("delivery is already in progress")),
        }
    }

    fn cancel(&self) -> Result<Option<String>> {
        let mut state = self.lock_state()?;
        match &*state {
            State::Idle => Ok(Some("nothing to cancel".into())),
            State::Listening(_) => {
                let State::Listening(active) = std::mem::replace(&mut *state, State::Idle) else {
                    return Err(Error::message("dictation state changed unexpectedly"));
                };
                active.cancel()?;
                Ok(Some("recording cancelled".into()))
            }
            State::Finalizing(cancelled) => {
                cancelled.store(true, Ordering::Release);
                Ok(Some("pending delivery cancelled".into()))
            }
            State::Delivering => Err(Error::message("delivery is already in progress")),
        }
    }

    fn spawn_finalizer(&self, active: Box<dyn ActiveDictation>, cancelled: Arc<AtomicBool>) {
        let service = self.clone();
        thread::spawn(move || {
            let result = active.finish(&cancelled);
            if cancelled.load(Ordering::Acquire) {
                service.set_idle();
                return;
            }
            let dictation = match result {
                Ok(dictation) if !dictation.text.trim().is_empty() => dictation,
                Ok(_) => {
                    service.set_idle();
                    return;
                }
                Err(error) => {
                    eprintln!("whispers: finalization failed: {error}");
                    service.set_idle();
                    return;
                }
            };
            if let Ok(mut state) = service.lock_state() {
                *state = State::Delivering;
            }
            if !cancelled.load(Ordering::Acquire) {
                if let Err(error) = service.pipeline.deliver(&dictation) {
                    eprintln!("whispers: delivery failed: {error}");
                }
            }
            service.set_idle();
        });
    }

    fn set_idle(&self) {
        if let Ok(mut state) = self.state.lock() {
            *state = State::Idle;
        }
    }

    fn response(&self, ok: bool, message: Option<String>) -> ControlResponse {
        let state = self
            .state
            .lock()
            .map_or(PublicState::Idle, |state| state.public());
        ControlResponse {
            ok,
            state: state.as_str().into(),
            message,
        }
    }

    fn lock_state(&self) -> Result<std::sync::MutexGuard<'_, State>> {
        self.state
            .lock()
            .map_err(|_| Error::message("dictation state lock is poisoned"))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    use super::*;

    struct FakePipeline {
        delivered: Arc<Mutex<Vec<String>>>,
        starts: Arc<AtomicUsize>,
    }

    struct FakeDictation;

    impl ActiveDictation for FakeDictation {
        fn finish(self: Box<Self>, _cancelled: &AtomicBool) -> Result<CompletedDictation> {
            Ok(CompletedDictation {
                text: "this/that".into(),
                target: None,
            })
        }

        fn cancel(self: Box<Self>) -> Result<()> {
            Ok(())
        }
    }

    impl Pipeline for FakePipeline {
        fn start(&self) -> Result<Box<dyn ActiveDictation>> {
            self.starts.fetch_add(1, Ordering::Relaxed);
            Ok(Box::new(FakeDictation))
        }

        fn deliver(&self, dictation: &CompletedDictation) -> Result<()> {
            self.delivered
                .lock()
                .map_err(|_| Error::message("test lock poisoned"))?
                .push(dictation.text.clone());
            Ok(())
        }
    }

    fn service() -> (Service<FakePipeline>, Arc<Mutex<Vec<String>>>) {
        let delivered = Arc::new(Mutex::new(Vec::new()));
        let service = Service::new(FakePipeline {
            delivered: Arc::clone(&delivered),
            starts: Arc::new(AtomicUsize::new(0)),
        });
        (service, delivered)
    }

    #[test]
    fn toggle_traverses_the_complete_boundary() {
        let (service, delivered) = service();
        assert_eq!(service.handle(&ControlRequest::Toggle).state, "listening");
        assert!(service.handle(&ControlRequest::Toggle).ok);
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            if delivered
                .lock()
                .is_ok_and(|items| items.as_slice() == ["this/that"])
            {
                break;
            }
            assert!(Instant::now() < deadline, "delivery did not complete");
            thread::yield_now();
        }
        assert_eq!(service.handle(&ControlRequest::Status).state, "idle");
    }

    #[test]
    fn cancel_while_listening_returns_to_idle() {
        let (service, _) = service();
        service.handle(&ControlRequest::Toggle);
        assert_eq!(service.handle(&ControlRequest::Cancel).state, "idle");
    }

    #[test]
    fn delivery_is_not_reported_as_cancellable_after_dispatch() {
        let (service, _) = service();
        if let Ok(mut state) = service.state.lock() {
            *state = State::Delivering;
        }
        let response = service.handle(&ControlRequest::Cancel);
        assert!(!response.ok);
        assert_eq!(response.state, "delivering");
        assert_eq!(
            response.message.as_deref(),
            Some("delivery is already in progress")
        );
    }
}

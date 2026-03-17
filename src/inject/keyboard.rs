use evdev::uinput::VirtualDevice;
use evdev::{AttributeSet, EventType, InputEvent, KeyCode};

use crate::error::{Result, WhsprError};

pub(super) struct VirtualKeyboardAdapter {
    device: VirtualDevice,
}

impl VirtualKeyboardAdapter {
    pub(super) fn new() -> Result<Self> {
        Ok(Self {
            device: build_virtual_device()?,
        })
    }

    pub(super) fn emit_paste_combo(&mut self) -> Result<()> {
        self.device
            .emit(&[
                InputEvent::new(EventType::KEY.0, KeyCode::KEY_LEFTCTRL.0, 1),
                InputEvent::new(EventType::KEY.0, KeyCode::KEY_LEFTSHIFT.0, 1),
            ])
            .map_err(|e| WhsprError::Injection(format!("paste modifier press: {e}")))?;
        std::thread::sleep(std::time::Duration::from_millis(12));

        self.device
            .emit(&[
                InputEvent::new(EventType::KEY.0, KeyCode::KEY_V.0, 1),
                InputEvent::new(EventType::KEY.0, KeyCode::KEY_V.0, 0),
            ])
            .map_err(|e| WhsprError::Injection(format!("paste key press: {e}")))?;
        std::thread::sleep(std::time::Duration::from_millis(12));

        self.device
            .emit(&[
                InputEvent::new(EventType::KEY.0, KeyCode::KEY_LEFTSHIFT.0, 0),
                InputEvent::new(EventType::KEY.0, KeyCode::KEY_LEFTCTRL.0, 0),
            ])
            .map_err(|e| WhsprError::Injection(format!("paste modifier release: {e}")))?;

        Ok(())
    }

    pub(super) fn emit_backspaces(&mut self, count: usize) -> Result<()> {
        for _ in 0..count {
            self.device
                .emit(&[
                    InputEvent::new(EventType::KEY.0, KeyCode::KEY_BACKSPACE.0, 1),
                    InputEvent::new(EventType::KEY.0, KeyCode::KEY_BACKSPACE.0, 0),
                ])
                .map_err(|e| WhsprError::Injection(format!("backspace key press: {e}")))?;
            std::thread::sleep(std::time::Duration::from_millis(6));
        }

        Ok(())
    }
}

fn build_virtual_device() -> Result<VirtualDevice> {
    let mut keys = AttributeSet::<KeyCode>::new();
    keys.insert(KeyCode::KEY_LEFTCTRL);
    keys.insert(KeyCode::KEY_LEFTSHIFT);
    keys.insert(KeyCode::KEY_V);
    keys.insert(KeyCode::KEY_BACKSPACE);

    VirtualDevice::builder()
        .map_err(|e| WhsprError::Injection(format!("uinput: {e}")))?
        .name("whispers-keyboard")
        .with_keys(&keys)
        .map_err(|e| WhsprError::Injection(format!("uinput keys: {e}")))?
        .build()
        .map_err(|e| WhsprError::Injection(format!("uinput build: {e}")))
}

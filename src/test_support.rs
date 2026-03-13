use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, OnceLock};

static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

pub fn env_lock() -> MutexGuard<'static, ()> {
    ENV_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("env lock should not be poisoned")
}

pub struct EnvVarGuard {
    saved: Vec<(String, Option<String>)>,
}

impl EnvVarGuard {
    pub fn capture(keys: &[&str]) -> Self {
        let mut saved = Vec::with_capacity(keys.len());
        for key in keys {
            saved.push(((*key).to_string(), std::env::var(key).ok()));
        }
        Self { saved }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        for (key, value) in &self.saved {
            match value {
                Some(v) => set_env(key, v),
                None => remove_env(key),
            }
        }
    }
}

pub fn set_env(key: &str, value: &str) {
    // SAFETY: Unit tests serialize env mutations via env_lock().
    unsafe {
        std::env::set_var(key, value);
    }
}

pub fn remove_env(key: &str) {
    // SAFETY: Unit tests serialize env mutations via env_lock().
    unsafe {
        std::env::remove_var(key);
    }
}

pub fn unique_temp_path(prefix: &str, ext: &str) -> PathBuf {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("time should be after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{}-{now}.{ext}", std::process::id()))
}

pub fn unique_temp_dir(prefix: &str) -> PathBuf {
    let path = unique_temp_path(prefix, "dir");
    std::fs::create_dir_all(&path).expect("create temp directory");
    path
}

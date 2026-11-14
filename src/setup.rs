use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Deserialize;

use crate::config::{CONFIG_VERSION, Config, DEFAULT_FINAL_MODEL, LEGACY_ENGLISH_FINAL_MODEL};
use crate::vocabulary::Vocabulary;
use crate::{Error, Result, paths};

const RUNTIME_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.4/sherpa-onnx-v1.13.4-linux-x64-shared-no-tts-lib.tar.bz2";
const RUNTIME_SHA256: &str = "8a2c6d5f8d04e651c90e71ba3ee8b08dedb2b741ff5f316e12aa629423f91c9f";
const RUNTIME_ROOT: &str = "sherpa-onnx-v1.13.4-linux-x64-shared-no-tts-lib";
const MODEL_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11.tar.bz2";
const MODEL_SHA256: &str = "c6bf5e0df765f9d5b43bc9e0536d4b4b3e7d40bdf5ecf13e45f134c51c05ae3a";
const MODEL_ROOT: &str = "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11";
const MODEL_NAME: &str = "nemotron-3.5-asr-streaming-0.6b-560ms-int8";
const FINAL_MODEL_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2";
const FINAL_MODEL_SHA256: &str = "1495e91b59d7fc53ddca91f3431758ec99f8fa253fdb0e1e851be0f72b14b0f9";
const FINAL_MODEL_ROOT: &str = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8";
const LLAMA_URL: &str = "https://github.com/ggml-org/llama.cpp/releases/download/b10258/llama-b10258-bin-ubuntu-vulkan-x64.tar.gz";
const LLAMA_SHA256: &str = "21e2ca510aa05f5814adfab82e4cb2d31641596aa9a8780ae05132c731ebe75a";
const LLAMA_ROOT: &str = "llama-b10258";
const ENHANCEMENT_URL: &str = "https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF/resolve/4168f45a16a1290d65a4ec0fa312ae917a4c15d6/Qwen_Qwen3.5-4B-Q4_K_M.gguf";
const ENHANCEMENT_SHA256: &str = "13c16f426047e2de38cd075bdade4a7bcbc8c774384876f677740cda65f8a983";
const ENHANCEMENT_FILE: &str = "Qwen_Qwen3.5-4B-Q4_K_M.gguf";

pub struct SetupReport {
    pub migrated_config: bool,
    pub vocabulary_entries: usize,
    pub service_started: bool,
}

struct ConfigMigration {
    migrated: bool,
    legacy_vocabulary: Option<LegacyVocabularyPaths>,
    backup: Option<PathBuf>,
}

struct LegacyVocabularyPaths {
    dictionary: PathBuf,
    glossary: PathBuf,
}

pub fn run(offline_pack: Option<&Path>, install_service: bool) -> Result<SetupReport> {
    if std::env::consts::OS != "linux" || std::env::consts::ARCH != "x86_64" {
        return Err(Error::message(
            "managed runtime setup currently supports Linux x86_64 only",
        ));
    }
    let data_dir = paths::data_dir()?;
    fs::create_dir_all(&data_dir)?;
    let config_path = Config::path()?;
    let migration = migrate_config(&config_path, &data_dir)?;
    let vocabulary_entries = match migrate_vocabulary(
        &data_dir,
        migration.legacy_vocabulary.as_ref(),
    ) {
        Ok(entries) => entries,
        Err(error) => {
            if let Some(backup) = &migration.backup {
                fs::copy(backup, &config_path).map_err(|restore_error| {
                        Error::message(format!(
                            "{error}; additionally could not restore legacy configuration: {restore_error}"
                        ))
                    })?;
            }
            return Err(error);
        }
    };
    let config = Config::load(None)?;
    let install_asr = !config.recognition.uses_whisper_cli();
    if let Some(pack) = offline_pack {
        install_from_pack(pack, &data_dir, config.enhancement.enabled, install_asr)?;
    } else {
        install_from_network(&data_dir, config.enhancement.enabled, install_asr)?;
    }
    if install_asr {
        validate_asr_installation(&data_dir)?;
    } else {
        validate_whisper_cli_installation(&config)?;
    }
    if config.enhancement.enabled {
        validate_enhancement_installation(&data_dir)?;
    }
    let service_started = install_service && install_user_service()?;
    Ok(SetupReport {
        migrated_config: migration.migrated,
        vocabulary_entries,
        service_started,
    })
}

fn migrate_config(path: &Path, data_dir: &Path) -> Result<ConfigMigration> {
    let contents = match fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Config::default().save_to(path)?;
            return Ok(ConfigMigration {
                migrated: false,
                legacy_vocabulary: None,
                backup: None,
            });
        }
        Err(error) => return Err(error.into()),
    };
    let value: toml::Value = toml::from_str(&contents)?;
    if value.get("version").and_then(toml::Value::as_integer) == Some(i64::from(CONFIG_VERSION)) {
        let mut config = Config::load(Some(path))?;
        if config.recognition.final_model == LEGACY_ENGLISH_FINAL_MODEL {
            config.recognition.final_model = DEFAULT_FINAL_MODEL.into();
        }
        config.save_to(path)?;
        return Ok(ConfigMigration {
            migrated: false,
            legacy_vocabulary: None,
            backup: None,
        });
    }

    let dictionary = value_at(&value, &["personalization", "dictionary_path"])
        .filter(|path| !path.is_empty())
        .map_or_else(
            || legacy_data_path(data_dir, "dictionary.toml"),
            expand_home,
        );
    let glossary = value_at(&value, &["rewrite", "glossary_path"])
        .or_else(|| value_at(&value, &["agentic_rewrite", "glossary_path"]))
        .filter(|path| !path.is_empty())
        .map_or_else(
            || legacy_data_path(data_dir, "technical-glossary.toml"),
            expand_home,
        );

    let backup = backup_path(path);
    fs::copy(path, &backup)?;
    let mut config = Config::default();
    if let Some(device) = value_at(&value, &["audio", "device"]) {
        config.audio.device = device.to_string();
    }
    if let Some(enabled) = bool_at(&value, &["feedback", "enabled"]) {
        config.audio.feedback = enabled;
    }
    if let Some(language) = value_at(&value, &["transcription", "language"]) {
        config.language = language.to_string();
    }
    if let Some(model) =
        value_at(&value, &["transcription", "model_path"]).filter(|model| !model.is_empty())
    {
        config.recognition.fallback_model = expand_home(model).to_string_lossy().into_owned();
    }
    if let Some(model) =
        value_at(&value, &["rewrite", "model_path"]).filter(|model| !model.is_empty())
    {
        config.enhancement.model = expand_home(model).to_string_lossy().into_owned();
    } else {
        let existing = data_dir.join("rewrite").join(&config.enhancement.model);
        if existing.is_file() {
            config.enhancement.model = existing.to_string_lossy().into_owned();
        }
    }
    if let Some(timeout) = integer_at(&value, &["rewrite", "timeout_ms"]) {
        config.enhancement.timeout_ms = u64::try_from(timeout.max(1))
            .map_err(|_| Error::message("rewrite timeout is invalid"))?;
    }
    config.save_to(path)?;
    Ok(ConfigMigration {
        migrated: true,
        legacy_vocabulary: Some(LegacyVocabularyPaths {
            dictionary,
            glossary,
        }),
        backup: Some(backup),
    })
}

fn migrate_vocabulary(
    data_dir: &Path,
    legacy_paths: Option<&LegacyVocabularyPaths>,
) -> Result<usize> {
    let target = data_dir.join("vocabulary.toml");
    let mut vocabulary = Vocabulary::load_from(&target)?;
    if let Some(paths) = legacy_paths {
        import_dictionary(&paths.dictionary, &mut vocabulary)?;
        import_glossary(&paths.glossary, &mut vocabulary)?;
    }
    vocabulary.save_to(&target)?;
    Ok(vocabulary.entries.len())
}

#[derive(Deserialize)]
struct LegacyDictionary {
    #[serde(default)]
    entries: Vec<LegacyReplacement>,
}

#[derive(Deserialize)]
struct LegacyReplacement {
    phrase: String,
    replace: String,
}

#[derive(Deserialize)]
struct LegacyGlossary {
    #[serde(default)]
    entries: Vec<LegacyTerm>,
}

#[derive(Deserialize)]
struct LegacyTerm {
    term: String,
    #[serde(default)]
    aliases: Vec<String>,
}

fn import_dictionary(path: &Path, vocabulary: &mut Vocabulary) -> Result<()> {
    let Some(contents) = read_optional(path)? else {
        return Ok(());
    };
    let legacy: LegacyDictionary = toml::from_str(&contents)?;
    for entry in legacy.entries {
        vocabulary.add(&entry.replace, vec![entry.phrase])?;
    }
    Ok(())
}

fn import_glossary(path: &Path, vocabulary: &mut Vocabulary) -> Result<()> {
    let Some(contents) = read_optional(path)? else {
        return Ok(());
    };
    let legacy: LegacyGlossary = toml::from_str(&contents)?;
    for entry in legacy.entries {
        if !entry.aliases.is_empty() {
            vocabulary.add(&entry.term, entry.aliases)?;
        }
    }
    Ok(())
}

fn install_from_network(
    data_dir: &Path,
    install_enhancement: bool,
    install_asr: bool,
) -> Result<()> {
    let downloads = data_dir.join("downloads");
    fs::create_dir_all(&downloads)?;
    if install_asr {
        install_archive(
            Asset {
                name: "sherpa-runtime",
                extension: "tar.bz2",
                extract_flag: "-xjf",
                url: RUNTIME_URL,
                sha256: RUNTIME_SHA256,
                archive_root: RUNTIME_ROOT,
            },
            &downloads,
            &data_dir.join("runtimes/sherpa-onnx/current"),
        )?;
        install_archive(
            Asset {
                name: "nemotron-model",
                extension: "tar.bz2",
                extract_flag: "-xjf",
                url: MODEL_URL,
                sha256: MODEL_SHA256,
                archive_root: MODEL_ROOT,
            },
            &downloads,
            &data_dir.join("models").join(MODEL_NAME),
        )?;
        install_archive(
            Asset {
                name: "parakeet-v3-final-model",
                extension: "tar.bz2",
                extract_flag: "-xjf",
                url: FINAL_MODEL_URL,
                sha256: FINAL_MODEL_SHA256,
                archive_root: FINAL_MODEL_ROOT,
            },
            &downloads,
            &data_dir.join("models").join(DEFAULT_FINAL_MODEL),
        )?;
    }
    if install_enhancement {
        install_archive(
            Asset {
                name: "llama-runtime",
                extension: "tar.gz",
                extract_flag: "-xzf",
                url: LLAMA_URL,
                sha256: LLAMA_SHA256,
                archive_root: LLAMA_ROOT,
            },
            &downloads,
            &data_dir.join(paths::MANAGED_LLAMA_RUNTIME),
        )?;
        install_file(
            ENHANCEMENT_URL,
            ENHANCEMENT_SHA256,
            &downloads.join("enhancement-model.gguf"),
            &data_dir.join("rewrite").join(ENHANCEMENT_FILE),
        )?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Asset {
    name: &'static str,
    extension: &'static str,
    extract_flag: &'static str,
    url: &'static str,
    sha256: &'static str,
    archive_root: &'static str,
}

fn install_archive(asset: Asset, downloads: &Path, target: &Path) -> Result<()> {
    if target.is_dir() {
        return Ok(());
    }
    if target.exists() {
        return Err(Error::message(format!(
            "installation target is not a directory: {}",
            target.display()
        )));
    }
    let archive = downloads.join(format!("{}.{}", asset.name, asset.extension));
    download_verified(asset.url, &archive, asset.sha256, "download runtime asset")?;
    let temporary = TemporaryDirectory::create(downloads, asset.name)?;
    run_command(
        Command::new("tar")
            .arg(asset.extract_flag)
            .arg(&archive)
            .arg("-C")
            .arg(temporary.path()),
        "extract runtime asset",
    )?;
    let extracted = temporary.path().join(asset.archive_root);
    if !extracted.is_dir() {
        return Err(Error::message(format!(
            "runtime archive did not contain {}",
            asset.archive_root
        )));
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::rename(extracted, target)?;
    fs::remove_file(archive)?;
    Ok(())
}

fn install_file(url: &str, sha256: &str, download: &Path, target: &Path) -> Result<()> {
    if target.is_file() && verify_sha256(target, sha256).is_ok() {
        return Ok(());
    }
    if target.exists() && !target.is_file() {
        return Err(Error::message(format!(
            "installation target is not a file: {}",
            target.display()
        )));
    }
    download_verified(url, download, sha256, "download enhancement model")?;
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::rename(download, target)?;
    Ok(())
}

fn install_from_pack(
    pack: &Path,
    data_dir: &Path,
    install_enhancement: bool,
    install_asr: bool,
) -> Result<()> {
    if !pack.is_dir() {
        return Err(Error::message(
            "--runtime-pack must name an unpacked runtime directory",
        ));
    }
    if install_asr {
        copy_install(
            &pack.join("runtimes/sherpa-onnx/current"),
            &data_dir.join("runtimes/sherpa-onnx/current"),
        )?;
        copy_install(
            &pack.join("models").join(MODEL_NAME),
            &data_dir.join("models").join(MODEL_NAME),
        )?;
        copy_install(
            &pack.join("models").join(DEFAULT_FINAL_MODEL),
            &data_dir.join("models").join(DEFAULT_FINAL_MODEL),
        )?;
    }
    if install_enhancement {
        copy_install(
            &pack.join(paths::MANAGED_LLAMA_RUNTIME),
            &data_dir.join(paths::MANAGED_LLAMA_RUNTIME),
        )?;
        copy_file_install(
            &pack.join("rewrite").join(ENHANCEMENT_FILE),
            &data_dir.join("rewrite").join(ENHANCEMENT_FILE),
        )?;
    }
    Ok(())
}

fn copy_file_install(source: &Path, target: &Path) -> Result<()> {
    if target.is_file() {
        return Ok(());
    }
    if !source.is_file() {
        return Err(Error::message(format!(
            "runtime pack is missing {}",
            source.display()
        )));
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut temporary = target.as_os_str().to_os_string();
    temporary.push(format!(".{}.tmp", std::process::id()));
    let temporary = PathBuf::from(temporary);
    fs::copy(source, &temporary)?;
    fs::rename(temporary, target)?;
    Ok(())
}

fn copy_install(source: &Path, target: &Path) -> Result<()> {
    if target.is_dir() {
        return Ok(());
    }
    if !source.is_dir() {
        return Err(Error::message(format!(
            "runtime pack is missing {}",
            source.display()
        )));
    }
    let parent = target
        .parent()
        .ok_or_else(|| Error::message("runtime target has no parent"))?;
    fs::create_dir_all(parent)?;
    let temporary = TemporaryDirectory::create(parent, "offline-pack")?;
    let staged = temporary.path().join("content");
    copy_directory(source, &staged)?;
    fs::rename(staged, target)?;
    Ok(())
}

fn copy_directory(source: &Path, target: &Path) -> Result<()> {
    fs::create_dir_all(target)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let destination = target.join(entry.file_name());
        if file_type.is_dir() {
            copy_directory(&entry.path(), &destination)?;
        } else if file_type.is_file() {
            fs::copy(entry.path(), destination)?;
        } else {
            return Err(Error::message(format!(
                "runtime packs cannot contain links or special files: {}",
                entry.path().display()
            )));
        }
    }
    Ok(())
}

pub fn validate_asr_installation(data_dir: &Path) -> Result<()> {
    let required = [
        data_dir.join("runtimes/sherpa-onnx/current/lib/libsherpa-onnx-c-api.so"),
        data_dir
            .join("models")
            .join(MODEL_NAME)
            .join("encoder.int8.onnx"),
        data_dir
            .join("models")
            .join(MODEL_NAME)
            .join("decoder.int8.onnx"),
        data_dir
            .join("models")
            .join(MODEL_NAME)
            .join("joiner.int8.onnx"),
        data_dir.join("models").join(MODEL_NAME).join("tokens.txt"),
        data_dir
            .join("models")
            .join(DEFAULT_FINAL_MODEL)
            .join("encoder.int8.onnx"),
        data_dir
            .join("models")
            .join(DEFAULT_FINAL_MODEL)
            .join("decoder.int8.onnx"),
        data_dir
            .join("models")
            .join(DEFAULT_FINAL_MODEL)
            .join("joiner.int8.onnx"),
        data_dir
            .join("models")
            .join(DEFAULT_FINAL_MODEL)
            .join("tokens.txt"),
    ];
    for path in required {
        if !path.is_file() {
            return Err(Error::message(format!(
                "runtime installation is incomplete: {} is missing",
                path.display()
            )));
        }
    }
    Ok(())
}

fn validate_whisper_cli_installation(config: &Config) -> Result<()> {
    let available = Command::new("whisper-cli")
        .arg("--help")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success());
    if !available {
        return Err(Error::message(
            "[recognition].runtime = \"whisper-cli\" requires whisper-cli in PATH",
        ));
    }
    let model = paths::fallback_model(&config.recognition.fallback_model)?;
    if !model.is_file() {
        return Err(Error::message(format!(
            "whisper-cli model is missing: {}",
            model.display()
        )));
    }
    Ok(())
}

pub fn validate_enhancement_installation(data_dir: &Path) -> Result<()> {
    for path in [
        data_dir
            .join(paths::MANAGED_LLAMA_RUNTIME)
            .join("llama-server"),
        data_dir
            .join(paths::MANAGED_LLAMA_RUNTIME)
            .join("libggml-vulkan.so"),
        data_dir.join("rewrite").join(ENHANCEMENT_FILE),
    ] {
        if !path.is_file() {
            return Err(Error::message(format!(
                "enhancement installation is incomplete: {} is missing",
                path.display()
            )));
        }
    }
    Ok(())
}

pub fn validate_installation(data_dir: &Path) -> Result<()> {
    validate_asr_installation(data_dir)?;
    validate_enhancement_installation(data_dir)
}

fn install_user_service() -> Result<bool> {
    let executable = std::env::current_exe()?.canonicalize()?;
    let unit_dir = paths::config_dir()?
        .parent()
        .ok_or_else(|| Error::message("configuration directory has no parent"))?
        .join("systemd/user");
    fs::create_dir_all(&unit_dir)?;
    let unit = user_service_unit(&executable);
    fs::write(unit_dir.join("whispers.service"), unit)?;
    let reloaded = Command::new("systemctl")
        .args(["--user", "daemon-reload"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success());
    let enabled = reloaded
        && Command::new("systemctl")
            .args(["--user", "enable", "whispers.service"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success());
    let started = enabled
        && Command::new("systemctl")
            .args(["--user", "restart", "whispers.service"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success());
    Ok(started)
}

fn user_service_unit(executable: &Path) -> String {
    format!(
        "[Unit]\nDescription=Whispers local dictation service\nAfter=graphical-session.target\nConditionEnvironment=WAYLAND_DISPLAY\n\n[Service]\nExecStart={} daemon\nRestart=on-failure\n\n[Install]\nWantedBy=default.target\n",
        systemd_escape(executable)
    )
}

fn systemd_escape(path: &Path) -> String {
    let mut escaped = String::from("\"");
    for character in path.to_string_lossy().chars() {
        match character {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            _ => escaped.push(character),
        }
    }
    escaped.push('"');
    escaped
}

fn verify_sha256(path: &Path, expected: &str) -> Result<()> {
    let output = Command::new("sha256sum").arg(path).output()?;
    if !output.status.success() {
        return Err(Error::message("sha256sum failed"));
    }
    let actual = String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_string();
    if actual != expected {
        return Err(Error::message(format!(
            "checksum mismatch for {}: expected {expected}, got {actual}",
            path.display()
        )));
    }
    Ok(())
}

fn download_verified(url: &str, path: &Path, sha256: &str, description: &str) -> Result<()> {
    if path.is_file() && verify_sha256(path, sha256).is_ok() {
        return Ok(());
    }
    let first_attempt = download(url, path, description).and_then(|()| verify_sha256(path, sha256));
    if first_attempt.is_ok() {
        return Ok(());
    }
    if path.exists() {
        fs::remove_file(path)?;
    }
    download(url, path, description)?;
    verify_sha256(path, sha256)
}

fn download(url: &str, path: &Path, description: &str) -> Result<()> {
    run_command(
        Command::new("curl")
            .args(["--fail", "--location", "--show-error", "--continue-at", "-"])
            .arg("--output")
            .arg(path)
            .arg(url),
        description,
    )
}

fn run_command(command: &mut Command, description: &str) -> Result<()> {
    let status = command
        .status()
        .map_err(|error| Error::message(format!("could not {description}: {error}")))?;
    if !status.success() {
        return Err(Error::message(format!("could not {description}")));
    }
    Ok(())
}

struct TemporaryDirectory(PathBuf);

impl TemporaryDirectory {
    fn create(parent: &Path, name: &str) -> Result<Self> {
        let path = parent.join(format!(".{name}.{}.tmp", std::process::id()));
        if path.exists() {
            fs::remove_dir_all(&path)?;
        }
        fs::create_dir(&path)?;
        Ok(Self(path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn read_optional(path: &Path) -> Result<Option<String>> {
    match fs::read_to_string(path) {
        Ok(contents) => Ok(Some(contents)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn value_at<'a>(value: &'a toml::Value, path: &[&str]) -> Option<&'a str> {
    path.iter()
        .try_fold(value, |current, key| current.get(key))?
        .as_str()
}

fn integer_at(value: &toml::Value, path: &[&str]) -> Option<i64> {
    path.iter()
        .try_fold(value, |current, key| current.get(key))?
        .as_integer()
}

fn bool_at(value: &toml::Value, path: &[&str]) -> Option<bool> {
    path.iter()
        .try_fold(value, |current, key| current.get(key))?
        .as_bool()
}

fn expand_home(path: &str) -> PathBuf {
    path.strip_prefix("~/").map_or_else(
        || PathBuf::from(path),
        |suffix| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map_or_else(|| PathBuf::from(path), |home| home.join(suffix))
        },
    )
}

fn legacy_data_path(data_dir: &Path, name: &str) -> PathBuf {
    std::env::var_os("HOME").map_or_else(
        || data_dir.join(name),
        |home| PathBuf::from(home).join(".local/share/whispers").join(name),
    )
}

fn backup_path(path: &Path) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs());
    let mut backup = path.as_os_str().to_os_string();
    backup.push(format!(".backup-{timestamp}"));
    PathBuf::from(backup)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrates_legacy_configuration_without_deleting_it() {
        let root = std::env::temp_dir().join(format!("whispers-setup-test-{}", std::process::id()));
        let config = root.join("config.toml");
        let data = root.join("data");
        let _ = fs::remove_dir_all(&root);
        assert!(fs::create_dir_all(&root).is_ok());
        assert!(
            fs::write(
                &config,
                "[audio]\ndevice = \"mic\"\n[transcription]\nlanguage = \"en\"\nmodel_path = \"/tmp/model.bin\"\n[rewrite]\ntimeout_ms = 9000\n"
            )
            .is_ok()
        );
        let migration = migrate_config(&config, &data);
        assert!(migration.is_ok_and(|migration| migration.migrated));
        let migrated = Config::load(Some(&config));
        assert!(migrated.is_ok());
        let migrated = migrated.unwrap_or_default();
        assert_eq!(migrated.audio.device, "mic");
        assert_eq!(migrated.language, "en");
        assert_eq!(migrated.enhancement.timeout_ms, 9_000);
        let backups = fs::read_dir(&root)
            .map(|entries| {
                entries
                    .filter_map(std::result::Result::ok)
                    .filter(|entry| entry.file_name().to_string_lossy().contains("backup"))
                    .count()
            })
            .unwrap_or_default();
        assert_eq!(backups, 1);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn refreshes_default_fields_in_current_configuration() {
        let root = std::env::temp_dir().join(format!(
            "whispers-current-config-test-{}",
            std::process::id()
        ));
        let config = root.join("config.toml");
        let _ = fs::remove_dir_all(&root);
        assert!(fs::create_dir_all(&root).is_ok());
        assert!(fs::write(&config, "version = 2\n").is_ok());

        let migration = migrate_config(&config, &root);
        assert!(migration.is_ok_and(|migration| !migration.migrated));
        let contents = fs::read_to_string(&config).unwrap_or_default();
        assert!(contents.contains("final_model = \"nemo-parakeet-tdt-0.6b-v3-int8\""));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn upgrades_the_previous_english_only_default() {
        let root = std::env::temp_dir().join(format!(
            "whispers-english-default-test-{}",
            std::process::id()
        ));
        let config = root.join("config.toml");
        let _ = fs::remove_dir_all(&root);
        assert!(fs::create_dir_all(&root).is_ok());
        let mut previous = Config::default();
        previous.recognition.final_model = LEGACY_ENGLISH_FINAL_MODEL.into();
        assert!(previous.save_to(&config).is_ok());

        let migration = migrate_config(&config, &root);
        assert!(migration.is_ok_and(|migration| !migration.migrated));
        let upgraded = Config::load(Some(&config)).unwrap_or_default();
        assert_eq!(upgraded.recognition.final_model, DEFAULT_FINAL_MODEL);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn escapes_systemd_paths() {
        assert_eq!(
            systemd_escape(Path::new("/tmp/a path/whispers")),
            "\"/tmp/a path/whispers\""
        );
    }

    #[test]
    fn user_service_waits_for_wayland() {
        let unit = user_service_unit(Path::new("/tmp/a path/whispers"));

        assert!(unit.contains("ConditionEnvironment=WAYLAND_DISPLAY"));
        assert!(unit.contains("ExecStart=\"/tmp/a path/whispers\" daemon"));
    }

    #[test]
    fn replaces_an_existing_file_with_the_pinned_asset() {
        let root =
            std::env::temp_dir().join(format!("whispers-pinned-file-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        assert!(fs::create_dir_all(&root).is_ok());
        let source = root.join("source.gguf");
        let download = root.join("download.gguf");
        let target = root.join("target.gguf");
        assert!(fs::write(&source, b"replacement").is_ok());
        assert!(fs::write(&target, b"truncated").is_ok());
        let url = format!("file://{}", source.display());
        assert!(
            install_file(
                &url,
                "95713e9cbdd1dfcb2d4080c2537f418d43ca0da25f0d7d6631f4f7c97b89dc47",
                &download,
                &target,
            )
            .is_ok()
        );
        assert_eq!(
            fs::read(&target).ok().as_deref(),
            Some(b"replacement".as_slice())
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn imports_both_legacy_vocabulary_formats() {
        let root = std::env::temp_dir().join(format!(
            "whispers-vocabulary-migration-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        assert!(fs::create_dir_all(&root).is_ok());
        assert!(
            fs::write(
                root.join("dictionary.toml"),
                "[[entries]]\nphrase = \"seashell\"\nreplace = \"zsh\"\n"
            )
            .is_ok()
        );
        assert!(
            fs::write(
                root.join("technical-glossary.toml"),
                "[[entries]]\nterm = \"chezmoi\"\naliases = [\"chasse-moi\"]\n"
            )
            .is_ok()
        );
        let paths = LegacyVocabularyPaths {
            dictionary: root.join("dictionary.toml"),
            glossary: root.join("technical-glossary.toml"),
        };
        assert_eq!(migrate_vocabulary(&root, Some(&paths)).ok(), Some(2));
        let vocabulary = Vocabulary::load_from(&root.join("vocabulary.toml"));
        assert!(vocabulary.is_ok_and(|vocabulary| {
            vocabulary.apply("seashell and chasse-moi") == "zsh and chezmoi"
        }));
        assert!(root.join("dictionary.toml").is_file());
        assert!(root.join("technical-glossary.toml").is_file());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn imports_custom_legacy_paths_only_during_migration() {
        let root = std::env::temp_dir().join(format!(
            "whispers-custom-vocabulary-migration-test-{}",
            std::process::id()
        ));
        let config = root.join("config.toml");
        let data = root.join("data");
        let custom = root.join("custom");
        let _ = fs::remove_dir_all(&root);
        assert!(fs::create_dir_all(&custom).is_ok());
        assert!(
            fs::write(
                custom.join("dictionary.toml"),
                "[[entries]]\nphrase = \"seashell\"\nreplace = \"zsh\"\n"
            )
            .is_ok()
        );
        assert!(
            fs::write(
                &config,
                format!(
                    "[personalization]\ndictionary_path = {:?}\n[rewrite]\nglossary_path = {:?}\n",
                    custom.join("dictionary.toml").to_string_lossy(),
                    custom.join("glossary.toml").to_string_lossy()
                )
            )
            .is_ok()
        );
        let migration = migrate_config(&config, &data);
        assert!(migration.is_ok());
        let Some(migration) = migration.ok() else {
            return;
        };
        assert_eq!(
            migrate_vocabulary(&data, migration.legacy_vocabulary.as_ref()).ok(),
            Some(1)
        );
        let vocabulary = Vocabulary::load_from(&data.join("vocabulary.toml"));
        assert!(vocabulary.is_ok());
        let Some(mut vocabulary) = vocabulary.ok() else {
            return;
        };
        assert!(vocabulary.remove("zsh"));
        assert!(vocabulary.save_to(&data.join("vocabulary.toml")).is_ok());
        assert_eq!(migrate_vocabulary(&data, None).ok(), Some(0));
        assert!(
            Vocabulary::load_from(&data.join("vocabulary.toml"))
                .is_ok_and(|vocabulary| vocabulary.entries.is_empty())
        );
        let _ = fs::remove_dir_all(root);
    }
}

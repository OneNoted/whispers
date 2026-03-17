use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::error::{Result, WhsprError};

use super::normalized_words;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct DictionaryEntry {
    pub phrase: String,
    pub replace: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct SnippetEntry {
    pub name: String,
    pub text: String,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
struct DictionaryFile {
    entries: Vec<DictionaryEntry>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
struct SnippetFile {
    snippets: Vec<SnippetEntry>,
}

pub fn list_dictionary(config_override: Option<&Path>) -> Result<()> {
    let config = Config::load(config_override)?;
    let entries = read_dictionary_file(&config.resolved_dictionary_path())?;
    if entries.is_empty() {
        println!("No dictionary entries configured.");
        return Ok(());
    }

    for entry in entries {
        println!("{} -> {}", entry.phrase, entry.replace);
    }

    Ok(())
}

pub fn add_dictionary(config_override: Option<&Path>, phrase: &str, replace: &str) -> Result<()> {
    let config = Config::load(config_override)?;
    let path = config.resolved_dictionary_path();
    let mut entries = read_dictionary_file(&path)?;
    upsert_dictionary_entry(&mut entries, phrase, replace);
    write_dictionary_file(&path, &entries)?;
    println!("Added dictionary entry: {} -> {}", phrase, replace);
    println!("Dictionary updated: {}", path.display());
    Ok(())
}

pub fn remove_dictionary(config_override: Option<&Path>, phrase: &str) -> Result<()> {
    let config = Config::load(config_override)?;
    let path = config.resolved_dictionary_path();
    let mut entries = read_dictionary_file(&path)?;
    let removed = remove_dictionary_entry(&mut entries, phrase);
    write_dictionary_file(&path, &entries)?;
    if removed {
        println!("Removed dictionary entry: {}", phrase);
    } else {
        println!("No dictionary entry matched: {}", phrase);
    }
    println!("Dictionary updated: {}", path.display());
    Ok(())
}

pub fn list_snippets(config_override: Option<&Path>) -> Result<()> {
    let config = Config::load(config_override)?;
    let snippets = read_snippet_file(&config.resolved_snippets_path())?;
    if snippets.is_empty() {
        println!("No snippets configured.");
        return Ok(());
    }

    for snippet in snippets {
        println!("{} -> {}", snippet.name, snippet.text.replace('\n', "\\n"));
    }

    Ok(())
}

pub fn add_snippet(config_override: Option<&Path>, name: &str, text: &str) -> Result<()> {
    let config = Config::load(config_override)?;
    let path = config.resolved_snippets_path();
    let mut snippets = read_snippet_file(&path)?;
    upsert_snippet(&mut snippets, name, text);
    write_snippet_file(&path, &snippets)?;
    println!("Added snippet: {}", name);
    println!("Snippets updated: {}", path.display());
    Ok(())
}

pub fn remove_snippet(config_override: Option<&Path>, name: &str) -> Result<()> {
    let config = Config::load(config_override)?;
    let path = config.resolved_snippets_path();
    let mut snippets = read_snippet_file(&path)?;
    let removed = remove_snippet_entry(&mut snippets, name);
    write_snippet_file(&path, &snippets)?;
    if removed {
        println!("Removed snippet: {}", name);
    } else {
        println!("No snippet matched: {}", name);
    }
    println!("Snippets updated: {}", path.display());
    Ok(())
}

pub fn print_rewrite_instructions_path(config_override: Option<&Path>) -> Result<()> {
    let config = Config::load(config_override)?;
    match config.resolved_rewrite_instructions_path() {
        Some(path) => println!("{}", path.display()),
        None => println!("No rewrite instructions path configured."),
    }
    Ok(())
}

pub(super) fn load_custom_instructions(config: &Config) -> Result<String> {
    let Some(path) = config.resolved_rewrite_instructions_path() else {
        return Ok(String::new());
    };

    match std::fs::read_to_string(&path) {
        Ok(contents) => Ok(contents.trim().to_string()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(String::new()),
        Err(err) => Err(WhsprError::Config(format!(
            "failed to read rewrite instructions {}: {err}",
            path.display()
        ))),
    }
}

pub(super) fn read_dictionary_file(path: &Path) -> Result<Vec<DictionaryEntry>> {
    if !path.exists() {
        return Ok(Vec::new());
    }

    let contents = std::fs::read_to_string(path).map_err(|e| {
        WhsprError::Config(format!("failed to read dictionary {}: {e}", path.display()))
    })?;
    let file: DictionaryFile = toml::from_str(&contents).map_err(|e| {
        WhsprError::Config(format!(
            "failed to parse dictionary {}: {e}",
            path.display()
        ))
    })?;
    Ok(file.entries)
}

pub(super) fn write_dictionary_file(path: &Path, entries: &[DictionaryEntry]) -> Result<()> {
    write_parent(path)?;
    let file = DictionaryFile {
        entries: entries.to_vec(),
    };
    let contents = toml::to_string_pretty(&file)
        .map_err(|e| WhsprError::Config(format!("failed to encode dictionary: {e}")))?;
    std::fs::write(path, contents).map_err(|e| {
        WhsprError::Config(format!(
            "failed to write dictionary {}: {e}",
            path.display()
        ))
    })?;
    Ok(())
}

pub(super) fn read_snippet_file(path: &Path) -> Result<Vec<SnippetEntry>> {
    if !path.exists() {
        return Ok(Vec::new());
    }

    let contents = std::fs::read_to_string(path).map_err(|e| {
        WhsprError::Config(format!("failed to read snippets {}: {e}", path.display()))
    })?;
    let file: SnippetFile = toml::from_str(&contents).map_err(|e| {
        WhsprError::Config(format!("failed to parse snippets {}: {e}", path.display()))
    })?;
    Ok(file.snippets)
}

pub(super) fn write_snippet_file(path: &Path, snippets: &[SnippetEntry]) -> Result<()> {
    write_parent(path)?;
    let file = SnippetFile {
        snippets: snippets.to_vec(),
    };
    let contents = toml::to_string_pretty(&file)
        .map_err(|e| WhsprError::Config(format!("failed to encode snippets: {e}")))?;
    std::fs::write(path, contents).map_err(|e| {
        WhsprError::Config(format!("failed to write snippets {}: {e}", path.display()))
    })?;
    Ok(())
}

fn write_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            WhsprError::Config(format!(
                "failed to create directory {}: {e}",
                parent.display()
            ))
        })?;
    }
    Ok(())
}

fn upsert_dictionary_entry(entries: &mut Vec<DictionaryEntry>, phrase: &str, replace: &str) {
    let target = normalized_words(phrase);
    if let Some(existing) = entries
        .iter_mut()
        .find(|entry| normalized_words(&entry.phrase) == target)
    {
        existing.phrase = phrase.to_string();
        existing.replace = replace.to_string();
        return;
    }

    entries.push(DictionaryEntry {
        phrase: phrase.to_string(),
        replace: replace.to_string(),
    });
}

fn remove_dictionary_entry(entries: &mut Vec<DictionaryEntry>, phrase: &str) -> bool {
    let target = normalized_words(phrase);
    let before = entries.len();
    entries.retain(|entry| normalized_words(&entry.phrase) != target);
    before != entries.len()
}

fn upsert_snippet(snippets: &mut Vec<SnippetEntry>, name: &str, text: &str) {
    let target = normalized_words(name);
    if let Some(existing) = snippets
        .iter_mut()
        .find(|entry| normalized_words(&entry.name) == target)
    {
        existing.name = name.to_string();
        existing.text = text.to_string();
        return;
    }

    snippets.push(SnippetEntry {
        name: name.to_string(),
        text: text.to_string(),
    });
}

fn remove_snippet_entry(snippets: &mut Vec<SnippetEntry>, name: &str) -> bool {
    let target = normalized_words(name);
    let before = snippets.len();
    snippets.retain(|entry| normalized_words(&entry.name) != target);
    before != snippets.len()
}

#[cfg(test)]
mod tests {
    use super::{
        add_dictionary, add_snippet, load_custom_instructions, read_dictionary_file,
        read_snippet_file, remove_dictionary, remove_snippet,
    };
    use crate::config::Config;

    #[test]
    fn add_and_remove_dictionary_entries_roundtrip() {
        let _env_lock = crate::test_support::env_lock();
        let _guard = crate::test_support::EnvVarGuard::capture(&[
            "HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
        ]);
        let home = crate::test_support::unique_temp_dir("personalization-dict-home");
        crate::test_support::set_env("HOME", &home.to_string_lossy());
        crate::test_support::remove_env("XDG_CONFIG_HOME");
        crate::test_support::remove_env("XDG_DATA_HOME");

        add_dictionary(None, "wisper flow", "Wispr Flow").expect("add dictionary");
        let config = Config::load(None).expect("config");
        let entries = read_dictionary_file(&config.resolved_dictionary_path()).expect("read");
        assert_eq!(entries.len(), 1);

        remove_dictionary(None, "wisper flow").expect("remove dictionary");
        let entries = read_dictionary_file(&config.resolved_dictionary_path()).expect("read");
        assert!(entries.is_empty());
    }

    #[test]
    fn add_and_remove_snippets_roundtrip() {
        let _env_lock = crate::test_support::env_lock();
        let _guard = crate::test_support::EnvVarGuard::capture(&[
            "HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
        ]);
        let home = crate::test_support::unique_temp_dir("personalization-snippet-home");
        crate::test_support::set_env("HOME", &home.to_string_lossy());
        crate::test_support::remove_env("XDG_CONFIG_HOME");
        crate::test_support::remove_env("XDG_DATA_HOME");

        add_snippet(None, "signature", "Best regards,\nNotes").expect("add snippet");
        let config = Config::load(None).expect("config");
        let entries = read_snippet_file(&config.resolved_snippets_path()).expect("read");
        assert_eq!(entries.len(), 1);

        remove_snippet(None, "signature").expect("remove snippet");
        let entries = read_snippet_file(&config.resolved_snippets_path()).expect("read");
        assert!(entries.is_empty());
    }

    #[test]
    fn load_custom_instructions_tolerates_missing_file() {
        let mut config = Config::default();
        config.rewrite.instructions_path = "/tmp/whispers-missing-instructions.txt".into();
        let loaded = load_custom_instructions(&config).expect("load");
        assert!(loaded.is_empty());
    }
}

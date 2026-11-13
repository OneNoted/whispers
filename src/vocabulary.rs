use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{Error, Result, paths};

const VOCABULARY_VERSION: u16 = 1;

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct VocabularyEntry {
    pub term: String,
    #[serde(default)]
    pub aliases: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Vocabulary {
    pub version: u16,
    #[serde(default)]
    pub entries: Vec<VocabularyEntry>,
}

impl Default for Vocabulary {
    fn default() -> Self {
        Self {
            version: VOCABULARY_VERSION,
            entries: Vec::new(),
        }
    }
}

impl Vocabulary {
    pub fn load() -> Result<Self> {
        Self::load_from(&paths::vocabulary()?)
    }

    pub fn load_from(path: &Path) -> Result<Self> {
        match fs::read_to_string(path) {
            Ok(contents) => {
                let vocabulary: Self = toml::from_str(&contents)?;
                if vocabulary.version != VOCABULARY_VERSION {
                    return Err(Error::message(format!(
                        "vocabulary version {} is not supported",
                        vocabulary.version
                    )));
                }
                Ok(vocabulary)
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(error) => Err(error.into()),
        }
    }

    pub fn save(&self) -> Result<()> {
        self.save_to(&paths::vocabulary()?)
    }

    pub fn save_to(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let temporary = temporary_path(path);
        fs::write(&temporary, toml::to_string_pretty(self)?)?;
        fs::rename(temporary, path)?;
        Ok(())
    }

    pub fn add(&mut self, term: &str, aliases: Vec<String>) -> Result<()> {
        let term = term.trim();
        if term.is_empty() {
            return Err(Error::message("vocabulary term cannot be empty"));
        }
        let aliases = aliases
            .into_iter()
            .map(|alias| alias.trim().to_string())
            .filter(|alias| !alias.is_empty())
            .collect::<Vec<_>>();
        if aliases.is_empty() {
            return Err(Error::message("vocabulary aliases cannot be empty"));
        }
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| entry.term.eq_ignore_ascii_case(term))
        {
            for alias in aliases {
                if !entry
                    .aliases
                    .iter()
                    .any(|existing| existing.eq_ignore_ascii_case(&alias))
                {
                    entry.aliases.push(alias);
                }
            }
        } else {
            self.entries.push(VocabularyEntry {
                term: term.to_string(),
                aliases,
            });
        }
        self.entries
            .sort_by_key(|entry| entry.term.to_ascii_lowercase());
        Ok(())
    }

    pub fn remove(&mut self, term: &str) -> bool {
        let previous = self.entries.len();
        self.entries
            .retain(|entry| !entry.term.eq_ignore_ascii_case(term));
        self.entries.len() != previous
    }

    pub fn apply(&self, input: &str) -> String {
        let mut replacements = self
            .entries
            .iter()
            .flat_map(|entry| {
                entry
                    .aliases
                    .iter()
                    .map(move |alias| (alias.as_str(), entry.term.as_str()))
            })
            .collect::<Vec<_>>();
        replacements.sort_by_key(|(alias, _)| std::cmp::Reverse(alias.len()));
        replace_phrases(input, &replacements)
    }

    /// Joins the canonical terms into a whisper initial prompt that biases
    /// decoding toward them. whisper.cpp keeps at most `n_text_ctx / 2` prompt
    /// tokens, so very large vocabularies are truncated by the decoder.
    pub fn initial_prompt(&self) -> String {
        self.entries
            .iter()
            .map(|entry| entry.term.trim())
            .filter(|term| !term.is_empty())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

fn replace_phrases(input: &str, replacements: &[(&str, &str)]) -> String {
    let mut output = String::with_capacity(input.len());
    let mut cursor = 0;
    while cursor < input.len() {
        let replacement = replacements.iter().find(|(phrase, _)| {
            !phrase.is_empty()
                && cursor + phrase.len() <= input.len()
                && input.is_char_boundary(cursor + phrase.len())
                && input[cursor..cursor + phrase.len()].eq_ignore_ascii_case(phrase)
                && boundary_before(input, cursor)
                && boundary_after(input, cursor + phrase.len())
        });
        if let Some((phrase, replacement)) = replacement {
            output.push_str(replacement);
            cursor += phrase.len();
        } else {
            let Some(character) = input[cursor..].chars().next() else {
                break;
            };
            output.push(character);
            cursor += character.len_utf8();
        }
    }
    output
}

fn boundary_before(input: &str, index: usize) -> bool {
    index == 0
        || input[..index]
            .chars()
            .next_back()
            .is_none_or(|character| !word_character(character))
}

fn boundary_after(input: &str, index: usize) -> bool {
    index == input.len()
        || input[index..]
            .chars()
            .next()
            .is_none_or(|character| !word_character(character))
}

fn word_character(character: char) -> bool {
    character.is_alphanumeric() || character == '_'
}

fn temporary_path(path: &Path) -> PathBuf {
    let mut temporary = path.as_os_str().to_os_string();
    temporary.push(format!(".{}.tmp", std::process::id()));
    PathBuf::from(temporary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replaces_aliases_at_word_boundaries() {
        let vocabulary = Vocabulary {
            version: VOCABULARY_VERSION,
            entries: vec![VocabularyEntry {
                term: "serde_json".into(),
                aliases: vec!["serde json".into(), "surdy json".into()],
            }],
        };
        assert_eq!(
            vocabulary.apply("Use Surdy JSON, not serde jsonish."),
            "Use serde_json, not serde jsonish."
        );
    }

    #[test]
    fn initial_prompt_joins_nonempty_terms() {
        let entry = |term: &str| VocabularyEntry {
            term: term.into(),
            aliases: vec!["alias".into()],
        };
        let vocabulary = Vocabulary {
            version: VOCABULARY_VERSION,
            entries: vec![entry(" Vulkan "), entry(""), entry("RenderDoc")],
        };
        assert_eq!(vocabulary.initial_prompt(), "Vulkan, RenderDoc");
        assert_eq!(Vocabulary::default().initial_prompt(), "");
    }

    #[test]
    fn adding_merges_without_duplicates() {
        let mut vocabulary = Vocabulary::default();
        assert!(vocabulary.add("zsh", vec!["seashell".into()]).is_ok());
        assert!(
            vocabulary
                .add("ZSH", vec!["SeaShell".into(), "z shell".into()])
                .is_ok()
        );
        assert_eq!(vocabulary.entries.len(), 1);
        assert_eq!(vocabulary.entries[0].aliases.len(), 2);
    }

    #[test]
    fn does_not_replace_inside_inserted_terms() {
        let vocabulary = Vocabulary {
            version: VOCABULARY_VERSION,
            entries: vec![
                VocabularyEntry {
                    term: "TypeScript".into(),
                    aliases: vec!["type script".into()],
                },
                VocabularyEntry {
                    term: "TS".into(),
                    aliases: vec!["TypeScript".into()],
                },
            ],
        };
        assert_eq!(vocabulary.apply("type script"), "TypeScript");
    }
}

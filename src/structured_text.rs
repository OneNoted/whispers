#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StructuredCandidate {
    pub normalized: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Word(String),
    Sep { ch: char, spoken: bool },
}

#[derive(Debug, Clone)]
struct ParsedCandidate {
    normalized: String,
    end: usize,
    word_count: usize,
    dot_clusters: usize,
    has_non_dot_cluster: bool,
    has_spoken_separator: bool,
}

pub(crate) fn extract_structured_candidate(text: &str) -> Option<StructuredCandidate> {
    let tokens = tokenize(text);
    let mut best = None::<ParsedCandidate>;

    for start in 0..tokens.len() {
        let Some(parsed) = parse_candidate(&tokens, start) else {
            continue;
        };
        if !candidate_is_confident(&parsed) {
            continue;
        }

        let replace = match &best {
            Some(best) => parsed.normalized.len() > best.normalized.len(),
            None => true,
        };
        if replace {
            best = Some(parsed);
        }
    }

    best.map(|parsed| StructuredCandidate {
        normalized: parsed.normalized,
    })
}

pub(crate) fn normalize_strict_structured_text(text: &str) -> Option<String> {
    let tokens = tokenize(text);
    if tokens.is_empty() {
        return None;
    }
    let parsed = parse_candidate(&tokens, 0)?;
    (parsed.end == tokens.len() && candidate_is_confident(&parsed)).then_some(parsed.normalized)
}

pub(crate) fn output_matches_candidate(text: &str, candidate: &str) -> bool {
    normalize_strict_structured_text(text)
        .as_deref()
        .is_some_and(|normalized| normalized == candidate)
        || meta_wrapped_candidate_matches(text, candidate)
}

fn meta_wrapped_candidate_matches(text: &str, candidate: &str) -> bool {
    let tokens = tokenize(text);
    if tokens.is_empty() {
        return false;
    }

    (0..tokens.len()).any(|start| {
        let Some(parsed) = parse_candidate(&tokens, start) else {
            return false;
        };
        candidate_is_confident(&parsed)
            && parsed.normalized == candidate
            && non_candidate_tokens_are_meta(&tokens[..start])
            && non_candidate_tokens_are_meta(&tokens[parsed.end..])
    })
}

fn non_candidate_tokens_are_meta(tokens: &[Token]) -> bool {
    tokens.iter().all(|token| match token {
        Token::Sep { .. } => true,
        Token::Word(word) => is_meta_word(word),
    })
}

fn is_meta_word(word: &str) -> bool {
    matches!(
        word,
        "a" | "an"
            | "the"
            | "is"
            | "are"
            | "was"
            | "were"
            | "be"
            | "being"
            | "called"
            | "named"
            | "written"
            | "spelled"
            | "literal"
            | "literally"
            | "just"
            | "this"
            | "that"
            | "it"
            | "its"
            | "s"
            | "url"
            | "link"
            | "website"
            | "site"
            | "domain"
            | "address"
    )
}

fn tokenize(text: &str) -> Vec<Token> {
    fold_spoken_separator_tokens(raw_tokens(text))
}

fn raw_tokens(text: &str) -> Vec<Token> {
    let chars = text
        .chars()
        .flat_map(|ch| ch.to_lowercase())
        .collect::<Vec<_>>();
    let mut tokens = Vec::new();
    let mut index = 0usize;

    while index < chars.len() {
        let ch = chars[index];
        if ch.is_ascii_alphanumeric() {
            let start = index;
            index += 1;
            while index < chars.len() && chars[index].is_ascii_alphanumeric() {
                index += 1;
            }
            tokens.push(Token::Word(chars[start..index].iter().collect()));
            continue;
        }

        if matches!(ch, '.' | '/' | ':' | '_' | '-' | '@') {
            tokens.push(Token::Sep { ch, spoken: false });
        }

        index += 1;
    }

    tokens
}

fn fold_spoken_separator_tokens(tokens: Vec<Token>) -> Vec<Token> {
    let mut output = Vec::with_capacity(tokens.len());
    let mut index = 0usize;

    while index < tokens.len() {
        if matches_word_slice(&tokens, index, &["at", "sign"]) {
            output.push(Token::Sep {
                ch: '@',
                spoken: true,
            });
            index += 2;
            continue;
        }
        if matches_word_slice(&tokens, index, &["forward", "slash"]) {
            output.push(Token::Sep {
                ch: '/',
                spoken: true,
            });
            index += 2;
            continue;
        }
        if matches_word_slice(&tokens, index, &["full", "stop"]) {
            output.push(Token::Sep {
                ch: '.',
                spoken: true,
            });
            index += 2;
            continue;
        }

        match tokens.get(index) {
            Some(Token::Word(word)) => {
                let separator = match word.as_str() {
                    "dot" => Some('.'),
                    "period" => Some('.'),
                    "slash" => Some('/'),
                    "colon" => Some(':'),
                    "underscore" => Some('_'),
                    "dash" | "hyphen" => Some('-'),
                    _ => None,
                };
                if let Some(ch) = separator {
                    output.push(Token::Sep { ch, spoken: true });
                } else {
                    output.push(tokens[index].clone());
                }
            }
            Some(_) => output.push(tokens[index].clone()),
            None => {}
        }

        index += 1;
    }

    output
}

fn matches_word_slice(tokens: &[Token], index: usize, words: &[&str]) -> bool {
    if index + words.len() > tokens.len() {
        return false;
    }

    tokens[index..index + words.len()]
        .iter()
        .zip(words)
        .all(|(token, expected)| matches!(token, Token::Word(word) if word == expected))
}

fn parse_candidate(tokens: &[Token], start: usize) -> Option<ParsedCandidate> {
    let Token::Word(first_word) = tokens.get(start)? else {
        return None;
    };

    let mut normalized = first_word.clone();
    let mut index = start + 1;
    let mut word_count = 1usize;
    let mut dot_clusters = 0usize;
    let mut has_non_dot_cluster = false;
    let mut has_spoken_separator = false;

    loop {
        let cluster_start = index;
        let mut cluster = String::new();
        let mut cluster_has_spoken = false;

        while let Some(Token::Sep { ch, spoken }) = tokens.get(index) {
            cluster.push(*ch);
            cluster_has_spoken |= *spoken;
            index += 1;
        }

        if cluster.is_empty() || !allowed_separator_cluster(&cluster) {
            index = cluster_start;
            break;
        }

        let Some(Token::Word(word)) = tokens.get(index) else {
            index = cluster_start;
            break;
        };

        if cluster.contains('.') {
            dot_clusters += 1;
        }
        has_non_dot_cluster |= cluster.chars().any(|ch| ch != '.');
        has_spoken_separator |= cluster_has_spoken;
        normalized.push_str(&cluster);
        normalized.push_str(word);
        word_count += 1;
        index += 1;
    }

    (word_count >= 2).then_some(ParsedCandidate {
        normalized,
        end: index,
        word_count,
        dot_clusters,
        has_non_dot_cluster,
        has_spoken_separator,
    })
}

fn allowed_separator_cluster(cluster: &str) -> bool {
    matches!(
        cluster,
        "." | "-" | "_" | "@" | "/" | "//" | ":" | ":/" | "://"
    )
}

fn candidate_is_confident(candidate: &ParsedCandidate) -> bool {
    candidate.dot_clusters >= 2
        || candidate.has_non_dot_cluster
        || (candidate.dot_clusters >= 1 && candidate.has_spoken_separator)
        || (candidate.dot_clusters >= 1 && candidate.word_count >= 3)
        || looks_like_bare_hostname(candidate)
}

fn looks_like_bare_hostname(candidate: &ParsedCandidate) -> bool {
    if candidate.dot_clusters != 1
        || candidate.word_count != 2
        || candidate.has_non_dot_cluster
        || candidate.has_spoken_separator
    {
        return false;
    }

    let mut labels = candidate.normalized.split('.');
    let (Some(host), Some(tld), None) = (labels.next(), labels.next(), labels.next()) else {
        return false;
    };

    hostname_label_is_valid(host) && likely_hostname_tld(tld)
}

fn hostname_label_is_valid(label: &str) -> bool {
    !label.is_empty()
        && label.len() <= 63
        && !label.starts_with('-')
        && !label.ends_with('-')
        && label
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-')
        && label.chars().any(|ch| ch.is_ascii_alphabetic())
}

fn likely_hostname_tld(tld: &str) -> bool {
    hostname_label_is_valid(tld)
        && ((tld.len() == 2 && tld.chars().all(|ch| ch.is_ascii_alphabetic()))
            || matches!(
                tld,
                "com"
                    | "net"
                    | "org"
                    | "edu"
                    | "gov"
                    | "mil"
                    | "app"
                    | "dev"
                    | "info"
                    | "me"
                    | "io"
                    | "ai"
                    | "co"
                    | "tv"
                    | "fm"
                    | "gg"
                    | "blog"
                    | "tech"
                    | "site"
                    | "online"
                    | "store"
                    | "cloud"
            ))
}

#[cfg(test)]
mod tests {
    use super::{
        extract_structured_candidate, normalize_strict_structured_text, output_matches_candidate,
    };

    #[test]
    fn extracts_dotted_hostname_from_literal_periods() {
        let candidate = extract_structured_candidate("portfolio. Notes. Supply is the URL")
            .expect("structured candidate");
        assert_eq!(candidate.normalized, "portfolio.notes.supply");
    }

    #[test]
    fn extracts_two_label_hostname_from_literal_periods() {
        let candidate = extract_structured_candidate("Check example.com tomorrow")
            .expect("structured candidate");
        assert_eq!(candidate.normalized, "example.com");
    }

    #[test]
    fn extracts_dotted_hostname_from_spoken_separators() {
        let candidate = extract_structured_candidate("portfolio dot notes dot supply")
            .expect("structured candidate");
        assert_eq!(candidate.normalized, "portfolio.notes.supply");
    }

    #[test]
    fn extracts_full_url_from_spoken_separators() {
        let candidate = extract_structured_candidate(
            "https colon slash slash portfolio dot notes dot supply slash blog",
        )
        .expect("structured candidate");
        assert_eq!(candidate.normalized, "https://portfolio.notes.supply/blog");
    }

    #[test]
    fn strict_normalization_rejects_prose_suffixes() {
        assert_eq!(
            normalize_strict_structured_text("portfolio. Notes. Supply"),
            Some("portfolio.notes.supply".into())
        );
        assert_eq!(
            normalize_strict_structured_text("portfolio. Notes. Supply is the URL"),
            None
        );
    }

    #[test]
    fn avoids_false_positive_for_normal_sentence() {
        assert_eq!(extract_structured_candidate("Section 2. Next step"), None);
        assert_eq!(extract_structured_candidate("Check two.next steps"), None);
        assert_eq!(
            normalize_strict_structured_text("Section 2. Next step"),
            None
        );
    }

    #[test]
    fn candidate_match_requires_full_structured_output() {
        assert!(output_matches_candidate(
            "portfolio . notes . supply",
            "portfolio.notes.supply"
        ));
        assert!(output_matches_candidate(
            "portfolio. Notes. Supply is the URL",
            "portfolio.notes.supply"
        ));
        assert!(!output_matches_candidate(
            "check portfolio. Notes. Supply tomorrow",
            "portfolio.notes.supply"
        ));
    }
}

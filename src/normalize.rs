#[derive(Clone, Debug, Eq, PartialEq)]
enum Token {
    Word(String),
    Symbol(&'static str),
    Newline,
}

pub fn normalize_spoken_text(input: &str) -> String {
    let number_normalized = crate::numbers::normalize_spoken_numbers(input);
    let words: Vec<&str> = number_normalized.split_whitespace().collect();
    let mut tokens = Vec::with_capacity(words.len());
    let mut index = 0;
    while index < words.len() {
        let lower = clean_word(words[index]).to_ascii_lowercase();
        let next = words
            .get(index + 1)
            .map(|word| clean_word(word).to_ascii_lowercase());

        if lower == "new" && next.as_deref() == Some("paragraph") {
            tokens.push(Token::Newline);
            tokens.push(Token::Newline);
            index += 2;
            continue;
        }
        if lower == "new" && next.as_deref() == Some("line") {
            tokens.push(Token::Newline);
            index += 2;
            continue;
        }
        if lower == "literal" {
            if let Some((symbol, consumed)) = spoken_symbol(&words, index + 1) {
                tokens.push(Token::Symbol(symbol));
                index += consumed + 1;
                continue;
            }
        }
        if let Some((symbol, consumed)) = spoken_symbol(&words, index) {
            if is_strong_symbol_context(&words, index, consumed, symbol) {
                tokens.push(Token::Symbol(symbol));
                index += consumed;
                continue;
            }
        }
        tokens.push(Token::Word(words[index].to_string()));
        index += 1;
    }
    render(&tokens)
}

fn clean_word(word: &str) -> &str {
    word.trim_matches(|character: char| !character.is_alphanumeric())
}

fn spoken_symbol(words: &[&str], index: usize) -> Option<(&'static str, usize)> {
    let first = clean_word(words.get(index)?).to_ascii_lowercase();
    let second = words
        .get(index + 1)
        .map(|word| clean_word(word).to_ascii_lowercase());
    let pair = match (first.as_str(), second.as_deref()) {
        ("forward", Some("slash")) => Some("/"),
        ("back", Some("slash")) => Some("\\"),
        ("question", Some("mark")) => Some("?"),
        ("exclamation", Some("mark" | "point")) => Some("!"),
        ("at", Some("sign")) => Some("@"),
        _ => None,
    };
    if let Some(symbol) = pair {
        return Some((symbol, 2));
    }
    let symbol = match first.as_str() {
        "slash" => "/",
        "backslash" => "\\",
        "comma" => ",",
        "period" => ".",
        "colon" => ":",
        "semicolon" => ";",
        "dash" | "hyphen" => "-",
        "underscore" => "_",
        "pipe" => "|",
        "ampersand" => "&",
        "asterisk" => "*",
        "plus" => "+",
        "equals" => "=",
        "hash" => "#",
        _ => return None,
    };
    Some((symbol, 1))
}

fn is_strong_symbol_context(words: &[&str], index: usize, consumed: usize, symbol: &str) -> bool {
    let previous = index.checked_sub(1).and_then(|value| words.get(value));
    let next = words.get(index + consumed);
    let lexical_prefix = previous.is_some_and(|word| {
        matches!(
            clean_word(word).to_ascii_lowercase().as_str(),
            "word" | "say" | "called"
        )
    });
    if lexical_prefix {
        return false;
    }
    let lexical_symbol_use = symbol == "/"
        && (previous.is_some_and(|word| {
            matches!(
                clean_word(word).to_ascii_lowercase().as_str(),
                "like" | "love" | "read" | "reading"
            )
        }) || next.is_some_and(|word| {
            matches!(
                clean_word(word).to_ascii_lowercase().as_str(),
                "fiction" | "genre" | "command" | "character" | "symbol"
            )
        }));
    if lexical_symbol_use {
        return false;
    }
    match symbol {
        "," | "." | ":" | ";" | "?" | "!" => previous.is_some(),
        "/" | "\\" | "_" | "|" | "&" | "*" | "+" | "=" | "#" | "-" => {
            previous.is_some() && next.is_some()
        }
        _ => false,
    }
}

fn render(tokens: &[Token]) -> String {
    let mut output = String::new();
    for token in tokens {
        match token {
            Token::Newline => {
                while output.ends_with(' ') {
                    output.pop();
                }
                output.push('\n');
            }
            Token::Word(word) => {
                if needs_space_before_word(&output) {
                    output.push(' ');
                }
                output.push_str(word);
            }
            Token::Symbol(symbol) => {
                while output.ends_with(' ') {
                    output.pop();
                }
                output.push_str(symbol);
            }
        }
    }
    output.trim().to_string()
}

fn needs_space_before_word(output: &str) -> bool {
    output.chars().last().is_some_and(|character| {
        !character.is_whitespace()
            && !matches!(
                character,
                '/' | '\\' | '_' | '-' | '@' | '#' | '|' | '&' | '*' | '+' | '=' | '(' | '[' | '{'
            )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_slash_between_terms() {
        assert_eq!(normalize_spoken_text("this slash that"), "this/that");
        assert_eq!(normalize_spoken_text("and forward slash or"), "and/or");
    }

    #[test]
    fn keeps_lexical_slash() {
        assert_eq!(
            normalize_spoken_text("the word slash here"),
            "the word slash here"
        );
        assert_eq!(normalize_spoken_text("slash fiction"), "slash fiction");
        assert_eq!(
            normalize_spoken_text("I like slash fiction"),
            "I like slash fiction"
        );
    }

    #[test]
    fn explicit_literals_are_guaranteed() {
        assert_eq!(normalize_spoken_text("literal slash"), "/");
        assert_eq!(normalize_spoken_text("literal question mark"), "?");
    }

    #[test]
    fn renders_operators_without_half_spacing() {
        assert_eq!(normalize_spoken_text("x plus y"), "x+y");
        assert_eq!(normalize_spoken_text("left equals right"), "left=right");
        assert_eq!(normalize_spoken_text("one ampersand two"), "one&two");
    }

    #[test]
    fn renders_line_structure() {
        assert_eq!(
            normalize_spoken_text("first line new paragraph second line"),
            "first line\n\nsecond line"
        );
    }

    #[test]
    fn formats_spoken_number_sequences_before_symbols() {
        assert_eq!(
            normalize_spoken_text("One, two, three, put a comma between those."),
            "1, 2, 3."
        );
        assert_eq!(
            normalize_spoken_text("One, two, three. Put a hyphen between these."),
            "1-2-3."
        );
    }
}

#[derive(Clone, Debug)]
struct Word {
    start: usize,
    end: usize,
    lower: String,
}

pub(crate) fn normalize_spoken_numbers(input: &str) -> String {
    apply_separator_command(input).unwrap_or_else(|| normalize_number_lists(input))
}

fn apply_separator_command(input: &str) -> Option<String> {
    let words = word_spans(input);
    let (command, separator) = find_separator_command(&words)?;
    let last = command.checked_sub(1)?;
    if !is_number_component(&words[last].lower) {
        return None;
    }

    let mut first = last;
    while first > 0 && is_number_component(&words[first - 1].lower) {
        let gap = &input[words[first - 1].end..words[first].start];
        if gap
            .chars()
            .any(|character| matches!(character, '.' | '!' | '?'))
            || !gap
                .chars()
                .all(|character| character.is_whitespace() || matches!(character, ',' | '-'))
        {
            break;
        }
        first -= 1;
    }

    let spoken = &input[words[first].start..words[last].end];
    let numbers = parse_number_items(spoken)?;
    if numbers.len() < 2 {
        return None;
    }
    let punctuation = input
        .trim_end()
        .chars()
        .last()
        .filter(|character| matches!(character, '.' | '!' | '?'))
        .or_else(|| {
            input[words[last].end..words[command].start]
                .chars()
                .find(|character| matches!(character, '.' | '!' | '?'))
        });
    let mut output = input[..words[first].start].to_string();
    output.push_str(&numbers.join(separator));
    if let Some(punctuation) = punctuation {
        output.push(punctuation);
    }
    Some(output.trim().to_string())
}

fn find_separator_command(words: &[Word]) -> Option<(usize, &'static str)> {
    for start in 0..words.len() {
        let remaining = &words[start..];
        if remaining.first()?.lower == "put" {
            let mut index = 1;
            if remaining.get(index).is_some_and(|word| word.lower == "a") {
                index += 1;
            }
            let Some(separator) = remaining.get(index).and_then(|word| separator(&word.lower))
            else {
                continue;
            };
            index += 1;
            if remaining.get(index).is_some_and(|word| word.lower == "in") {
                index += 1;
            }
            if remaining
                .get(index)
                .is_none_or(|word| word.lower != "between")
            {
                continue;
            }
            index += 1;
            if remaining
                .get(index)
                .is_none_or(|word| !matches!(word.lower.as_str(), "these" | "those" | "them"))
            {
                continue;
            }
            if start + index + 1 == words.len() {
                return Some((start, separator));
            }
        }
    }
    None
}

fn separator(word: &str) -> Option<&'static str> {
    match word {
        "comma" | "commas" => Some(", "),
        "hyphen" | "hyphens" | "dash" | "dashes" => Some("-"),
        "slash" | "slashes" => Some("/"),
        "space" | "spaces" => Some(" "),
        _ => None,
    }
}

fn normalize_number_lists(input: &str) -> String {
    let words = word_spans(input);
    let mut replacements = Vec::new();
    let mut index = 0;
    while index < words.len() {
        if !is_number_start(&words[index].lower) {
            index += 1;
            continue;
        }
        let first = index;
        let mut last = index;
        let mut comma_count = 0;
        while let Some(next) = words.get(last + 1) {
            if !is_number_component(&next.lower) {
                break;
            }
            let gap = &input[words[last].end..next.start];
            if !gap
                .chars()
                .all(|character| character.is_whitespace() || matches!(character, ',' | '-'))
            {
                break;
            }
            comma_count += gap.matches(',').count();
            last += 1;
        }
        if comma_count > 0 {
            let followed_by_open_list = words.get(last + 1).is_some_and(|next| {
                input[words[last].end..next.start].contains(',')
                    || matches!(next.lower.as_str(), "and" | "or")
            });
            let spoken = &input[words[first].start..words[last].end];
            if !followed_by_open_list
                && let Some(numbers) = parse_comma_separated_numbers(spoken)
                && numbers.len() >= 2
            {
                replacements.push((words[first].start, words[last].end, numbers.join(", ")));
            }
        }
        index = last + 1;
    }

    let mut output = input.to_string();
    for (start, end, replacement) in replacements.into_iter().rev() {
        output.replace_range(start..end, &replacement);
    }
    output
}

fn parse_number_items(input: &str) -> Option<Vec<String>> {
    if input.contains(',') {
        return parse_comma_separated_numbers(input);
    }
    let items = word_spans(input)
        .into_iter()
        .map(|word| parse_number_phrase(&word.lower).map(|number| number.to_string()))
        .collect::<Option<Vec<_>>>()?;
    (!items.is_empty()).then_some(items)
}

fn parse_comma_separated_numbers(input: &str) -> Option<Vec<String>> {
    input
        .split(',')
        .map(|phrase| parse_number_phrase(phrase).map(|number| number.to_string()))
        .collect()
}

fn parse_number_phrase(input: &str) -> Option<u64> {
    let words = word_spans(input);
    if words.is_empty() {
        return None;
    }
    if words.len() == 1 {
        if let Ok(number) = words[0].lower.parse() {
            return Some(number);
        }
    }

    let mut total = 0_u64;
    let mut current = 0_u64;
    let mut saw_number = false;
    for word in words {
        if word.lower == "and" {
            continue;
        }
        if word.lower == "hundred" {
            current = current.max(1).checked_mul(100)?;
            saw_number = true;
            continue;
        }
        let scale = match word.lower.as_str() {
            "thousand" => Some(1_000),
            "million" => Some(1_000_000),
            _ => None,
        };
        if let Some(scale) = scale {
            total = total.checked_add(current.max(1).checked_mul(scale)?)?;
            current = 0;
            saw_number = true;
            continue;
        }
        current = current.checked_add(number_value(&word.lower)?)?;
        saw_number = true;
    }
    saw_number.then(|| total + current)
}

fn number_value(word: &str) -> Option<u64> {
    match word {
        "zero" | "oh" => Some(0),
        "one" => Some(1),
        "two" => Some(2),
        "three" => Some(3),
        "four" => Some(4),
        "five" => Some(5),
        "six" => Some(6),
        "seven" => Some(7),
        "eight" => Some(8),
        "nine" => Some(9),
        "ten" => Some(10),
        "eleven" => Some(11),
        "twelve" => Some(12),
        "thirteen" => Some(13),
        "fourteen" => Some(14),
        "fifteen" => Some(15),
        "sixteen" => Some(16),
        "seventeen" => Some(17),
        "eighteen" => Some(18),
        "nineteen" => Some(19),
        "twenty" => Some(20),
        "thirty" => Some(30),
        "forty" => Some(40),
        "fifty" => Some(50),
        "sixty" => Some(60),
        "seventy" => Some(70),
        "eighty" => Some(80),
        "ninety" => Some(90),
        _ => word.parse().ok(),
    }
}

fn is_number_start(word: &str) -> bool {
    word != "and" && is_number_component(word)
}

fn is_number_component(word: &str) -> bool {
    matches!(word, "and" | "hundred" | "thousand" | "million") || number_value(word).is_some()
}

fn word_spans(input: &str) -> Vec<Word> {
    let mut words = Vec::new();
    let mut start = None;
    for (index, character) in input.char_indices() {
        if character.is_alphanumeric() {
            start.get_or_insert(index);
        } else if let Some(start) = start.take() {
            words.push(Word {
                start,
                end: index,
                lower: input[start..index].to_ascii_lowercase(),
            });
        }
    }
    if let Some(start) = start {
        words.push(Word {
            start,
            end: input.len(),
            lower: input[start..].to_ascii_lowercase(),
        });
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_clear_number_lists() {
        assert_eq!(normalize_spoken_numbers("One, two, three."), "1, 2, 3.");
        assert_eq!(
            normalize_spoken_numbers("Twenty one, thirty two, forty three."),
            "21, 32, 43."
        );
        assert_eq!(
            normalize_spoken_numbers("Choose twenty one, thirty two, or forty three."),
            "Choose twenty one, thirty two, or forty three."
        );
        assert_eq!(
            normalize_spoken_numbers("One of those is enough."),
            "One of those is enough."
        );
    }

    #[test]
    fn executes_explicit_separator_commands() {
        assert_eq!(
            normalize_spoken_numbers("One, two, three, put a comma between those."),
            "1, 2, 3."
        );
        assert_eq!(
            normalize_spoken_numbers(
                "Let's show them how smart you are. One, two, three. Put a hyphen between these"
            ),
            "Let's show them how smart you are. 1-2-3."
        );
        assert_eq!(
            normalize_spoken_numbers("One two three, put a slash between them."),
            "1/2/3."
        );
    }
}

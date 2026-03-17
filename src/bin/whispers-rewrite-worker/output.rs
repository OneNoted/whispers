pub fn sanitize_rewrite_output(raw: &str) -> String {
    let mut text = raw.replace("\r\n", "\n");

    for stop in ["<|eot_id|>", "<|end_of_text|>", "</s>"] {
        if let Some(index) = text.find(stop) {
            text.truncate(index);
        }
    }

    if let Some(index) = text.find("</output>") {
        text.truncate(index);
    }

    text = strip_tagged_section(&text, "<think>", "</think>");

    let mut text = text.trim().to_string();

    if let Some(stripped) = text.strip_prefix("<output>") {
        text = stripped.trim().to_string();
    }

    for prefix in ["Final text:", "Output:", "Rewritten text:"] {
        if text
            .get(..prefix.len())
            .map(|candidate| candidate.eq_ignore_ascii_case(prefix))
            .unwrap_or(false)
        {
            text = text[prefix.len()..].trim().to_string();
            break;
        }
    }

    if text.starts_with('"') && text.ends_with('"') && text.len() >= 2 {
        text = text[1..text.len() - 1].trim().to_string();
    }

    text
}

fn strip_tagged_section(input: &str, open: &str, close: &str) -> String {
    let mut output = input.to_string();

    while let Some(start) = output.find(open) {
        let close_start = match output[start + open.len()..].find(close) {
            Some(offset) => start + open.len() + offset,
            None => {
                output.truncate(start);
                break;
            }
        };
        output.replace_range(start..close_start + close.len(), "");
    }

    output
}

#[cfg(test)]
mod tests {
    use super::sanitize_rewrite_output;

    #[test]
    fn sanitize_rewrite_output_strips_wrapper_and_label() {
        let cleaned = sanitize_rewrite_output("<output>\nFinal text: Hi there.\n</output>");
        assert_eq!(cleaned, "Hi there.");
    }

    #[test]
    fn sanitize_rewrite_output_strips_llama_stop_tokens() {
        let cleaned = sanitize_rewrite_output("Hi there.<|eot_id|>ignored");
        assert_eq!(cleaned, "Hi there.");
    }

    #[test]
    fn sanitize_rewrite_output_strips_think_blocks() {
        let cleaned = sanitize_rewrite_output("<think>reasoning</think>\nHi there.");
        assert_eq!(cleaned, "Hi there.");
    }
}

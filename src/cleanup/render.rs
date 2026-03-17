use super::{BreakKind, Piece};

pub(super) fn render_pieces(pieces: &[Piece]) -> String {
    let mut rendered = String::new();
    let mut capitalize_next = true;

    for piece in pieces {
        match piece {
            Piece::Word(word) => {
                if !rendered.is_empty() && !rendered.ends_with([' ', '\n']) {
                    rendered.push(' ');
                }
                if capitalize_next {
                    rendered.push_str(&capitalize_first(word));
                } else {
                    rendered.push_str(word);
                }
                capitalize_next = false;
            }
            Piece::Punctuation(ch) => {
                trim_trailing_spaces(&mut rendered);
                rendered.push(*ch);
                capitalize_next = matches!(ch, '.' | '?' | '!');
            }
            Piece::Break(BreakKind::Line) => {
                trim_trailing_spaces(&mut rendered);
                if !rendered.is_empty() {
                    if !rendered.ends_with('\n') {
                        rendered.push('\n');
                    }
                    capitalize_next = true;
                }
            }
            Piece::Break(BreakKind::Paragraph) => {
                trim_trailing_spaces(&mut rendered);
                if !rendered.is_empty() {
                    while rendered.ends_with('\n') {
                        rendered.pop();
                    }
                    rendered.push('\n');
                    rendered.push('\n');
                    capitalize_next = true;
                }
            }
        }
    }

    trim_trailing_spaces(&mut rendered);
    rendered
}

fn trim_trailing_spaces(text: &mut String) {
    while text.ends_with(' ') {
        text.pop();
    }
}

fn capitalize_first(word: &str) -> String {
    let mut chars = word.chars();
    let Some(first) = chars.next() else {
        return String::new();
    };

    let mut result = String::new();
    result.extend(first.to_uppercase());
    result.extend(chars);
    result
}

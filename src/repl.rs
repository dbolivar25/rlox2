use nu_ansi_term::{Color, Style};
use reedline::{
    Highlighter, Prompt, PromptEditMode, PromptHistorySearch, PromptHistorySearchStatus,
    StyledText, ValidationResult, Validator,
};
use std::borrow::Cow;

use crate::tokenizer::{tokenize, TokenType};

#[derive(Clone)]
pub struct REPLPrompt;

impl Prompt for REPLPrompt {
    fn render_prompt_left(&self) -> Cow<str> {
        Cow::Borrowed("rlox")
    }

    fn render_prompt_right(&self) -> Cow<str> {
        Cow::Borrowed("")
    }

    fn render_prompt_indicator(&self, _edit_mode: PromptEditMode) -> Cow<str> {
        Cow::Borrowed("\x1b[38;2;191;168;227m❯\x1b[0m ")
    }

    fn render_prompt_multiline_indicator(&self) -> Cow<str> {
        Cow::Borrowed("\x1b[90m  ...\x1b[0m ")
    }

    fn render_prompt_history_search_indicator(
        &self,
        history_search: PromptHistorySearch,
    ) -> Cow<str> {
        let (status_color, prefix) = match history_search.status {
            PromptHistorySearchStatus::Passing => ("\x1b[38;2;66;113;139m", "SEARCH"), // Green for success
            PromptHistorySearchStatus::Failing => ("\x1b[31m", "NOT FOUND"), // Red for failure
        };

        let search_term = history_search.term;

        Cow::Owned(format!(
            " {}[{}]\x1b[0m \x1b[36m{}\x1b[0m\x1b[38;2;191;168;227m❯\x1b[0m ",
            status_color, prefix, search_term
        ))
    }
}

pub struct REPLValidator;

impl Validator for REPLValidator {
    fn validate(&self, line: &str) -> ValidationResult {
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            return ValidationResult::Complete;
        }

        let mut delimiters = Vec::new();
        let mut in_string = false;
        let mut escaped = false;

        for c in line.chars() {
            match c {
                '"' if !escaped => in_string = !in_string,
                '\\' if in_string => escaped = !escaped,
                _ if in_string => {
                    escaped = false;
                    continue;
                }

                '{' | '(' | '[' => delimiters.push(c),
                '}' => {
                    if delimiters.pop() != Some('{') {
                        return ValidationResult::Complete;
                    }
                }
                ')' => {
                    if delimiters.pop() != Some('(') {
                        return ValidationResult::Complete;
                    }
                }
                ']' => {
                    if delimiters.pop() != Some('[') {
                        return ValidationResult::Complete;
                    }
                }

                _ => escaped = false,
            }
        }

        if in_string {
            return ValidationResult::Incomplete;
        }

        if delimiters.is_empty() {
            ValidationResult::Complete
        } else {
            ValidationResult::Incomplete
        }
    }
}

pub static KEYWORD_COLOR: Color = Color::LightBlue;
pub static LITERAL_COLOR: Color = Color::Yellow;
pub static DEFAULT_COLOR: Color = Color::White;
pub static OPERATOR_COLOR: Color = Color::DarkGray;

pub struct SyntaxHighlighter;

impl Highlighter for SyntaxHighlighter {
    fn highlight(&self, line: &str, _cursor: usize) -> StyledText {
        let mut styled_text = StyledText::new();

        let tokens = match tokenize(line.as_bytes()) {
            Ok(t) => t,
            Err(_) => {
                styled_text.push((Style::new().fg(DEFAULT_COLOR), line.to_string()));
                return styled_text;
            }
        };

        let mut remaining = line;
        let in_multiline_string = false;
        let multiline_start = 0;

        for (i, token) in tokens.iter().enumerate() {
            if token.token_type == TokenType::EOF {
                break;
            }

            // Get the actual text from the original line using the byte_span
            let token_slice = &line[token.byte_span.clone()];

            if let Some(pos) = remaining.find(token_slice) {
                if pos > 0 {
                    styled_text
                        .push((Style::new().fg(DEFAULT_COLOR), remaining[..pos].to_string()));
                }

                let color = if in_multiline_string && i >= multiline_start {
                    LITERAL_COLOR
                } else {
                    match &token.token_type {
                        // Keywords
                        TokenType::Let
                        | TokenType::If
                        | TokenType::Else
                        | TokenType::Fn
                        | TokenType::While
                        | TokenType::For
                        | TokenType::Return
                        | TokenType::Struct
                        | TokenType::And
                        | TokenType::Or => KEYWORD_COLOR,
                        // Literals
                        TokenType::String(_)
                        | TokenType::Number(_)
                        | TokenType::Nil
                        | TokenType::True
                        | TokenType::False => LITERAL_COLOR,
                        // Operators
                        TokenType::Semicolon
                        | TokenType::LeftBrace
                        | TokenType::RightBrace
                        | TokenType::LeftParen
                        | TokenType::RightParen
                        | TokenType::LeftSquare
                        | TokenType::RightSquare
                        | TokenType::Plus
                        | TokenType::Minus
                        | TokenType::Star
                        | TokenType::Slash
                        | TokenType::Percent
                        | TokenType::Bang
                        | TokenType::Equal
                        | TokenType::Greater
                        | TokenType::Less
                        | TokenType::BangEqual
                        | TokenType::EqualEqual
                        | TokenType::GreaterEqual
                        | TokenType::LessEqual
                        | TokenType::Concat => OPERATOR_COLOR,
                        // Everything else
                        _ => DEFAULT_COLOR,
                    }
                };

                styled_text.push((Style::new().fg(color), token_slice.to_string()));
                remaining = &remaining[pos + token_slice.len()..];
            }
        }

        if !remaining.is_empty() {
            styled_text.push((Style::new().fg(DEFAULT_COLOR), remaining.to_string()));
        }

        styled_text
    }
}

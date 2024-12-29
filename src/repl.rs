use nu_ansi_term::{Color, Style};
use reedline::{
    Highlighter, Prompt, PromptEditMode, PromptHistorySearch, PromptHistorySearchStatus,
    StyledText, ValidationResult, Validator,
};
use std::borrow::Cow;

use crate::tokenizer::{tokenize, Token};

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
        Cow::Borrowed("❯ ")
    }

    fn render_prompt_multiline_indicator(&self) -> Cow<str> {
        Cow::Borrowed("  ... ")
    }

    fn render_prompt_history_search_indicator(
        &self,
        history_search: PromptHistorySearch,
    ) -> Cow<str> {
        let prefix = match history_search.status {
            PromptHistorySearchStatus::Passing => "",
            PromptHistorySearchStatus::Failing => "failing ",
        };
        Cow::Owned(format!(
            "({}reverse-search: {}) ",
            prefix, history_search.term
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

        if trimmed.ends_with('\\') {
            return ValidationResult::Incomplete;
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

        for token in tokens {
            if token == Token::EOF {
                break;
            }

            let token_str = match &token {
                Token::String(s) => format!("\"{}\"", s),
                Token::Number(n) => n.to_string(),
                Token::Identifier(s) => s.clone(),
                Token::And => "and".to_string(),
                Token::Or => "or".to_string(),
                Token::Struct => "struct".to_string(),
                Token::Let => "let".to_string(),
                Token::If => "if".to_string(),
                Token::Else => "else".to_string(),
                Token::Nil => "nil".to_string(),
                Token::Return => "return".to_string(),
                Token::True => "true".to_string(),
                Token::False => "false".to_string(),
                Token::For => "for".to_string(),
                Token::While => "while".to_string(),
                Token::Fn => "fn".to_string(),
                Token::LeftParen => "(".to_string(),
                Token::RightParen => ")".to_string(),
                Token::LeftBrace => "{".to_string(),
                Token::RightBrace => "}".to_string(),
                Token::LeftSquare => "[".to_string(),
                Token::RightSquare => "]".to_string(),
                Token::Comma => ",".to_string(),
                Token::Dot => ".".to_string(),
                Token::Semicolon => ";".to_string(),
                Token::Plus => "+".to_string(),
                Token::Minus => "-".to_string(),
                Token::Star => "*".to_string(),
                Token::Slash => "/".to_string(),
                Token::Percent => "%".to_string(),
                Token::Bang => "!".to_string(),
                Token::Equal => "=".to_string(),
                Token::Greater => ">".to_string(),
                Token::Less => "<".to_string(),
                Token::BangEqual => "!=".to_string(),
                Token::EqualEqual => "==".to_string(),
                Token::GreaterEqual => ">=".to_string(),
                Token::LessEqual => "<=".to_string(),
                Token::Concat => "<>".to_string(),
                Token::NewLine => "\n".to_string(),
                Token::EOF => "".to_string(),
            };

            if let Some(pos) = remaining.find(&token_str) {
                if pos > 0 {
                    styled_text
                        .push((Style::new().fg(DEFAULT_COLOR), remaining[..pos].to_string()));
                }

                // Updated color selection to include operators
                let color = match &token {
                    // Keywords
                    Token::Let
                    | Token::If
                    | Token::Else
                    | Token::Fn
                    | Token::While
                    | Token::For
                    | Token::Return
                    | Token::Struct
                    | Token::And
                    | Token::Or
                    | Token::True
                    | Token::False
                    | Token::Nil => KEYWORD_COLOR,
                    // Literals
                    Token::String(_) | Token::Number(_) => LITERAL_COLOR,
                    // Operators
                    Token::Semicolon
                    | Token::LeftBrace
                    | Token::RightBrace
                    | Token::LeftParen
                    | Token::RightParen
                    | Token::LeftSquare
                    | Token::RightSquare
                    | Token::Plus
                    | Token::Minus
                    | Token::Star
                    | Token::Slash
                    | Token::Percent
                    | Token::Bang
                    | Token::Equal
                    | Token::Greater
                    | Token::Less
                    | Token::BangEqual
                    | Token::EqualEqual
                    | Token::GreaterEqual
                    | Token::LessEqual
                    | Token::Concat => OPERATOR_COLOR,
                    // Everything else
                    _ => DEFAULT_COLOR,
                };

                styled_text.push((Style::new().fg(color), token_str.clone()));
                remaining = &remaining[pos + token_str.len()..];
            }
        }

        if !remaining.is_empty() {
            styled_text.push((Style::new().fg(DEFAULT_COLOR), remaining.to_string()));
        }

        styled_text
    }
}

use crate::{parser::Expr, tokenizer::Token};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

const RED: &str = "\x1b[31m";
const YELLOW: &str = "\x1b[33m";
const GRAY: &str = "\x1b[90m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

#[derive(Error, Debug)]
pub enum Error {
    #[error("{RED}io error:{RESET} {0}")]
    IO(#[from] std::io::Error),

    #[error("{RED}parse float error:{RESET} {0}")]
    ParseFloat(#[from] std::num::ParseFloatError),

    #[error("{RED}{BOLD}Tokenizer Error:{RESET} at line {YELLOW}{line}{RESET}, column {YELLOW}{column}{RESET}:\n    {GRAY}{context}{RESET}\n{BOLD}{pointer}{RESET}\n{message}", 
        line = .line,
        column = .column,
        context = .context,
        pointer = format_error_pointer(4 + *column),
        message = .message
    )]
    Tokenizer {
        message: String,
        line: usize,
        column: usize,
        context: String,
    },

    #[error("{RED}{BOLD}Parser Error:{RESET} at line {YELLOW}{line}{RESET}, column {YELLOW}{column}{RESET}:\n    {GRAY}{context}{RESET}\n{BOLD}{pointer}{RESET}\n{message}", 
        line = .line,
        column = .column,
        context = .context,
        pointer = format_error_pointer(4 + *column),
        message = .message
    )]
    Parser {
        message: String,
        line: usize,
        column: usize,
        context: String,
    },

    #[error("{RED}{BOLD}Runtime Error:{RESET} at line {YELLOW}{line}{RESET}, column {YELLOW}{column}{RESET}:\n    {GRAY}{context}{RESET}\n{BOLD}{pointer}{RESET}\n{message}", 
        line = .line,
        column = .column,
        context = .context,
        pointer = format_error_pointer(4 + *column),
        message = .message
    )]
    Runtime {
        message: String,
        line: usize,
        column: usize,
        context: String,
    },
}

pub fn tokenizer_error<T>(source: &[u8], message: &str, start_byte: usize) -> Result<T> {
    let (line, column, context) = get_location(source, start_byte);
    let message = message.to_string();
    Err(Error::Tokenizer {
        message,
        line,
        column,
        context,
    })
}

pub fn parser_error<T>(source: &[u8], message: &str, token: &Token) -> Result<T> {
    let (line, column, context) = get_location(source, token.byte_span.start);
    Err(Error::Parser {
        message: message.to_string(),
        line,
        column,
        context,
    })
}

pub fn runtime_error<T>(source: &[u8], message: &str, expr: &Expr) -> Result<T> {
    let (line, column, context) = get_location(source, expr.byte_span.start);
    Err(Error::Runtime {
        message: message.to_string(),
        line,
        column,
        context,
    })
}

fn get_location(source: &[u8], byte_pos: usize) -> (usize, usize, String) {
    let mut line = 1;
    let mut last_line_start = 0;
    for (pos, &byte) in source[..byte_pos].iter().enumerate() {
        if byte == b'\n' {
            line += 1;
            last_line_start = pos + 1;
        }
    }
    let column = byte_pos - last_line_start;

    let line_end = source[byte_pos..]
        .iter()
        .position(|&b| b == b'\n')
        .map_or(source.len(), |pos| byte_pos + pos);
    let context = String::from_utf8_lossy(&source[last_line_start..line_end]).into_owned();

    (line, column, context)
}

fn format_error_pointer(column: usize) -> String {
    let mut pointer = String::with_capacity(column + 1);
    for _ in 0..column {
        pointer.push(' ');
    }
    pointer.push('^');

    pointer
}

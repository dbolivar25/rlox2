use crate::error::{tokenizer_error, Result};
use std::{f64, ops::Range};

#[derive(Debug)]
pub struct Token {
    pub token_type: TokenType,
    pub byte_span: Range<usize>,
}

#[derive(Debug)]
pub enum TokenType {
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    LeftSquare,
    RightSquare,

    Comma,
    Dot,
    Semicolon,

    Plus,
    Minus,
    Star,
    Slash,
    Percent,

    Bang,
    Equal,
    Greater,
    Less,

    BangEqual,
    EqualEqual,
    GreaterEqual,
    LessEqual,
    Concat,

    Identifier(String),
    String(String),
    Number(f64),

    And,
    Or,
    Struct,
    Let,
    Rec,
    If,
    Else,
    Nil,
    Return,
    True,
    False,
    For,
    While,
    Fn,

    Skip,
    EOF,
}

impl PartialEq for TokenType {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TokenType::Number(a), TokenType::Number(b)) => {
                const EPSILON: f64 = 1e-10;
                (a - b).abs() < EPSILON
            }
            (TokenType::String(a), TokenType::String(b)) => a == b,
            (TokenType::Identifier(a), TokenType::Identifier(b)) => a == b,
            _ => std::mem::discriminant(self) == std::mem::discriminant(other),
        }
    }
}

impl Eq for TokenType {}

pub fn tokenize(source: &[u8]) -> Result<Vec<Token>> {
    let n = source.len();
    let mut cursor = 0;
    let mut tokens = Vec::with_capacity(n); // upper bound

    loop {
        let (bytes_read, token) = next_token(source, cursor)?;
        cursor += bytes_read;

        match token.token_type {
            TokenType::EOF => {
                tokens.push(token);
                return Ok(tokens);
            }
            TokenType::Skip => continue,
            _ => tokens.push(token),
        }
    }
}

fn next_token(source: &[u8], offset: usize) -> Result<(usize, Token)> {
    let bytes = &source[offset..];
    let n = bytes.len();
    if n == 0 {
        return Ok((
            0,
            Token {
                token_type: TokenType::EOF,
                byte_span: offset.saturating_sub(1)..offset.saturating_sub(1),
            },
        ));
    }

    let whitespace = bytes
        .iter()
        .take_while(|&&b| matches!(b, b' ' | b'\t' | b'\r' | b'\n'))
        .count();

    if whitespace == n {
        return Ok((
            whitespace,
            Token {
                token_type: TokenType::EOF,
                byte_span: (offset + whitespace.saturating_sub(1))
                    ..(offset + whitespace.saturating_sub(1)),
            },
        ));
    }

    let token_start = offset + whitespace;
    let cursor = whitespace;

    if cursor + 1 < n && bytes[cursor..cursor + 2] == *b"//" {
        let comment_start = cursor + 2;
        let mut comment_end = comment_start;

        // Consume until newline or EOF
        while comment_end < n && bytes[comment_end] != b'\n' {
            comment_end += 1;
        }

        // Include the newline in the consumed characters if we found one
        if comment_end < n && bytes[comment_end] == b'\n' {
            comment_end += 1;
            return Ok((
                comment_end,
                Token {
                    token_type: TokenType::Skip,
                    byte_span: token_start..(token_start + comment_end - cursor),
                },
            ));
        }

        // If we reached EOF, return EOF token
        return Ok((
            comment_end,
            Token {
                token_type: TokenType::EOF,
                byte_span: (offset + comment_end.saturating_sub(1))
                    ..(offset + comment_end.saturating_sub(1)),
            },
        ));
    }

    if cursor + 1 < n {
        let token_type = match &bytes[cursor..cursor + 2] {
            b"==" => Some(TokenType::EqualEqual),
            b"!=" => Some(TokenType::BangEqual),
            b"<=" => Some(TokenType::LessEqual),
            b">=" => Some(TokenType::GreaterEqual),
            b"<>" => Some(TokenType::Concat),
            _ => None,
        };

        if let Some(token_type) = token_type {
            return Ok((
                cursor + 2,
                Token {
                    token_type,
                    byte_span: token_start..(token_start + 2),
                },
            ));
        }
    }

    let token = match bytes[cursor] {
        b'(' => Some(TokenType::LeftParen),
        b')' => Some(TokenType::RightParen),
        b'{' => Some(TokenType::LeftBrace),
        b'}' => Some(TokenType::RightBrace),
        b'[' => Some(TokenType::LeftSquare),
        b']' => Some(TokenType::RightSquare),
        b',' => Some(TokenType::Comma),
        b'.' => Some(TokenType::Dot),
        b';' => Some(TokenType::Semicolon),
        b'+' => Some(TokenType::Plus),
        b'-' => Some(TokenType::Minus),
        b'*' => Some(TokenType::Star),
        b'/' => Some(TokenType::Slash),
        b'%' => Some(TokenType::Percent),
        b'=' => Some(TokenType::Equal),
        b'!' => Some(TokenType::Bang),
        b'<' => Some(TokenType::Less),
        b'>' => Some(TokenType::Greater),
        _ => None,
    };

    if let Some(token_type) = token {
        return Ok((
            cursor + 1,
            Token {
                token_type,
                byte_span: token_start..(token_start + 1),
            },
        ));
    }

    if bytes[cursor] == b'"' {
        let str_start = cursor + 1;
        let mut str_end = str_start;

        while str_end < n && bytes[str_end] != b'"' {
            str_end += 1;
        }

        if str_end >= n {
            return tokenizer_error(
                source,
                "Unterminated string literal: missing closing double quote",
                token_start + (str_end - cursor),
            );
        }

        return Ok((
            str_end + 1,
            Token {
                token_type: TokenType::String(bytes[str_start..str_end].escape_ascii().to_string()),
                byte_span: token_start..(token_start + (str_end - cursor) + 1),
            },
        ));
    }

    if bytes[cursor] == b'\'' {
        let str_start = cursor + 1;
        let mut str_end = str_start;

        while str_end < n && bytes[str_end] != b'\'' {
            str_end += 1;
        }

        if str_end >= n {
            return tokenizer_error(
                source,
                "Unterminated string literal: missing closing single quote",
                token_start + (str_end - cursor),
            );
        }

        return Ok((
            str_end + 1,
            Token {
                token_type: TokenType::String(bytes[str_start..str_end].escape_ascii().to_string()),
                byte_span: token_start..(token_start + (str_end - cursor) + 1),
            },
        ));
    }

    if bytes[cursor].is_ascii_digit() {
        let num_start = cursor;
        let mut num_end = cursor;

        while num_end < n && bytes[num_end].is_ascii_digit() {
            num_end += 1;
        }

        if num_end < n && bytes[num_end] == b'.' {
            num_end += 1;

            if num_end >= n || !bytes[num_end].is_ascii_digit() {
                return tokenizer_error(
                    source,
                    &format!(
                        "Invalid number format: expected digit after decimal point, found '{}'",
                        if num_end >= n {
                            "end of file"
                        } else {
                            std::str::from_utf8(&bytes[num_end..num_end + 1]).unwrap_or("?")
                        }
                    ),
                    token_start + num_end,
                );
            }

            while num_end < n && bytes[num_end].is_ascii_digit() {
                num_end += 1;
            }
        }

        return Ok((
            num_end,
            Token {
                token_type: TokenType::Number(
                    bytes[num_start..num_end]
                        .escape_ascii()
                        .to_string()
                        .parse()?,
                ),
                byte_span: token_start..(token_start + num_end - cursor),
            },
        ));
    }

    if bytes[cursor].is_ascii_alphabetic() || bytes[cursor] == b'_' {
        let ident_start = cursor;
        let mut ident_end = cursor + 1;

        while ident_end < n
            && (bytes[ident_end].is_ascii_alphanumeric() || bytes[ident_end] == b'_')
        {
            ident_end += 1;
        }

        let token_type = match &bytes[ident_start..ident_end] {
            b"and" => TokenType::And,
            b"or" => TokenType::Or,
            b"struct" => TokenType::Struct,
            b"let" => TokenType::Let,
            b"rec" => TokenType::Rec,
            b"if" => TokenType::If,
            b"else" => TokenType::Else,
            b"nil" => TokenType::Nil,
            b"return" => TokenType::Return,
            b"true" => TokenType::True,
            b"false" => TokenType::False,
            b"for" => TokenType::For,
            b"while" => TokenType::While,
            b"fn" => TokenType::Fn,
            _ => TokenType::Identifier(bytes[ident_start..ident_end].escape_ascii().to_string()),
        };

        return Ok((
            ident_end,
            Token {
                token_type,
                byte_span: token_start..(token_start + ident_end - cursor),
            },
        ));
    }

    tokenizer_error(
        source,
        &format!(
            "Invalid character '{}' encountered in input",
            bytes[cursor].escape_ascii()
        ),
        token_start,
    )
}

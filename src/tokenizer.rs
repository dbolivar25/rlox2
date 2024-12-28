use log::debug;

use crate::error::{Error, Result, TokenizerError};

#[derive(Debug)]
pub enum Token {
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
    If,
    Else,
    Nil,
    Return,
    True,
    False,
    For,
    While,
    Fn,

    NewLine,
    EOF,
}

pub fn tokenize(bytes: &[u8]) -> Result<Vec<Token>> {
    let n = bytes.len();
    let mut cursor = 0;
    let mut line_number = 1;
    let mut tokens = Vec::with_capacity(n); // upper bound

    loop {
        match next_token(&bytes[cursor..]) {
            Ok((bytes_read, Token::EOF)) => {
                cursor += bytes_read;
                tokens.push(Token::EOF);

                assert_eq!(cursor, n);

                return Ok(tokens);
            }
            Ok((bytes_read, Token::NewLine)) => {
                cursor += bytes_read;
                line_number += 1;
            }
            Ok((bytes_read, token)) => {
                cursor += bytes_read;
                tokens.push(token);
            }
            Err(err) => {
                debug!("{} on line {}", err, line_number);
                return Err(err);
            }
        }
    }
}

fn next_token(bytes: &[u8]) -> Result<(usize, Token)> {
    let n = bytes.len();
    if n == 0 {
        return Ok((0, Token::EOF));
    }

    let cursor = bytes
        .iter()
        .take_while(|&&b| matches!(b, b' ' | b'\t' | b'\r' | b'\\'))
        .count();

    if cursor == n {
        return Ok((cursor, Token::EOF));
    }

    if cursor + 1 < n {
        let token = match &bytes[cursor..cursor + 2] {
            b"==" => Some(Token::EqualEqual),
            b"!=" => Some(Token::BangEqual),
            b"<=" => Some(Token::LessEqual),
            b">=" => Some(Token::GreaterEqual),
            b"<>" => Some(Token::Concat),
            _ => None,
        };

        if let Some(token) = token {
            return Ok((cursor + 2, token));
        }
    }

    let token = match bytes[cursor] {
        b'\n' => Some(Token::NewLine),
        b'(' => Some(Token::LeftParen),
        b')' => Some(Token::RightParen),
        b'{' => Some(Token::LeftBrace),
        b'}' => Some(Token::RightBrace),
        b'[' => Some(Token::LeftSquare),
        b']' => Some(Token::RightSquare),
        b',' => Some(Token::Comma),
        b'.' => Some(Token::Dot),
        b';' => Some(Token::Semicolon),
        b'+' => Some(Token::Plus),
        b'-' => Some(Token::Minus),
        b'*' => Some(Token::Star),
        b'/' => Some(Token::Slash),
        b'=' => Some(Token::Equal),
        b'!' => Some(Token::Bang),
        b'<' => Some(Token::Less),
        b'>' => Some(Token::Greater),
        _ => None,
    };

    if let Some(token) = token {
        return Ok((cursor + 1, token));
    }

    if bytes[cursor] == b'"' {
        let start_byte = cursor + 1;
        let mut end_byte = cursor + 1;

        while end_byte < n && bytes[end_byte] != b'"' {
            end_byte += 1;
        }

        if end_byte >= n {
            return Err(Error::Tokenizer(TokenizerError {}));
        }

        return Ok((
            end_byte + 1,
            Token::String(bytes[start_byte..end_byte].escape_ascii().to_string()),
        ));
    }

    if bytes[cursor] == b'\'' {
        let start_byte = cursor + 1;
        let mut end_byte = cursor + 1;

        while end_byte < n && bytes[end_byte] != b'\'' {
            end_byte += 1;
        }

        if end_byte >= n {
            return Err(Error::Tokenizer(TokenizerError {}));
        }

        return Ok((
            end_byte + 1,
            Token::String(bytes[start_byte..end_byte].escape_ascii().to_string()),
        ));
    }

    if bytes[cursor].is_ascii_digit() {
        let start_byte = cursor;
        let mut end_byte = cursor;

        while end_byte < n && bytes[end_byte].is_ascii_digit() {
            end_byte += 1;
        }

        if end_byte < n && bytes[end_byte] == b'.' {
            end_byte += 1;

            if end_byte >= n || !bytes[end_byte].is_ascii_digit() {
                return Err(Error::Tokenizer(TokenizerError {}));
            }

            while end_byte < n && bytes[end_byte].is_ascii_digit() {
                end_byte += 1;
            }
        }

        return Ok((
            end_byte,
            Token::Number(
                bytes[start_byte..end_byte]
                    .escape_ascii()
                    .to_string()
                    .parse()?,
            ),
        ));
    }

    if bytes[cursor].is_ascii_alphabetic() || bytes[cursor] == b'_' {
        let start_byte = cursor;
        let mut end_byte = cursor + 1;

        while end_byte < n && (bytes[end_byte].is_ascii_alphanumeric() || bytes[end_byte] == b'_') {
            end_byte += 1;
        }

        let token = match &bytes[start_byte..end_byte] {
            b"and" => Token::And,
            b"or" => Token::Or,
            b"struct" => Token::Struct,
            b"let" => Token::Let,
            b"if" => Token::If,
            b"else" => Token::Else,
            b"nil" => Token::Nil,
            b"return" => Token::Return,
            b"true" => Token::True,
            b"false" => Token::False,
            b"for" => Token::For,
            b"while" => Token::While,
            b"fn" => Token::Fn,
            _ => Token::Identifier(bytes[start_byte..end_byte].escape_ascii().to_string()),
        };

        return Ok((end_byte, token));
    }

    Err(Error::Tokenizer(TokenizerError {}))
}

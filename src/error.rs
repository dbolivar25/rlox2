use std::fmt::Display;

use thiserror::Error;

#[derive(Error, Debug)]
pub struct TokenizerError {}

impl Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TODO")
    }
}

#[derive(Error, Debug)]
pub struct ParserError {}

impl Display for ParserError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TODO")
    }
}

#[derive(Error, Debug)]
pub struct RuntimeError {}

impl Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TODO")
    }
}

#[derive(Error, Debug)]
pub enum Error {
    #[error("io error: {0}")]
    IO(#[from] std::io::Error),
    #[error("parse float error: {0}")]
    ParseFloat(#[from] std::num::ParseFloatError),
    #[error("tokenizer error: {0}")]
    Tokenizer(TokenizerError),
    #[error("parser error: {0:?}")]
    Parser(Vec<ParserError>),
    #[error("runtime error: {0:?}")]
    Runtime(Vec<RuntimeError>),
}

pub type Result<T> = std::result::Result<T, Error>;

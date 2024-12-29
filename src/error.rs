use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("io error: {0}")]
    IO(#[from] std::io::Error),
    #[error("parse float error: {0}")]
    ParseFloat(#[from] std::num::ParseFloatError),
    #[error("tokenizer error: {message}")]
    Tokenizer { message: String },
    #[error("parser error: {message}")]
    Parser { message: String },
    #[error("runtime error: {message}")]
    Runtime { message: String },
}

pub type Result<T> = std::result::Result<T, Error>;

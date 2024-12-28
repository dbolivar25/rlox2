use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[clap(author = "Daniel Bolivar", version)]
pub struct Args {
    pub file: Option<PathBuf>,
    pub args: Vec<String>,
}

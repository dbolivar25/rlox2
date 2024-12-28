use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[clap(author = "Daniel Bolivar", version)]
pub struct Args {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Run a source file with optional arguments
    Run {
        /// Path to the source file
        file: PathBuf,

        /// Optional arguments to pass to the program
        args: Vec<String>,
    },

    /// Check a source file for syntax errors
    Check {
        /// Path to the source file to check
        file: PathBuf,
    },

    /// Start an interactive REPL session
    Repl,
}

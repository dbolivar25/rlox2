use clap::Parser;
use log::{debug, info};
use nu_ansi_term::{Color, Style};
use reedline::{DefaultHinter, Reedline, Signal};
use rlox2::{
    cli::Args,
    error::Result,
    repl::{REPLPrompt, REPLValidator},
    tokenizer::tokenize,
};
use std::{fs, path::PathBuf};

fn run_repl() -> Result<()> {
    let mut line_editor = Reedline::create()
        .with_hinter(Box::new(
            DefaultHinter::default().with_style(Style::new().italic().fg(Color::LightGray)),
        ))
        .with_validator(Box::new(REPLValidator));
    let prompt = REPLPrompt;

    loop {
        match line_editor.read_line(&prompt) {
            Ok(Signal::Success(buffer)) => {
                let tokens = tokenize(buffer.as_bytes()).ok();

                dbg!(tokens);
            }
            Ok(Signal::CtrlD | Signal::CtrlC) => {
                return Ok(());
            }
            Err(e) => {
                eprintln!("error: {}", e);
                return Ok(());
            }
        }
    }
}

fn run_file(file: PathBuf, _args: Vec<String>) -> Result<()> {
    let bytes = fs::read(file)?;
    let tokens = tokenize(&bytes)?;

    dbg!(tokens);

    Ok(())
}

fn main() -> Result<()> {
    env_logger::init();

    match Args::parse() {
        Args { file: None, .. } => {
            info!("REPL MODE");

            run_repl()?;
        }
        Args {
            file: Some(file),
            args,
        } => {
            info!("FILE MODE");
            debug!("file: {:?}", file);
            debug!("args: {:?}", args);

            run_file(file, args)?;
        }
    }

    Ok(())
}

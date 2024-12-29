use clap::Parser;
use dirs::home_dir;
use log::{debug, info};
use nu_ansi_term::{Color, Style};
use reedline::{DefaultHinter, FileBackedHistory, Reedline, Signal};
use rlox2::{
    cli::{Args, Commands},
    error::Result,
    parser::parse,
    repl::{REPLPrompt, REPLValidator, SyntaxHighlighter},
    tokenizer::tokenize,
};
use std::{fs, path::PathBuf};

fn run_file(file: PathBuf, _args: Vec<String>) -> Result<()> {
    let bytes = fs::read(file)?;

    let tokens = tokenize(&bytes)?;
    dbg!(&tokens);

    let ast = parse(&tokens)?;
    dbg!(&ast);

    Ok(())
}

fn check_file(file: PathBuf) -> Result<()> {
    let bytes = fs::read(file)?;

    let tokens = tokenize(&bytes)?;
    dbg!(&tokens);

    let ast = parse(&tokens)?;
    dbg!(&ast);

    Ok(())
}

fn run_repl() -> Result<()> {
    let history_file = home_dir()
        .expect("Could not find home directory")
        .join(".rlox_history");

    let history = Box::new(
        FileBackedHistory::with_file(20, history_file)
            .expect("Error configuring history with file"),
    );
    let mut line_editor = Reedline::create()
        .with_hinter(Box::new(
            DefaultHinter::default().with_style(Style::new().italic().fg(Color::LightGray)),
        ))
        .with_highlighter(Box::new(SyntaxHighlighter))
        .with_validator(Box::new(REPLValidator))
        .with_history(history);
    let prompt = REPLPrompt;

    loop {
        match line_editor.read_line(&prompt) {
            Ok(Signal::Success(buffer)) => match tokenize(buffer.as_bytes()) {
                Ok(tokens) => match parse(&tokens) {
                    Ok(expression) => {
                        dbg!(expression);
                        continue;
                    }
                    Err(parse_error) => {
                        dbg!(tokens);
                        dbg!(parse_error);
                        continue;
                    }
                },
                Err(tokenize_error) => {
                    dbg!(buffer);
                    dbg!(tokenize_error);
                    continue;
                }
            },
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

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    match args.command {
        Commands::Run { file, args } => {
            info!("FILE MODE");
            debug!("file: {:?}", file);
            debug!("args: {:?}", args);

            run_file(file, args)?;
        }
        Commands::Check { file } => {
            info!("CHECK MODE");
            debug!("file: {:?}", file);

            check_file(file)?;
        }
        Commands::Repl => {
            info!("REPL MODE");

            run_repl()?;
        }
    }
    Ok(())
}

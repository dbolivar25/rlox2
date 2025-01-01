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
    runtime::evaluate,
    stdlib::create_standard_env,
    tokenizer::tokenize,
};
use std::{fs, path::PathBuf};

fn run_file(file: PathBuf, _args: Vec<String>) -> Result<()> {
    let source = fs::read(file)?;
    let tokens = tokenize(&source)?;
    let ast = parse(&source, &tokens)?;

    let mut environment = create_standard_env();
    let evaluation = evaluate(&ast, &mut environment)?;

    debug!("evaluation: {}", evaluation);

    Ok(())
}

fn check_file(file: PathBuf) -> Result<()> {
    let source = fs::read(file)?;

    let tokens = tokenize(&source)?;
    dbg!(&tokens);

    let ast = parse(&source, &tokens)?;
    dbg!(&ast);

    Ok(())
}

fn run_repl() -> Result<()> {
    let mut line_editor = Reedline::create()
        .with_hinter(Box::new(
            DefaultHinter::default().with_style(Style::new().italic().fg(Color::LightGray)),
        ))
        .with_highlighter(Box::new(SyntaxHighlighter))
        .with_validator(Box::new(REPLValidator));

    // Add file-backed history if possible
    if let Some(history) = home_dir()
        .map(|home| home.join(".rlox_history"))
        .and_then(|path| FileBackedHistory::with_file(20, path).ok())
        .map(Box::new)
    {
        line_editor = line_editor.with_history(history);
    } else {
        eprintln!("NOTE: Failed to load history. Persistence is now disabled.")
    }

    let prompt = REPLPrompt;

    let mut environment = create_standard_env();

    loop {
        match line_editor.read_line(&prompt)? {
            Signal::Success(buffer) => {
                let source = buffer.as_bytes();
                tokenize(source)
                    .and_then(|tokens| parse(source, &tokens))
                    .and_then(|ast| evaluate(&ast, &mut environment))
                    .inspect(|evaluation| {
                        debug!("evaluation: {}", evaluation);
                    })
                    .inspect_err(|err| {
                        eprintln!("{}", err);
                    })
                    .ok();
            }
            Signal::CtrlD | Signal::CtrlC => {
                break Ok(());
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

            run_file(file, args)
                .inspect_err(|err| {
                    eprintln!("{}", err);
                })
                .ok();
        }
        Commands::Check { file } => {
            info!("CHECK MODE");
            debug!("file: {:?}", file);

            check_file(file)
                .inspect_err(|err| {
                    eprintln!("{}", err);
                })
                .ok();
        }
        Commands::Repl => {
            info!("REPL MODE");

            run_repl()
                .inspect_err(|err| {
                    eprintln!("{}", err);
                })
                .ok();
        }
    }
    Ok(())
}

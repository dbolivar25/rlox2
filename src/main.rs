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
    runtime::{evaluate, Value},
    stdlib::default_env,
    tokenizer::tokenize,
};
use std::{fs, path::PathBuf};

fn run_file(file: PathBuf, _args: Vec<String>) -> Result<()> {
    let source = fs::read(file)?;
    debug!("source: {:?}", &source);

    let tokens = tokenize(&source)?;
    debug!("tokens: {:?}", &tokens);

    let ast = parse(&source, &tokens)?;
    debug!("ast: {:?}", &ast);

    let mut environment = default_env();
    let evaluation = evaluate(&source, &ast, &mut environment)?;
    debug!("evaluation: {:?}", evaluation);

    if let Value::Nil = evaluation {
    } else {
        println!("{:?}", evaluation);
    }

    Ok(())
}

fn check_file(file: PathBuf) -> Result<()> {
    let source = fs::read(file)?;
    debug!("source: {:?}", &source);

    let tokens = tokenize(&source)?;
    debug!("tokens: {:?}", &tokens);

    let ast = parse(&source, &tokens)?;
    debug!("ast: {:?}", &ast);

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

    let mut environment = default_env();

    loop {
        match line_editor.read_line(&prompt)? {
            Signal::Success(buffer) => {
                let source = buffer.as_bytes();
                tokenize(source)
                    .inspect(|tokens| {
                        debug!("tokens: {:?}", tokens);
                    })
                    .and_then(|tokens| parse(source, &tokens))
                    .inspect(|ast| {
                        debug!("ast: {:?}", ast);
                    })
                    .and_then(|ast| evaluate(source, &ast, &mut environment))
                    .inspect(|evaluation| {
                        debug!("evaluation: {:?}", evaluation);

                        if let Value::Nil = evaluation {
                        } else {
                            println!("{:?}", evaluation);
                        }
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

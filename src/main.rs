use clap::{error::ErrorKind, Parser, Subcommand};
use humanify::cli::{anthropic, gemini, ollama, openai, openrouter};
use std::path::PathBuf;

const EXIT_CLI_USAGE: i32 = 64;

#[derive(Parser)]
#[command(
    name = "humanify",
    version,
    about = "Un-minify JavaScript with LLM help"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Openai(SubArgs),
    Gemini(SubArgs),
    Anthropic(SubArgs),
    Ollama(SubArgs),
    Openrouter(SubArgs),
    Resume { state_file: PathBuf },
}

#[derive(Parser)]
struct SubArgs {
    /// Filename, or `-` for stdin
    input: String,

    /// Output file (default: stdout)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Override preset's default model
    #[arg(short, long)]
    model: Option<String>,

    /// Override env-var-based API key
    #[arg(short = 'k', long)]
    api_key: Option<String>,

    /// Override preset's base URL
    #[arg(long)]
    base_url: Option<String>,

    /// Surrounding code chars per identifier
    #[arg(long, default_value_t = 500)]
    context_size: usize,

    /// JSON strategy mode
    #[arg(long, default_value = "ladder")]
    json_mode: String,

    /// Per-request HTTP timeout in seconds. Overrides the preset default
    /// (60s for hosted APIs, 1800s for Ollama).
    #[arg(long)]
    timeout_seconds: Option<u64>,

    /// Enable raw LLM request/response debug logging
    #[arg(long)]
    enable_llm_log: bool,

    /// Append raw LLM request/response debug records to this JSONL file
    #[arg(long)]
    llm_log_file: Option<PathBuf>,

    /// Append program run logs to this file (default: humanify.log)
    #[arg(long)]
    log_file: Option<PathBuf>,

    /// Disable program log file output
    #[arg(long)]
    no_log_file: bool,

    /// Only write program errors to stderr; file logging remains detailed
    #[arg(long)]
    quiet_log: bool,

    /// Debug log to stderr
    #[arg(short, long)]
    verbose: bool,

    /// Continue from a matching saved rename state when available
    #[arg(long)]
    resume: bool,

    /// Override the saved rename state path
    #[arg(long)]
    state_file: Option<PathBuf>,

    /// Maximum LLM attempts before pausing
    #[arg(long, default_value_t = 5)]
    max_llm_attempts: u32,

    /// Estimated token budget for each LLM batch
    #[arg(long)]
    llm_batch_token_budget: Option<usize>,

    /// Maximum symbols in each LLM batch
    #[arg(long)]
    llm_batch_max_symbols: Option<usize>,

    /// Run deterministic planner only; fail if LLM would be needed
    #[arg(long, hide = true)]
    dry_run_no_llm: bool,
}

fn into_openai_args(a: SubArgs) -> openai::Args {
    openai::Args {
        input: a.input,
        output: a.output,
        model: a.model,
        api_key: a.api_key,
        base_url: a.base_url,
        context_size: a.context_size,
        json_mode: a.json_mode,
        verbose: a.verbose,
        timeout_seconds: a.timeout_seconds,
        enable_llm_log: a.enable_llm_log,
        llm_log_file: a.llm_log_file,
        log_file: a.log_file,
        no_log_file: a.no_log_file,
        quiet_log: a.quiet_log,
        resume: a.resume,
        state_file: a.state_file,
        max_llm_attempts: a.max_llm_attempts,
        llm_batch_token_budget: a.llm_batch_token_budget,
        llm_batch_max_symbols: a.llm_batch_max_symbols,
        dry_run_no_llm: a.dry_run_no_llm,
    }
}

fn into_gemini_args(a: SubArgs) -> gemini::Args {
    gemini::Args {
        input: a.input,
        output: a.output,
        model: a.model,
        api_key: a.api_key,
        base_url: a.base_url,
        context_size: a.context_size,
        json_mode: a.json_mode,
        verbose: a.verbose,
        timeout_seconds: a.timeout_seconds,
        enable_llm_log: a.enable_llm_log,
        llm_log_file: a.llm_log_file,
        log_file: a.log_file,
        no_log_file: a.no_log_file,
        quiet_log: a.quiet_log,
        resume: a.resume,
        state_file: a.state_file,
        max_llm_attempts: a.max_llm_attempts,
        llm_batch_token_budget: a.llm_batch_token_budget,
        llm_batch_max_symbols: a.llm_batch_max_symbols,
        dry_run_no_llm: a.dry_run_no_llm,
    }
}

fn into_anthropic_args(a: SubArgs) -> anthropic::Args {
    anthropic::Args {
        input: a.input,
        output: a.output,
        model: a.model,
        api_key: a.api_key,
        base_url: a.base_url,
        context_size: a.context_size,
        json_mode: a.json_mode,
        verbose: a.verbose,
        timeout_seconds: a.timeout_seconds,
        enable_llm_log: a.enable_llm_log,
        llm_log_file: a.llm_log_file,
        log_file: a.log_file,
        no_log_file: a.no_log_file,
        quiet_log: a.quiet_log,
        resume: a.resume,
        state_file: a.state_file,
        max_llm_attempts: a.max_llm_attempts,
        llm_batch_token_budget: a.llm_batch_token_budget,
        llm_batch_max_symbols: a.llm_batch_max_symbols,
        dry_run_no_llm: a.dry_run_no_llm,
    }
}

fn into_ollama_args(a: SubArgs) -> ollama::Args {
    ollama::Args {
        input: a.input,
        output: a.output,
        model: a.model,
        api_key: a.api_key,
        base_url: a.base_url,
        context_size: a.context_size,
        json_mode: a.json_mode,
        verbose: a.verbose,
        timeout_seconds: a.timeout_seconds,
        enable_llm_log: a.enable_llm_log,
        llm_log_file: a.llm_log_file,
        log_file: a.log_file,
        no_log_file: a.no_log_file,
        quiet_log: a.quiet_log,
        resume: a.resume,
        state_file: a.state_file,
        max_llm_attempts: a.max_llm_attempts,
        llm_batch_token_budget: a.llm_batch_token_budget,
        llm_batch_max_symbols: a.llm_batch_max_symbols,
        dry_run_no_llm: a.dry_run_no_llm,
    }
}

fn into_openrouter_args(a: SubArgs) -> openrouter::Args {
    openrouter::Args {
        input: a.input,
        output: a.output,
        model: a.model,
        api_key: a.api_key,
        base_url: a.base_url,
        context_size: a.context_size,
        json_mode: a.json_mode,
        verbose: a.verbose,
        timeout_seconds: a.timeout_seconds,
        enable_llm_log: a.enable_llm_log,
        llm_log_file: a.llm_log_file,
        log_file: a.log_file,
        no_log_file: a.no_log_file,
        quiet_log: a.quiet_log,
        resume: a.resume,
        state_file: a.state_file,
        max_llm_attempts: a.max_llm_attempts,
        llm_batch_token_budget: a.llm_batch_token_budget,
        llm_batch_max_symbols: a.llm_batch_max_symbols,
        dry_run_no_llm: a.dry_run_no_llm,
    }
}

fn main() {
    let cli = match Cli::try_parse() {
        Ok(c) => c,
        Err(e) => match e.kind() {
            ErrorKind::DisplayHelp | ErrorKind::DisplayVersion => {
                e.exit();
            }
            _ => {
                let _ = e.print();
                std::process::exit(EXIT_CLI_USAGE);
            }
        },
    };

    let exit_code = match cli.command {
        Commands::Openai(args) => openai::run(into_openai_args(args)),
        Commands::Gemini(args) => gemini::run(into_gemini_args(args)),
        Commands::Anthropic(args) => anthropic::run(into_anthropic_args(args)),
        Commands::Ollama(args) => ollama::run(into_ollama_args(args)),
        Commands::Openrouter(args) => openrouter::run(into_openrouter_args(args)),
        Commands::Resume { state_file } => humanify::cli::preset::resume_from_state(&state_file),
    };

    if exit_code != 0 {
        std::process::exit(exit_code);
    }
}

use std::path::PathBuf;

use crate::cli::preset::{run_preset, PresetArgs, PresetDefaults, ProviderKind};

pub const DEFAULTS: PresetDefaults = PresetDefaults {
    provider_name: "openai",
    base_url: "https://api.openai.com/v1",
    model: "gpt-5-mini",
    api_key_env: "OPENAI_API_KEY",
    provider_kind: ProviderKind::OpenAICompat,
    timeout_seconds: 60,
};

/// Plain args carrier — not a clap struct so it can be constructed without clap.
pub struct Args {
    pub input: String,
    pub output: Option<PathBuf>,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub context_size: usize,
    pub json_mode: String,
    pub verbose: bool,
    pub timeout_seconds: Option<u64>,
    pub enable_llm_log: bool,
    pub llm_log_file: Option<PathBuf>,
    pub log_file: Option<PathBuf>,
    pub no_log_file: bool,
    pub quiet_log: bool,
}

impl From<Args> for PresetArgs {
    fn from(a: Args) -> Self {
        PresetArgs {
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
        }
    }
}

/// Run the openai subcommand. Returns a process exit code (0 / 1 / 2 / 64).
pub fn run(args: Args) -> i32 {
    run_preset(args.into(), DEFAULTS)
}

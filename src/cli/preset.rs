use std::env;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;

use crate::cli::app_log::{AppLogger, RunTimer, StderrMode};
use crate::llm::log::LlmLogger;
use crate::llm::{
    http::HttpClient, AnthropicNativeJsonSchema, AnthropicToolCallAndPrompt, ForcedToolCall,
    JsonStrategy, Ladder, LlmRenamer, OpenAIJsonSchema, PromptToJson, ToolCallAndPrompt,
};
use crate::pipe;
use crate::rename::inventory::build_symbol_inventory;
use crate::rename::rules::plan_deterministic_renames;
use crate::rename::{rename_all_identifiers_with_progress, RenameError, RenameProgressPhase, Renamer};
use crate::rename::plan::PlanItemState;
use crate::rename::state::load_state;

pub struct PresetConfig {
    pub base_url: String,
    pub model: String,
    pub api_key: Option<String>,
    pub json_mode: JsonMode,
    pub context_size: usize,
    pub verbose: bool,
}

#[derive(Clone, Copy)]
pub enum ProviderKind {
    OpenAICompat,
    Anthropic,
}

#[derive(Clone, Copy)]
pub struct PresetDefaults {
    pub provider_name: &'static str,
    pub base_url: &'static str,
    pub model: &'static str,
    pub api_key_env: &'static str,
    pub provider_kind: ProviderKind,
    /// Per-request HTTP timeout. Set generously for local providers (Ollama on a
    /// CPU runner can take ~10–15 min for a single constrained completion) and
    /// tight for hosted APIs that answer in seconds.
    pub timeout_seconds: u64,
}

/// Generic args carrier for all presets.
pub struct PresetArgs {
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
    pub resume: bool,
    pub state_file: Option<PathBuf>,
    pub max_llm_attempts: u32,
    pub llm_batch_token_budget: Option<usize>,
    pub llm_batch_max_symbols: Option<usize>,
    pub dry_run_no_llm: bool,
}

/// Returns Err with a user-facing message if `mode` is not valid for `kind`.
pub fn validate_json_mode_for_provider(mode: &JsonMode, kind: ProviderKind) -> Result<(), String> {
    match (mode, kind) {
        (JsonMode::AnthropicNative, ProviderKind::OpenAICompat) => Err(
            "--json-mode anthropic-native is only valid for the `anthropic` subcommand".to_string(),
        ),
        (
            JsonMode::OpenAIJsonSchema | JsonMode::ForcedToolCall | JsonMode::ToolCallAndPrompt | JsonMode::Prompt,
            ProviderKind::Anthropic,
        ) => Err(format!(
            "--json-mode {} is not valid for the `anthropic` subcommand; use anthropic-native or ladder",
            mode.as_str()
        )),
        _ => Ok(()),
    }
}

/// Drives the full pipeline for any preset. Returns process exit code (0 / 1 / 2 / 64).
pub fn run_preset(args: PresetArgs, defaults: PresetDefaults) -> i32 {
    let timer = RunTimer::start();
    let app_log_path = resolve_app_log_file(args.no_log_file, args.log_file.as_ref());
    let app_logger = match AppLogger::open(app_log_path.as_deref(), stderr_mode(args.quiet_log)) {
        Ok(logger) => logger,
        Err(e) => {
            eprintln!("humanify: failed to open program log: {e}");
            return 1;
        }
    };

    let json_mode = match JsonMode::parse(&args.json_mode) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("humanify: {e}");
            log_error_and_finish(&app_logger, &timer, "argument_error", &e, 64);
            return 64;
        }
    };

    if let Err(msg) = validate_json_mode_for_provider(&json_mode, defaults.provider_kind) {
        eprintln!("humanify: {msg}");
        log_error_and_finish(&app_logger, &timer, "argument_error", &msg, 64);
        return 64;
    }

    let model = args.model.unwrap_or_else(|| defaults.model.to_string());
    let llm_log_file = resolve_llm_log_file(
        &args.input,
        args.enable_llm_log,
        args.llm_log_file.as_ref(),
        defaults.provider_name,
        &model,
    );
    let timeout_seconds = args.timeout_seconds.unwrap_or(defaults.timeout_seconds);
    let output = args.output;
    let base_url_source = option_source(args.base_url.as_ref());

    app_logger.info(
        "start",
        [
            ("provider", defaults.provider_name),
            ("input", args.input.as_str()),
            ("output", output_label(output.as_ref()).as_str()),
            ("model", model.as_str()),
            ("base_url_source", base_url_source),
            ("json_mode", json_mode.as_str()),
            ("context_size", args.context_size.to_string().as_str()),
            ("timeout_seconds", timeout_seconds.to_string().as_str()),
            ("llm_log", on_off(llm_log_file.is_some())),
        ],
    );

    let cfg = PresetConfig {
        base_url: args
            .base_url
            .unwrap_or_else(|| defaults.base_url.to_string()),
        model,
        api_key: args.api_key.or_else(|| env_api_key(defaults.api_key_env)),
        json_mode,
        context_size: args.context_size,
        verbose: args.verbose,
    };
    let llm_logger = match llm_log_file.as_deref() {
        Some(path) => match LlmLogger::open(path) {
            Ok(logger) => Some(logger),
            Err(e) => {
                eprintln!("humanify: failed to open LLM log: {e}");
                log_error_and_finish(&app_logger, &timer, "llm_log_open_error", &e.to_string(), 1);
                return 1;
            }
        },
        None => None,
    };

    let source = match pipe::read_input(&args.input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("humanify: failed to read input: {e}");
            log_error_and_finish(&app_logger, &timer, "input_read_error", &e.to_string(), 1);
            return 1;
        }
    };
    app_logger.info("input_read", [("bytes", source.len().to_string().as_str())]);

    if args.dry_run_no_llm {
        let renamed = match run_deterministic_rename_only(&source, cfg.context_size, &app_logger) {
            Ok(renamed) => renamed,
            Err(RenameError::Parse(msg)) => {
                eprintln!("humanify: parse error: {msg}");
                log_error_and_finish(&app_logger, &timer, "parse_error", &msg, 2);
                return 2;
            }
        }; if let Err(e) = pipe::write_output(output.as_deref(), &renamed) {
            eprintln!("humanify: failed to write output: {e}");
            log_error_and_finish(&app_logger, &timer, "output_write_error", &e.to_string(), 1);
            return 1;
        }
        app_logger.info("output_write", [("target", output_label(output.as_ref()).as_str()), ("bytes", renamed.len().to_string().as_str())]);
        log_finish(&app_logger, &timer, 0);
        return 0;
    }

    let rt = match tokio::runtime::Runtime::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("humanify: failed to create tokio runtime: {e}");
            log_error_and_finish(&app_logger, &timer, "runtime_error", &e.to_string(), 1);
            return 1;
        }
    };

    let timeout = std::time::Duration::from_secs(timeout_seconds);
    let client = HttpClient::with_timeout_and_logger(timeout, llm_logger);
    let ladder = Arc::new(build_ladder(client, &cfg, defaults.provider_kind));
    let mut renamer = LlmRenamer::new(Arc::clone(&ladder), rt.handle().clone());
    let context_size = cfg.context_size;
    let progress_logger = app_logger.clone();
    let mut progress_timer = timer.clone();

    let result = rt.block_on(async move {
        tokio::task::spawn_blocking(move || {
            rename_all_identifiers_with_progress(&source, &mut renamer, context_size, |progress| {
                match progress.phase {
                    RenameProgressPhase::Start => {
                        eprintln!(
                            "humanify: renaming {}/{}: {}",
                            progress.current, progress.total, progress.original_name
                        );
                        progress_logger.info(
                            "progress_start",
                            [
                                ("step", progress.current.to_string().as_str()),
                                ("total", progress.total.to_string().as_str()),
                                ("original_name", progress.original_name),
                                ("elapsed_ms", progress_timer.elapsed_ms().as_str()),
                            ],
                        );
                    }
                    RenameProgressPhase::Finish => {
                        let (step_ms, elapsed_ms) = progress_timer.step_ms_and_elapsed_ms();
                        progress_logger.info(
                            "progress_finish",
                            [
                                ("step", progress.current.to_string().as_str()),
                                ("total", progress.total.to_string().as_str()),
                                ("original_name", progress.original_name),
                                (
                                    "new_name",
                                    progress.new_name.unwrap_or(progress.original_name),
                                ),
                                ("elapsed_ms", elapsed_ms.as_str()),
                                ("step_ms", step_ms.as_str()),
                            ],
                        );
                    }
                }
            })
        })
        .await
    });

    let renamed = match result {
        Ok(Ok(s)) => s,
        Ok(Err(RenameError::Parse(msg))) => {
            eprintln!("humanify: parse error: {msg}");
            log_error_and_finish(&app_logger, &timer, "parse_error", &msg, 2);
            return 2;
        }
        Err(join_err) => {
            eprintln!("humanify: internal error: {join_err}");
            log_error_and_finish(
                &app_logger,
                &timer,
                "internal_error",
                &join_err.to_string(),
                1,
            );
            return 1;
        }
    };

    if cfg.verbose {
        let locked = ladder.locked_strategy_name().unwrap_or("none");
        eprintln!("humanify: locked strategy: {locked}");
        app_logger.info("strategy", [("locked", locked)]);
    }

    if let Err(e) = pipe::write_output(output.as_deref(), &renamed) {
        eprintln!("humanify: failed to write output: {e}");
        log_error_and_finish(&app_logger, &timer, "output_write_error", &e.to_string(), 1);
        return 1;
    }

    app_logger.info(
        "output_write",
        [
            ("target", output_label(output.as_ref()).as_str()),
            ("bytes", renamed.len().to_string().as_str()),
        ],
    );
    log_finish(&app_logger, &timer, 0);
    0
}

fn resolve_app_log_file(no_log_file: bool, explicit_file: Option<&PathBuf>) -> Option<PathBuf> {
    if no_log_file {
        None
    } else {
        Some(
            explicit_file
                .cloned()
                .unwrap_or_else(|| AppLogger::default_log_file().to_path_buf()),
        )
    }
}

fn stderr_mode(quiet_log: bool) -> StderrMode {
    if quiet_log {
        StderrMode::ErrorsOnly
    } else {
        StderrMode::All
    }
}

fn option_source<T>(value: Option<&T>) -> &'static str {
    if value.is_some() {
        "custom"
    } else {
        "default"
    }
}

fn output_label(output: Option<&PathBuf>) -> String {
    output
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "stdout".to_string())
}

fn on_off(value: bool) -> &'static str {
    if value {
        "on"
    } else {
        "off"
    }
}

fn log_error_and_finish(
    logger: &AppLogger,
    timer: &RunTimer,
    event: &str,
    message: &str,
    exit_code: i32,
) {
    logger.error(event, [("message", message)]);
    log_finish(logger, timer, exit_code);
}

fn log_finish(logger: &AppLogger, timer: &RunTimer, exit_code: i32) {
    logger.info(
        "finish",
        [
            ("exit_code", exit_code.to_string().as_str()),
            ("elapsed_ms", timer.elapsed_ms().as_str()),
        ],
    );
}

fn build_ladder(client: HttpClient, cfg: &PresetConfig, kind: ProviderKind) -> Ladder {
    match cfg.json_mode {
        JsonMode::Ladder => build_default_ladder(client, cfg, kind),
        JsonMode::OpenAIJsonSchema => Ladder::pinned(Arc::new(OpenAIJsonSchema::new(
            client,
            cfg.base_url.clone(),
            cfg.api_key.clone(),
            cfg.model.clone(),
        ))),
        JsonMode::ForcedToolCall => Ladder::pinned(Arc::new(ForcedToolCall::new(
            client,
            cfg.base_url.clone(),
            cfg.api_key.clone(),
            cfg.model.clone(),
        ))),
        JsonMode::ToolCallAndPrompt => Ladder::pinned(Arc::new(ToolCallAndPrompt::new(
            client,
            cfg.base_url.clone(),
            cfg.api_key.clone(),
            cfg.model.clone(),
        ))),
        JsonMode::Prompt => Ladder::pinned(Arc::new(PromptToJson::new(
            client,
            cfg.base_url.clone(),
            cfg.api_key.clone(),
            cfg.model.clone(),
        ))),
        JsonMode::AnthropicNative => Ladder::pinned(Arc::new(AnthropicNativeJsonSchema::new(
            client,
            cfg.base_url.clone(),
            cfg.api_key.clone(),
            cfg.model.clone(),
        ))),
    }
}

pub(crate) fn build_default_ladder(
    client: HttpClient,
    cfg: &PresetConfig,
    kind: ProviderKind,
) -> Ladder {
    let strategies: Vec<Arc<dyn JsonStrategy>> = match kind {
        ProviderKind::OpenAICompat => vec![
            Arc::new(OpenAIJsonSchema::new(
                client.clone(),
                cfg.base_url.clone(),
                cfg.api_key.clone(),
                cfg.model.clone(),
            )),
            Arc::new(ForcedToolCall::new(
                client.clone(),
                cfg.base_url.clone(),
                cfg.api_key.clone(),
                cfg.model.clone(),
            )),
            Arc::new(PromptToJson::new(
                client,
                cfg.base_url.clone(),
                cfg.api_key.clone(),
                cfg.model.clone(),
            )),
        ],
        // AnthropicNativeJsonSchema uses a beta API whose response shape we
        // haven't validated against a live call — its parser frequently rejects
        // real responses as "no JSON block in content", and the ladder cannot
        // fall back from a Transient error. Default to the tool-call strategy,
        // which is well-tested. AnthropicNativeJsonSchema is still reachable
        // via `--json-mode anthropic-native` for anyone wanting to opt in.
        ProviderKind::Anthropic => vec![Arc::new(AnthropicToolCallAndPrompt::new(
            client,
            cfg.base_url.clone(),
            cfg.api_key.clone(),
            cfg.model.clone(),
        ))],
    };
    Ladder::new(strategies)
}

/// Selects which JSON strategy (or ladder) to use for a run.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonMode {
    Ladder,
    OpenAIJsonSchema,
    AnthropicNative,
    ForcedToolCall,
    ToolCallAndPrompt,
    Prompt,
}

impl JsonMode {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "ladder" => Ok(JsonMode::Ladder),
            "openai-json-schema" => Ok(JsonMode::OpenAIJsonSchema),
            "anthropic-native" => Ok(JsonMode::AnthropicNative),
            "forced-tool-call" => Ok(JsonMode::ForcedToolCall),
            "tool-call-and-prompt" => Ok(JsonMode::ToolCallAndPrompt),
            "prompt" => Ok(JsonMode::Prompt),
            other => Err(format!(
                "unknown json-mode '{}'. Valid values: ladder, openai-json-schema, \
                 anthropic-native, forced-tool-call, tool-call-and-prompt, prompt",
                other
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            JsonMode::Ladder => "ladder",
            JsonMode::OpenAIJsonSchema => "openai-json-schema",
            JsonMode::AnthropicNative => "anthropic-native",
            JsonMode::ForcedToolCall => "forced-tool-call",
            JsonMode::ToolCallAndPrompt => "tool-call-and-prompt",
            JsonMode::Prompt => "prompt",
        }
    }
}

/// Read an API key from the given env var. Returns `None` if unset or empty.
pub fn env_api_key(var_name: &str) -> Option<String> {
    env::var(var_name).ok().filter(|s| !s.is_empty())
}

fn resolve_llm_log_file(
    input: &str,
    enable_llm_log: bool,
    explicit_file: Option<&PathBuf>,
    provider: &str,
    model: &str,
) -> Option<PathBuf> {
    explicit_file
        .cloned()
        .or_else(|| enable_llm_log.then(|| default_llm_log_file(input, provider, model)))
}

pub fn default_llm_log_file(input: &str, provider: &str, model: &str) -> PathBuf {
    let input_name = input_file_name(input);
    let provider = sanitize_log_component(provider);
    let model = sanitize_log_component(model);
    PathBuf::from(format!("{input_name}-llm-{provider}-{model}.jsonl"))
}

fn input_file_name(input: &str) -> String {
    if input == "-" {
        return "stdin".to_string();
    }

    let name = std::path::Path::new(input)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(input);
    sanitize_log_component(name)
}

fn sanitize_log_component(value: &str) -> String {
    let mut out = String::new();
    let mut previous_dash = false;

    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
            out.push(ch);
            previous_dash = false;
        } else if !previous_dash {
            out.push('-');
            previous_dash = true;
        }
    }

    let trimmed = out.trim_matches('-');
    if trimmed.is_empty() {
        "unknown".to_string()
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- JsonMode::parse ---

    #[test]
    fn ladder_parses() {
        assert_eq!(JsonMode::parse("ladder"), Ok(JsonMode::Ladder));
    }

    #[test]
    fn openai_json_schema_parses() {
        assert_eq!(
            JsonMode::parse("openai-json-schema"),
            Ok(JsonMode::OpenAIJsonSchema)
        );
    }

    #[test]
    fn anthropic_native_parses() {
        assert_eq!(
            JsonMode::parse("anthropic-native"),
            Ok(JsonMode::AnthropicNative)
        );
    }

    #[test]
    fn forced_tool_call_parses() {
        assert_eq!(
            JsonMode::parse("forced-tool-call"),
            Ok(JsonMode::ForcedToolCall)
        );
    }

    #[test]
    fn tool_call_and_prompt_parses() {
        assert_eq!(
            JsonMode::parse("tool-call-and-prompt"),
            Ok(JsonMode::ToolCallAndPrompt)
        );
    }

    #[test]
    fn prompt_parses() {
        assert_eq!(JsonMode::parse("prompt"), Ok(JsonMode::Prompt));
    }

    #[test]
    fn unknown_returns_err_with_valid_values() {
        let err = JsonMode::parse("garbage").unwrap_err();
        assert!(err.contains("garbage"), "err: {err}");
        assert!(err.contains("ladder"), "err: {err}");
    }

    #[test]
    fn empty_string_returns_err() {
        assert!(JsonMode::parse("").is_err());
    }

    #[test]
    fn case_sensitive_rejects_uppercase() {
        assert!(JsonMode::parse("Ladder").is_err());
    }

    // --- env_api_key ---

    #[test]
    fn env_api_key_returns_value_when_set() {
        std::env::set_var("_TEST_KEY_SET", "mykey");
        assert_eq!(env_api_key("_TEST_KEY_SET"), Some("mykey".to_string()));
        std::env::remove_var("_TEST_KEY_SET");
    }

    #[test]
    fn env_api_key_returns_none_when_unset() {
        std::env::remove_var("_TEST_KEY_UNSET");
        assert_eq!(env_api_key("_TEST_KEY_UNSET"), None);
    }

    #[test]
    fn env_api_key_returns_none_when_empty() {
        std::env::set_var("_TEST_KEY_EMPTY", "");
        assert_eq!(env_api_key("_TEST_KEY_EMPTY"), None);
        std::env::remove_var("_TEST_KEY_EMPTY");
    }

    // --- validate_json_mode_for_provider ---

    #[test]
    fn validate_anthropic_native_on_openai_compat_returns_err() {
        assert!(validate_json_mode_for_provider(
            &JsonMode::AnthropicNative,
            ProviderKind::OpenAICompat
        )
        .is_err());
    }

    #[test]
    fn validate_anthropic_native_on_anthropic_returns_ok() {
        assert!(validate_json_mode_for_provider(
            &JsonMode::AnthropicNative,
            ProviderKind::Anthropic
        )
        .is_ok());
    }

    #[test]
    fn validate_valid_mode_on_openai_compat_returns_ok() {
        assert!(
            validate_json_mode_for_provider(&JsonMode::Ladder, ProviderKind::OpenAICompat).is_ok()
        );
    }

    #[test]
    fn validate_openai_json_schema_on_anthropic_returns_err() {
        assert!(validate_json_mode_for_provider(
            &JsonMode::OpenAIJsonSchema,
            ProviderKind::Anthropic
        )
        .is_err());
    }

    #[test]
    fn validate_forced_tool_call_on_anthropic_returns_err() {
        assert!(validate_json_mode_for_provider(
            &JsonMode::ForcedToolCall,
            ProviderKind::Anthropic
        )
        .is_err());
    }

    #[test]
    fn validate_tool_call_and_prompt_on_anthropic_returns_err() {
        assert!(validate_json_mode_for_provider(
            &JsonMode::ToolCallAndPrompt,
            ProviderKind::Anthropic
        )
        .is_err());
    }

    #[test]
    fn validate_prompt_on_anthropic_returns_err() {
        assert!(
            validate_json_mode_for_provider(&JsonMode::Prompt, ProviderKind::Anthropic).is_err()
        );
    }

    #[test]
    fn validate_ladder_on_anthropic_returns_ok() {
        assert!(
            validate_json_mode_for_provider(&JsonMode::Ladder, ProviderKind::Anthropic).is_ok()
        );
    }

    #[test]
    fn validate_prompt_on_openai_compat_returns_ok() {
        assert!(
            validate_json_mode_for_provider(&JsonMode::Prompt, ProviderKind::OpenAICompat).is_ok()
        );
    }

    // --- PresetDefaults sanity ---

    #[test]
    fn openai_defaults_constants() {
        assert_eq!(
            crate::cli::openai::DEFAULTS.base_url,
            "https://api.openai.com/v1"
        );
        assert_eq!(crate::cli::openai::DEFAULTS.model, "gpt-5-mini");
        assert_eq!(crate::cli::openai::DEFAULTS.api_key_env, "OPENAI_API_KEY");
    }

    #[test]
    fn gemini_defaults_constants() {
        assert_eq!(
            crate::cli::gemini::DEFAULTS.base_url,
            "https://generativelanguage.googleapis.com/v1beta/openai/"
        );
        assert_eq!(crate::cli::gemini::DEFAULTS.model, "gemini-3.1-flash-lite");
        assert_eq!(crate::cli::gemini::DEFAULTS.api_key_env, "GEMINI_API_KEY");
    }

    #[test]
    fn anthropic_defaults_constants() {
        assert_eq!(
            crate::cli::anthropic::DEFAULTS.base_url,
            "https://api.anthropic.com/v1"
        );
        assert_eq!(crate::cli::anthropic::DEFAULTS.model, "claude-sonnet-4-6");
        assert_eq!(
            crate::cli::anthropic::DEFAULTS.api_key_env,
            "ANTHROPIC_API_KEY"
        );
    }

    #[test]
    fn hosted_providers_share_short_timeout() {
        // Hosted APIs answer in seconds; a tight per-request budget surfaces
        // upstream stalls quickly instead of letting the run hang.
        assert_eq!(crate::cli::openai::DEFAULTS.timeout_seconds, 60);
        assert_eq!(crate::cli::gemini::DEFAULTS.timeout_seconds, 60);
        assert_eq!(crate::cli::anthropic::DEFAULTS.timeout_seconds, 60);
        assert_eq!(crate::cli::openrouter::DEFAULTS.timeout_seconds, 60);
    }

    #[test]
    fn ollama_gets_generous_timeout_for_local_inference() {
        assert_eq!(crate::cli::ollama::DEFAULTS.timeout_seconds, 1800);
    }

    // --- run_preset early-exit paths (no I/O reached) ---

    fn preset_args_no_io(json_mode: &str) -> PresetArgs {
        PresetArgs {
            input: "irrelevant".to_string(),
            output: None,
            model: None,
            api_key: None,
            base_url: None,
            context_size: 500,
            json_mode: json_mode.to_string(),
            verbose: false,
            timeout_seconds: None,
            enable_llm_log: false,
            llm_log_file: None,
            log_file: None,
            no_log_file: true,
            quiet_log: false,
            resume: false,
            state_file: None,
            max_llm_attempts: 5,
            llm_batch_token_budget: None,
            llm_batch_max_symbols: None,
            dry_run_no_llm: false,
        }
    }

    #[test]
    fn anthropic_native_on_openai_compat_returns_64() {
        let code = run_preset(
            preset_args_no_io("anthropic-native"),
            crate::cli::openai::DEFAULTS,
        );
        assert_eq!(code, 64);
    }

    #[test]
    fn unknown_json_mode_returns_64() {
        let code = run_preset(preset_args_no_io("garbage"), crate::cli::openai::DEFAULTS);
        assert_eq!(code, 64);
    }

    #[test]
    fn preset_args_can_carry_llm_log_file_path() {
        let mut args = preset_args_no_io("ladder");
        args.llm_log_file = Some(PathBuf::from("llm.jsonl"));
        assert_eq!(
            args.llm_log_file.as_deref(),
            Some(std::path::Path::new("llm.jsonl"))
        );
    }

    #[test]
    fn default_llm_log_file_uses_input_provider_and_sanitized_model() {
        assert_eq!(
            default_llm_log_file("fixtures/app.min.js", "openrouter", "qwen/qwen3-coder:free"),
            PathBuf::from("app.min.js-llm-openrouter-qwen-qwen3-coder-free.jsonl")
        );
    }

    #[test]
    fn default_llm_log_file_uses_stdin_for_dash_input() {
        assert_eq!(
            default_llm_log_file("-", "ollama", "qwen3.5:4b"),
            PathBuf::from("stdin-llm-ollama-qwen3.5-4b.jsonl")
        );
    }

    #[test]
    fn resolve_llm_log_file_prefers_explicit_file() {
        let mut args = preset_args_no_io("ladder");
        args.enable_llm_log = true;
        args.llm_log_file = Some(PathBuf::from("custom.jsonl"));
        let cfg = PresetConfig {
            base_url: "http://localhost:11434/v1".to_string(),
            model: "qwen3.5:4b".to_string(),
            api_key: None,
            json_mode: JsonMode::Ladder,
            context_size: 500,
            verbose: false,
        };

        assert_eq!(
            resolve_llm_log_file(
                &args.input,
                args.enable_llm_log,
                args.llm_log_file.as_ref(),
                crate::cli::ollama::DEFAULTS.provider_name,
                &cfg.model,
            ),
            Some(PathBuf::from("custom.jsonl"))
        );
    }

    // --- Anthropic default ladder shape ---

    fn anthropic_preset_cfg() -> PresetConfig {
        PresetConfig {
            base_url: crate::cli::anthropic::DEFAULTS.base_url.to_string(),
            model: crate::cli::anthropic::DEFAULTS.model.to_string(),
            api_key: None,
            json_mode: JsonMode::Ladder,
            context_size: 500,
            verbose: false,
        }
    }

    #[test]
    fn anthropic_default_ladder_uses_tool_call_only() {
        // The native json-schema strategy lives behind `--json-mode anthropic-native`;
        // the ladder ships tool-call exclusively because the native path's response
        // shape isn't validated end-to-end yet.
        let ladder = build_default_ladder(
            HttpClient::new(),
            &anthropic_preset_cfg(),
            ProviderKind::Anthropic,
        );
        assert_eq!(ladder.strategy_count(), 1);
    }
}

pub fn resume_from_state(state_file: &std::path::Path) -> i32 {
    match load_state(state_file) {
        Ok(_) => {
            eprintln!("humanify: resume is not fully implemented yet for this state file");
            1
        }
        Err(e) => {
            eprintln!("humanify: failed to load state: {e}");
            1
        }
    }
}

fn run_deterministic_rename_only(
    source: &str,
    context_size: usize,
    app_logger: &AppLogger,
) -> Result<String, RenameError> {
    app_logger.info("inventory_start", [("context_size", context_size.to_string().as_str())]);
    let inventory = build_symbol_inventory(source, context_size)?;
    app_logger.info("inventory_finish", [("symbols", inventory.entries.len().to_string().as_str())]);
    let plan = plan_deterministic_renames(&inventory);
    let needs_llm = plan.needs_llm_count();
    app_logger.info("planner_finish", [("resolved", (plan.items.len() - needs_llm).to_string().as_str()), ("needs_llm", needs_llm.to_string().as_str())]);
    if needs_llm > 0 {
        eprintln!("humanify: paused because {needs_llm} symbols still require LLM");
        return Ok(source.to_string());
    }
    let names = plan
        .items
        .iter()
        .map(|item| match &item.state {
            PlanItemState::Resolved { name, .. } => name.clone(),
            PlanItemState::Keep { .. } => item.original_name.clone(),
            PlanItemState::NeedsLlm { .. } | PlanItemState::Failed { .. } => item.original_name.clone(),
        })
        .collect::<VecDeque<_>>();
    let mut renamer = QueuePlanRenamer { names };
    let renamed = rename_all_identifiers_with_progress(source, &mut renamer, context_size, |_| {})?;
    app_logger.info("apply_finish", [("symbols", plan.items.len().to_string().as_str())]);
    Ok(renamed)
}

struct QueuePlanRenamer {
    names: VecDeque<String>,
}

impl Renamer for QueuePlanRenamer {
    fn rename(&mut self, original: &str, _: &str) -> String {
        self.names
            .pop_front()
            .unwrap_or_else(|| original.to_string())
    }
}

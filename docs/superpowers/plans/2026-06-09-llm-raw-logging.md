# LLM Raw Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in `--llm-log <PATH>` JSONL logging for raw LLM HTTP requests and responses.

**Architecture:** Add a focused `src/llm/log.rs` module that owns JSONL event serialization and synchronized append writes. Thread an optional `LlmLogger` through `HttpClient`, and wire a shared CLI `--llm-log` argument through the provider presets. Keep stdout reserved for transformed JavaScript and omit headers/API keys from logs.

**Tech Stack:** Rust 2021, `serde`, `serde_json`, `reqwest`, `tokio`, `clap`, existing `assert_cmd` and `tempfile` dev dependencies.

---

## File Structure

- Create `src/llm/log.rs`: owns `LlmLogger`, `LlmLogEvent`, `LlmLogResponse`, and append-only JSONL writing.
- Modify `src/llm/mod.rs`: expose the new log module inside the crate.
- Modify `src/llm/http.rs`: add optional logging to `HttpClient` and record each `post_json` result.
- Modify `src/cli/preset.rs`: add `llm_log` to `PresetArgs`, open the logger early, and construct a logging `HttpClient` when requested.
- Modify provider CLI files under `src/cli/*.rs`: add `llm_log` to each public `Args` struct and pass it to `PresetArgs`.
- Modify `src/main.rs`: add `--llm-log <PATH>` to shared subcommand args and pass it through every provider converter.
- Modify `README.md`: document the new debug logging flag and mention JSONL viewers.
- Modify or add tests in `tests/` only if a CLI smoke test can be written without live provider credentials.

### Task 1: Add The LLM JSONL Logger

**Files:**
- Create: `src/llm/log.rs`
- Modify: `src/llm/mod.rs`

- [ ] **Step 1: Write failing unit tests for JSONL append logging**

Create `src/llm/log.rs` with the test module first:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;

    #[test]
    fn appends_one_valid_json_line_per_event() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("llm.jsonl");
        let logger = LlmLogger::open(&path).unwrap();

        logger
            .log(&LlmLogEvent {
                timestamp_ms: 123,
                url: "http://localhost:11434/v1/chat/completions".to_string(),
                request: json!({"model":"qwen3.5:4b"}),
                response: Some(LlmLogResponse::Json(json!({"ok":true}))),
                error: None,
            })
            .unwrap();
        logger
            .log(&LlmLogEvent {
                timestamp_ms: 456,
                url: "http://localhost:11434/v1/chat/completions".to_string(),
                request: json!({"model":"qwen3.5:4b"}),
                response: None,
                error: Some("network timeout".to_string()),
            })
            .unwrap();

        let contents = fs::read_to_string(path).unwrap();
        let lines: Vec<_> = contents.lines().collect();
        assert_eq!(lines.len(), 2);

        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first["timestamp_ms"], 123);
        assert_eq!(first["url"], "http://localhost:11434/v1/chat/completions");
        assert_eq!(first["request"]["model"], "qwen3.5:4b");
        assert_eq!(first["response"]["json"]["ok"], true);
        assert!(first["error"].is_null());

        let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(second["timestamp_ms"], 456);
        assert!(second["response"].is_null());
        assert_eq!(second["error"], "network timeout");
    }

    #[test]
    fn open_fails_for_missing_parent_directory() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing").join("llm.jsonl");
        assert!(LlmLogger::open(path).is_err());
    }
}
```

- [ ] **Step 2: Run logger tests to verify they fail**

Run: `cargo test llm::log --lib`

Expected: compile failure because `LlmLogger`, `LlmLogEvent`, and `LlmLogResponse` are not defined yet.

- [ ] **Step 3: Implement the minimal logger**

Replace `src/llm/log.rs` with:

```rust
use serde::Serialize;
use serde_json::Value;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone)]
pub struct LlmLogger {
    file: Arc<Mutex<File>>,
}

#[derive(Serialize)]
pub struct LlmLogEvent {
    pub timestamp_ms: u128,
    pub url: String,
    pub request: Value,
    pub response: Option<LlmLogResponse>,
    pub error: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmLogResponse {
    Json(Value),
    Text { status: u16, body: String },
}

impl LlmLogger {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            file: Arc::new(Mutex::new(file)),
        })
    }

    pub fn log(&self, event: &LlmLogEvent) -> io::Result<()> {
        let line = serde_json::to_string(event).map_err(io::Error::other)?;
        let mut file = self
            .file
            .lock()
            .map_err(|_| io::Error::other("LLM log file lock poisoned"))?;
        writeln!(file, "{line}")?;
        file.flush()
    }
}

pub fn unix_timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;

    #[test]
    fn appends_one_valid_json_line_per_event() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("llm.jsonl");
        let logger = LlmLogger::open(&path).unwrap();

        logger
            .log(&LlmLogEvent {
                timestamp_ms: 123,
                url: "http://localhost:11434/v1/chat/completions".to_string(),
                request: json!({"model":"qwen3.5:4b"}),
                response: Some(LlmLogResponse::Json(json!({"ok":true}))),
                error: None,
            })
            .unwrap();
        logger
            .log(&LlmLogEvent {
                timestamp_ms: 456,
                url: "http://localhost:11434/v1/chat/completions".to_string(),
                request: json!({"model":"qwen3.5:4b"}),
                response: None,
                error: Some("network timeout".to_string()),
            })
            .unwrap();

        let contents = fs::read_to_string(path).unwrap();
        let lines: Vec<_> = contents.lines().collect();
        assert_eq!(lines.len(), 2);

        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first["timestamp_ms"], 123);
        assert_eq!(first["url"], "http://localhost:11434/v1/chat/completions");
        assert_eq!(first["request"]["model"], "qwen3.5:4b");
        assert_eq!(first["response"]["json"]["ok"], true);
        assert!(first["error"].is_null());

        let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(second["timestamp_ms"], 456);
        assert!(second["response"].is_null());
        assert_eq!(second["error"], "network timeout");
    }

    #[test]
    fn open_fails_for_missing_parent_directory() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing").join("llm.jsonl");
        assert!(LlmLogger::open(path).is_err());
    }
}
```

Modify `src/llm/mod.rs` to include the module:

```rust
pub mod anthropic;
pub mod http;
pub mod ladder;
pub mod log;
pub mod openai_compat;
pub mod renamer;
```

- [ ] **Step 4: Run logger tests to verify they pass**

Run: `cargo test llm::log --lib`

Expected: PASS for both logger tests.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/llm/log.rs src/llm/mod.rs
git commit -m "feat: add LLM JSONL logger"
```

### Task 2: Record HTTP Request And Response Bodies

**Files:**
- Modify: `src/llm/http.rs`
- Test: `src/llm/http.rs`

- [ ] **Step 1: Write failing tests for logging constructor behavior**

In `src/llm/http.rs` tests, add:

```rust
#[test]
fn http_client_default_has_no_logger() {
    let client = HttpClient::new();
    assert!(!client.has_logger());
}

#[test]
fn http_client_with_logger_reports_logger_present() {
    let dir = tempfile::tempdir().unwrap();
    let logger = crate::llm::log::LlmLogger::open(dir.path().join("llm.jsonl")).unwrap();
    let client = HttpClient::with_timeout_and_logger(Duration::from_secs(1), Some(logger));
    assert!(client.has_logger());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test http_client_ --lib`

Expected: compile failure because `has_logger` and `with_timeout_and_logger` are not defined.

- [ ] **Step 3: Add optional logger to `HttpClient`**

Update the top of `src/llm/http.rs`:

```rust
use anyhow::anyhow;
use serde_json::Value;
use std::time::Duration;

use crate::llm::log::{unix_timestamp_ms, LlmLogEvent, LlmLogResponse, LlmLogger};

#[derive(Clone)]
pub struct HttpClient {
    inner: reqwest::Client,
    logger: Option<LlmLogger>,
}
```

Update constructors:

```rust
impl HttpClient {
    pub fn new() -> Self {
        Self::with_timeout(Duration::from_secs(600))
    }

    pub fn with_timeout(timeout: Duration) -> Self {
        Self::with_timeout_and_logger(timeout, None)
    }

    pub fn with_timeout_and_logger(timeout: Duration, logger: Option<LlmLogger>) -> Self {
        let inner = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .expect("reqwest client init failed");
        Self { inner, logger }
    }

    #[cfg(test)]
    pub fn has_logger(&self) -> bool {
        self.logger.is_some()
    }
```

- [ ] **Step 4: Add a helper that records log write failures as transient errors**

Still inside `impl HttpClient`, add this private method before `post_json`:

```rust
    fn log_event(&self, event: LlmLogEvent) -> Result<(), StrategyError> {
        if let Some(logger) = &self.logger {
            logger
                .log(&event)
                .map_err(|e| StrategyError::Transient(anyhow!("failed to write LLM log: {e}")))?;
        }
        Ok(())
    }
```

- [ ] **Step 5: Log every post_json outcome**

Replace the body of `post_json` with:

```rust
    pub async fn post_json(
        &self,
        url: &str,
        api_key: Option<&str>,
        extra_headers: &[(&str, &str)],
        body: &Value,
    ) -> Result<Value, StrategyError> {
        let mut request = self
            .inner
            .post(url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json");

        if let Some(key) = api_key {
            request = request.header("Authorization", format!("Bearer {key}"));
        }

        for (name, value) in extra_headers {
            request = request.header(*name, *value);
        }

        request = request.json(body);

        let response = match request.send().await {
            Ok(response) => response,
            Err(e) => {
                let error = e.to_string();
                self.log_event(LlmLogEvent {
                    timestamp_ms: unix_timestamp_ms(),
                    url: url.to_string(),
                    request: body.clone(),
                    response: None,
                    error: Some(error.clone()),
                })?;
                return Err(StrategyError::Transient(anyhow!(error)));
            }
        };

        let status = response.status().as_u16();
        let body_text = match response.text().await {
            Ok(text) => text,
            Err(e) => {
                let error = e.to_string();
                self.log_event(LlmLogEvent {
                    timestamp_ms: unix_timestamp_ms(),
                    url: url.to_string(),
                    request: body.clone(),
                    response: None,
                    error: Some(error.clone()),
                })?;
                return Err(StrategyError::Transient(anyhow!(error)));
            }
        };

        if (200..300).contains(&status) {
            let value: Value = match serde_json::from_str(&body_text) {
                Ok(value) => value,
                Err(e) => {
                    self.log_event(LlmLogEvent {
                        timestamp_ms: unix_timestamp_ms(),
                        url: url.to_string(),
                        request: body.clone(),
                        response: Some(LlmLogResponse::Text {
                            status,
                            body: body_text,
                        }),
                        error: Some(format!("response was not valid JSON: {e}")),
                    })?;
                    return Err(StrategyError::Transient(anyhow!(
                        "response was not valid JSON: {e}"
                    )));
                }
            };

            self.log_event(LlmLogEvent {
                timestamp_ms: unix_timestamp_ms(),
                url: url.to_string(),
                request: body.clone(),
                response: Some(LlmLogResponse::Json(value.clone())),
                error: None,
            })?;
            Ok(value)
        } else {
            self.log_event(LlmLogEvent {
                timestamp_ms: unix_timestamp_ms(),
                url: url.to_string(),
                request: body.clone(),
                response: Some(LlmLogResponse::Text {
                    status,
                    body: body_text.clone(),
                }),
                error: Some(format!("http {status}")),
            })?;
            Err(classify_error(status, &body_text))
        }
    }
```

- [ ] **Step 6: Run HTTP tests**

Run: `cargo test http_client_ --lib`

Expected: PASS for the new constructor tests.

- [ ] **Step 7: Run classification tests to guard existing behavior**

Run: `cargo test classify_error --lib`

Expected: PASS. Existing error classification remains unchanged.

- [ ] **Step 8: Commit Task 2**

```bash
git add src/llm/http.rs
git commit -m "feat: log raw LLM HTTP exchanges"
```

### Task 3: Thread `--llm-log` Through Presets And Providers

**Files:**
- Modify: `src/main.rs`
- Modify: `src/cli/preset.rs`
- Modify: `src/cli/openai.rs`
- Modify: `src/cli/gemini.rs`
- Modify: `src/cli/anthropic.rs`
- Modify: `src/cli/ollama.rs`
- Modify: `src/cli/openrouter.rs`

- [ ] **Step 1: Add failing preset test for argument carrier shape**

In `src/cli/preset.rs`, update `preset_args_no_io` to include the new field after `timeout_seconds`:

```rust
            timeout_seconds: None,
            llm_log: None,
```

Then add this test:

```rust
#[test]
fn preset_args_can_carry_llm_log_path() {
    let mut args = preset_args_no_io("ladder");
    args.llm_log = Some(PathBuf::from("llm.jsonl"));
    assert_eq!(args.llm_log.as_deref(), Some(std::path::Path::new("llm.jsonl")));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test preset_args_can_carry_llm_log_path --lib`

Expected: compile failure because `PresetArgs` has no `llm_log` field.

- [ ] **Step 3: Add `llm_log` to shared preset args and open logger early**

In `src/cli/preset.rs`, import the logger:

```rust
use crate::llm::log::LlmLogger;
```

Add the field to `PresetArgs`:

```rust
    pub timeout_seconds: Option<u64>,
    pub llm_log: Option<PathBuf>,
```

In `run_preset`, after `let output = args.output;`, add:

```rust
    let llm_logger = match args.llm_log.as_deref() {
        Some(path) => match LlmLogger::open(path) {
            Ok(logger) => Some(logger),
            Err(e) => {
                eprintln!("humanify: failed to open LLM log: {e}");
                return 1;
            }
        },
        None => None,
    };
```

Replace client construction:

```rust
    let client = HttpClient::with_timeout_and_logger(timeout, llm_logger);
```

- [ ] **Step 4: Add `llm_log` to provider Args structs and conversions**

For each of `src/cli/openai.rs`, `src/cli/gemini.rs`, `src/cli/anthropic.rs`, `src/cli/ollama.rs`, and `src/cli/openrouter.rs`, add to the public `Args` struct:

```rust
    pub llm_log: Option<std::path::PathBuf>,
```

When constructing `PresetArgs`, include:

```rust
        llm_log: a.llm_log,
```

- [ ] **Step 5: Add the CLI flag in main shared subcommand args**

In `src/main.rs`, add to `SubArgs` after `timeout_seconds`:

```rust
    /// Append raw LLM request/response debug records to a JSONL file
    #[arg(long)]
    llm_log: Option<PathBuf>,
```

In every `into_*_args` function, pass:

```rust
        llm_log: a.llm_log,
```

- [ ] **Step 6: Run the targeted preset test**

Run: `cargo test preset_args_can_carry_llm_log_path --lib`

Expected: PASS.

- [ ] **Step 7: Run CLI/preset compile tests**

Run: `cargo test cli::preset --lib`

Expected: PASS.

- [ ] **Step 8: Commit Task 3**

```bash
git add src/main.rs src/cli/preset.rs src/cli/openai.rs src/cli/gemini.rs src/cli/anthropic.rs src/cli/ollama.rs src/cli/openrouter.rs
git commit -m "feat: add llm log CLI option"
```

### Task 4: Add A CLI Smoke Test For Early Log File Errors

**Files:**
- Modify: `tests/common/mod.rs` if the command builder needs a helper
- Modify or create: `tests/smoke_llm_log.rs`

- [ ] **Step 1: Inspect the existing test command builder**

Run: `rg -n "struct HumanifyCommand|fn arg|fn output|assert" tests/common/mod.rs tests`

Expected: find the existing helper shape for adding arbitrary arguments or provider-specific methods.

- [ ] **Step 2: Write a failing smoke test for missing log parent**

Create `tests/smoke_llm_log.rs`:

```rust
mod common;

use assert_cmd::prelude::*;
use common::humanify;
use std::process::Command;

#[test]
fn llm_log_missing_parent_fails_before_llm_call() {
    let dir = tempfile::tempdir().unwrap();
    let missing_log = dir.path().join("missing").join("llm.jsonl");

    let mut cmd = Command::cargo_bin("humanify").unwrap();
    cmd.arg("ollama")
        .arg("--llm-log")
        .arg(&missing_log)
        .arg("fixtures/splitstring.min.js");

    let output = cmd.output().unwrap();
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("humanify: failed to open LLM log"),
        "stderr was: {stderr}"
    );
    assert!(output.stdout.is_empty());
}
```

If `common::humanify` is unused, remove that import before committing.

- [ ] **Step 3: Run smoke test to verify behavior**

Run: `cargo test --test smoke_llm_log`

Expected before implementation is complete: FAIL or compile error. After Tasks 1-3 are complete: PASS without contacting Ollama because `run_preset` opens the log before reading input or constructing the runtime.

- [ ] **Step 4: Commit Task 4**

```bash
git add tests/smoke_llm_log.rs tests/common/mod.rs
git commit -m "test: cover llm log open failures"
```

If `tests/common/mod.rs` was not modified, omit it from `git add`.

### Task 5: Document The Flag And Viewer Options

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add README documentation**

In the usage/options section of `README.md`, add a bullet near other flags:

```markdown
* `--llm-log <PATH>` appends raw LLM HTTP request and response records to a JSON Lines file for debugging. Headers and API keys are not logged, but prompts and source context may be sensitive.
```

Add a short subsection near troubleshooting or provider notes:

```markdown
### Raw LLM logs

For debugging prompt construction or provider responses, pass `--llm-log`:

```bash
humanify ollama --llm-log llm.jsonl app.min.js -o app.js
```

The file uses JSON Lines: one JSON object per LLM HTTP call. You can inspect it with tools such as VS Code JSON Lines extensions, `jq`, `jless`, `fx`, Logdy, or lnav. The format is also suitable for later forwarding to LLM observability tools such as Langfuse.
```
```

- [ ] **Step 2: Run README grep check**

Run: `rg -n "llm-log|Raw LLM logs|JSON Lines" README.md`

Expected: the new flag and section appear exactly once each.

- [ ] **Step 3: Commit Task 5**

```bash
git add README.md
git commit -m "docs: document raw LLM logging"
```

### Task 6: Final Verification

**Files:**
- No code files expected unless verification reveals a defect.

- [ ] **Step 1: Format the code**

Run: `cargo fmt --check`

Expected: PASS. If it fails, run `cargo fmt`, then rerun `cargo fmt --check`.

- [ ] **Step 2: Run the non-live test suite**

Run: `cargo test --lib --tests`

Expected: PASS for unit and smoke tests. If live E2E tests require credentials or a local model, rerun the subset that excludes those label-gated tests and record the skipped environment-dependent suites in the final handoff.

- [ ] **Step 3: Inspect git diff**

Run: `git diff --stat HEAD~5..HEAD`

Expected: changes are limited to LLM logging implementation, tests, and README documentation.

- [ ] **Step 4: Commit any verification fixes**

If formatting or test fixes changed files:

```bash
git add <changed-files>
git commit -m "chore: finalize llm logging"
```

If no files changed, do not create an empty commit.

## Self-Review Notes

Spec coverage:

- `--llm-log <PATH>` is covered by Task 3.
- JSONL append logging is covered by Task 1.
- Raw request/response/error capture is covered by Task 2.
- Early failure for missing log path is covered by Task 3 and Task 4.
- Header/API key omission is covered by Task 2 because only URL and JSON body are passed into log events.
- Viewer documentation and Langfuse-compatible future path are covered by Task 5.

Placeholder scan: no TBD/TODO placeholders remain.

Type consistency: the plan consistently uses `LlmLogger`, `LlmLogEvent`, `LlmLogResponse`, `HttpClient::with_timeout_and_logger`, and `PresetArgs::llm_log`.

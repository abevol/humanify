# LLM Raw Logging Design

## Context

`humanify` is a Rust CLI that reads JavaScript, asks an LLM for identifier rename suggestions, and writes transformed JavaScript to stdout or an output file. The LLM request paths are shared across provider presets through `src/cli/preset.rs`, `src/llm/http.rs`, and provider strategy modules under `src/llm/`.

The CLI already keeps stdout reserved for transformed JavaScript. Diagnostics and progress go to stderr. Raw LLM request and response logging must follow the same rule: it should never contaminate stdout and should be enabled only when the user asks for it.

## Goal

Add opt-in debug logging that records the raw JSON request sent to the LLM and the raw JSON response returned by the provider. The log must be writable to a file so users can inspect failed or low-quality rename suggestions after a run.

## User Experience

Add a shared CLI flag available on every provider subcommand:

```text
--llm-log <PATH>
```

When present, `humanify` appends one JSON object per HTTP LLM call to the given file using JSON Lines format. A representative record is:

```json
{"timestamp":"2026-06-09T10:15:30.123Z","url":"http://localhost:11434/v1/chat/completions","request":{"model":"qwen3.5:4b"},"response":{"choices":[]},"error":null}
```

The file is append-only for the current run. This allows repeated runs to accumulate history and lets users follow the log with tail-like tools while a long Ollama run is still active.

The log intentionally omits request headers and API keys. Request and response bodies are logged verbatim as JSON values because they are the data needed to debug prompt construction, provider behavior, and model output parsing.

## Format And UI Compatibility

Use JSONL because it is simple, stream-friendly, and compatible with common viewers and log tools. Users can inspect it with VS Code JSON Lines extensions, `jq`, `jless`, `fx`, Logdy, lnav, or import/forward it later into LLM observability tools such as Langfuse.

This design does not directly integrate with Langfuse yet. A future exporter can transform these JSONL records into Langfuse traces/generations, including local Ollama models such as `qwen3.5:4b`, without changing the basic logging surface.

## Architecture

Extend the shared CLI argument carrier with `llm_log: Option<PathBuf>` and thread it through every provider-specific `Args` type into `PresetArgs`.

Add a small logging module in the LLM layer, such as `src/llm/log.rs`, responsible for:

- Opening the target file in append/create mode.
- Serializing a log event as one JSON line.
- Synchronizing writes safely across cloned HTTP clients and concurrent async calls.
- Returning ordinary I/O errors when the log file cannot be opened or written.

Add an optional logger to `HttpClient`. `HttpClient::with_timeout` keeps existing behavior with no logging. A new constructor or builder attaches the logger when `--llm-log` is provided. `HttpClient::post_json` logs each call after the response body or transport error is known.

The event should include:

- UTC timestamp.
- URL.
- Raw request JSON body.
- Raw response JSON body on successful JSON responses.
- Raw response text and HTTP status for non-2xx or non-JSON responses when available.
- Error string for network, timeout, response read, JSON parse, or log write failures.

Provider strategy names are useful, but the current central HTTP call does not know which `JsonStrategy` invoked it. The first implementation can rely on the request body and URL, plus the selected mode visible in CLI args. If strategy-level names are later required, add a thin context parameter to the strategy-to-HTTP call path.

## Error Handling

If `--llm-log` is omitted, there is no logging work and no behavior change.

If the log file cannot be opened, fail early before reading input or making LLM calls. This prevents users from thinking a run was captured when it was not.

If a log write fails during a run, treat it as a user-visible error and abort the run. Debug logging is explicitly requested output; silently dropping it would make the feature unreliable. The error should go to stderr and return exit code 1.

HTTP classification behavior remains unchanged. Logging must record what happened, then allow the existing success, not-supported, and transient-error paths to behave as they do today.

## Testing

Add unit tests for the logging module to verify JSONL append behavior, valid JSON records, request/response/error fields, and file-open failures.

Add focused `HttpClient` tests if the existing code can test against a local mock server without new dependencies. If that is too heavy, keep HTTP behavior covered through logger unit tests plus strategy/client constructor tests.

Add CLI argument propagation tests for every provider carrier or shared preset path so `--llm-log` reaches `run_preset`.

Run the normal Rust test suite. Live E2E provider tests remain environment-dependent and are not required for this feature.

## Scope

This design does not add a web UI, a Langfuse exporter, token accounting, cost tracking, or header logging. It also does not redact prompt content, because the purpose is raw request and response debugging. Users should treat the log file as sensitive when source code or prompts contain private information.

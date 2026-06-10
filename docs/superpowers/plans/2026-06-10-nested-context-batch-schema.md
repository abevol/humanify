# Nested Context Batch Schema Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace repeated per-item LLM evidence and long prompt-visible stable keys with nested unique contexts and numeric item ids.

**Architecture:** `src/llm/batch.rs` keeps internal stable-key identity but serializes prompt-facing jobs as contexts with nested items. Validation accepts numeric response ids and maps them back to `StableSymbolKey`. `src/llm/jobs.rs` updates the response schema to require `id` instead of `symbol_key`.

**Tech Stack:** Rust 2021, serde, serde_json, existing LLM batch/job modules, cargo test.

---

## File Structure

- Modify `src/llm/batch.rs`: add prompt-facing nested context DTOs, build batches by unique evidence context, validate numeric ids, and update tests.
- Modify `src/llm/jobs.rs`: serialize prompt DTO instead of internal job and update JSON schema to `id`.
- Modify tests in `src/llm/jobs.rs`: scripted responses return numeric ids.
- Add docs in `docs/superpowers/specs/2026-06-10-nested-context-batch-schema-design.md` and this plan.

## Task 1: Refactor Batch DTOs and Validation

- [ ] **Step 1: Write/adjust tests in `src/llm/batch.rs`**

Add tests that assert repeated evidence produces one context with two nested items, prompt ids are numeric, and validation maps numeric id responses to stable keys.

- [ ] **Step 2: Run failing focused tests**

Run: `cargo test llm::batch -- --nocapture`
Expected: FAIL while old `symbol_key` schema is still in place.

- [ ] **Step 3: Implement nested prompt schema**

Add prompt-facing structs: `PromptBatchJob`, `PromptContext`, and `PromptBatchItem`. Keep internal `LlmBatchItem` with `prompt_id`, `symbol_key`, `original_name`, and `evidence`. The prompt item field is `original` to distinguish source names from response `name`. Add `LlmBatchJob::prompt_payload()`.

- [ ] **Step 4: Update validation**

Change validation to read numeric `id`, reject missing/duplicate/unknown ids, sanitize names, and return `AcceptedRename`/`RejectedRename` with internal stable keys.

- [ ] **Step 5: Run focused tests**

Run: `cargo test llm::batch -- --nocapture`
Expected: PASS.

## Task 2: Update Job Runner Prompt and Schema

- [ ] **Step 1: Update `src/llm/jobs.rs` schema**

Require numeric `id` in each response item and remove prompt-visible `symbol_key`.

- [ ] **Step 2: Serialize prompt payload**

Change `serde_json::to_string(job)` to `serde_json::to_string(&job.prompt_payload())`.

- [ ] **Step 3: Update tests**

Change scripted successful responses from `{"symbol_key":"s1"}` to `{"id":0}`.

- [ ] **Step 4: Run focused tests**

Run: `cargo test llm::jobs -- --nocapture`
Expected: PASS.

## Task 3: Verify End-to-End Token Shape

- [ ] **Step 1: Run formatting and full tests**

Run:

```powershell
cargo fmt --check
cargo test
```

Expected: PASS.

- [ ] **Step 2: Build release**

Run: `cargo build --release`
Expected: PASS.

- [ ] **Step 3: Run the real Ollama command**

Run from `target/release`:

```powershell
.\humanify.exe ollama .\deobfuscated.js -o .\ollama\readable-qwen3.5-4b.js -m qwen3.5:4b -v --context-size 350 --json-mode tool-call-and-prompt --enable-llm-log
```

Expected: request JSON contains `contexts`, nested `items`, numeric `id`, no prompt-visible `symbol_key`, and no repeated full evidence strings.

- [ ] **Step 4: Commit**

```powershell
git add src/llm/batch.rs src/llm/jobs.rs docs/superpowers/specs/2026-06-10-nested-context-batch-schema-design.md docs/superpowers/plans/2026-06-10-nested-context-batch-schema.md
git commit -m "feat: compact llm batch prompt schema"
```

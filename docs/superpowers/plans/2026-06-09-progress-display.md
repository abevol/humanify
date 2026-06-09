# Default Progress Display Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show default per-identifier rename progress on stderr while preserving stdout as transformed JavaScript only.

**Architecture:** Add an opt-in progress callback variant of the rename walker. Keep the existing `rename_all_identifiers` function as the quiet compatibility API, and have the CLI call the progress-aware variant from `run_preset`.

**Tech Stack:** Rust 2021, clap, tokio, assert_cmd, tempfile, existing oxc parser/semantic/codegen stack.

---

## File Structure

- Modify `src/rename/walker.rs`: define `RenameProgress`, add `rename_all_identifiers_with_progress`, delegate the existing function to the new function, emit progress before each `Renamer::rename` call, and add focused walker unit tests.
- Modify `src/rename/mod.rs`: re-export `RenameProgress` and `rename_all_identifiers_with_progress` beside the existing rename API.
- Modify `src/cli/preset.rs`: import the progress-aware function and print default progress lines to stderr from inside the existing blocking rename closure.
- Modify `tests/cli_smoke.rs`: assert that the offline CLI run writes progress to stderr while final code remains in the output file.

### Task 1: Add Walker Progress API

**Files:**
- Modify: `src/rename/walker.rs`
- Modify: `src/rename/mod.rs`

- [ ] **Step 1: Write the failing walker progress tests**

Add these tests to the existing `#[cfg(test)] mod tests` in `src/rename/walker.rs`:

```rust
#[test]
fn progress_reports_processed_identifiers_in_order() {
    let mut events = Vec::new();
    let output = rename_all_identifiers_with_progress(
        "function f(a){ const b = a + 1; return b; }",
        &mut test_dsl::identity(),
        200,
        |progress| {
            events.push((
                progress.current,
                progress.total,
                progress.original_name.to_string(),
            ));
        },
    )
    .expect("rename_all_identifiers_with_progress failed");

    assert!(output.contains("function f"), "output: {output}");
    assert_eq!(events.len(), 3, "events: {events:?}");
    assert_eq!(events[0].0, 1);
    assert_eq!(events[0].1, 3);
    assert_eq!(events[1].0, 2);
    assert_eq!(events[1].1, 3);
    assert_eq!(events[2].0, 3);
    assert_eq!(events[2].1, 3);
    assert_eq!(
        events.iter().map(|(_, _, name)| name.as_str()).collect::<Vec<_>>(),
        vec!["f", "a", "b"]
    );
}

#[test]
fn progress_is_quiet_for_empty_source() {
    let mut events = Vec::new();
    let output = rename_all_identifiers_with_progress(
        "",
        &mut test_dsl::identity(),
        200,
        |progress| events.push(progress.current),
    )
    .expect("rename_all_identifiers_with_progress failed");

    assert_eq!(output, "");
    assert!(events.is_empty(), "events: {events:?}");
}
```

These tests intentionally use `test_dsl::identity()` so no LLM or async runtime is involved. The expected order follows the walker sort rule: largest scope first, then source position.

- [ ] **Step 2: Run the failing walker tests**

Run:

```bash
cargo test progress_reports_processed_identifiers_in_order progress_is_quiet_for_empty_source
```

Expected: compilation fails because `rename_all_identifiers_with_progress` and `RenameProgress` do not exist yet.

- [ ] **Step 3: Implement the progress callback variant**

In `src/rename/walker.rs`, insert this public event type above `rename_all_identifiers`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenameProgress<'a> {
    pub current: usize,
    pub total: usize,
    pub original_name: &'a str,
}
```

Replace the top-level function signature block with this wrapper plus the new callback variant:

```rust
pub fn rename_all_identifiers(
    source: &str,
    renamer: &mut dyn Renamer,
    context_size: usize,
) -> Result<String, RenameError> {
    rename_all_identifiers_with_progress(source, renamer, context_size, |_| {})
}

pub fn rename_all_identifiers_with_progress<F>(
    source: &str,
    renamer: &mut dyn Renamer,
    context_size: usize,
    mut on_progress: F,
) -> Result<String, RenameError>
where
    F: FnMut(RenameProgress<'_>),
{
```

Keep the existing function body under the new callback variant. After `entries.sort_by(...)`, add:

```rust
    let total = entries.len();
```

Change the loop header from:

```rust
    for (sym_id, _, _) in &entries {
```

to:

```rust
    for (index, (sym_id, _, _)) in entries.iter().enumerate() {
```

Immediately before the existing `let new_name = renamer.rename(&original_name, &surrounding);`, add:

```rust
        on_progress(RenameProgress {
            current: index + 1,
            total,
            original_name: &original_name,
        });
```

In `src/rename/mod.rs`, update the public exports to include the new API. The export line should become:

```rust
pub use walker::{rename_all_identifiers, rename_all_identifiers_with_progress, RenameProgress};
```

- [ ] **Step 4: Run the walker tests again**

Run:

```bash
cargo test progress_reports_processed_identifiers_in_order progress_is_quiet_for_empty_source
```

Expected: both tests pass.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add src/rename/walker.rs src/rename/mod.rs
git commit -m "Add rename progress callback API"
```

### Task 2: Print Progress From CLI

**Files:**
- Modify: `src/cli/preset.rs`
- Modify: `tests/cli_smoke.rs`

- [ ] **Step 1: Write the failing CLI smoke assertion**

In `tests/cli_smoke.rs`, change the command assertion in `gemini_offline_identity` from chained `.assert().success();` to capture stderr:

```rust
    let assert = Command::cargo_bin("humanify")
        .unwrap()
        .args([
            "gemini",
            "-",
            "-o",
            out_path.to_str().unwrap(),
            "--base-url",
            "http://127.0.0.1:1",
        ])
        .write_stdin("const x = 1;")
        .assert()
        .success();

    let stderr = String::from_utf8(assert.get_output().stderr.clone()).unwrap();
    assert!(
        stderr.contains("humanify: renaming 1/1: x"),
        "stderr: {stderr}"
    );
```

Keep the existing output-file assertion below it:

```rust
    let contents = std::fs::read_to_string(&out_path).unwrap();
    assert_eq!(contents.trim(), "const x = 1;");
```

- [ ] **Step 2: Run the failing CLI smoke test**

Run:

```bash
cargo test --test cli_smoke gemini_offline_identity
```

Expected: test fails because the CLI does not print progress yet.

- [ ] **Step 3: Wire CLI progress output**

In `src/cli/preset.rs`, update the import that currently brings in `rename_all_identifiers` so it brings in `rename_all_identifiers_with_progress` instead. The import should read:

```rust
use crate::{
    llm::{http::HttpClient, renamer::LlmRenamer},
    pipe,
    rename::{rename_all_identifiers_with_progress, RenameError},
};
```

Then change the blocking rename call in `run_preset` from:

```rust
        tokio::task::spawn_blocking(move || {
            rename_all_identifiers(&source, &mut renamer, context_size)
        })
```

to:

```rust
        tokio::task::spawn_blocking(move || {
            rename_all_identifiers_with_progress(&source, &mut renamer, context_size, |progress| {
                eprintln!(
                    "humanify: renaming {}/{}: {}",
                    progress.current, progress.total, progress.original_name
                );
            })
        })
```

- [ ] **Step 4: Run the CLI smoke test again**

Run:

```bash
cargo test --test cli_smoke gemini_offline_identity
```

Expected: the test passes, and the output file still contains only `const x = 1;`.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add src/cli/preset.rs tests/cli_smoke.rs
git commit -m "Show rename progress in CLI"
```

### Task 3: Final Verification

**Files:**
- No new edits expected unless verification exposes a bug.

- [ ] **Step 1: Run the focused tests**

Run:

```bash
cargo test progress_reports_processed_identifiers_in_order progress_is_quiet_for_empty_source
cargo test --test cli_smoke gemini_offline_identity
```

Expected: all focused tests pass.

- [ ] **Step 2: Run the full test suite**

Run:

```bash
cargo test
```

Expected: all unit and smoke tests pass. If live E2E tests require provider credentials or local model availability and fail for environmental reasons, record the failing test names and stderr, then rerun the non-E2E subset that is available locally.

- [ ] **Step 3: Inspect git status and recent commits**

Run:

```bash
git status --short
git log -3 --oneline
```

Expected: only intentional changes remain, and the two implementation commits appear above the design/spec commit.

- [ ] **Step 4: Final response**

Report the files changed, the progress format, and the exact verification commands that passed. Mention any tests that could not be run or any environment-only failures.

## Self-Review

Spec coverage: Task 1 implements the callback API and preserves the quiet library wrapper. Task 2 implements default stderr progress in the shared CLI path and verifies stdout/output behavior. Task 3 verifies focused and broad behavior.

Placeholder scan: no placeholder or deferred implementation notes remain.

Type consistency: the plan consistently uses `RenameProgress<'a>`, `rename_all_identifiers_with_progress`, `current`, `total`, and `original_name` across tests and implementation.

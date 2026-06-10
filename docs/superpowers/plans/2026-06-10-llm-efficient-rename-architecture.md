# LLM-Efficient Rename Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-symbol LLM renaming with a plan-first rename pipeline that resolves obvious names locally, batches remaining LLM work globally, and can pause/resume from durable state.

**Architecture:** Introduce a file-level symbol inventory and rename plan before any semantic mutation. Run deterministic naming rules first, then send only unresolved items through globally packed durable LLM jobs with validation, retry, checkpointing, and resume support.

**Tech Stack:** Rust 2021, oxc parser/semantic/codegen, serde/serde_json, tokio, clap, assert_cmd, tempfile.

---

## File Structure

- Create `src/rename/inventory.rs`: parse/semantic symbol inventory builder and stable symbol keys.
- Create `src/rename/plan.rs`: `RenamePlan`, plan item states, validation helpers, and final-plan application inputs.
- Create `src/rename/rules.rs`: deterministic high-confidence naming rules for the initial vertical slice.
- Modify `src/rename/walker.rs`: split current online walker into inventory collection plus final plan application while preserving compatibility APIs.
- Modify `src/rename/mod.rs`: export new inventory/plan APIs and keep old `Renamer` path available during migration.
- Create `src/llm/batch.rs`: global unresolved-symbol batching and response schema helpers.
- Create `src/llm/jobs.rs`: durable LLM job runner, retry state, partial-success splitting, and pause result.
- Create `src/rename/state.rs`: atomic state persistence and resume validation.
- Modify `src/llm/renamer.rs`: add batch-oriented LLM renamer while retaining the existing single-symbol adapter for compatibility tests.
- Modify `src/llm/mod.rs`: export `batch` and `jobs` modules.
- Modify `src/cli/preset.rs`: wire plan-first pipeline, state file options, retry options, phase logging, pause exit handling.
- Modify provider arg files under `src/cli/*.rs`: pass new preset args through.
- Modify `src/main.rs`: add resume command and new subcommand flags.
- Add or modify tests in `src/rename/*`, `src/llm/*`, `src/cli/preset.rs`, and `tests/cli_smoke.rs`.

## Task 1: Add Plan Data Model and State Persistence

**Files:**
- Create: `src/rename/plan.rs`
- Create: `src/rename/state.rs`
- Modify: `src/rename/mod.rs`
- Modify: `Cargo.toml`

- [ ] **Step 1: Add dependencies**

Add `sha2` for stable keys and `hex` for readable hashes:

```toml
sha2 = "0.10"
hex = "0.4"
```

- [ ] **Step 2: Write failing plan/state tests**

Add unit tests in `src/rename/state.rs` under `#[cfg(test)]`:

```rust
#[test]
fn stable_symbol_key_is_repeatable() {
    let key1 = StableSymbolKey::new("input", 10, 12, "a", "function", "0/1");
    let key2 = StableSymbolKey::new("input", 10, 12, "a", "function", "0/1");
    assert_eq!(key1, key2);
}

#[test]
fn atomic_state_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("state.json");
    let state = RenameState::new_for_test("abc123", "input.js", "out.js");
    save_state_atomic(&path, &state).unwrap();
    let loaded = load_state(&path).unwrap();
    assert_eq!(loaded.input_hash, "abc123");
    assert_eq!(loaded.input_path, "input.js");
    assert_eq!(loaded.output_path, "out.js");
}

#[test]
fn resume_rejects_mismatched_input_hash() {
    let state = RenameState::new_for_test("old", "input.js", "out.js");
    let err = state.validate_resume("new", "input.js").unwrap_err();
    assert!(err.to_string().contains("input hash"));
}
```

- [ ] **Step 3: Run failing tests**

Run: `cargo test rename::state -- --nocapture`

Expected: FAIL because `rename::state` does not exist.

- [ ] **Step 4: Implement minimal plan/state types**

Create `src/rename/plan.rs` with serializable plan states:

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StableSymbolKey(pub String);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PlanItemState {
    Resolved { name: String, source: String, confidence: u8 },
    Keep { reason: String },
    NeedsLlm { evidence: String, attempts: u32 },
    Failed { reason: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenamePlanItem {
    pub key: StableSymbolKey,
    pub original_name: String,
    pub scope_path: String,
    pub state: PlanItemState,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RenamePlan {
    pub items: Vec<RenamePlanItem>,
}

impl RenamePlan {
    pub fn needs_llm_count(&self) -> usize {
        self.items
            .iter()
            .filter(|item| matches!(item.state, PlanItemState::NeedsLlm { .. }))
            .count()
    }
}
```

Create `src/rename/state.rs`:

```rust
use std::fs;
use std::io;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::plan::{RenamePlan, StableSymbolKey};

impl StableSymbolKey {
    pub fn new(input_hash: &str, start: u32, end: u32, original: &str, kind: &str, scope_path: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(input_hash.as_bytes());
        hasher.update(start.to_le_bytes());
        hasher.update(end.to_le_bytes());
        hasher.update(original.as_bytes());
        hasher.update(kind.as_bytes());
        hasher.update(scope_path.as_bytes());
        Self(hex::encode(hasher.finalize()))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub backoff_ms: Vec<u64>,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self { max_attempts: 5, backoff_ms: vec![1000, 3000, 10000, 30000, 120000] }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenameState {
    pub version: u32,
    pub input_hash: String,
    pub input_path: String,
    pub output_path: String,
    pub phase: String,
    pub retry_policy: RetryPolicy,
    pub plan: RenamePlan,
}

impl RenameState {
    pub fn validate_resume(&self, input_hash: &str, input_path: &str) -> Result<()> {
        if self.input_hash != input_hash {
            return Err(anyhow!("state input hash mismatch"));
        }
        if self.input_path != input_path {
            return Err(anyhow!("state input path mismatch"));
        }
        Ok(())
    }

    #[cfg(test)]
    pub fn new_for_test(input_hash: &str, input_path: &str, output_path: &str) -> Self {
        Self {
            version: 1,
            input_hash: input_hash.to_string(),
            input_path: input_path.to_string(),
            output_path: output_path.to_string(),
            phase: "test".to_string(),
            retry_policy: RetryPolicy::default(),
            plan: RenamePlan::default(),
        }
    }
}

pub fn save_state_atomic(path: &Path, state: &RenameState) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create state dir {}", parent.display()))?;
    }
    let tmp = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(state)?;
    fs::write(&tmp, bytes).with_context(|| format!("write temp state {}", tmp.display()))?;
    fs::rename(&tmp, path).or_else(|e| {
        if e.kind() == io::ErrorKind::AlreadyExists {
            fs::remove_file(path)?;
            fs::rename(&tmp, path)
        } else {
            Err(e)
        }
    }).with_context(|| format!("rename state {}", path.display()))?;
    Ok(())
}

pub fn load_state(path: &Path) -> Result<RenameState> {
    let bytes = fs::read(path).with_context(|| format!("read state {}", path.display()))?;
    Ok(serde_json::from_slice(&bytes)?)
}
```

Update `src/rename/mod.rs`:

```rust
pub mod plan;
pub mod state;
```

- [ ] **Step 5: Run tests**

Run: `cargo test rename::state -- --nocapture`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add Cargo.toml src/rename/mod.rs src/rename/plan.rs src/rename/state.rs
git commit -m "feat: add durable rename state model"
```

## Task 2: Build Symbol Inventory Without Calling LLM

**Files:**
- Create: `src/rename/inventory.rs`
- Modify: `src/rename/walker.rs`
- Modify: `src/rename/mod.rs`

- [ ] **Step 1: Write failing inventory tests**

Add tests to `src/rename/inventory.rs`:

```rust
#[test]
fn inventory_collects_fibonacci_sample_symbols() {
    let source = "function a(e){var t=[0,1];while(t.length<e){var n=t.length;t.push(t[n-1]+t[n-2])}return t}var c=a(10);";
    let inventory = build_symbol_inventory(source, 350).unwrap();
    let names = inventory.entries.iter().map(|e| e.original_name.as_str()).collect::<Vec<_>>();
    assert!(names.contains(&"a"), "names: {names:?}");
    assert!(names.contains(&"e"), "names: {names:?}");
    assert!(names.contains(&"t"), "names: {names:?}");
    assert!(names.contains(&"n"), "names: {names:?}");
    assert!(names.contains(&"c"), "names: {names:?}");
    assert!(inventory.entries.iter().all(|e| !e.key.0.is_empty()));
}

#[test]
fn inventory_keeps_context_windows() {
    let source = "function sumItems(items){var total=0;for(var i=0;i<items.length;i++){total+=items[i]}return total}";
    let inventory = build_symbol_inventory(source, 40).unwrap();
    let total = inventory.entries.iter().find(|e| e.original_name == "total").unwrap();
    assert!(total.context.len() <= source.len());
    assert!(total.context.contains("total"));
}
```

- [ ] **Step 2: Run failing inventory tests**

Run: `cargo test rename::inventory -- --nocapture`

Expected: FAIL because inventory module does not exist.

- [ ] **Step 3: Implement inventory builder**

Create `SymbolInventory` and `SymbolEntry` using the existing parser/semantic setup from `walker.rs`. Move or expose `find_binding_ancestor_span` and `compute_context_window` as `pub(crate)` helpers so inventory and legacy walker share the same context logic.

The implementation should collect entries sorted with the current rule: largest binding scope first, then declaration position.

- [ ] **Step 4: Preserve legacy API behavior**

Update `rename_all_identifiers_with_progress` to call `build_symbol_inventory` internally for symbol ordering/context, then keep using the old single-symbol `Renamer` loop. This makes inventory a pure refactor before the new planner is introduced.

- [ ] **Step 5: Run focused tests**

Run: `cargo test rename::inventory rename::walker -- --nocapture`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/rename/inventory.rs src/rename/walker.rs src/rename/mod.rs
git commit -m "refactor: collect rename symbol inventory"
```

## Task 3: Add Deterministic Rename Planner Rules

**Files:**
- Create: `src/rename/rules.rs`
- Modify: `src/rename/plan.rs`
- Modify: `src/rename/mod.rs`

- [ ] **Step 1: Write failing planner tests**

Add tests in `src/rename/rules.rs`:

```rust
#[test]
fn planner_resolves_fibonacci_sample_without_llm_for_core_symbols() {
    let source = "function a(e){var t=[0,1];if(e<=2)return t.slice(0,e);while(t.length<e){var n=t.length;t.push(t[n-1]+t[n-2])}return t}function b(e){var t=0;for(var n=0;n<e.length;n++){t+=e[n]}return t}var c=a(10);var d=b(c);";
    let inventory = crate::rename::inventory::build_symbol_inventory(source, 350).unwrap();
    let plan = plan_deterministic_renames(&inventory);
    assert_eq!(plan.needs_llm_count(), 0, "plan: {plan:#?}");
    assert!(plan.items.iter().any(|i| i.original_name == "a" && i.resolved_name() == Some("generateFibonacciSequence")));
    assert!(plan.items.iter().any(|i| i.original_name == "b" && i.resolved_name() == Some("sumValues")));
}
```

- [ ] **Step 2: Run failing test**

Run: `cargo test rename::rules -- --nocapture`

Expected: FAIL because rules are not implemented.

- [ ] **Step 3: Implement initial high-confidence rules**

Implement a conservative first slice:

- Detect Fibonacci-like sequence builders from `[0,1]`, `push(prev + prevPrev)`, and return of the array.
- Detect sum functions from `var x=0`, loop over `.length`, `x += collection[i]`, return accumulator.
- Detect loop index variables and keep or resolve them as `i`.
- Propagate call results from resolved function names.
- Mark remaining unresolved entries as `NeedsLlm` with compact evidence.

Add this helper to `RenamePlanItem`:

```rust
impl RenamePlanItem {
    pub fn resolved_name(&self) -> Option<&str> {
        match &self.state {
            PlanItemState::Resolved { name, .. } => Some(name.as_str()),
            PlanItemState::Keep { .. } => Some(self.original_name.as_str()),
            _ => None,
        }
    }
}
```

- [ ] **Step 4: Run planner tests**

Run: `cargo test rename::rules -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/rename/rules.rs src/rename/plan.rs src/rename/mod.rs
git commit -m "feat: add deterministic rename planner"
```

## Task 4: Add Global LLM Batching and Validation

**Files:**
- Create: `src/llm/batch.rs`
- Modify: `src/llm/mod.rs`
- Modify: `src/rename/plan.rs`

- [ ] **Step 1: Write failing batch tests**

Add tests in `src/llm/batch.rs`:

```rust
#[test]
fn batcher_packs_unresolved_items_globally() {
    let plan = RenamePlan {
        items: vec![
            unresolved("s1", "a", "function evidence"),
            unresolved("s2", "b", "variable evidence"),
        ],
    };
    let jobs = build_llm_batches(&plan, BatchLimits { max_symbols: 10, token_budget: 1000 });
    assert_eq!(jobs.len(), 1);
    assert_eq!(jobs[0].items.len(), 2);
}

#[test]
fn validation_splits_invalid_items() {
    let job = LlmBatchJob::for_test(&["s1", "s2"]);
    let response = serde_json::json!({"renames":[{"symbol_key":"s1","name":"goodName","confidence":90},{"symbol_key":"s2","name":"123bad","confidence":90}]});
    let validated = validate_batch_response(&job, &response);
    assert_eq!(validated.accepted.len(), 1);
    assert_eq!(validated.rejected.len(), 1);
}
```

- [ ] **Step 2: Run failing tests**

Run: `cargo test llm::batch -- --nocapture`

Expected: FAIL because batch module does not exist.

- [ ] **Step 3: Implement batch models and validation**

Define `BatchLimits`, `LlmBatchJob`, `ValidatedBatch`, and `validate_batch_response`. Use `safe_name::to_identifier` and reserved-word checks during validation. Keep token budgeting simple initially with `chars / 4` estimation and a conservative schema overhead constant.

- [ ] **Step 4: Run tests**

Run: `cargo test llm::batch -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/llm/batch.rs src/llm/mod.rs src/rename/plan.rs
git commit -m "feat: batch unresolved rename prompts"
```

## Task 5: Add Durable LLM Job Runner

**Files:**
- Create: `src/llm/jobs.rs`
- Modify: `src/llm/mod.rs`
- Modify: `src/rename/state.rs`

- [ ] **Step 1: Write failing job runner tests**

Add tests in `src/llm/jobs.rs` using `llm::test_dsl::ScriptedStrategy`:

```rust
#[tokio::test]
async fn transient_failure_retries_and_succeeds() {
    let strategy = crate::llm::test_dsl::script("scripted", vec![
        crate::llm::test_dsl::ScriptedResponse::Transient("offline".to_string()),
        crate::llm::test_dsl::ScriptedResponse::Ok(serde_json::json!({"renames":[{"symbol_key":"s1","name":"goodName","confidence":90}]})),
    ]);
    let runner = JobRunner::for_test(strategy, RetryPolicy { max_attempts: 2, backoff_ms: vec![0, 0] });
    let result = runner.run_job(LlmBatchJob::for_test(&["s1"])).await.unwrap();
    assert_eq!(result.accepted.len(), 1);
}

#[tokio::test]
async fn exhausted_job_returns_pause() {
    let strategy = crate::llm::test_dsl::transient("scripted", "offline");
    let runner = JobRunner::for_test(strategy, RetryPolicy { max_attempts: 1, backoff_ms: vec![0] });
    let err = runner.run_job(LlmBatchJob::for_test(&["s1"])).await.unwrap_err();
    assert!(err.to_string().contains("paused"));
}
```

- [ ] **Step 2: Run failing tests**

Run: `cargo test llm::jobs -- --nocapture`

Expected: FAIL because jobs module does not exist.

- [ ] **Step 3: Implement job runner**

Implement `JobRunner` over `Arc<dyn JsonStrategy>`. It should:

- save or expose attempt count per item,
- retry transient errors,
- validate successful responses through `llm::batch`,
- return accepted/rejected results,
- return a pause error after max attempts.

Do not add concurrency in this task.

- [ ] **Step 4: Run tests**

Run: `cargo test llm::jobs -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/llm/jobs.rs src/llm/mod.rs src/rename/state.rs
git commit -m "feat: add durable llm job runner"
```

## Task 6: Wire Plan-First Pipeline Into CLI With Resume

**Files:**
- Modify: `src/cli/preset.rs`
- Modify: `src/cli/anthropic.rs`
- Modify: `src/cli/gemini.rs`
- Modify: `src/cli/ollama.rs`
- Modify: `src/cli/openai.rs`
- Modify: `src/cli/openrouter.rs`
- Modify: `src/main.rs`
- Modify: `tests/cli_smoke.rs`

- [ ] **Step 1: Write failing CLI tests**

Add tests to `tests/cli_smoke.rs`:

```rust
#[test]
fn ollama_offline_sample_uses_plan_first_without_llm_for_fibonacci() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.js");
    let output = dir.path().join("out.js");
    std::fs::write(&input, "function a(e){var t=[0,1];if(e<=2)return t.slice(0,e);while(t.length<e){var n=t.length;t.push(t[n-1]+t[n-2])}return t}function b(e){var t=0;for(var n=0;n<e.length;n++){t+=e[n]}return t}var c=a(10);var d=b(c);console.log(c,d);").unwrap();

    let mut cmd = assert_cmd::Command::cargo_bin("humanify").unwrap();
    cmd.args(["ollama", input.to_str().unwrap(), "-o", output.to_str().unwrap(), "--dry-run-no-llm"]);
    cmd.assert().success();
    let renamed = std::fs::read_to_string(output).unwrap();
    assert!(renamed.contains("generateFibonacciSequence"), "{renamed}");
    assert!(renamed.contains("sumValues"), "{renamed}");
}
```

Add a second test for state option parsing once `--state-file` is wired:

```rust
#[test]
fn resume_rejects_missing_state_file() {
    let mut cmd = assert_cmd::Command::cargo_bin("humanify").unwrap();
    cmd.args(["resume", "missing-state.json"]);
    cmd.assert().failure();
}
```

- [ ] **Step 2: Run failing CLI tests**

Run: `cargo test --test cli_smoke -- --nocapture`

Expected: FAIL because `--dry-run-no-llm`, resume command, and new pipeline are not wired.

- [ ] **Step 3: Add CLI fields**

Add to `SubArgs` in `src/main.rs`:

```rust
#[arg(long)]
resume: bool,
#[arg(long)]
state_file: Option<PathBuf>,
#[arg(long, default_value_t = 5)]
max_llm_attempts: u32,
#[arg(long)]
llm_batch_token_budget: Option<usize>,
#[arg(long)]
llm_batch_max_symbols: Option<usize>,
#[arg(long, hide = true)]
dry_run_no_llm: bool,
```

Add `Commands::Resume { state_file: PathBuf }` and route it to a new `preset::resume_from_state` helper.

Pass these fields through all provider `Args` and `PresetArgs` structs.

- [ ] **Step 4: Implement plan-first path in `run_preset`**

Replace the blocking call to `rename_all_identifiers_with_progress` with:

1. read source,
2. compute input hash,
3. load matching state when `--resume` is set,
4. build inventory when no usable state exists,
5. run deterministic planner,
6. if unresolved exists and `--dry-run-no-llm` is set, pause with non-zero exit,
7. otherwise batch unresolved and run durable jobs,
8. apply completed plan to semantic scoping,
9. write output.

Log phase events from the spec: `inventory_start`, `planner_finish`, `llm_job_start`, `state_saved`, `paused`, and `apply_finish`.

- [ ] **Step 5: Run CLI tests**

Run: `cargo test --test cli_smoke -- --nocapture`

Expected: PASS.

- [ ] **Step 6: Run full tests**

Run: `cargo test`

Expected: PASS. If provider e2e tests require external services and are ignored by default, confirm only default tests ran.

- [ ] **Step 7: Commit**

```powershell
git add src/main.rs src/cli src/rename src/llm tests/cli_smoke.rs
git commit -m "feat: wire plan-first rename pipeline"
```

## Task 7: End-to-End Verification Against Provided Sample

**Files:**
- Modify only if verification exposes bugs in files touched above.

- [ ] **Step 1: Build release binary**

Run: `cargo build --release`

Expected: PASS.

- [ ] **Step 2: Run deterministic dry run on provided sample**

Run:

```powershell
.\target\release\humanify.exe ollama .\target\release\deobfuscated.js -o .\target\release\ollama\readable-qwen3.5-4b.js -m qwen3.5:4b -v --context-size 350 --json-mode tool-call-and-prompt --enable-llm-log --dry-run-no-llm
```

Expected: PASS if deterministic rules resolve all sample symbols; otherwise PAUSED with a state file and clear unresolved count. For the sample in the spec, target behavior is PASS with 0 LLM requests.

- [ ] **Step 3: Inspect logs**

Check `target/release/humanify.log` and the LLM JSONL file. Expected:

- phase logs exist,
- no `progress_start` per-symbol LLM loop,
- LLM JSONL has 0 new lines for deterministic sample, or only batched job lines for unresolved cases.

- [ ] **Step 4: Run normal command without dry-run flag**

Run the user's original command:

```powershell
.\target\release\humanify.exe ollama .\target\release\deobfuscated.js -o .\target\release\ollama\readable-qwen3.5-4b.js -m qwen3.5:4b -v --context-size 350 --json-mode tool-call-and-prompt --enable-llm-log
```

Expected: PASS. Runtime should be materially lower than the previous 95s sample when deterministic rules resolve the file.

- [ ] **Step 5: Commit fixes if needed**

If verification required code changes:

```powershell
git status --short
git add Cargo.toml Cargo.lock src tests docs/superpowers/plans/2026-06-10-llm-efficient-rename-architecture.md
git commit -m "fix: verify plan-first rename flow"
```

If no changes were needed, do not create an empty commit.

## Final Verification

Run these before declaring the branch ready:

```powershell
cargo fmt --check
cargo test
cargo build --release
```

Then report:

- final branch name,
- commits created,
- whether sample run used 0, 1, or more LLM requests,
- state/resume behavior observed,
- any e2e tests skipped because external LLM services were unavailable.

# LLM-Efficient Rename Architecture Design

## Purpose

The current rename pipeline calls the LLM once per semantic symbol. A small 232-byte input file with 10 symbols produced 10 serialized LLM requests and took about 95 seconds with `qwen3.5:4b`. This design changes the architecture so LLM calls become the fallback for unresolved naming decisions, not the default mechanism for every binding.

The primary optimization is to reduce the number of LLM requests and LLM tokens consumed. Concurrency is not a primary optimization because it does not reduce LLM work and can hit provider concurrency limits.

## Goals

- Replace per-symbol LLM calls with a plan-first rename pipeline.
- Resolve common identifier names locally through deterministic analysis passes.
- Merge all remaining LLM work into the fewest provider-safe requests, without using function or scope as the fixed batching boundary.
- Retry every failed LLM item until it succeeds or exceeds the configured retry limit.
- Pause, save state, and allow later resume when the LLM service is down, the process crashes, the machine loses power, or validation repeatedly fails.
- Preserve existing safe identifier normalization, reserved-word handling, and scope collision protection.
- Keep the old behavior available during migration as a compatibility baseline or test fixture.

## Non-Goals

- Optimizing elapsed time by increasing request concurrency.
- Accepting partial rename failures silently.
- Requiring the LLM to process an entire file in a single request regardless of context or provider limits.
- Rewriting the JavaScript parser, semantic analysis, or code generation layers.

## Current Architecture

Current flow:

```text
parse source
-> build semantic model
-> collect symbols sorted by binding scope size
-> for each symbol:
   -> compute surrounding code
   -> call Renamer::rename(original, surrounding)
   -> normalize and resolve conflicts
   -> mutate semantic scoping
-> codegen
```

The core limitation is that `Renamer::rename` is a synchronous single-symbol API. The walker mutates semantic scoping while iterating, so the system cannot first inspect the whole file, infer relationships, reuse decisions, or consolidate LLM work.

## Proposed Architecture

New flow:

```text
parse source
-> build semantic model
-> build SymbolInventory
-> run deterministic RenamePlanner passes
-> build unresolved LLM work queue
-> merge unresolved work into provider-safe LLM jobs
-> execute durable LLM jobs with retries and checkpointing
-> validate and merge LLM results into RenamePlan
-> apply final plan to semantic scoping
-> codegen
```

The important shift is from online mutation to a durable planning pass. The system first builds a complete rename plan, then applies it once the plan is complete.

## Module Boundaries

### `src/rename/inventory.rs`

Builds a file-level symbol inventory from the parser and semantic model.

Each inventory entry should include:

- Stable symbol key for persisted state.
- Original name.
- Symbol kind, such as function, parameter, variable, catch parameter, import, or class.
- Declaration span and declaration snippet.
- Scope id and parent scope relationship.
- Reference spans and compact usage snippets.
- Binding ancestor span.
- Initializer, callee, return use, member access, loop use, and assignment facts when available.
- Rename eligibility.

The persisted symbol key cannot rely only on in-memory `SymbolId`, because resumed runs must reconstruct the same identity. Use a deterministic key derived from input hash, declaration span, original name, symbol kind, and scope path.

### `src/rename/planner.rs`

Owns `RenamePlan` and the deterministic analysis passes.

A plan item can be:

```text
Resolved { name, source, confidence }
Keep { reason }
NeedsLlm { evidence, constraints, attempts }
Failed { reason }
```

Planner passes run repeatedly until no new local conclusions are produced. Later passes can use earlier names for propagation.

### `src/rename/rules/`

Contains local naming rules. Initial rules should target high-confidence patterns:

- Loop indexes: `for (var n = 0; n < items.length; n++)` can become `i` or stay compact without LLM.
- Numeric accumulators: `var t = 0; t += items[i]` can become `sum` or `total`.
- Collection builders: arrays initialized and pushed to can become `items`, `values`, or a domain-specific sequence name if the expression pattern is known.
- Function body shape: a function returning a built collection, summing values, mapping, filtering, or wrapping a call can get a local candidate.
- Call-result propagation: `var c = generateFibSequence(10)` can become a result name derived from the callee.
- Return propagation: if a function returns a local variable with a strong name, the function name can inherit the same concept with a verb form.
- Existing meaningful names: skip LLM for already descriptive identifiers.
- Trivial temporaries: keep or normalize conventional local names when descriptive renaming would add noise.

Rules must produce confidence and source metadata. Low-confidence candidates remain unresolved for LLM.

### `src/llm/batch.rs`

Builds LLM jobs from all unresolved plan items.

Batching is global across the file. It is not fixed to functions or scopes. The batcher should pack as many unresolved symbols as possible into each job while respecting:

- Provider token limits.
- Configured maximum symbols per request.
- Schema size.
- Evidence size.
- Model-specific stability limits.

Each LLM candidate should send compact evidence, not raw full source by default:

- Stable symbol key.
- Original name.
- Symbol kind.
- Declaration snippet.
- Important references.
- Known local facts.
- Already resolved neighboring names.
- Unavailable names in the target scope.
- Naming constraints.

The expected LLM response is an array keyed by symbol key:

```json
{
  "renames": [
    {
      "symbol_key": "...",
      "name": "fibonacciSequence",
      "confidence": 0.87
    }
  ]
}
```

### `src/llm/jobs.rs`

Executes LLM jobs with durable state. A job can be:

```text
pending
in_progress
succeeded
retrying
exhausted
paused
```

Job execution requirements:

- Every LLM request is checkpointed before it is sent.
- A successful response is checkpointed before continuing to the next job.
- If a batch partially succeeds, successful items are persisted and failed items are requeued into smaller jobs.
- Network failures retry the same job.
- Invalid JSON retries the same job and stores the raw response for diagnostics.
- Missing, unknown, duplicate, or invalid symbol results retry only the affected symbols where possible.
- Exceeding the retry limit pauses the run instead of silently keeping original names.

### `src/rename/state.rs`

Persists and restores execution state.

Default state location:

```text
target/release/.humanify-state/<input-file-name>.<input-hash>.json
```

State writes must be atomic: write to a temporary file in the same directory, flush, then rename over the previous state file.

State should include:

- State format version.
- Input path and input hash.
- Output path.
- Provider, model, JSON mode, context size, and relevant command options.
- Current phase.
- Symbol inventory.
- Rename plan.
- LLM jobs and attempts.
- Completed LLM results.
- Retry policy.
- Last error and raw failed response when available.

Resume must reject state when the input hash or incompatible command options do not match.

## Resume and Pause Semantics

The CLI should support both explicit and automatic resume flows.

Explicit resume:

```powershell
.\humanify.exe resume .\target\release\.humanify-state\deobfuscated.js.<hash>.json
```

Optional automatic resume:

```powershell
.\humanify.exe ollama .\deobfuscated.js -o .\ollama\readable.js --resume ...
```

When a run is paused because retries are exhausted, the program should exit with a non-zero status and log the state file path. It should not produce a normal final output as if the rename completed. If a partial output mode is later added, it must use a distinct filename and be clearly marked as partial.

On resume, completed deterministic work and successful LLM items are reused. Pending, retrying, or exhausted jobs can continue according to the configured retry policy.

## Validation and Merge

LLM output is never applied directly. The merge layer validates:

- Every returned symbol key belongs to the requested job.
- Required symbols are present, unless the job is explicitly partial-success capable.
- Names are non-empty valid JavaScript identifiers after normalization.
- Names are not reserved words.
- Names do not collide in the target scope after deterministic conflict resolution.
- Names do not degrade already meaningful names without enough confidence.

Collision handling remains local and deterministic. If a conflict can be safely resolved by prefixing or suffixing, the plan records that adjustment. If the conflict indicates semantic ambiguity, the affected item is requeued or paused after retry exhaustion.

## CLI and Logging

New options:

- `--resume`: continue from matching saved state when available.
- `--state-file <path>`: override the default state path.
- `--max-llm-attempts <n>`: configure retry exhaustion threshold.
- `--llm-batch-token-budget <n>`: cap estimated tokens per LLM job.
- `--llm-batch-max-symbols <n>`: cap symbols per LLM job.

Existing progress logging should move from symbol-by-symbol progress to phase-aware progress:

```text
inventory_start / inventory_finish
planner_start / planner_finish resolved=... needs_llm=...
llm_job_start job=... symbols=... attempt=...
llm_job_finish job=... succeeded=... failed=...
state_saved phase=...
paused reason=... state_file=...
apply_start / apply_finish
```

Raw LLM logging remains useful and should include the job id and symbol keys for correlation.

## Migration Plan

1. Introduce `SymbolInventory` and `RenamePlan` while preserving the existing final behavior.
2. Change the walker so it can collect inventory first and apply a completed plan later.
3. Add deterministic rules for the common patterns seen in the sample file: Fibonacci-like sequence construction, sum accumulator, loop index, call-result propagation, and simple function naming.
4. Replace the single-symbol `Renamer` execution path with a batch-oriented LLM job API for unresolved items.
5. Add durable state saving, retry policy, pause behavior, and resume command support.
6. Keep the old single-symbol path behind a compatibility option or tests until the new pipeline is verified.

## Test Strategy

Unit tests:

- Inventory creates stable keys for symbols across parse runs.
- Deterministic rules resolve loop indexes, sum accumulators, collection builders, and call-result variables.
- Planner reaches a fixed point when propagation rules depend on earlier decisions.
- Batcher packs unresolved symbols globally and splits by configured token or symbol limits.
- LLM validation rejects missing, unknown, duplicate, illegal, and conflicting names.
- State writes are atomic and can be read back.
- Resume rejects mismatched input hash and incompatible options.

Integration tests:

- The sample Fibonacci file requires 0 or 1 LLM requests instead of 10.
- A simulated network failure retries without losing prior successful results.
- A partially invalid batch response persists valid results and requeues invalid items only.
- A simulated crash after a successful LLM job does not repeat that job after resume.
- Exceeding max attempts pauses the run and records a resumable state file.
- Final code generation preserves behavior and applies collision-safe names.

## Initial Defaults

These defaults make the first implementation deterministic while leaving room for later tuning.

- Stable symbol keys use `sha256(input_hash, declaration_start, declaration_end, original_name, symbol_kind, scope_path)`.
- Default retry limit is 5 attempts per LLM job item.
- Default retry backoff is 1s, 3s, 10s, 30s, and 120s.
- Default LLM batch token budget is conservative and provider-specific. For Ollama-compatible local models, start with 60 percent of the configured context window after subtracting prompt/schema overhead.
- Automatic resume is opt-in through `--resume`; without it, a matching state file is logged but not used.
- Exhausted jobs preserve attempt counts on normal resume. A future explicit `--reset-llm-attempts` option may reset them, but the initial implementation should not reset attempts implicitly.

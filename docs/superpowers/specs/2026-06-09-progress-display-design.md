# Default Progress Display Design

## Context

`humanify` is a Rust CLI that reads one JavaScript input, renames identifiers through an LLM-backed `Renamer`, and writes the final JavaScript to stdout or an output file. The current CLI keeps stdout reserved for transformed code and uses stderr for diagnostics and verbose logging.

The rename loop in `src/rename/walker.rs` already collects every symbol before processing them. That is the right place to compute accurate progress because both the total number of identifiers and each current identifier are available there.

## Goal

Show progress by default while identifiers are being renamed, without contaminating stdout or changing existing library behavior for callers that do not opt in to progress events.

## User Experience

During normal CLI runs, `humanify` prints progress lines to stderr as each identifier is processed. A representative line is:

```text
humanify: renaming 3/42: foo
```

The final renamed JavaScript remains the only stdout content when no `-o` output file is provided. `-v/--verbose` keeps its current meaning and still only adds the locked strategy diagnostic after the run.

Empty input returns without progress output. Inputs with no renameable identifiers also remain quiet to avoid unnecessary stderr noise.

## Architecture

Add a lightweight progress event type in the rename module with these fields:

- `current`: one-based index of the identifier being processed.
- `total`: total number of identifiers scheduled for processing.
- `original_name`: the identifier name before renaming.

Keep `rename_all_identifiers(source, renamer, context_size)` as the stable no-progress API. Add a sibling function that accepts a callback, such as `rename_all_identifiers_with_progress(source, renamer, context_size, on_progress)`. The existing function delegates to the callback variant with a no-op callback.

`src/cli/preset.rs` calls the callback variant inside the existing `spawn_blocking` closure and prints each event to stderr. This keeps provider-specific modules unchanged because all subcommands already route through `run_preset`.

## Error Handling

Progress reporting is best-effort stderr output. It should not alter parse errors, LLM fallback behavior, collision handling, or final output writing. If a rename falls back to the original name, progress has still advanced because the identifier was processed.

## Testing

Add focused unit coverage around the rename walker to confirm progress events are emitted in order with correct totals and original names. Existing tests for `rename_all_identifiers` should continue to pass unchanged, proving the no-progress API remains compatible.

Add or adjust a CLI smoke test only if the existing smoke harness can assert stderr without requiring real API success. The key CLI-level assertion is that progress goes to stderr, not stdout.

## Scope

This design does not add a CLI flag to disable progress, does not introduce a TTY-only progress bar, and does not change provider behavior. It intentionally uses plain line-oriented stderr output so it works consistently in terminals, CI, and redirected runs.

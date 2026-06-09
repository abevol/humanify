# Nested Context Batch Schema Design

## Goal

Reduce LLM prompt tokens for batched rename jobs by removing repeated per-item evidence and long stable symbol keys from the prompt.

## Problem

The current LLM batch request stores full evidence on every item. In the shopping-cart sample, 9 items contained only 2 unique evidence strings, and 8 items repeated the same 331-character context. The request spent 2709 evidence characters where 392 unique context characters were sufficient. The prompt also exposes 64-character `symbol_key` values that the model does not need to understand.

## Design

LLM-facing requests use nested contexts:

```json
{
  "id": "llm-job-0",
  "contexts": [
    {
      "code": "function calculateShoppingCart(a,t,e){...}",
      "items": [
        { "id": 0, "original": "calculateShoppingCart" },
        { "id": 1, "original": "a" }
      ]
    }
  ]
}
```

Each item id is a job-local integer. Contexts do not need prompt-visible ids because items are nested inside the code they refer to. The internal `LlmBatchJob` still keeps `StableSymbolKey` for durable plan/state/resume identity and maps prompt item ids back to stable keys during validation.

Responses use the short numeric id:

```json
{
  "renames": [
    { "id": 1, "name": "cartItems", "confidence": 95 }
  ]
}
```

Validation rejects missing, duplicate, out-of-range, or invalid identifier responses and maps accepted/rejected items back to `StableSymbolKey`.

## Token Budgeting

Batch token estimation counts each unique context once plus compact item metadata. Packing still respects `max_symbols` and `token_budget`, but no longer overestimates or transmits repeated evidence.

## Scope

This changes only the LLM batch request/response shape and validation. Durable state continues to store stable keys. The CLI retry, pause, resume, and apply flow remains unchanged.

## Verification

Add batch unit tests proving repeated evidence becomes one context with multiple nested items and validation maps numeric ids back to stable keys. Update job-runner tests to use numeric response ids. Run `cargo fmt --check`, `cargo test`, `cargo build --release`, and the real Ollama command with LLM logging to confirm prompt shape and reduced token usage.

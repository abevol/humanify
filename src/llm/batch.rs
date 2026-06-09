use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::rename::plan::{PlanItemState, RenamePlan, StableSymbolKey};
use crate::rename::safe_name;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchLimits {
    pub max_symbols: usize,
    pub token_budget: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmBatchItem {
    pub symbol_key: String,
    pub original_name: String,
    pub evidence: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmBatchJob {
    pub id: String,
    pub items: Vec<LlmBatchItem>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceptedRename {
    pub symbol_key: StableSymbolKey,
    pub name: String,
    pub confidence: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RejectedRename {
    pub symbol_key: StableSymbolKey,
    pub reason: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ValidatedBatch {
    pub accepted: Vec<AcceptedRename>,
    pub rejected: Vec<RejectedRename>,
}

pub fn build_llm_batches(plan: &RenamePlan, limits: BatchLimits) -> Vec<LlmBatchJob> {
    let max_symbols = limits.max_symbols.max(1);
    let token_budget = limits.token_budget.max(1);
    let mut jobs = Vec::new();
    let mut current = Vec::new();
    let mut current_tokens = 0usize;

    for item in &plan.items {
        let PlanItemState::NeedsLlm { evidence, .. } = &item.state else {
            continue;
        };
        let batch_item = LlmBatchItem {
            symbol_key: item.key.0.clone(),
            original_name: item.original_name.clone(),
            evidence: evidence.clone(),
        };
        let estimated = estimate_tokens(&batch_item);
        let would_exceed_symbols = current.len() >= max_symbols;
        let would_exceed_tokens = !current.is_empty() && current_tokens + estimated > token_budget;
        if would_exceed_symbols || would_exceed_tokens {
            jobs.push(job_from_items(jobs.len(), std::mem::take(&mut current)));
            current_tokens = 0;
        }
        current_tokens += estimated;
        current.push(batch_item);
    }

    if !current.is_empty() {
        jobs.push(job_from_items(jobs.len(), current));
    }

    jobs
}

pub fn validate_batch_response(job: &LlmBatchJob, response: &Value) -> ValidatedBatch {
    let requested = job
        .items
        .iter()
        .map(|item| item.symbol_key.as_str())
        .collect::<HashSet<_>>();
    let mut seen = HashSet::new();
    let mut accepted = Vec::new();
    let mut rejected = Vec::new();

    let Some(renames) = response.get("renames").and_then(|value| value.as_array()) else {
        return ValidatedBatch {
            accepted,
            rejected: job
                .items
                .iter()
                .map(|item| RejectedRename {
                    symbol_key: StableSymbolKey(item.symbol_key.clone()),
                    reason: "response missing renames array".to_string(),
                })
                .collect(),
        };
    };

    for rename in renames {
        let Some(symbol_key) = rename.get("symbol_key").and_then(|value| value.as_str()) else {
            continue;
        };
        if !requested.contains(symbol_key) {
            continue;
        }
        if !seen.insert(symbol_key.to_string()) {
            rejected.push(RejectedRename {
                symbol_key: StableSymbolKey(symbol_key.to_string()),
                reason: "duplicate symbol result".to_string(),
            });
            continue;
        }
        let Some(raw_name) = rename.get("name").and_then(|value| value.as_str()) else {
            rejected.push(RejectedRename {
                symbol_key: StableSymbolKey(symbol_key.to_string()),
                reason: "missing name".to_string(),
            });
            continue;
        };
        let safe = safe_name::to_identifier(raw_name.trim());
        if safe != raw_name.trim() || safe_name::is_reserved_word(&safe) {
            rejected.push(RejectedRename {
                symbol_key: StableSymbolKey(symbol_key.to_string()),
                reason: "invalid identifier".to_string(),
            });
            continue;
        }
        let confidence = rename
            .get("confidence")
            .and_then(|value| value.as_u64())
            .unwrap_or(0)
            .min(100) as u8;
        accepted.push(AcceptedRename {
            symbol_key: StableSymbolKey(symbol_key.to_string()),
            name: safe,
            confidence,
        });
    }

    let accepted_or_rejected = accepted
        .iter()
        .map(|item| item.symbol_key.0.clone())
        .chain(rejected.iter().map(|item| item.symbol_key.0.clone()))
        .collect::<HashSet<_>>();
    let by_key = job
        .items
        .iter()
        .map(|item| (item.symbol_key.as_str(), item))
        .collect::<HashMap<_, _>>();

    for missing_key in requested {
        if accepted_or_rejected.contains(missing_key) {
            continue;
        }
        if by_key.contains_key(missing_key) {
            rejected.push(RejectedRename {
                symbol_key: StableSymbolKey((*missing_key).to_string()),
                reason: "missing symbol result".to_string(),
            });
        }
    }

    ValidatedBatch { accepted, rejected }
}

fn job_from_items(index: usize, items: Vec<LlmBatchItem>) -> LlmBatchJob {
    LlmBatchJob {
        id: format!("llm-job-{index}"),
        items,
    }
}

fn estimate_tokens(item: &LlmBatchItem) -> usize {
    16 + (item.symbol_key.len() + item.original_name.len() + item.evidence.len()) / 4
}

impl LlmBatchJob {
    pub fn for_test(keys: &[&str]) -> Self {
        Self {
            id: "test-job".to_string(),
            items: keys
                .iter()
                .map(|key| LlmBatchItem {
                    symbol_key: (*key).to_string(),
                    original_name: "x".to_string(),
                    evidence: "evidence".to_string(),
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rename::plan::{RenamePlanItem, StableSymbolKey};

    fn unresolved(key: &str, original: &str, evidence: &str) -> RenamePlanItem {
        RenamePlanItem {
            key: StableSymbolKey(key.to_string()),
            original_name: original.to_string(),
            scope_path: "0".to_string(),
            state: PlanItemState::NeedsLlm {
                evidence: evidence.to_string(),
                attempts: 0,
            },
        }
    }

    #[test]
    fn batcher_packs_unresolved_items_globally() {
        let plan = RenamePlan {
            items: vec![
                unresolved("s1", "a", "function evidence"),
                unresolved("s2", "b", "variable evidence"),
            ],
        };
        let jobs = build_llm_batches(
            &plan,
            BatchLimits {
                max_symbols: 10,
                token_budget: 1000,
            },
        );
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
}

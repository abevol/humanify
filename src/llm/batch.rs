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
    pub prompt_id: u32,
    pub symbol_key: String,
    pub original_name: String,
    pub evidence: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmBatchJob {
    pub id: String,
    pub items: Vec<LlmBatchItem>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PromptBatchJob {
    pub id: String,
    pub contexts: Vec<PromptContext>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PromptContext {
    pub code: String,
    pub items: Vec<PromptBatchItem>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PromptBatchItem {
    pub id: u32,
    pub original: String,
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
    let mut next_prompt_id = 0u32;

    for item in &plan.items {
        let PlanItemState::NeedsLlm { evidence, .. } = &item.state else {
            continue;
        };
        let batch_item = LlmBatchItem {
            prompt_id: next_prompt_id,
            symbol_key: item.key.0.clone(),
            original_name: item.original_name.clone(),
            evidence: evidence.clone(),
        };
        next_prompt_id = next_prompt_id.saturating_add(1);

        let estimated = estimate_incremental_tokens(&current, &batch_item);
        let would_exceed_symbols = current.len() >= max_symbols;
        let would_exceed_tokens = !current.is_empty() && current_tokens + estimated > token_budget;
        if would_exceed_symbols || would_exceed_tokens {
            jobs.push(job_from_items(jobs.len(), std::mem::take(&mut current)));
            current_tokens = 0;
        }
        current_tokens += estimate_incremental_tokens(&current, &batch_item);
        current.push(batch_item);
    }

    if !current.is_empty() {
        jobs.push(job_from_items(jobs.len(), current));
    }

    jobs
}

pub fn validate_batch_response(job: &LlmBatchJob, response: &Value) -> ValidatedBatch {
    let by_prompt_id = job
        .items
        .iter()
        .map(|item| (item.prompt_id, item))
        .collect::<HashMap<_, _>>();
    let requested = by_prompt_id.keys().copied().collect::<HashSet<_>>();
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
        let Some(prompt_id) = rename
            .get("id")
            .and_then(|value| value.as_u64())
            .and_then(|value| u32::try_from(value).ok())
        else {
            continue;
        };
        let Some(item) = by_prompt_id.get(&prompt_id) else {
            continue;
        };
        if !seen.insert(prompt_id) {
            rejected.push(RejectedRename {
                symbol_key: StableSymbolKey(item.symbol_key.clone()),
                reason: "duplicate symbol result".to_string(),
            });
            continue;
        }
        let Some(raw_name) = rename.get("name").and_then(|value| value.as_str()) else {
            rejected.push(RejectedRename {
                symbol_key: StableSymbolKey(item.symbol_key.clone()),
                reason: "missing name".to_string(),
            });
            continue;
        };
        let safe = safe_name::to_identifier(raw_name.trim());
        if safe != raw_name.trim() || safe_name::is_reserved_word(&safe) {
            rejected.push(RejectedRename {
                symbol_key: StableSymbolKey(item.symbol_key.clone()),
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
            symbol_key: StableSymbolKey(item.symbol_key.clone()),
            name: safe,
            confidence,
        });
    }

    let accepted_or_rejected = accepted
        .iter()
        .map(|item| item.symbol_key.0.clone())
        .chain(rejected.iter().map(|item| item.symbol_key.0.clone()))
        .collect::<HashSet<_>>();

    for prompt_id in requested {
        let item = by_prompt_id[&prompt_id];
        if accepted_or_rejected.contains(&item.symbol_key) {
            continue;
        }
        rejected.push(RejectedRename {
            symbol_key: StableSymbolKey(item.symbol_key.clone()),
            reason: "missing symbol result".to_string(),
        });
    }

    ValidatedBatch { accepted, rejected }
}

fn job_from_items(index: usize, items: Vec<LlmBatchItem>) -> LlmBatchJob {
    LlmBatchJob {
        id: format!("llm-job-{index}"),
        items,
    }
}

fn estimate_incremental_tokens(current: &[LlmBatchItem], item: &LlmBatchItem) -> usize {
    let context_cost = if current
        .iter()
        .any(|current_item| current_item.evidence == item.evidence)
    {
        0
    } else {
        item.evidence.len() / 4
    };
    12 + context_cost + item.original_name.len() / 4
}

impl LlmBatchJob {
    pub fn prompt_payload(&self) -> PromptBatchJob {
        let mut contexts: Vec<PromptContext> = Vec::new();
        let mut by_evidence: HashMap<&str, usize> = HashMap::new();

        for item in &self.items {
            let context_index = match by_evidence.get(item.evidence.as_str()) {
                Some(index) => *index,
                None => {
                    let index = contexts.len();
                    by_evidence.insert(item.evidence.as_str(), index);
                    contexts.push(PromptContext {
                        code: item.evidence.clone(),
                        items: Vec::new(),
                    });
                    index
                }
            };
            contexts[context_index].items.push(PromptBatchItem {
                id: item.prompt_id,
                original: item.original_name.clone(),
            });
        }

        PromptBatchJob {
            id: self.id.clone(),
            contexts,
        }
    }

    pub fn for_test(keys: &[&str]) -> Self {
        Self {
            id: "test-job".to_string(),
            items: keys
                .iter()
                .enumerate()
                .map(|(index, key)| LlmBatchItem {
                    prompt_id: index as u32,
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
    fn prompt_payload_nests_items_under_unique_contexts() {
        let plan = RenamePlan {
            items: vec![
                unresolved("stable-a", "a", "shared code"),
                unresolved("stable-b", "b", "shared code"),
                unresolved("stable-c", "c", "other code"),
            ],
        };
        let jobs = build_llm_batches(
            &plan,
            BatchLimits {
                max_symbols: 10,
                token_budget: 1000,
            },
        );
        let payload = jobs[0].prompt_payload();
        assert_eq!(payload.contexts.len(), 2);
        assert_eq!(payload.contexts[0].code, "shared code");
        assert_eq!(payload.contexts[0].items.len(), 2);
        assert_eq!(payload.contexts[0].items[0].id, 0);
        assert_eq!(payload.contexts[0].items[1].id, 1);
        assert_eq!(payload.contexts[1].items[0].id, 2);
    }

    #[test]
    fn prompt_payload_omits_stable_symbol_keys() {
        let job = LlmBatchJob::for_test(&["0123456789abcdef"]);
        let json = serde_json::to_string(&job.prompt_payload()).unwrap();
        assert!(!json.contains("symbol_key"), "json: {json}");
        assert!(!json.contains("0123456789abcdef"), "json: {json}");
        assert!(json.contains("\"id\":0"), "json: {json}");
    }

    #[test]
    fn validation_maps_numeric_ids_to_stable_keys() {
        let job = LlmBatchJob::for_test(&["s1", "s2"]);
        let response = serde_json::json!({"renames":[{"id":0,"name":"goodName","confidence":90},{"id":1,"name":"123bad","confidence":90}]});
        let validated = validate_batch_response(&job, &response);
        assert_eq!(validated.accepted.len(), 1);
        assert_eq!(validated.rejected.len(), 1);
        assert_eq!(
            validated.accepted[0].symbol_key,
            StableSymbolKey("s1".to_string())
        );
        assert_eq!(
            validated.rejected[0].symbol_key,
            StableSymbolKey("s2".to_string())
        );
    }
}

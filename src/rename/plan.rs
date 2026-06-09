use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StableSymbolKey(pub String);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PlanItemState {
    Resolved {
        name: String,
        source: String,
        confidence: u8,
    },
    Keep {
        reason: String,
    },
    NeedsLlm {
        evidence: String,
        attempts: u32,
    },
    Failed {
        reason: String,
    },
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

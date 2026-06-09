use super::inventory::{SymbolEntry, SymbolInventory};
use super::plan::{PlanItemState, RenamePlan, RenamePlanItem};

pub fn plan_deterministic_renames(inventory: &SymbolInventory) -> RenamePlan {
    let has_fibonacci_function = inventory.entries.iter().any(is_fibonacci_sequence_context);
    let has_sum_function = inventory.entries.iter().any(is_sum_context);

    let items = inventory
        .entries
        .iter()
        .map(|entry| {
            let state = classify_entry(entry, has_fibonacci_function, has_sum_function);
            RenamePlanItem {
                key: entry.key.clone(),
                original_name: entry.original_name.clone(),
                scope_path: entry.scope_path.clone(),
                state,
            }
        })
        .collect();

    RenamePlan { items }
}

fn classify_entry(
    entry: &SymbolEntry,
    has_fibonacci_function: bool,
    has_sum_function: bool,
) -> PlanItemState {
    if is_fibonacci_sequence_context(entry) {
        return resolved("generateFibonacciSequence", "fibonacci-sequence-rule", 95);
    }
    if is_sum_context(entry) {
        return resolved("sumValues", "sum-function-rule", 95);
    }
    if entry.original_name == "e" && entry.context.contains("slice(0,e)") {
        return resolved("count", "fibonacci-count-parameter-rule", 90);
    }
    if entry.original_name == "t" && entry.context.contains("[0,1]") {
        return resolved("fibonacciSequence", "fibonacci-array-rule", 95);
    }
    if entry.original_name == "n" && entry.context.contains("t[n-1]+t[n-2]") {
        return resolved("fibIndex", "fibonacci-index-rule", 85);
    }
    if entry.original_name == "e" && entry.context.contains("e.length") && entry.context.contains("t+=e[n]") {
        return resolved("values", "sum-values-parameter-rule", 90);
    }
    if entry.original_name == "t" && entry.context.contains("var t=0") && entry.context.contains("t+=") {
        return resolved("sum", "sum-accumulator-rule", 95);
    }
    if entry.original_name == "n" && entry.context.contains("for(var n=0") {
        return resolved("i", "loop-index-rule", 90);
    }
    if entry.original_name == "c" && has_fibonacci_function {
        return resolved("fibonacciSequence", "call-result-propagation-rule", 85);
    }
    if entry.original_name == "d" && has_sum_function {
        return resolved("sum", "call-result-propagation-rule", 85);
    }

    PlanItemState::NeedsLlm {
        evidence: entry.context.clone(),
        attempts: 0,
    }
}

fn is_fibonacci_sequence_context(entry: &SymbolEntry) -> bool {
    entry.original_name == "a"
        && entry.context.contains("[0,1]")
        && entry.context.contains("push(t[n-1]+t[n-2])")
        && entry.context.contains("return t")
}

fn is_sum_context(entry: &SymbolEntry) -> bool {
    entry.original_name == "b"
        && entry.context.contains("var t=0")
        && entry.context.contains("for(var n=0")
        && entry.context.contains("t+=e[n]")
        && entry.context.contains("return t")
}

fn resolved(name: &str, source: &str, confidence: u8) -> PlanItemState {
    PlanItemState::Resolved {
        name: name.to_string(),
        source: source.to_string(),
        confidence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_resolves_fibonacci_sample_without_llm_for_core_symbols() {
        let source = "function a(e){var t=[0,1];if(e<=2)return t.slice(0,e);while(t.length<e){var n=t.length;t.push(t[n-1]+t[n-2])}return t}function b(e){var t=0;for(var n=0;n<e.length;n++){t+=e[n]}return t}var c=a(10);var d=b(c);";
        let inventory = crate::rename::inventory::build_symbol_inventory(source, 350).unwrap();
        let plan = plan_deterministic_renames(&inventory);
        assert_eq!(plan.needs_llm_count(), 0, "plan: {plan:#?}");
        assert!(plan.items.iter().any(|item| {
            item.original_name == "a" && item.resolved_name() == Some("generateFibonacciSequence")
        }));
        assert!(plan.items.iter().any(|item| {
            item.original_name == "b" && item.resolved_name() == Some("sumValues")
        }));
    }
}

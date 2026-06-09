use oxc_allocator::Allocator;
use oxc_parser::Parser;
use oxc_semantic::{SemanticBuilder, SymbolId};
use oxc_span::SourceType;
use sha2::{Digest, Sha256};

use super::plan::StableSymbolKey;
use super::walker::{compute_context_window, find_binding_ancestor_span};
use super::RenameError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolInventory {
    pub entries: Vec<SymbolEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolEntry {
    pub key: StableSymbolKey,
    pub symbol_id: SymbolId,
    pub original_name: String,
    pub symbol_kind: String,
    pub scope_path: String,
    pub declaration_start: u32,
    pub declaration_end: u32,
    pub context: String,
}

pub fn build_symbol_inventory(
    source: &str,
    context_size: usize,
) -> Result<SymbolInventory, RenameError> {
    if source.is_empty() {
        return Ok(SymbolInventory { entries: Vec::new() });
    }

    let allocator = Allocator::default();
    let parse_result = Parser::new(&allocator, source, SourceType::default()).parse();
    if !parse_result.errors.is_empty() {
        let msg = parse_result
            .errors
            .iter()
            .map(|e| e.message.to_string())
            .collect::<Vec<_>>()
            .join("; ");
        return Err(RenameError::Parse(msg));
    }
    let program = parse_result.program;

    let semantic_result = SemanticBuilder::new().build(&program);
    if !semantic_result.errors.is_empty() {
        let msg = semantic_result
            .errors
            .iter()
            .map(|e| e.message.to_string())
            .collect::<Vec<_>>()
            .join("; ");
        return Err(RenameError::Parse(msg));
    }
    let semantic = semantic_result.semantic;
    let scoping = semantic.scoping();
    let nodes = semantic.nodes();
    let input_hash = input_hash(source);

    let mut entries = scoping
        .symbol_ids()
        .map(|sym_id| {
            let decl_node_id = scoping.symbol_declaration(sym_id);
            let original_name = scoping.symbol_name(sym_id).to_string();
            let span = scoping.symbol_span(sym_id);
            let binding_scope = scoping.symbol_scope_id(sym_id);
            let ctx_span = find_binding_ancestor_span(
                nodes,
                scoping,
                decl_node_id,
                &original_name,
                source,
                binding_scope,
            );
            let context = compute_context_window(source, span, ctx_span, context_size);
            let scope_path = format!("{binding_scope:?}");
            let symbol_kind = "binding".to_string();
            let key = StableSymbolKey::new(
                &input_hash,
                span.start,
                span.end,
                &original_name,
                &symbol_kind,
                &scope_path,
            );
            let scope_size = ctx_span.end.saturating_sub(ctx_span.start);

            (
                SymbolEntry {
                    key,
                    symbol_id: sym_id,
                    original_name,
                    symbol_kind,
                    scope_path,
                    declaration_start: span.start,
                    declaration_end: span.end,
                    context,
                },
                scope_size,
                span.start,
            )
        })
        .collect::<Vec<_>>();

    entries.sort_by(|a, b| b.1.cmp(&a.1).then(a.2.cmp(&b.2)));

    Ok(SymbolInventory {
        entries: entries.into_iter().map(|(entry, _, _)| entry).collect(),
    })
}

fn input_hash(source: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(source.as_bytes());
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inventory_collects_fibonacci_sample_symbols() {
        let source = "function a(e){var t=[0,1];while(t.length<e){var n=t.length;t.push(t[n-1]+t[n-2])}return t}var c=a(10);";
        let inventory = build_symbol_inventory(source, 350).unwrap();
        let names = inventory
            .entries
            .iter()
            .map(|entry| entry.original_name.as_str())
            .collect::<Vec<_>>();
        assert!(names.contains(&"a"), "names: {names:?}");
        assert!(names.contains(&"e"), "names: {names:?}");
        assert!(names.contains(&"t"), "names: {names:?}");
        assert!(names.contains(&"n"), "names: {names:?}");
        assert!(names.contains(&"c"), "names: {names:?}");
        assert!(inventory.entries.iter().all(|entry| !entry.key.0.is_empty()));
    }

    #[test]
    fn inventory_keeps_context_windows() {
        let source = "function sumItems(items){var total=0;for(var i=0;i<items.length;i++){total+=items[i]}return total}";
        let inventory = build_symbol_inventory(source, 40).unwrap();
        let total = inventory
            .entries
            .iter()
            .find(|entry| entry.original_name == "total")
            .unwrap();
        assert!(total.context.len() <= source.len());
        assert!(total.context.contains("total"));
    }
}

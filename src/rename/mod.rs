pub mod inventory;
pub mod plan;
pub mod rules;
pub(crate) mod safe_name;
pub mod state;
#[cfg(test)]
pub mod test_dsl;
mod walker;

pub use walker::{
    rename_all_identifiers, rename_all_identifiers_with_progress, RenameProgress,
    RenameProgressPhase,
};

pub trait Renamer {
    /// Returns the new name for the identifier. Returning the same string means "leave it alone".
    fn rename(&mut self, original: &str, surrounding_code: &str) -> String;
}

#[derive(Debug, thiserror::Error)]
pub enum RenameError {
    #[error("failed to parse JavaScript: {0}")]
    Parse(String),
}

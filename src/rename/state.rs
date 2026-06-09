use std::fs;
use std::io;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::plan::{RenamePlan, StableSymbolKey};

pub fn hash_source(source: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(source.as_bytes());
    hex::encode(hasher.finalize())
}

impl StableSymbolKey {
    pub fn new(
        input_hash: &str,
        start: u32,
        end: u32,
        original: &str,
        kind: &str,
        scope_path: &str,
    ) -> Self {
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
        Self {
            max_attempts: 5,
            backoff_ms: vec![1000, 3000, 10000, 30000, 120000],
        }
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
        fs::create_dir_all(parent)
            .with_context(|| format!("create state dir {}", parent.display()))?;
    }

    let tmp = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(state)?;
    fs::write(&tmp, bytes).with_context(|| format!("write temp state {}", tmp.display()))?;
    fs::rename(&tmp, path)
        .or_else(|e| {
            if e.kind() == io::ErrorKind::AlreadyExists {
                fs::remove_file(path)?;
                fs::rename(&tmp, path)
            } else {
                Err(e)
            }
        })
        .with_context(|| format!("rename state {}", path.display()))?;
    Ok(())
}

pub fn load_state(path: &Path) -> Result<RenameState> {
    let bytes = fs::read(path).with_context(|| format!("read state {}", path.display()))?;
    Ok(serde_json::from_slice(&bytes)?)
}

#[cfg(test)]
mod tests {
    use super::*;

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
}

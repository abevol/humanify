use serde::Serialize;
use serde_json::Value;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone)]
pub struct LlmLogger {
    file: Arc<Mutex<File>>,
}

#[derive(Serialize)]
pub struct LlmLogEvent {
    pub timestamp_ms: u128,
    pub url: String,
    pub request: Value,
    pub response: Option<LlmLogResponse>,
    pub error: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmLogResponse {
    Json(Value),
    Text { status: u16, body: String },
}

impl LlmLogger {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            file: Arc::new(Mutex::new(file)),
        })
    }

    pub fn log(&self, event: &LlmLogEvent) -> io::Result<()> {
        let line = serde_json::to_string(event).map_err(io::Error::other)?;
        let mut file = self
            .file
            .lock()
            .map_err(|_| io::Error::other("LLM log file lock poisoned"))?;
        writeln!(file, "{line}")?;
        file.flush()
    }
}

pub fn unix_timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;

    #[test]
    fn appends_one_valid_json_line_per_event() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("llm.jsonl");
        let logger = LlmLogger::open(&path).unwrap();

        logger
            .log(&LlmLogEvent {
                timestamp_ms: 123,
                url: "http://localhost:11434/v1/chat/completions".to_string(),
                request: json!({"model":"qwen3.5:4b"}),
                response: Some(LlmLogResponse::Json(json!({"ok":true}))),
                error: None,
            })
            .unwrap();
        logger
            .log(&LlmLogEvent {
                timestamp_ms: 456,
                url: "http://localhost:11434/v1/chat/completions".to_string(),
                request: json!({"model":"qwen3.5:4b"}),
                response: None,
                error: Some("network timeout".to_string()),
            })
            .unwrap();

        let contents = fs::read_to_string(path).unwrap();
        let lines: Vec<_> = contents.lines().collect();
        assert_eq!(lines.len(), 2);

        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first["timestamp_ms"], 123);
        assert_eq!(first["url"], "http://localhost:11434/v1/chat/completions");
        assert_eq!(first["request"]["model"], "qwen3.5:4b");
        assert_eq!(first["response"]["json"]["ok"], true);
        assert!(first["error"].is_null());

        let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(second["timestamp_ms"], 456);
        assert!(second["response"].is_null());
        assert_eq!(second["error"], "network timeout");
    }

    #[test]
    fn open_fails_for_missing_parent_directory() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing").join("llm.jsonl");
        assert!(LlmLogger::open(path).is_err());
    }
}

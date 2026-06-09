use assert_cmd::prelude::*;
use assert_cmd::Command as AssertCommand;
use std::process::Command;
use tempfile::NamedTempFile;

#[test]
fn llm_log_missing_parent_fails_before_llm_call() {
    let dir = tempfile::tempdir().unwrap();
    let missing_log = dir.path().join("missing").join("llm.jsonl");

    let mut cmd = Command::cargo_bin("humanify").unwrap();
    cmd.arg("ollama")
        .arg("--llm-log-file")
        .arg(&missing_log)
        .arg("fixtures/splitstring.min.js");

    let output = cmd.output().unwrap();
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("humanify: failed to open LLM log"),
        "stderr was: {stderr}"
    );
    assert!(output.stdout.is_empty());
}

#[test]
fn enable_llm_log_uses_default_file_name() {
    let dir = tempfile::tempdir().unwrap();
    let out = NamedTempFile::new_in(dir.path()).unwrap();
    let out_path = out.path().to_owned();
    let expected_log = dir
        .path()
        .join("stdin-llm-gemini-gemini-3.1-flash-lite.jsonl");

    let assert = AssertCommand::cargo_bin("humanify")
        .unwrap()
        .current_dir(dir.path())
        .args([
            "gemini",
            "-",
            "-o",
            out_path.to_str().unwrap(),
            "--base-url",
            "http://127.0.0.1:1",
            "--enable-llm-log",
        ])
        .write_stdin("const x = 1;")
        .assert()
        .success();

    let stderr = String::from_utf8(assert.get_output().stderr.clone()).unwrap();
    assert!(
        stderr.contains("humanify: renaming 1/1: x"),
        "stderr: {stderr}"
    );
    assert!(expected_log.exists(), "missing log file: {expected_log:?}");
}

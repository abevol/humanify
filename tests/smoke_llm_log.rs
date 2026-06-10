use assert_cmd::prelude::*;
use assert_cmd::Command as AssertCommand;
use std::process::Command;
use tempfile::NamedTempFile;

const FIB_SAMPLE: &str = "function a(e){var t=[0,1];if(e<=2)return t.slice(0,e);while(t.length<e){var n=t.length;t.push(t[n-1]+t[n-2])}return t}function b(e){var t=0;for(var n=0;n<e.length;n++){t+=e[n]}return t}var c=a(10);var d=b(c);console.log(c,d);";

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
    let input = dir.path().join("input.js");
    std::fs::write(&input, FIB_SAMPLE).unwrap();
    let expected_log = dir
        .path()
        .join("input.js-llm-gemini-gemini-3.1-flash-lite.jsonl");

    let assert = AssertCommand::cargo_bin("humanify")
        .unwrap()
        .current_dir(dir.path())
        .args([
            "gemini",
            input.to_str().unwrap(),
            "-o",
            out_path.to_str().unwrap(),
            "--base-url",
            "http://127.0.0.1:1",
            "--enable-llm-log",
        ])
        .assert()
        .success();

    let stderr = String::from_utf8(assert.get_output().stderr.clone()).unwrap();
    assert!(
        stderr.contains("INFO planner_finish") && stderr.contains("needs_llm=0"),
        "stderr: {stderr}"
    );
    assert!(expected_log.exists(), "missing log file: {expected_log:?}");
}

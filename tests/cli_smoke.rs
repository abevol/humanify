use assert_cmd::Command;
use tempfile::tempdir;
const FIB_SAMPLE: &str = "function a(e){var t=[0,1];if(e<=2)return t.slice(0,e);while(t.length<e){var n=t.length;t.push(t[n-1]+t[n-2])}return t}function b(e){var t=0;for(var n=0;n<e.length;n++){t+=e[n]}return t}var c=a(10);var d=b(c);console.log(c,d);";

#[test]
fn gemini_offline_unresolved_symbols_pause_and_save_state() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.js");
    let out_path = dir.path().join("out.js");
    let state_path = dir.path().join("state.json");
    std::fs::write(&input, "const x = 1;").unwrap();

    let assert = Command::cargo_bin("humanify")
        .unwrap()
        .args([
            "gemini",
            input.to_str().unwrap(),
            "-o",
            out_path.to_str().unwrap(),
            "--base-url",
            "http://127.0.0.1:1",
            "--state-file",
            state_path.to_str().unwrap(),
            "--max-llm-attempts",
            "1",
        ])
        .assert()
        .failure();

    let stderr = String::from_utf8(assert.get_output().stderr.clone()).unwrap();
    assert!(
        stderr.contains("humanify: paused after LLM job failure"),
        "stderr: {stderr}"
    );

    assert!(!out_path.exists());
    let state = std::fs::read_to_string(state_path).unwrap();
    assert!(state.contains("paused"), "state: {state}");
    assert!(state.contains("NeedsLlm"), "state: {state}");
}

#[test]
fn default_program_log_appends_humanify_log() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.js");
    let out_path = dir.path().join("out.js");
    std::fs::write(&input, FIB_SAMPLE).unwrap();

    let assert = Command::cargo_bin("humanify")
        .unwrap()
        .current_dir(dir.path())
        .args([
            "gemini",
            input.to_str().unwrap(),
            "-o",
            out_path.to_str().unwrap(),
            "--base-url",
            "http://127.0.0.1:1",
        ])
        .assert()
        .success();

    let stderr = String::from_utf8(assert.get_output().stderr.clone()).unwrap();
    assert!(
        stderr.contains("INFO start provider=gemini"),
        "stderr: {stderr}"
    );
    assert!(
        stderr.contains("INFO finish exit_code=0"),
        "stderr: {stderr}"
    );

    let log_path = dir.path().join("humanify.log");
    let log = std::fs::read_to_string(log_path).unwrap();
    assert!(log.contains("INFO start provider=gemini"), "log: {log}");
    assert!(log.contains("INFO input_read bytes="), "log: {log}");
    assert!(log.contains("INFO planner_finish"), "log: {log}");
    assert!(log.contains("INFO apply_finish"), "log: {log}");
    assert!(log.contains("INFO finish exit_code=0"), "log: {log}");
    assert!(!log.contains("api_key"), "log: {log}");
}

#[test]
fn no_log_file_disables_program_log_file() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.js");
    let out_path = dir.path().join("out.js");
    std::fs::write(&input, FIB_SAMPLE).unwrap();

    Command::cargo_bin("humanify")
        .unwrap()
        .current_dir(dir.path())
        .args([
            "gemini",
            input.to_str().unwrap(),
            "-o",
            out_path.to_str().unwrap(),
            "--base-url",
            "http://127.0.0.1:1",
            "--no-log-file",
        ])
        .assert()
        .success();

    assert!(!dir.path().join("humanify.log").exists());
}

#[test]
fn quiet_log_keeps_file_detail_but_hides_info_from_stderr() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.js");
    let out_path = dir.path().join("out.js");
    std::fs::write(&input, FIB_SAMPLE).unwrap();

    let assert = Command::cargo_bin("humanify")
        .unwrap()
        .current_dir(dir.path())
        .args([
            "gemini",
            input.to_str().unwrap(),
            "-o",
            out_path.to_str().unwrap(),
            "--base-url",
            "http://127.0.0.1:1",
            "--quiet-log",
        ])
        .assert()
        .success();

    let stderr = String::from_utf8(assert.get_output().stderr.clone()).unwrap();
    assert!(!stderr.contains("INFO start"), "stderr: {stderr}");

    let log = std::fs::read_to_string(dir.path().join("humanify.log")).unwrap();
    assert!(log.contains("INFO start provider=gemini"), "log: {log}");
    assert!(log.contains("INFO planner_finish"), "log: {log}");
    assert!(log.contains("INFO apply_finish"), "log: {log}");
    assert!(log.contains("INFO finish exit_code=0"), "log: {log}");
}

#[test]
fn ollama_offline_sample_uses_plan_first_without_llm_for_fibonacci() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.js");
    let output = dir.path().join("out.js");
    std::fs::write(&input, FIB_SAMPLE).unwrap();

    let mut cmd = Command::cargo_bin("humanify").unwrap();
    cmd.args([
        "ollama",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--dry-run-no-llm",
    ]);
    cmd.assert().success();
    let renamed = std::fs::read_to_string(output).unwrap();
    assert!(renamed.contains("generateFibonacciSequence"), "{renamed}");
    assert!(renamed.contains("sumValues"), "{renamed}");
}

#[test]
fn resume_rejects_missing_state_file() {
    let mut cmd = Command::cargo_bin("humanify").unwrap();
    cmd.args(["resume", "missing-state.json"]);
    cmd.assert().failure();
}

#[test]
fn ollama_sample_uses_deterministic_plan_before_llm() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.js");
    let output = dir.path().join("out.js");
    std::fs::write(&input, FIB_SAMPLE).unwrap();

    Command::cargo_bin("humanify")
        .unwrap()
        .args([
            "ollama",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--base-url",
            "http://127.0.0.1:1",
        ])
        .assert()
        .success();
    let renamed = std::fs::read_to_string(output).unwrap();
    assert!(renamed.contains("generateFibonacciSequence"), "{renamed}");
    assert!(renamed.contains("sumValues"), "{renamed}");
}

#[test]
fn dry_run_no_llm_pauses_and_writes_state_for_unresolved_symbols() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.js");
    let output = dir.path().join("out.js");
    let state = dir.path().join("state.json");
    std::fs::write(&input, "const x = window.location.href;").unwrap();

    Command::cargo_bin("humanify")
        .unwrap()
        .args([
            "ollama",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--dry-run-no-llm",
            "--state-file",
            state.to_str().unwrap(),
        ])
        .assert()
        .failure();
    let state_json = std::fs::read_to_string(state).unwrap();
    assert!(state_json.contains("paused"), "{state_json}");
    assert!(state_json.contains("NeedsLlm"), "{state_json}");
    assert!(!output.exists());
}

#[test]
fn resume_reads_existing_state_file() {
    let dir = tempdir().unwrap();
    let state = dir.path().join("state.json");
    std::fs::write(&state, r#"{"version":1,"input_hash":"h","input_path":"input.js","output_path":"out.js","phase":"paused","retry_policy":{"max_attempts":5,"backoff_ms":[1]},"plan":{"items":[]}}"#).unwrap();

    Command::cargo_bin("humanify")
        .unwrap()
        .args(["resume", state.to_str().unwrap()])
        .assert()
        .success();
}

#[test]
fn provider_resume_loads_saved_plan_and_pauses_again_when_llm_is_offline() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.js");
    let output = dir.path().join("out.js");
    let state = dir.path().join("state.json");
    std::fs::write(&input, "const x = window.location.href;").unwrap();

    Command::cargo_bin("humanify")
        .unwrap()
        .args([
            "ollama",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--dry-run-no-llm",
            "--state-file",
            state.to_str().unwrap(),
        ])
        .assert()
        .failure();

    let assert = Command::cargo_bin("humanify")
        .unwrap()
        .args([
            "ollama",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--resume",
            "--state-file",
            state.to_str().unwrap(),
            "--base-url",
            "http://127.0.0.1:1",
            "--max-llm-attempts",
            "1",
        ])
        .assert()
        .failure();

    let stderr = String::from_utf8(assert.get_output().stderr.clone()).unwrap();
    assert!(stderr.contains("state_loaded"), "stderr: {stderr}");
    assert!(stderr.contains("llm_job_start"), "stderr: {stderr}");
    let state_json = std::fs::read_to_string(state).unwrap();
    assert!(state_json.contains("paused"), "state: {state_json}");
    assert!(
        state_json.contains("\"attempts\": 1"),
        "state: {state_json}"
    );
    assert!(!output.exists());
}

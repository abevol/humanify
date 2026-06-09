use assert_cmd::Command;
use tempfile::tempdir;
use tempfile::NamedTempFile;

// Gemini is fully wired; point at an unreachable base-url so all renames
// get Transient errors → walker returns original names → identity output.
#[test]
fn gemini_offline_identity() {
    let out = NamedTempFile::new().unwrap();
    let out_path = out.path().to_owned();

    let assert = Command::cargo_bin("humanify")
        .unwrap()
        .args([
            "gemini",
            "-",
            "-o",
            out_path.to_str().unwrap(),
            "--base-url",
            "http://127.0.0.1:1",
        ])
        .write_stdin("const x = 1;")
        .assert()
        .success();

    let stderr = String::from_utf8(assert.get_output().stderr.clone()).unwrap();
    assert!(
        stderr.contains("humanify: renaming 1/1: x"),
        "stderr: {stderr}"
    );

    let contents = std::fs::read_to_string(&out_path).unwrap();
    assert_eq!(contents.trim(), "const x = 1;");
}

#[test]
fn default_program_log_appends_humanify_log() {
    let dir = tempdir().unwrap();
    let out = NamedTempFile::new_in(dir.path()).unwrap();
    let out_path = out.path().to_owned();

    let assert = Command::cargo_bin("humanify")
        .unwrap()
        .current_dir(dir.path())
        .args([
            "gemini",
            "-",
            "-o",
            out_path.to_str().unwrap(),
            "--base-url",
            "http://127.0.0.1:1",
        ])
        .write_stdin("const x = 1;")
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
    assert!(log.contains("INFO input_read bytes=12"), "log: {log}");
    assert!(
        log.contains("INFO progress_start step=1 total=1 original_name=x"),
        "log: {log}"
    );
    assert!(
        log.contains("INFO progress_finish step=1 total=1 original_name=x new_name=x"),
        "log: {log}"
    );
    assert!(log.contains("step_ms="), "log: {log}");
    let finish_line = log
        .lines()
        .find(|line| line.contains("INFO progress_finish"))
        .unwrap();
    assert!(
        finish_line.find("elapsed_ms=") < finish_line.find("step_ms="),
        "finish_line: {finish_line}"
    );
    assert!(log.contains("INFO finish exit_code=0"), "log: {log}");
    assert!(!log.contains("api_key"), "log: {log}");
}

#[test]
fn no_log_file_disables_program_log_file() {
    let dir = tempdir().unwrap();
    let out = NamedTempFile::new_in(dir.path()).unwrap();
    let out_path = out.path().to_owned();

    Command::cargo_bin("humanify")
        .unwrap()
        .current_dir(dir.path())
        .args([
            "gemini",
            "-",
            "-o",
            out_path.to_str().unwrap(),
            "--base-url",
            "http://127.0.0.1:1",
            "--no-log-file",
        ])
        .write_stdin("const x = 1;")
        .assert()
        .success();

    assert!(!dir.path().join("humanify.log").exists());
}

#[test]
fn quiet_log_keeps_file_detail_but_hides_info_from_stderr() {
    let dir = tempdir().unwrap();
    let out = NamedTempFile::new_in(dir.path()).unwrap();
    let out_path = out.path().to_owned();

    let assert = Command::cargo_bin("humanify")
        .unwrap()
        .current_dir(dir.path())
        .args([
            "gemini",
            "-",
            "-o",
            out_path.to_str().unwrap(),
            "--base-url",
            "http://127.0.0.1:1",
            "--quiet-log",
        ])
        .write_stdin("const x = 1;")
        .assert()
        .success();

    let stderr = String::from_utf8(assert.get_output().stderr.clone()).unwrap();
    assert!(!stderr.contains("INFO start"), "stderr: {stderr}");
    assert!(
        stderr.contains("humanify: renaming 1/1: x"),
        "stderr: {stderr}"
    );

    let log = std::fs::read_to_string(dir.path().join("humanify.log")).unwrap();
    assert!(log.contains("INFO start provider=gemini"), "log: {log}");
    assert!(
        log.contains("INFO progress_start step=1 total=1 original_name=x"),
        "log: {log}"
    );
    assert!(
        log.contains("INFO progress_finish step=1 total=1 original_name=x new_name=x"),
        "log: {log}"
    );
    assert!(log.contains("INFO finish exit_code=0"), "log: {log}");
}

#[test]
fn ollama_offline_sample_uses_plan_first_without_llm_for_fibonacci() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.js");
    let output = dir.path().join("out.js");
    std::fs::write(&input, "function a(e){var t=[0,1];if(e<=2)return t.slice(0,e);while(t.length<e){var n=t.length;t.push(t[n-1]+t[n-2])}return t}function b(e){var t=0;for(var n=0;n<e.length;n++){t+=e[n]}return t}var c=a(10);var d=b(c);console.log(c,d);").unwrap();

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

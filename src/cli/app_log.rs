use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(windows)]
#[repr(C)]
struct SystemTimeParts {
    year: u16,
    month: u16,
    day_of_week: u16,
    day: u16,
    hour: u16,
    minute: u16,
    second: u16,
    milliseconds: u16,
}

#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    fn GetLocalTime(system_time: *mut SystemTimeParts);
}

const DEFAULT_LOG_FILE: &str = "humanify.log";

#[derive(Clone)]
pub struct AppLogger {
    file: Option<Arc<Mutex<File>>>,
    stderr_mode: StderrMode,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum StderrMode {
    All,
    ErrorsOnly,
}

#[derive(Clone)]
pub struct RunTimer {
    state: Arc<Mutex<RunTimerState>>,
}

struct RunTimerState {
    started: std::time::Instant,
    last_step: std::time::Instant,
}

impl AppLogger {
    pub fn open(path: Option<&Path>, stderr_mode: StderrMode) -> io::Result<Self> {
        let file = match path {
            Some(path) => Some(Arc::new(Mutex::new(
                OpenOptions::new().create(true).append(true).open(path)?,
            ))),
            None => None,
        };
        Ok(Self { file, stderr_mode })
    }

    pub fn default_log_file() -> &'static Path {
        Path::new(DEFAULT_LOG_FILE)
    }

    pub fn info<K, V, I>(&self, event: &str, fields: I)
    where
        K: AsRef<str>,
        V: AsRef<str>,
        I: IntoIterator<Item = (K, V)>,
    {
        self.write("INFO", event, fields, true, &mut io::stderr());
    }

    pub fn error<K, V, I>(&self, event: &str, fields: I)
    where
        K: AsRef<str>,
        V: AsRef<str>,
        I: IntoIterator<Item = (K, V)>,
    {
        self.write("ERROR", event, fields, true, &mut io::stderr());
    }

    fn write<K, V, I, W>(
        &self,
        level: &str,
        event: &str,
        fields: I,
        is_stderr: bool,
        stderr: &mut W,
    ) where
        K: AsRef<str>,
        V: AsRef<str>,
        I: IntoIterator<Item = (K, V)>,
        W: Write,
    {
        let line = format_line(level, event, fields);
        if let Some(file) = &self.file {
            if let Ok(mut file) = file.lock() {
                let _ = writeln!(file, "{line}");
                let _ = file.flush();
            }
        }

        let should_write_stderr = match self.stderr_mode {
            StderrMode::All => true,
            StderrMode::ErrorsOnly => level == "ERROR",
        };
        if is_stderr && should_write_stderr {
            let _ = writeln!(stderr, "humanify: {line}");
        }
    }

    #[cfg(test)]
    fn open_with_stderr<'a, W: Write>(
        path: Option<&Path>,
        stderr: &'a mut W,
    ) -> io::Result<TestLogger<'a, W>> {
        Self::open_with_options(path, StderrMode::All, stderr)
    }

    #[cfg(test)]
    fn open_with_options<'a, W: Write>(
        path: Option<&Path>,
        stderr_mode: StderrMode,
        stderr: &'a mut W,
    ) -> io::Result<TestLogger<'a, W>> {
        let file = match path {
            Some(path) => Some(Arc::new(Mutex::new(
                OpenOptions::new().create(true).append(true).open(path)?,
            ))),
            None => None,
        };
        Ok(TestLogger {
            file,
            stderr_mode,
            stderr,
        })
    }
}

impl RunTimer {
    pub fn start() -> Self {
        let now = std::time::Instant::now();
        Self {
            state: Arc::new(Mutex::new(RunTimerState {
                started: now,
                last_step: now,
            })),
        }
    }

    pub fn elapsed_ms(&self) -> String {
        self.state
            .lock()
            .map(|state| state.started.elapsed().as_millis().to_string())
            .unwrap_or_else(|_| "0".to_string())
    }

    pub fn step_ms_and_elapsed_ms(&mut self) -> (String, String) {
        let now = std::time::Instant::now();
        match self.state.lock() {
            Ok(mut state) => {
                let step_ms = now.duration_since(state.last_step).as_millis().to_string();
                state.last_step = now;
                let elapsed_ms = now.duration_since(state.started).as_millis().to_string();
                (step_ms, elapsed_ms)
            }
            Err(_) => ("0".to_string(), "0".to_string()),
        }
    }
}

fn format_line<K, V, I>(level: &str, event: &str, fields: I) -> String
where
    K: AsRef<str>,
    V: AsRef<str>,
    I: IntoIterator<Item = (K, V)>,
{
    let mut line = format!("{} {level} {event}", human_timestamp());
    for (key, value) in fields {
        line.push(' ');
        line.push_str(key.as_ref());
        line.push('=');
        line.push_str(&format_value(value.as_ref()));
    }
    line
}

fn human_timestamp() -> String {
    if let Some(timestamp) = local_timestamp() {
        return timestamp;
    }

    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let seconds = duration.as_secs() as i64;
    let millis = duration.subsec_millis();
    let (year, month, day, hour, minute, second) = unix_seconds_to_utc_parts(seconds);
    format!("{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02}.{millis:03}")
}

#[cfg(windows)]
fn local_timestamp() -> Option<String> {
    let mut parts = SystemTimeParts {
        year: 0,
        month: 0,
        day_of_week: 0,
        day: 0,
        hour: 0,
        minute: 0,
        second: 0,
        milliseconds: 0,
    };
    unsafe {
        GetLocalTime(&mut parts);
    }
    Some(format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:03}",
        parts.year,
        parts.month,
        parts.day,
        parts.hour,
        parts.minute,
        parts.second,
        parts.milliseconds
    ))
}

#[cfg(not(windows))]
fn local_timestamp() -> Option<String> {
    None
}

fn unix_seconds_to_utc_parts(seconds: i64) -> (i32, u32, u32, u32, u32, u32) {
    let days = seconds.div_euclid(86_400);
    let seconds_of_day = seconds.rem_euclid(86_400) as u32;
    let (year, month, day) = civil_from_days(days);
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;
    (year, month, day, hour, minute, second)
}

fn civil_from_days(days_since_epoch: i64) -> (i32, u32, u32) {
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    let year = y + if month <= 2 { 1 } else { 0 };
    (year as i32, month as u32, day as u32)
}

fn format_value(value: &str) -> String {
    if value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.' | '/' | ':' | '='))
    {
        value.to_string()
    } else {
        let escaped = value.replace('\\', "\\\\").replace('"', "\\\"");
        format!("\"{escaped}\"")
    }
}

#[cfg(test)]
struct TestLogger<'a, W: Write> {
    file: Option<Arc<Mutex<File>>>,
    stderr_mode: StderrMode,
    stderr: &'a mut W,
}

#[cfg(test)]
impl<W: Write> TestLogger<'_, W> {
    fn info<K, V, I>(&mut self, event: &str, fields: I)
    where
        K: AsRef<str>,
        V: AsRef<str>,
        I: IntoIterator<Item = (K, V)>,
    {
        self.write("INFO", event, fields);
    }

    fn error<K, V, I>(&mut self, event: &str, fields: I)
    where
        K: AsRef<str>,
        V: AsRef<str>,
        I: IntoIterator<Item = (K, V)>,
    {
        self.write("ERROR", event, fields);
    }

    fn write<K, V, I>(&mut self, level: &str, event: &str, fields: I)
    where
        K: AsRef<str>,
        V: AsRef<str>,
        I: IntoIterator<Item = (K, V)>,
    {
        let line = format_line(level, event, fields);
        if let Some(file) = &self.file {
            let mut file = file.lock().unwrap();
            writeln!(file, "{line}").unwrap();
            file.flush().unwrap();
        }

        let should_write_stderr = match self.stderr_mode {
            StderrMode::All => true,
            StderrMode::ErrorsOnly => level == "ERROR",
        };
        if should_write_stderr {
            writeln!(self.stderr, "humanify: {line}").unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn appends_program_log_lines_to_file_and_stderr() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("humanify.log");
        let mut stderr = Vec::new();
        let mut logger = AppLogger::open_with_stderr(Some(&path), &mut stderr).unwrap();

        logger.info("start", [("provider", "gemini"), ("input", "-")]);
        logger.error("finish", [("exit_code", "1"), ("elapsed_ms", "25")]);

        let file_contents = fs::read_to_string(path).unwrap();
        assert_eq!(file_contents.lines().count(), 2);
        assert!(file_contents.contains("INFO start provider=gemini input=-"));
        assert!(file_contents.contains("ERROR finish exit_code=1 elapsed_ms=25"));

        let stderr_contents = String::from_utf8(stderr).unwrap();
        assert_eq!(stderr_contents.lines().count(), 2);
        assert!(!stderr_contents.contains("humanify: time="));
        assert!(!stderr_contents.contains("timestamp_ms="));
        assert!(stderr_contents.contains("INFO start provider=gemini input=-"));
        assert!(stderr_contents.contains("ERROR finish exit_code=1 elapsed_ms=25"));
    }

    #[test]
    fn formats_timestamp_for_humans() {
        let line = format_line("INFO", "start", [("provider", "gemini")]);
        assert!(
            line.chars().next().unwrap().is_ascii_digit(),
            "line: {line}"
        );
        assert!(!line.starts_with("time="), "line: {line}");
        assert!(!line.contains('Z'), "line: {line}");
        assert!(line.contains(" INFO start provider=gemini"), "line: {line}");
        assert_eq!(line.chars().nth(4), Some('-'), "line: {line}");
        assert_eq!(line.chars().nth(7), Some('-'), "line: {line}");
        assert_eq!(line.chars().nth(10), Some(' '), "line: {line}");
        assert_eq!(line.chars().nth(13), Some(':'), "line: {line}");
        assert_eq!(line.chars().nth(16), Some(':'), "line: {line}");
        assert_eq!(line.chars().nth(19), Some('.'), "line: {line}");
    }

    #[test]
    fn quotes_values_with_spaces_and_quotes() {
        let mut stderr = Vec::new();
        let mut logger = AppLogger::open_with_stderr(None, &mut stderr).unwrap();

        logger.error("parse_error", [("message", "bad \"thing\" happened")]);

        let stderr_contents = String::from_utf8(stderr).unwrap();
        assert!(stderr_contents.contains("message=\"bad \\\"thing\\\" happened\""));
    }

    #[test]
    fn quiet_logger_only_writes_errors_to_stderr_but_keeps_file_detail() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("humanify.log");
        let mut stderr = Vec::new();
        let mut logger =
            AppLogger::open_with_options(Some(&path), StderrMode::ErrorsOnly, &mut stderr).unwrap();

        logger.info("start", [("provider", "openai")]);
        logger.error("finish", [("exit_code", "1")]);

        let file_contents = fs::read_to_string(path).unwrap();
        assert!(file_contents.contains("INFO start provider=openai"));
        assert!(file_contents.contains("ERROR finish exit_code=1"));

        let stderr_contents = String::from_utf8(stderr).unwrap();
        assert!(!stderr_contents.contains("INFO start"));
        assert!(stderr_contents.contains("ERROR finish exit_code=1"));
    }
}

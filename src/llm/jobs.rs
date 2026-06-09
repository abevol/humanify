use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use serde_json::json;

use super::batch::{validate_batch_response, LlmBatchJob, ValidatedBatch};
use super::JsonStrategy;
use crate::llm::http::StrategyError;
use crate::rename::state::RetryPolicy;

const BATCH_SYSTEM_PROMPT: &str = "You rename obfuscated JavaScript identifiers. Return JSON only.";

pub struct JobRunner {
    strategy: Arc<dyn JsonStrategy>,
    retry_policy: RetryPolicy,
}

impl JobRunner {
    pub fn new(strategy: Arc<dyn JsonStrategy>, retry_policy: RetryPolicy) -> Self {
        Self {
            strategy,
            retry_policy,
        }
    }

    pub async fn run_job(&self, job: LlmBatchJob) -> Result<ValidatedBatch> {
        let attempts = self.retry_policy.max_attempts.max(1);
        let mut last_error = None;

        for attempt in 1..=attempts {
            match self.call_once(&job).await {
                Ok(validated) => return Ok(validated),
                Err(err) => {
                    last_error = Some(err);
                    if attempt < attempts {
                        let delay = self
                            .retry_policy
                            .backoff_ms
                            .get((attempt - 1) as usize)
                            .copied()
                            .unwrap_or(0);
                        if delay > 0 {
                            tokio::time::sleep(Duration::from_millis(delay)).await;
                        }
                    }
                }
            }
        }

        Err(anyhow!(
            "paused after {} LLM job attempts: {}",
            attempts,
            last_error
                .map(|err| err.to_string())
                .unwrap_or_else(|| "unknown error".to_string())
        ))
    }

    async fn call_once(&self, job: &LlmBatchJob) -> Result<ValidatedBatch> {
        let schema = json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["renames"],
            "properties": {
                "renames": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": false,
                        "required": ["symbol_key", "name", "confidence"],
                        "properties": {
                            "symbol_key": { "type": "string" },
                            "name": { "type": "string", "minLength": 1, "maxLength": 64 },
                            "confidence": { "type": "integer", "minimum": 0, "maximum": 100 }
                        }
                    }
                }
            }
        });
        let user = serde_json::to_string(job)?;
        let response = self
            .strategy
            .call(BATCH_SYSTEM_PROMPT, &user, &schema)
            .await
            .map_err(strategy_error_to_anyhow)?;
        Ok(validate_batch_response(job, &response))
    }

    #[cfg(test)]
    pub fn for_test<S>(strategy: Arc<S>, retry_policy: RetryPolicy) -> Self
    where
        S: JsonStrategy + 'static,
    {
        let strategy: Arc<dyn JsonStrategy> = strategy;
        Self::new(strategy, retry_policy)
    }
}

fn strategy_error_to_anyhow(err: StrategyError) -> anyhow::Error {
    match err {
        StrategyError::NotSupported(reason) => anyhow!("strategy not supported: {reason}"),
        StrategyError::Transient(err) => err,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::test_dsl::{script, transient, ScriptedResponse};

    #[tokio::test]
    async fn transient_failure_retries_and_succeeds() {
        let strategy = script(
            "scripted",
            vec![
                ScriptedResponse::Transient("offline".to_string()),
                ScriptedResponse::Ok(
                    serde_json::json!({"renames":[{"symbol_key":"s1","name":"goodName","confidence":90}]}),
                ),
            ],
        );
        let runner = JobRunner::for_test(
            strategy,
            RetryPolicy {
                max_attempts: 2,
                backoff_ms: vec![0, 0],
            },
        );
        let result = runner
            .run_job(crate::llm::batch::LlmBatchJob::for_test(&["s1"]))
            .await
            .unwrap();
        assert_eq!(result.accepted.len(), 1);
    }

    #[tokio::test]
    async fn exhausted_job_returns_pause() {
        let strategy = transient("scripted", "offline");
        let runner = JobRunner::for_test(
            strategy,
            RetryPolicy {
                max_attempts: 1,
                backoff_ms: vec![0],
            },
        );
        let err = runner
            .run_job(crate::llm::batch::LlmBatchJob::for_test(&["s1"]))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("paused"));
    }
}

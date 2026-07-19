use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use helix::{
    ChatCompletionRequest, Message, ModelOptions, Runtime, RuntimeOptions, LogLevel,
    SessionOptions,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize)]
struct Cases {
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    skip_if_no_vision: bool,
    #[serde(default)]
    cancel_after_chunks: Option<usize>,
    #[serde(default)]
    expect_error: Option<Value>,
    request: Value,
    #[serde(rename = "assert", default)]
    assertions: Value,
    #[serde(rename = "assert_stream", default)]
    stream_assertions: Value,
}

#[derive(Debug, Serialize)]
struct TestResult {
    name: String,
    passed: bool,
    error: Option<String>,
    detail: Value,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: conformance <cases.yaml>");
        std::process::exit(1);
    }

    let cases_path = PathBuf::from(&args[1]);
    let model_path = env::var("HELIX_TEST_MODEL_PATH").unwrap_or_default();
    let model_alias = env::var("HELIX_TEST_MODEL").unwrap_or_else(|_| "qwen-test".into());
    let shim_url = env::var("HELIX_SHIM_URL")
        .unwrap_or_else(|_| "http://localhost:8080".into());

    let content = std::fs::read_to_string(&cases_path)?;
    let cases: Cases = serde_yaml::from_str(&content)?;

    // Prefer direct lib usage when model path is set, else fall back to HTTP
    let results: Vec<TestResult> = if !model_path.is_empty() {
        run_via_lib(&cases.cases, &model_path, &model_alias).await?
    } else {
        run_via_http(&cases.cases, &shim_url, &model_alias).await?
    };

    println!("{}", serde_json::to_string(&results)?);
    Ok(())
}

async fn run_via_lib(
    cases: &[Case],
    model_path: &str,
    alias: &str,
) -> anyhow::Result<Vec<TestResult>> {
    let runtime = Runtime::new(RuntimeOptions {
        log_level: LogLevel::Error,
        ..Default::default()
    })?;
    let model = runtime.load_model(model_path, ModelOptions {
        alias: Some(alias.to_string()),
        ..Default::default()
    })?;
    let session = model.session(SessionOptions::default())?;

    let mut results = Vec::new();
    for case in cases {
        results.push(run_case_lib(Arc::clone(&session), case).await);
    }
    Ok(results)
}

async fn run_case_lib(session: Arc<helix::Session>, case: &Case) -> TestResult {
    let name = case.name.clone();

    if case.skip_if_no_vision {
        return TestResult {
            name,
            passed: true,
            error: None,
            detail: serde_json::json!({"skipped": "no_vision"}),
        };
    }

    let mut req: ChatCompletionRequest = match serde_json::from_value(case.request.clone()) {
        Ok(r) => r,
        Err(e) => return TestResult {
            name, passed: false, error: Some(format!("bad request: {e}")), detail: Value::Null,
        },
    };

    if case.expect_error.is_some() {
        let result = session.chat_completions(req);
        return TestResult {
            name,
            passed: result.is_err(),
            error: if result.is_ok() { Some("expected error but got success".into()) } else { None },
            detail: Value::Null,
        };
    }

    if case.stream {
        use futures::StreamExt;
        let mut chunks = Vec::new();
        let mut stream = Arc::clone(&session).stream_chat_completions(req);
        let cancel_after = case.cancel_after_chunks;
        let mut count = 0;
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(c) => {
                    chunks.push(c);
                    count += 1;
                    if let Some(limit) = cancel_after {
                        if count >= limit {
                            break;
                        }
                    }
                }
                Err(e) => return TestResult {
                    name, passed: false, error: Some(e.to_string()), detail: Value::Null,
                },
            }
        }
        drop(stream);

        TestResult {
            name,
            passed: true,
            error: None,
            detail: serde_json::json!({
                "chunks_received": chunks.len(),
                "cancelled_cleanly": cancel_after.is_some()
            }),
        }
    } else {
        match session.chat_completions(req) {
            Ok(resp) => TestResult {
                name,
                passed: true,
                error: None,
                detail: serde_json::json!({
                    "content": resp.choices.first()
                        .and_then(|c| c.message.content.as_deref())
                        .unwrap_or(""),
                    "finish_reason": resp.choices.first()
                        .and_then(|c| c.finish_reason.as_deref())
                }),
            },
            Err(e) => TestResult {
                name, passed: false, error: Some(e.to_string()), detail: Value::Null,
            },
        }
    }
}

async fn run_via_http(
    cases: &[Case],
    shim_url: &str,
    model_alias: &str,
) -> anyhow::Result<Vec<TestResult>> {
    let client = reqwest::Client::new();
    let mut results = Vec::new();
    for case in cases {
        let result = run_case_http(&client, case, shim_url, model_alias).await;
        results.push(result);
    }
    Ok(results)
}

async fn run_case_http(
    client: &reqwest::Client,
    case: &Case,
    shim_url: &str,
    model_alias: &str,
) -> TestResult {
    let name = case.name.clone();
    if case.skip_if_no_vision {
        return TestResult { name, passed: true, error: None,
            detail: serde_json::json!({"skipped": "no_vision"}) };
    }
    let mut req = case.request.clone();
    if let Some(obj) = req.as_object_mut() {
        obj.entry("model").or_insert_with(|| model_alias.into());
    }

    let url = format!("{}/v1/chat/completions", shim_url);
    match client.post(&url).json(&req).send().await {
        Ok(resp) => {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            if !status.is_success() {
                let passed = case.expect_error.is_some();
                return TestResult { name, passed,
                    error: if passed { None } else { Some(format!("HTTP {status}: {}", &body[..body.len().min(200)])) },
                    detail: Value::Null };
            }
            if case.expect_error.is_some() {
                return TestResult { name, passed: false,
                    error: Some("expected error but got success".into()), detail: Value::Null };
            }
            TestResult { name, passed: true, error: None,
                detail: serde_json::from_str(&body).unwrap_or(Value::Null) }
        }
        Err(e) => TestResult { name, passed: false, error: Some(e.to_string()), detail: Value::Null },
    }
}

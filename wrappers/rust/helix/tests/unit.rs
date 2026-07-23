use helix::types::*;

#[test]
fn message_user_constructor() {
    let msg = Message::user("hello");
    match msg {
        Message::User {
            content: MessageContent::Text(t),
        } => assert_eq!(t, "hello"),
        _ => panic!("unexpected variant"),
    }
}

#[test]
fn message_system_constructor() {
    let msg = Message::system("You are helpful");
    match msg {
        Message::System { content } => assert_eq!(content, "You are helpful"),
        _ => panic!("unexpected variant"),
    }
}

#[test]
fn chat_completion_request_serializes() {
    let req = ChatCompletionRequest {
        model: "qwen-test".into(),
        messages: vec![Message::user("hi")],
        temperature: Some(0.0),
        max_tokens: Some(8),
        ..Default::default()
    };
    let json = serde_json::to_string(&req).unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["model"], "qwen-test");
    assert_eq!(v["temperature"], 0.0);
    assert_eq!(v["max_tokens"], 8);
    assert!(v["messages"].is_array());
}

#[test]
fn chat_completion_deserializes() {
    let json = r#"{
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "qwen-test",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "pong"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    }"#;
    let cc: ChatCompletion = serde_json::from_str(json).unwrap();
    assert_eq!(cc.id, "chatcmpl-1");
    assert_eq!(cc.choices[0].message.content.as_deref(), Some("pong"));
    assert_eq!(cc.usage.total_tokens, 7);
}

#[test]
fn chunk_deserializes() {
    let json = r#"{
        "id": "x", "object": "chat.completion.chunk",
        "created": 0, "model": "m",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "po"}, "finish_reason": null}]
    }"#;
    let chunk: ChatCompletionChunk = serde_json::from_str(json).unwrap();
    assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("po"));
    assert_eq!(chunk.choices[0].delta.role.as_deref(), Some("assistant"));
}

#[test]
fn error_check_ok() {
    use helix::error::check;
    assert!(check(0, "{}").is_ok());
}

#[test]
fn error_check_validation() {
    use helix::error::{check, Error};
    let json = r#"{"error":{"message":"bad param","type":"invalid_request_error","param":"messages","code":null}}"#;
    let err = check(-3, json).unwrap_err();
    assert!(matches!(err, Error::Validation { .. }));
}

#[test]
fn error_check_model_not_found() {
    use helix::error::{check, Error};
    let json =
        r#"{"error":{"message":"not found","type":"not_found_error","param":null,"code":null}}"#;
    let err = check(-4, json).unwrap_err();
    assert!(matches!(err, Error::ModelNotFound { .. }));
}

#[test]
fn error_check_cancelled() {
    use helix::error::{check, Error};
    let err = check(-9, "{}").unwrap_err();
    assert!(matches!(err, Error::Cancelled));
}

#[test]
fn response_format_json_object() {
    let rf = ResponseFormat::json_object();
    assert_eq!(rf.format_type, "json_object");
    let json = serde_json::to_string(&rf).unwrap();
    assert!(json.contains("json_object"));
}

#[test]
fn request_skips_none_fields() {
    let req = ChatCompletionRequest {
        model: "m".into(),
        messages: vec![],
        ..Default::default()
    };
    let json = serde_json::to_string(&req).unwrap();
    assert!(!json.contains("temperature"));
    assert!(!json.contains("max_tokens"));
    assert!(!json.contains("tools"));
}

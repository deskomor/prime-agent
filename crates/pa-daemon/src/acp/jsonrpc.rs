//! Line-delimited JSON-RPC 2.0 framing for the ACP stdio transport: one
//! JSON object per line; requests carry `method` + `params` + `id`,
//! responses exactly one of `result` / `error`, notifications omit `id`.
//! Parse failures answer with the JSON-RPC error codes so a client never
//! sees a dropped line.

use serde_json::{json, Value};

/// The request could not be parsed as JSON.
pub const PARSE_ERROR: i64 = -32700;
/// The JSON is valid but not a JSON-RPC message.
pub const INVALID_REQUEST: i64 = -32600;
/// The method is not part of the served ACP surface.
pub const METHOD_NOT_FOUND: i64 = -32601;
/// The method exists but a parameter failed validation.
pub const INVALID_PARAMS: i64 = -32602;
/// A handler failed while processing the request.
pub const INTERNAL_ERROR: i64 = -32603;

/// One incoming frame off the stdio stream.
#[derive(Debug, Clone, PartialEq)]
pub enum Incoming {
    Request {
        id: Value,
        method: String,
        params: Value,
    },
    Notification {
        method: String,
        params: Value,
    },
    /// The client's answer to a request this agent sent
    /// (`session/request_permission`): `result` on success, `error` otherwise.
    Response {
        id: Value,
        result: Option<Value>,
        error: Option<Value>,
    },
}

/// Parse one line into an incoming message. `Err` carries the error
/// response value to write back (JSON-RPC requires an answer even for
/// malformed input, with `id: null` when unknowable).
pub fn parse_line(line: &str) -> Result<Incoming, Value> {
    let trimmed = line.trim();
    let bad_request = || error_response(&Value::Null, INVALID_REQUEST, "Invalid Request", None);
    let value: Value = match serde_json::from_str(trimmed) {
        Ok(value) => value,
        Err(_) => {
            return Err(error_response(
                &Value::Null,
                PARSE_ERROR,
                "Parse error",
                None,
            ))
        }
    };
    let Value::Object(object) = value else {
        return Err(bad_request());
    };
    if object.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Err(bad_request());
    }
    let Some(method) = object.get("method").and_then(Value::as_str) else {
        // A bare response (no method) answers a request this agent sent.
        let (Some(id), result, error) = (
            object.get("id").filter(|id| !id.is_null()),
            object.get("result"),
            object.get("error"),
        ) else {
            return Err(bad_request());
        };
        if result.is_none() == error.is_none() {
            return Err(bad_request());
        }
        return Ok(Incoming::Response {
            id: id.clone(),
            result: result.cloned(),
            error: error.cloned(),
        });
    };
    let params = object.get("params").cloned().unwrap_or_else(|| json!({}));
    match object.get("id") {
        Some(Value::Null) | None => Ok(Incoming::Notification {
            method: method.to_string(),
            params,
        }),
        Some(id) => Ok(Incoming::Request {
            id: id.clone(),
            method: method.to_string(),
            params,
        }),
    }
}

/// A successful response frame.
pub fn response(id: &Value, result: &Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

/// An error response frame. `data` rides under the error object.
pub fn error_response(id: &Value, code: i64, message: &str, data: Option<&Value>) -> Value {
    let error = match data {
        Some(data) => json!({ "code": code, "message": message, "data": data }),
        None => json!({ "code": code, "message": message }),
    };
    json!({ "jsonrpc": "2.0", "id": id, "error": error })
}

/// A notification frame (no id, no response expected).
pub fn notification(method: &str, params: &Value) -> Value {
    json!({ "jsonrpc": "2.0", "method": method, "params": params })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_request_notification_and_errors() {
        let request =
            parse_line(r#"{"jsonrpc":"2.0","id":7,"method":"initialize","params":{"a":1}}"#)
                .unwrap();
        assert_eq!(
            request,
            Incoming::Request {
                id: json!(7),
                method: "initialize".to_string(),
                params: json!({"a":1})
            }
        );
        let notification =
            parse_line(r#"{"jsonrpc":"2.0","method":"session/cancel","params":{"sessionId":"s"}}"#)
                .unwrap();
        assert_eq!(
            notification,
            Incoming::Notification {
                method: "session/cancel".to_string(),
                params: json!({"sessionId":"s"})
            }
        );
        let parse_error = parse_line("not json").unwrap_err();
        assert_eq!(parse_error["error"]["code"], PARSE_ERROR);
        let invalid = parse_line(r#"{"id":3,"result":null}"#).unwrap_err();
        assert_eq!(invalid["error"]["code"], INVALID_REQUEST);
        let answer = parse_line(
            r#"{"jsonrpc":"2.0","id":"p1","result":{"outcome":{"outcome":"cancelled"}}}"#,
        )
        .unwrap();
        assert_eq!(
            answer,
            Incoming::Response {
                id: json!("p1"),
                result: Some(json!({"outcome":{"outcome":"cancelled"}})),
                error: None,
            }
        );
        let both = parse_line(r#"{"jsonrpc":"2.0","id":"p1","result":{},"error":{}}"#).unwrap_err();
        assert_eq!(both["error"]["code"], INVALID_REQUEST);
    }

    #[test]
    fn response_shapes() {
        assert_eq!(
            response(&json!(1), &json!({"stopReason": "end_turn"})),
            json!({"jsonrpc":"2.0","id":1,"result":{"stopReason":"end_turn"}})
        );
        assert_eq!(
            error_response(
                &json!(2),
                INVALID_PARAMS,
                "Invalid params",
                Some(&json!({"details": "x"}))
            ),
            json!({"jsonrpc":"2.0","id":2,"error":{"code":-32602,"message":"Invalid params","data":{"details":"x"}}})
        );
        assert_eq!(
            notification("session/update", &json!({"sessionId": "s"})),
            json!({"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"s"}})
        );
    }
}

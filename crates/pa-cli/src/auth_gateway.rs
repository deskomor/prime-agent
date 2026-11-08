//! `prime-agent auth-gateway`: a loopback OpenAI chat-completions endpoint over
//! this agent dir's logins (fork-only).
//!
//! A benchmark judge, or any tool that only speaks the `OpenAI` wire format,
//! points its base URL here and sends the gateway's bearer token. Each request
//! resolves the model's credentials from `auth.json` (an API key, a Claude
//! Pro/Max login or a `ChatGPT` login, refreshed in one place) and runs one
//! tool-free completion through pa-ai. The caller never sees a provider token.
//!
//! Sub-verbs:
//!   - `serve [--bind 127.0.0.1:4100]`: listen on a loopback address only.
//!   - `token`: print the bearer token, creating it on first use.
//!
//! Routes: `GET /healthz` (no auth), `GET /v1/models`, `POST /v1/chat/completions`
//! (text only; `stream: true` gets the whole reply as one SSE chunk).

use std::future::Future;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use pa_types::ai::{
    AssistantContentBlock, AssistantMessage, Context, Message, Model, ModelThinkingLevel,
    StopReason, TextContent, UserContent, UserMessage,
};
use serde_json::{json, Map, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const DEFAULT_BIND: &str = "127.0.0.1:4100";
const TOKEN_FILE_NAME: &str = "auth-gateway-token";
const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_BODY_BYTES: usize = 32 * 1024 * 1024;

/// One resolved completion request: the model, its context and options.
pub(crate) struct Completion {
    pub model: Model,
    pub context: Context,
    pub options: pa_ai::types::SimpleStreamOptions,
}

type CompletionFuture = Pin<Box<dyn Future<Output = Result<AssistantMessage, String>> + Send>>;

/// The gateway's view of the agent dir: model lookup, credential resolution
/// and the completion call. Tests swap the whole seam.
pub(crate) trait Backend: Send + Sync {
    /// `provider/id` of every model this agent dir holds credentials for.
    fn models(&self) -> Vec<String>;
    /// Resolve `requested` (`provider/id`, or a bare id served by one provider)
    /// to a model plus its api key and headers.
    fn resolve(&self, requested: &str) -> Result<ResolvedModel, String>;
    fn complete(&self, completion: Completion) -> CompletionFuture;
}

pub(crate) struct ResolvedModel {
    pub model: Model,
    pub api_key: Option<String>,
    pub headers: Option<std::collections::HashMap<String, String>>,
}

/// The real backend: the agent dir's model registry and `auth.json`.
struct RegistryBackend {
    registry: Mutex<pa_core::models::ModelRegistry>,
}

impl Backend for RegistryBackend {
    fn models(&self) -> Vec<String> {
        let registry = self.registry.lock().expect("auth-gateway registry");
        registry
            .get_available()
            .into_iter()
            .map(|model| format!("{}/{}", model.provider, model.id))
            .collect()
    }

    fn resolve(&self, requested: &str) -> Result<ResolvedModel, String> {
        let mut registry = self.registry.lock().expect("auth-gateway registry");
        let model = find_model(&registry.get_available(), requested)?.clone();
        let auth = registry.get_api_key_and_headers(&model, None);
        if !auth.ok {
            return Err(auth
                .error
                .unwrap_or_else(|| format!("no credentials for provider {}", model.provider)));
        }
        Ok(ResolvedModel {
            model,
            api_key: auth.api_key,
            headers: auth.headers.map(|headers| headers.into_iter().collect()),
        })
    }

    fn complete(&self, completion: Completion) -> CompletionFuture {
        Box::pin(async move {
            pa_ai::complete_simple(
                &completion.model,
                &completion.context,
                Some(completion.options),
            )
            .await
            .map_err(|error| error.to_string())
        })
    }
}

fn find_model<'a>(available: &[&'a Model], requested: &str) -> Result<&'a Model, String> {
    let requested = requested.trim();
    if let Some((provider, id)) = requested.split_once('/') {
        if let Some(model) = available
            .iter()
            .find(|model| model.provider == provider && model.id == id)
        {
            return Ok(model);
        }
    }
    let bare: Vec<&&Model> = available
        .iter()
        .filter(|model| model.id == requested)
        .collect();
    match bare.as_slice() {
        [model] => Ok(model),
        [] => Err(format!(
            "unknown model {requested:?}: no logged-in provider serves it (GET /v1/models lists them)"
        )),
        _ => Err(format!(
            "model {requested:?} is served by several providers; ask for provider/id"
        )),
    }
}

pub fn run_auth_gateway_command(args: &[String]) -> i32 {
    let agent_dir = crate::config::get_agent_dir();
    match args.first().map(String::as_str) {
        Some("token") if args.len() == 1 => match ensure_token(&agent_dir) {
            Ok(token) => {
                println!("{token}");
                0
            }
            Err(error) => fail(&error),
        },
        Some("serve") => match parse_bind(&args[1..]) {
            Ok(bind) => serve_blocking(&agent_dir, bind),
            Err(error) => fail(&error),
        },
        _ => fail("usage: prime-agent auth-gateway <serve [--bind host:port]|token>"),
    }
}

fn fail(message: &str) -> i32 {
    eprintln!("Error: {message}");
    eprintln!("Run `prime-agent help auth-gateway` for usage.");
    1
}

fn parse_bind(args: &[String]) -> Result<SocketAddr, String> {
    let raw = match args {
        [] => DEFAULT_BIND.to_string(),
        [flag, value] if flag == "--bind" => value.clone(),
        [flag] if flag.starts_with("--bind=") => flag["--bind=".len()..].to_string(),
        _ => return Err("serve takes only --bind host:port".to_string()),
    };
    let addr: SocketAddr = raw
        .parse()
        .map_err(|_| format!("--bind wants host:port with an IP address, got {raw:?}"))?;
    if !addr.ip().is_loopback() {
        return Err(format!(
            "--bind {addr} is not a loopback address; the gateway hands out your logins, so it only listens on 127.0.0.1 or ::1"
        ));
    }
    Ok(addr)
}

fn token_path(agent_dir: &Path) -> PathBuf {
    agent_dir.join(TOKEN_FILE_NAME)
}

/// The bearer token, created (mode 0600) on first use.
fn ensure_token(agent_dir: &Path) -> Result<String, String> {
    let path = token_path(agent_dir);
    if let Ok(existing) = std::fs::read_to_string(&path) {
        let existing = existing.trim().to_string();
        if !existing.is_empty() {
            return Ok(existing);
        }
    }
    std::fs::create_dir_all(agent_dir).map_err(|error| error.to_string())?;
    let token = format!(
        "pag-{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    );
    std::fs::write(&path, format!("{token}\n")).map_err(|error| error.to_string())?;
    pa_core::platform::perms::restrict_file(&path).map_err(|error| error.to_string())?;
    Ok(token)
}

fn serve_blocking(agent_dir: &Path, bind: SocketAddr) -> i32 {
    let token = match ensure_token(agent_dir) {
        Ok(token) => token,
        Err(error) => return fail(&error),
    };
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => return fail(&error.to_string()),
    };
    let auth = pa_core::auth::AuthStorage::create(agent_dir);
    let registry = pa_core::models::ModelRegistry::create(auth, agent_dir.join("models.json"));
    if let Some(error) = registry.get_error() {
        eprintln!("Warning: errors loading models.json:\n{error}");
    }
    let backend: Arc<dyn Backend> = Arc::new(RegistryBackend {
        registry: Mutex::new(registry),
    });
    runtime.block_on(async move {
        let listener = match TcpListener::bind(bind).await {
            Ok(listener) => listener,
            Err(error) => return fail(&format!("cannot listen on {bind}: {error}")),
        };
        println!("auth-gateway listening on http://{bind}/v1");
        println!("bearer token: {}", token_path(agent_dir).display());
        serve(listener, backend, token).await;
        0
    })
}

/// Accept connections until the listener fails; one task per connection.
pub(crate) async fn serve(listener: TcpListener, backend: Arc<dyn Backend>, token: String) {
    let token = Arc::new(token);
    while let Ok((stream, _)) = listener.accept().await {
        let backend = Arc::clone(&backend);
        let token = Arc::clone(&token);
        tokio::spawn(async move {
            let _ = handle_connection(stream, backend.as_ref(), &token).await;
        });
    }
}

struct Request {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Request {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }
}

struct Response {
    status: u16,
    content_type: &'static str,
    body: Vec<u8>,
}

impl Response {
    fn json(status: u16, value: &Value) -> Self {
        Self {
            status,
            content_type: "application/json",
            body: value.to_string().into_bytes(),
        }
    }

    fn error(status: u16, kind: &str, message: &str) -> Self {
        Self::json(
            status,
            &json!({"error": {"message": message, "type": kind}}),
        )
    }
}

async fn handle_connection(
    mut stream: TcpStream,
    backend: &dyn Backend,
    token: &str,
) -> std::io::Result<()> {
    let response = match read_request(&mut stream).await {
        Ok(request) => route(&request, backend, token).await,
        Err(message) => Response::error(400, "invalid_request_error", &message),
    };
    let reason = match response.status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        _ => "Bad Gateway",
    };
    let head = format!(
        "HTTP/1.1 {} {reason}\r\ncontent-type: {}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        response.status,
        response.content_type,
        response.body.len()
    );
    stream.write_all(head.as_bytes()).await?;
    stream.write_all(&response.body).await?;
    stream.shutdown().await
}

async fn read_request(stream: &mut TcpStream) -> Result<Request, String> {
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 8192];
    let header_end = loop {
        if let Some(end) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
            break end;
        }
        if buffer.len() > MAX_HEADER_BYTES {
            return Err("request headers too large".to_string());
        }
        let read = stream
            .read(&mut chunk)
            .await
            .map_err(|error| error.to_string())?;
        if read == 0 {
            return Err("connection closed before the request ended".to_string());
        }
        buffer.extend_from_slice(&chunk[..read]);
    };
    let head = String::from_utf8_lossy(&buffer[..header_end]).to_string();
    let mut lines = head.split("\r\n");
    let mut request_line = lines.next().unwrap_or_default().split_whitespace();
    let method = request_line.next().unwrap_or_default().to_string();
    let path = request_line.next().unwrap_or_default().to_string();
    let headers: Vec<(String, String)> = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(key, value)| (key.trim().to_string(), value.trim().to_string()))
        .collect();
    let mut request = Request {
        method,
        path,
        headers,
        body: buffer[header_end + 4..].to_vec(),
    };
    if request
        .header("transfer-encoding")
        .is_some_and(|value| value.eq_ignore_ascii_case("chunked"))
    {
        return Err("chunked request bodies are not supported; send content-length".to_string());
    }
    let length: usize = request
        .header("content-length")
        .map_or(Ok(0), str::parse)
        .map_err(|_| "invalid content-length".to_string())?;
    if length > MAX_BODY_BYTES {
        return Err("request body too large".to_string());
    }
    while request.body.len() < length {
        let read = stream
            .read(&mut chunk)
            .await
            .map_err(|error| error.to_string())?;
        if read == 0 {
            return Err("connection closed before the body ended".to_string());
        }
        request.body.extend_from_slice(&chunk[..read]);
    }
    request.body.truncate(length);
    Ok(request)
}

async fn route(request: &Request, backend: &dyn Backend, token: &str) -> Response {
    let path = request.path.split('?').next().unwrap_or_default();
    if request.method == "GET" && path == "/healthz" {
        return Response::json(200, &json!({"ok": true}));
    }
    let presented = request
        .header("authorization")
        .and_then(|value| value.strip_prefix("Bearer "))
        .or_else(|| request.header("x-api-key"));
    if presented.map(str::trim) != Some(token) {
        return Response::error(
            401,
            "authentication_error",
            "missing or wrong bearer token (prime-agent auth-gateway token)",
        );
    }
    match (request.method.as_str(), path) {
        ("GET", "/v1/models") => {
            let data: Vec<Value> = backend
                .models()
                .into_iter()
                .map(|id| json!({"id": id, "object": "model", "owned_by": "prime-agent"}))
                .collect();
            Response::json(200, &json!({"object": "list", "data": data}))
        }
        ("POST", "/v1/chat/completions") => chat_completions(&request.body, backend).await,
        _ => Response::error(
            404,
            "not_found",
            &format!("no route {} {path}", request.method),
        ),
    }
}

/// What the request asks for beyond messages, kept for the reply.
struct ChatRequest {
    requested_model: String,
    stream: bool,
    json_reply: bool,
}

async fn chat_completions(body: &[u8], backend: &dyn Backend) -> Response {
    let body: Value = match serde_json::from_slice(body) {
        Ok(body) => body,
        Err(error) => return Response::error(400, "invalid_request_error", &error.to_string()),
    };
    let requested_model = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let resolved = match backend.resolve(&requested_model) {
        Ok(resolved) => resolved,
        Err(message) => return Response::error(400, "invalid_request_error", &message),
    };
    let (completion, request) = match build_completion(&body, resolved, requested_model) {
        Ok(built) => built,
        Err(message) => return Response::error(400, "invalid_request_error", &message),
    };
    let reply = match backend.complete(completion).await {
        Ok(reply) => reply,
        Err(message) => return Response::error(502, "upstream_error", &message),
    };
    if reply.stop_reason == StopReason::Error || reply.stop_reason == StopReason::Aborted {
        let message = reply
            .error_message
            .clone()
            .unwrap_or_else(|| "the provider returned an error".to_string());
        return Response::error(502, "upstream_error", &message);
    }
    completion_response(&reply, &request)
}

fn build_completion(
    body: &Value,
    resolved: ResolvedModel,
    requested_model: String,
) -> Result<(Completion, ChatRequest), String> {
    if body.get("tools").is_some_and(|tools| !tools.is_null()) {
        return Err("tools are not supported by the auth gateway".to_string());
    }
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or("messages must be an array")?;
    let model = resolved.model;
    let mut system: Vec<String> = Vec::new();
    let mut context_messages: Vec<Message> = Vec::new();
    for message in messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let text = message_text(message.get("content").unwrap_or(&Value::Null))?;
        match role {
            "system" | "developer" => system.push(text),
            "user" => context_messages.push(Message::User(UserMessage {
                content: UserContent::Text(text),
                timestamp: 0,
                rest: Map::default(),
            })),
            "assistant" => context_messages.push(Message::Assistant(AssistantMessage {
                content: vec![AssistantContentBlock::Text(TextContent {
                    text,
                    text_signature: None,
                    rest: Map::default(),
                })],
                api: model.api.clone(),
                provider: model.provider.clone(),
                model: model.id.clone(),
                response_model: None,
                response_id: None,
                diagnostics: None,
                usage: pa_types::ai::Usage::default(),
                stop_reason: StopReason::Stop,
                stop_reason_raw: None,
                error_message: None,
                timestamp: 0,
                rest: Map::default(),
            })),
            other => return Err(format!("message role {other:?} is not supported")),
        }
    }
    if context_messages.is_empty() {
        return Err("messages need at least one user message".to_string());
    }
    let format = body.get("response_format");
    let format_kind = format
        .and_then(|format| format.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("text");
    let json_reply = matches!(format_kind, "json_object" | "json_schema");
    match format_kind {
        "json_object" => system.push(
            "Reply with one JSON object only: no prose before or after it, no code fences."
                .to_string(),
        ),
        "json_schema" => {
            let schema = format
                .and_then(|format| format.get("json_schema"))
                .and_then(|schema| schema.get("schema"))
                .cloned()
                .unwrap_or(Value::Null);
            system.push(format!(
                "Reply with one JSON object only, matching this JSON schema: no prose, no code fences.\n{schema}"
            ));
        }
        _ => {}
    }
    let reasoning = match body.get("reasoning_effort").and_then(Value::as_str) {
        Some(effort) => {
            let level: ModelThinkingLevel = serde_json::from_value(json!(effort))
                .map_err(|_| format!("unknown reasoning_effort {effort:?}"))?;
            Some(pa_types::ai::clamp_thinking_level(&model, level))
                .filter(|level| *level != ModelThinkingLevel::Off)
        }
        None => None,
    };
    let max_tokens = body
        .get("max_completion_tokens")
        .or_else(|| body.get("max_tokens"))
        .and_then(Value::as_u64);
    let mut options = pa_ai::types::SimpleStreamOptions::from_base(pa_ai::types::StreamOptions {
        api_key: resolved.api_key,
        headers: resolved.headers,
        max_tokens,
        temperature: body.get("temperature").and_then(Value::as_f64),
        session_id: Some(uuid::Uuid::new_v4().to_string()),
        transport: Some(pa_ai::types::Transport::Sse),
        ..Default::default()
    });
    options.reasoning = reasoning;
    let context = Context {
        system_prompt: (!system.is_empty()).then(|| system.join("\n\n")),
        messages: context_messages,
        tools: None,
    };
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    Ok((
        Completion {
            model,
            context,
            options,
        },
        ChatRequest {
            requested_model,
            stream,
            json_reply,
        },
    ))
}

/// A message's text: a string, or an array of `text` parts.
fn message_text(content: &Value) -> Result<String, String> {
    match content {
        Value::String(text) => Ok(text.clone()),
        Value::Null => Ok(String::new()),
        Value::Array(parts) => {
            let mut texts = Vec::new();
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("text") => texts.push(
                        part.get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                    ),
                    other => {
                        return Err(format!(
                            "content part {other:?} is not supported; send text only"
                        ))
                    }
                }
            }
            Ok(texts.join("\n"))
        }
        _ => Err("message content must be a string or an array of parts".to_string()),
    }
}

/// A fenced reply (```json ... ```) unwrapped, for callers that asked for JSON.
fn unfence(text: &str) -> &str {
    let trimmed = text.trim();
    let Some(rest) = trimmed.strip_prefix("```") else {
        return text;
    };
    let Some(body) = rest.strip_suffix("```") else {
        return text;
    };
    body.split_once('\n')
        .map_or(body, |(_, inner)| inner)
        .trim()
}

fn completion_response(reply: &AssistantMessage, request: &ChatRequest) -> Response {
    let text: String = reply
        .content
        .iter()
        .filter_map(|block| match block {
            AssistantContentBlock::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("");
    let text = if request.json_reply {
        unfence(&text).to_string()
    } else {
        text
    };
    let finish_reason = if reply.stop_reason == StopReason::Length {
        "length"
    } else {
        "stop"
    };
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let created = reply.timestamp / 1000;
    let prompt_tokens = reply.usage.input + reply.usage.cache_read + reply.usage.cache_write;
    let usage = json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": reply.usage.output,
        "total_tokens": prompt_tokens + reply.usage.output,
    });
    if request.stream {
        let chunk = |delta: Value, finish: Value| {
            json!({
                "id": id, "object": "chat.completion.chunk", "created": created,
                "model": request.requested_model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            })
        };
        let body = format!(
            "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
            chunk(json!({"role": "assistant", "content": text}), Value::Null),
            chunk(json!({}), json!(finish_reason)),
        );
        return Response {
            status: 200,
            content_type: "text/event-stream",
            body: body.into_bytes(),
        };
    }
    Response::json(
        200,
        &json!({
            "id": id,
            "object": "chat.completion",
            "created": created,
            "model": request.requested_model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }],
            "usage": usage,
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> Model {
        serde_json::from_value(json!({
            "id": "claude-test", "name": "t", "api": "anthropic-messages", "provider": "anthropic",
            "baseUrl": "http://localhost", "reasoning": false, "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 1000, "maxTokens": 100
        }))
        .unwrap()
    }

    /// Answers every completion with the system prompt and last user text it
    /// was sent, so the test sees what reached the provider seam.
    struct EchoBackend {
        api_key: Mutex<Option<String>>,
    }

    impl Backend for EchoBackend {
        fn models(&self) -> Vec<String> {
            vec!["anthropic/claude-test".to_string()]
        }

        fn resolve(&self, requested: &str) -> Result<ResolvedModel, String> {
            find_model(&[&model()], requested).map(|model| ResolvedModel {
                model: model.clone(),
                api_key: Some("sk-ant-oat-subscription".to_string()),
                headers: None,
            })
        }

        fn complete(&self, completion: Completion) -> CompletionFuture {
            *self.api_key.lock().unwrap() = completion.options.base.api_key.clone();
            let system = completion.context.system_prompt.clone().unwrap_or_default();
            let user = match completion.context.messages.last() {
                Some(Message::User(UserMessage {
                    content: UserContent::Text(text),
                    ..
                })) => text.clone(),
                _ => String::new(),
            };
            let model = completion.model;
            Box::pin(async move {
                Ok(AssistantMessage {
                    content: vec![AssistantContentBlock::Text(TextContent {
                        text: format!("```json\n{{\"system\":{system:?},\"user\":{user:?}}}\n```"),
                        text_signature: None,
                        rest: Map::default(),
                    })],
                    api: model.api,
                    provider: model.provider,
                    model: model.id,
                    response_model: None,
                    response_id: None,
                    diagnostics: None,
                    usage: pa_types::ai::Usage::default(),
                    stop_reason: StopReason::Stop,
                    stop_reason_raw: None,
                    error_message: None,
                    timestamp: 0,
                    rest: Map::default(),
                })
            })
        }
    }

    async fn post(addr: SocketAddr, token: Option<&str>, body: &Value) -> (u16, Value) {
        let body = body.to_string();
        let auth = token.map_or(String::new(), |token| {
            format!("authorization: Bearer {token}\r\n")
        });
        let mut stream = TcpStream::connect(addr).await.unwrap();
        stream
            .write_all(
                format!(
                    "POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\n{auth}content-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
                    body.len()
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        let mut raw = String::new();
        stream.read_to_string(&mut raw).await.unwrap();
        let (head, body) = raw.split_once("\r\n\r\n").unwrap();
        let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
        (status, serde_json::from_str(body).unwrap())
    }

    #[tokio::test]
    async fn a_chat_completion_runs_on_the_logged_in_credential_behind_the_bearer_token() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let backend = Arc::new(EchoBackend {
            api_key: Mutex::new(None),
        });
        tokio::spawn(serve(listener, backend.clone(), "secret".to_string()));
        let request = json!({
            "model": "anthropic/claude-test",
            "messages": [
                {"role": "system", "content": "You grade rubrics."},
                {"role": "user", "content": [{"type": "text", "text": "Grade this."}]}
            ],
            "response_format": {"type": "json_object"}
        });

        let (status, _) = post(addr, Some("wrong"), &request).await;
        assert_eq!(status, 401);
        assert!(backend.api_key.lock().unwrap().is_none());

        let (status, reply) = post(addr, Some("secret"), &request).await;
        assert_eq!(status, 200, "{reply}");
        assert_eq!(
            backend.api_key.lock().unwrap().as_deref(),
            Some("sk-ant-oat-subscription")
        );
        let content: Value =
            serde_json::from_str(reply["choices"][0]["message"]["content"].as_str().unwrap())
                .expect("a JSON reply comes back unfenced");
        assert_eq!(content["user"], "Grade this.");
        assert!(content["system"]
            .as_str()
            .unwrap()
            .starts_with("You grade rubrics.\n\nReply with one JSON object only"));
    }

    #[test]
    fn serve_refuses_an_address_off_this_machine() {
        let error = parse_bind(&["--bind".to_string(), "0.0.0.0:4100".to_string()]).unwrap_err();
        assert!(error.contains("not a loopback address"), "{error}");
    }
}

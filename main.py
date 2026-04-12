"""
Anthropic-compatible /v1/messages proxy - Modular Implementation.

Translates Anthropic Messages API to OpenAI-compatible requests for
alternative backends like Kimi K2.5 via Baseten or other providers.
"""

import json
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Optional, Union

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import CONFIG, load_model_registry
from errors import (
    AuthenticationError,
    ProxyError,
    RateLimitError,
    UpstreamError,
    ValidationError,
)
from health import get_health_checker
from logging_config import get_logger, set_request_id, setup_logging
from middleware import MetricsMiddleware, RequestLoggingMiddleware
from models.registry import ModelHandlerRegistry
from performance import ConnectionPool
from rate_limit import get_rate_limiter
from retry import get_circuit_breaker
from security import MessageRequest, validate_request_size

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------------------


def setup_error_handlers(app: FastAPI) -> None:
    """Setup global error handlers for the application."""

    @app.exception_handler(ProxyError)
    async def handle_proxy_error(request: Request, exc: ProxyError) -> JSONResponse:
        """Handle custom proxy errors."""
        logger.error(
            "proxy_error",
            error_type=exc.error_type,
            status_code=exc.status_code,
            message=exc.message,
        )
        return JSONResponse(
            content=exc.to_dict(),
            status_code=exc.status_code,
        )

    @app.exception_handler(Exception)
    async def handle_generic_error(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected errors."""
        logger.exception("unexpected_error", error_type=type(exc).__name__)
        error = ProxyError(
            message="An unexpected error occurred",
            status_code=500,
            error_type="internal_error",
        )
        return JSONResponse(
            content=error.to_dict(),
            status_code=500,
        )


# ---------------------------------------------------------------------------
# Request Processing
# ---------------------------------------------------------------------------


def check_proxy_auth(authorization: Optional[str], x_api_key: Optional[str]) -> bool:
    """Validate proxy authentication."""
    if not CONFIG.proxy_auth_key:
        return True
    return (authorization == f"Bearer {CONFIG.proxy_auth_key}") or (
        x_api_key == CONFIG.proxy_auth_key
    )


def get_model_for_request(requested_model: str) -> tuple[str, Any]:
    """
    Get the appropriate model ID and handler for the requested model.

    Args:
        requested_model: The model ID from the request

    Returns:
        Tuple of (model_id, handler)
    """
    registry = load_model_registry()
    capability = registry.get_model(requested_model)

    if capability:
        model_id = capability.model_id
        handler = ModelHandlerRegistry.get_handler_instance(requested_model, capability)
    else:
        # Use default model from config
        model_id = CONFIG.default_model
        handler = ModelHandlerRegistry.get_handler_instance(model_id)

    return model_id, handler


# ---------------------------------------------------------------------------
# Streaming Response
# ---------------------------------------------------------------------------


async def stream_anthropic_response(
    upstream_response: httpx.Response,
    request_id: str,
    model: str,
) -> AsyncGenerator[str, None]:
    """
    Stream Anthropic-formatted SSE events from upstream OpenAI response.

    Args:
        upstream_response: The httpx streaming response
        request_id: Unique request ID
        model: Model ID

    Yields:
        SSE formatted strings
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Send message_start
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': model, 'content': [], 'stop_reason': None}})}\n\n"

    content_index = 0
    text_block_open = False
    tool_calls_buffer: dict[int, dict] = {}
    output_tokens = 0
    finish_reason = None

    async for line in upstream_response.aiter_lines():
        if not line or not line.startswith("data:"):
            continue

        raw = line[5:].strip()
        if raw == "[DONE]":
            break

        try:
            obj = json.loads(raw)
            choices = obj.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            content = delta.get("content")
            tool_calls = delta.get("tool_calls")
            finish_reason = choices[0].get("finish_reason") or finish_reason

            # Handle tool calls
            if tool_calls:
                if text_block_open:
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_index})}\n\n"
                    text_block_open = False
                    content_index += 1

                for tc in tool_calls:
                    idx = tc.get("index", 0)
                    if idx not in tool_calls_buffer:
                        tool_calls_buffer[idx] = {
                            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                            "name": tc.get("function", {}).get("name", ""),
                            "arguments": "",
                        }
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        tool_calls_buffer[idx]["name"] = fn["name"]
                    if fn.get("arguments"):
                        tool_calls_buffer[idx]["arguments"] += fn["arguments"]

                output_tokens += 1
                continue

            # Handle text content
            if content:
                if not text_block_open:
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                    text_block_open = True

                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': content_index, 'delta': {'type': 'text_delta', 'text': content}})}\n\n"
                output_tokens += 1

        except json.JSONDecodeError:
            continue

    # Close text block if open
    if text_block_open:
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_index})}\n\n"
        content_index += 1

    # Emit tool calls
    if tool_calls_buffer:
        for idx, tc in sorted(tool_calls_buffer.items()):
            try:
                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                args = {"raw": tc["arguments"]}

            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_index, 'content_block': {'type': 'tool_use', 'id': tc['id'], 'name': tc['name'], 'input': {}}})}\n\n"
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': content_index, 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(args)}})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_index})}\n\n"
            content_index += 1

        stop_reason = "tool_use"
    else:
        stop_reason = finish_reason or "end_turn"

    # Send message_delta and message_stop
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason}, 'usage': {'output_tokens': output_tokens}})}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


# ---------------------------------------------------------------------------
# Non-streaming Response
# ---------------------------------------------------------------------------


def build_anthropic_message(
    upstream_response: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    """
    Convert OpenAI response to Anthropic format.

    Args:
        upstream_response: The OpenAI response dict
        model: Model ID

    Returns:
        Anthropic-formatted response
    """
    choices = upstream_response.get("choices", [])
    usage = upstream_response.get("usage", {})

    if not choices:
        return {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": ""}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
        }

    choice = choices[0]
    message = choice.get("message", {})
    content = message.get("content", "")
    tool_calls = message.get("tool_calls", [])

    # Build content blocks
    blocks: list[dict[str, Any]] = []

    # Add text content if present
    if content:
        blocks.append({"type": "text", "text": content})

    # Add tool calls
    for tc in tool_calls:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {"raw": fn.get("arguments", "")}

        blocks.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
            "name": fn.get("name", ""),
            "input": args,
        })

    if not blocks:
        blocks.append({"type": "text", "text": ""})

    has_tool = any(b["type"] == "tool_use" for b in blocks)
    finish_reason = choice.get("finish_reason")

    # Map finish reason
    if has_tool:
        stop_reason = "tool_use"
    elif finish_reason == "stop":
        stop_reason = "end_turn"
    elif finish_reason == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = finish_reason or "end_turn"

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    setup_logging()
    logger.info(
        "proxy_starting",
        default_model=CONFIG.default_model,
        baseten_url=CONFIG.baseten_base_url,
    )

    # Initialize connection pool
    ConnectionPool.get_client()

    yield

    # Shutdown
    logger.info("proxy_shutting_down")
    await ConnectionPool.close()


# Create FastAPI application
app = FastAPI(
    title="Anthropic Proxy",
    description="Anthropic-to-OpenAI API proxy for model-agnostic LLM access",
    version="1.0.0",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(MetricsMiddleware)

# Setup error handlers
setup_error_handlers(app)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    checker = get_health_checker()
    status = await checker.check(include_upstream=False)
    return status.to_dict()


@app.get("/health/detailed")
async def detailed_health_check() -> dict[str, Any]:
    """Detailed health check with upstream validation."""
    checker = get_health_checker()
    status = await checker.check(include_upstream=True)
    return status.to_dict()


@app.post("/v1/messages", response_model=None)
async def messages(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
    x_request_id: Optional[str] = Header(default=None),
):
    """
    Anthropic /v1/messages endpoint.

    Translates Anthropic format to OpenAI format and proxies to upstream provider.
    """
    # Set request ID for logging context
    request_id = set_request_id(x_request_id)

    try:
        # Check authentication
        if not check_proxy_auth(authorization, x_api_key):
            raise AuthenticationError("Invalid API key")

        # Check Baseten API key is configured
        if not CONFIG.baseten_api_key:
            raise UpstreamError(
                "BASETEN_API_KEY not configured",
                provider="baseten",
            )

        # Read and validate request body
        body_bytes = await request.body()
        validate_request_size(body_bytes)

        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON body: {e}")

        # Validate request structure
        validated_request = MessageRequest(**body)

        # Check rate limits
        rate_limiter = get_rate_limiter()
        api_key = x_api_key or authorization or "anonymous"
        allowed, headers = rate_limiter.check_rate_limit(
            api_key, validated_request.model
        )

        if not allowed:
            logger.warning(
                "proxy_rate_limit_rejected",
                source="proxy_local",
                api_key_prefix=api_key[:8] if len(api_key) > 8 else api_key,
                model=validated_request.model,
                retry_after=headers.get("Retry-After"),
            )
            raise RateLimitError(
                "Rate limit exceeded",
                retry_after=int(headers.get("Retry-After", 60)),
                limit=int(headers["X-RateLimit-Limit"]) if headers.get("X-RateLimit-Limit") else None,
                remaining=0,
                details={"source": "proxy_local"},
            )

        # Get model handler
        model_id, handler = get_model_for_request(validated_request.model)

        # Prepare request for upstream
        messages = handler.prepare_messages(validated_request.messages)
        tools = handler.prepare_tools(
            [tool.model_dump() for tool in validated_request.tools]
            if validated_request.tools
            else None
        )
        system = handler.prepare_system(validated_request.system)

        # Build upstream payload
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "stream": validated_request.stream,
        }

        if system:
            payload["messages"] = [{"role": "system", "content": system}] + messages

        if validated_request.max_tokens:
            payload["max_tokens"] = validated_request.max_tokens

        if validated_request.temperature is not None:
            payload["temperature"] = validated_request.temperature

        if validated_request.top_p is not None:
            payload["top_p"] = validated_request.top_p

        if tools:
            payload["tools"] = tools

        if validated_request.stream:
            payload["stream_options"] = {"include_usage": True}

        # Get circuit breaker
        circuit_breaker = get_circuit_breaker()

        # Make upstream request
        url = f"{CONFIG.baseten_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {CONFIG.baseten_api_key}",
            "Content-Type": "application/json",
        }

        async def make_request():
            client = ConnectionPool.get_client()
            if validated_request.stream:
                return client.stream("POST", url, headers=headers, json=payload)
            else:
                return await client.post(url, headers=headers, json=payload)

        try:
            response = await circuit_breaker.call(make_request)
        except httpx.HTTPStatusError as e:
            # Upstream returned an explicit HTTP error code.
            # Preserve 429s as RateLimitError so Claude can honour Retry-After
            # and so logs clearly distinguish proxy-local vs upstream throttling.
            upstream_status = e.response.status_code
            if upstream_status == 429:
                retry_after_raw = e.response.headers.get("Retry-After", "60")
                try:
                    retry_after = int(retry_after_raw)
                except ValueError:
                    retry_after = 60
                logger.warning(
                    "upstream_rate_limit",
                    source="upstream",
                    provider="baseten",
                    retry_after=retry_after,
                )
                raise RateLimitError(
                    "Upstream rate limit exceeded",
                    retry_after=retry_after,
                    details={"source": "upstream", "provider": "baseten"},
                ) from e
            raise UpstreamError(
                f"Upstream returned HTTP {upstream_status}",
                upstream_status=upstream_status,
                provider="baseten",
            ) from e
        except Exception as e:
            raise UpstreamError(
                f"Upstream request failed: {e}",
                provider="baseten",
            ) from e

        if validated_request.stream:
            # Return streaming response
            async def stream_generator():
                async with response as resp:
                    resp.raise_for_status()
                    async for event in stream_anthropic_response(resp, request_id, model_id):
                        yield event

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers=headers,
            )
        else:
            # Return non-streaming response
            response.raise_for_status()
            upstream_data = response.json()

            # Parse with handler
            parsed = handler.parse_response(upstream_data)

            # Build Anthropic response
            anthropic_response = build_anthropic_message(parsed, model_id)

            return JSONResponse(
                content=anthropic_response,
                headers=headers,
            )

    except ProxyError:
        raise
    except Exception as e:
        logger.exception("unexpected_error_processing_request")
        raise ProxyError(
            message=f"Request processing failed: {e}",
            status_code=500,
            error_type="internal_error",
        ) from e


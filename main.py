"""
Anthropic-compatible /v1/messages proxy - Modular Implementation.

Translates Anthropic Messages API to OpenAI-compatible requests for
alternative backends like Kimi K2.5 via Baseten or other providers.
"""

import json
import re
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
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
        # Use `error_message` to avoid shadowing the positional `message` param of logger.error()
        logger.error(
            "proxy_error",
            error_type=exc.error_type,
            status_code=exc.status_code,
            error_message=exc.message,
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
        # Requested model not in registry — resolve the default model through the
        # registry so we get the real upstream model_id (e.g. "moonshotai/Kimi-K2.5")
        # rather than sending the alias (e.g. "kimi-k2.5") which the upstream won't recognise.
        default_capability = registry.get_model(CONFIG.default_model)
        if default_capability:
            model_id = default_capability.model_id
            handler = ModelHandlerRegistry.get_handler_instance(
                CONFIG.default_model, default_capability
            )
        else:
            model_id = CONFIG.default_model
            handler = ModelHandlerRegistry.get_handler_instance(model_id)

    return model_id, handler


# ---------------------------------------------------------------------------
# Streaming: helpers and state machine (ported from app_optimized.py)
# ---------------------------------------------------------------------------


def create_sse_event(event: str, data: dict[str, Any]) -> str:
    """Format a single SSE event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def map_finish_reason(finish: Optional[str]) -> str:
    """Map OpenAI finish_reason to Anthropic stop_reason."""
    mapping = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}
    return mapping.get(finish or "", finish or "end_turn")


_KIMI_CALL_RE = re.compile(
    r'<\|tool_call_begin\|>(?:functions\.)?(\w+)(?::\d+)?'
    r'<\|tool_call_argument_begin\|>(.*?)<\|tool_call_end\|>',
    re.DOTALL,
)


def parse_kimi_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Extract tool calls from Kimi's text-embedded syntax."""
    clean_parts = text.split("<|tool_calls_section_begin|>")
    before = clean_parts[0] if clean_parts else ""
    after_parts = text.split("<|tool_calls_section_end|>")
    after = after_parts[-1] if len(after_parts) > 1 else ""
    clean = (before + after).strip()

    tool_uses = [
        {
            "type": "tool_use",
            "id": f"toolu_{uuid.uuid4().hex[:24]}",
            "name": m.group(1),
            "input": json.loads(m.group(2).strip()) if m.group(2).strip() else {"raw": m.group(2).strip()},
        }
        for m in _KIMI_CALL_RE.finditer(text)
    ]

    return clean, tool_uses


@dataclass(frozen=True)
class StreamState:
    """Immutable streaming state container."""
    content_index: int
    text_block_open: bool
    accumulated_text: tuple[str, ...]
    structured_tcs: tuple[tuple[int, dict[str, str]], ...]
    kimi_buf: tuple[str, ...]
    in_kimi_section: bool
    output_tokens: int
    finish_reason: Optional[str]


@dataclass(frozen=True)
class DeltaUpdate:
    """Immutable delta update from OpenAI stream."""
    content: Optional[str]
    tool_calls: Optional[list[dict]]
    finish_reason: Optional[str]


def parse_delta(obj: dict[str, Any]) -> Optional[DeltaUpdate]:
    """Extract delta from an OpenAI stream chunk."""
    choices = obj.get("choices")
    if not choices:
        return None
    delta = choices[0].get("delta", {})
    return DeltaUpdate(
        content=delta.get("content"),
        tool_calls=delta.get("tool_calls") if delta.get("tool_calls") else None,
        finish_reason=choices[0].get("finish_reason"),
    )


def transition_state(state: StreamState, delta: DeltaUpdate) -> tuple[StreamState, list[str]]:
    """Compute next state and SSE events from a delta. Pure function."""
    events: list[str] = []
    new_state = state

    # --- native structured tool_calls ---
    if delta.tool_calls:
        if state.text_block_open:
            events.append(create_sse_event("content_block_stop", {"type": "content_block_stop", "index": state.content_index}))
            new_state = StreamState(
                content_index=state.content_index + 1,
                text_block_open=False,
                accumulated_text=state.accumulated_text,
                structured_tcs=state.structured_tcs,
                kimi_buf=state.kimi_buf,
                in_kimi_section=state.in_kimi_section,
                output_tokens=state.output_tokens,
                finish_reason=state.finish_reason,
            )

        for tc in delta.tool_calls:
            idx = tc.get("index", 0)
            existing = dict(new_state.structured_tcs)
            if idx not in existing:
                existing[idx] = {
                    "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": "",
                }
            fn = tc.get("function", {})
            if fn.get("name"):
                existing[idx]["name"] = fn["name"]
            if fn.get("arguments"):
                existing[idx]["arguments"] += fn["arguments"]

            new_state = StreamState(
                content_index=new_state.content_index,
                text_block_open=new_state.text_block_open,
                accumulated_text=new_state.accumulated_text,
                structured_tcs=tuple(existing.items()),
                kimi_buf=new_state.kimi_buf,
                in_kimi_section=new_state.in_kimi_section,
                output_tokens=new_state.output_tokens + 1,
                finish_reason=new_state.finish_reason,
            )

        return new_state, events

    # --- text content ---
    chunk = delta.content
    if not chunk:
        if delta.finish_reason:
            return StreamState(
                content_index=new_state.content_index,
                text_block_open=new_state.text_block_open,
                accumulated_text=new_state.accumulated_text,
                structured_tcs=new_state.structured_tcs,
                kimi_buf=new_state.kimi_buf,
                in_kimi_section=new_state.in_kimi_section,
                output_tokens=new_state.output_tokens,
                finish_reason=delta.finish_reason,
            ), events
        return new_state, events

    # --- detect Kimi embedded tool section ---
    if "<|tool_calls_section_begin|>" in chunk:
        before = chunk.split("<|tool_calls_section_begin|>")[0]
        if before.strip() and not new_state.text_block_open:
            events.append(create_sse_event("content_block_start", {
                "type": "content_block_start",
                "index": new_state.content_index,
                "content_block": {"type": "text", "text": ""},
            }))
            new_state = StreamState(
                content_index=new_state.content_index,
                text_block_open=True,
                accumulated_text=new_state.accumulated_text,
                structured_tcs=new_state.structured_tcs,
                kimi_buf=new_state.kimi_buf,
                in_kimi_section=new_state.in_kimi_section,
                output_tokens=new_state.output_tokens,
                finish_reason=new_state.finish_reason,
            )

        if before.strip():
            events.append(create_sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": new_state.content_index,
                "delta": {"type": "text_delta", "text": before},
            }))
            new_state = StreamState(
                content_index=new_state.content_index,
                text_block_open=new_state.text_block_open,
                accumulated_text=new_state.accumulated_text + (before,),
                structured_tcs=new_state.structured_tcs,
                kimi_buf=new_state.kimi_buf + (chunk,),
                in_kimi_section=True,
                output_tokens=new_state.output_tokens + 1,
                finish_reason=new_state.finish_reason,
            )
        else:
            new_state = StreamState(
                content_index=new_state.content_index,
                text_block_open=new_state.text_block_open,
                accumulated_text=new_state.accumulated_text,
                structured_tcs=new_state.structured_tcs,
                kimi_buf=new_state.kimi_buf + (chunk,),
                in_kimi_section=True,
                output_tokens=new_state.output_tokens + 1,
                finish_reason=new_state.finish_reason,
            )
        return new_state, events

    # --- inside Kimi section: buffer silently ---
    if new_state.in_kimi_section:
        new_kimi_buf = new_state.kimi_buf + (chunk,)
        new_in_kimi = "<|tool_calls_section_end|>" not in chunk
        return StreamState(
            content_index=new_state.content_index,
            text_block_open=new_state.text_block_open,
            accumulated_text=new_state.accumulated_text,
            structured_tcs=new_state.structured_tcs,
            kimi_buf=new_kimi_buf,
            in_kimi_section=new_in_kimi,
            output_tokens=new_state.output_tokens + 1,
            finish_reason=new_state.finish_reason,
        ), events

    # --- normal text: emit immediately ---
    if not new_state.text_block_open:
        events.append(create_sse_event("content_block_start", {
            "type": "content_block_start",
            "index": new_state.content_index,
            "content_block": {"type": "text", "text": ""},
        }))
        new_state = StreamState(
            content_index=new_state.content_index,
            text_block_open=True,
            accumulated_text=new_state.accumulated_text,
            structured_tcs=new_state.structured_tcs,
            kimi_buf=new_state.kimi_buf,
            in_kimi_section=new_state.in_kimi_section,
            output_tokens=new_state.output_tokens,
            finish_reason=new_state.finish_reason,
        )

    events.append(create_sse_event("content_block_delta", {
        "type": "content_block_delta",
        "index": new_state.content_index,
        "delta": {"type": "text_delta", "text": chunk},
    }))

    return StreamState(
        content_index=new_state.content_index,
        text_block_open=new_state.text_block_open,
        accumulated_text=new_state.accumulated_text + (chunk,),
        structured_tcs=new_state.structured_tcs,
        kimi_buf=new_state.kimi_buf,
        in_kimi_section=new_state.in_kimi_section,
        output_tokens=new_state.output_tokens + 1,
        finish_reason=new_state.finish_reason,
    ), events


async def stream_anthropic_response(
    upstream_response: httpx.Response,
    request_id: str,
    model: str,
) -> AsyncGenerator[str, None]:
    """
    Stream Anthropic-formatted SSE events from upstream OpenAI response.
    Uses immutable state transitions ported from app_optimized.py.
    """
    msg_id = f"msg_{uuid.uuid4().hex}"

    yield create_sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
        },
    })

    state = StreamState(
        content_index=0,
        text_block_open=False,
        accumulated_text=(),
        structured_tcs=(),
        kimi_buf=(),
        in_kimi_section=False,
        output_tokens=0,
        finish_reason=None,
    )

    async for line in upstream_response.aiter_lines():
        if not line or not line.startswith("data:"):
            continue

        raw = line[5:].strip()
        if raw == "[DONE]":
            break

        try:
            obj = json.loads(raw)
            delta = parse_delta(obj)
            if delta is None:
                continue

            state, events = transition_state(state, delta)
            for event in events:
                yield event

        except (json.JSONDecodeError, KeyError):
            continue

    # Finalize: close text block, emit tool_use blocks, send stop events
    if state.text_block_open:
        yield create_sse_event("content_block_stop", {"type": "content_block_stop", "index": state.content_index})

    # Build tool blocks from structured deltas or Kimi embedded-text fallback
    tool_blocks: list[dict[str, Any]] = []
    if state.structured_tcs:
        for idx, tc in sorted(state.structured_tcs, key=lambda x: x[0]):
            try:
                inp = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                inp = {"raw": tc["arguments"]}
            tool_blocks.append({"id": tc["id"], "name": tc["name"], "input": inp})
    elif state.kimi_buf:
        _, parsed = parse_kimi_tool_calls("".join(state.kimi_buf))
        for p in parsed:
            tool_blocks.append({"id": p["id"], "name": p["name"], "input": p["input"]})

    content_idx = state.content_index + (1 if state.text_block_open else 0)
    for tb in tool_blocks:
        yield create_sse_event("content_block_start", {
            "type": "content_block_start",
            "index": content_idx,
            "content_block": {"type": "tool_use", "id": tb["id"], "name": tb["name"], "input": {}},
        })
        yield create_sse_event("content_block_delta", {
            "type": "content_block_delta",
            "index": content_idx,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps(tb["input"])},
        })
        yield create_sse_event("content_block_stop", {"type": "content_block_stop", "index": content_idx})
        content_idx += 1

    stop = "tool_use" if tool_blocks else map_finish_reason(state.finish_reason)

    yield create_sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop},
        "usage": {"output_tokens": state.output_tokens},
    })
    yield create_sse_event("message_stop", {"type": "message_stop"})


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

    stop_reason = "tool_use" if has_tool else map_finish_reason(finish_reason)

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

        # Convert Anthropic tool_choice format to OpenAI format.
        # Anthropic: {"type": "auto"} / {"type": "any"} / {"type": "tool", "name": "X"}
        # OpenAI:    "auto" / "required" / {"type": "function", "function": {"name": "X"}}
        if validated_request.tool_choice is not None:
            tc = validated_request.tool_choice
            tc_type = tc.get("type") if isinstance(tc, dict) else tc
            if tc_type in ("auto", "none"):
                payload["tool_choice"] = tc_type
            elif tc_type == "any":
                payload["tool_choice"] = "required"
            elif tc_type == "tool" and isinstance(tc, dict):
                payload["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tc.get("name", "")},
                }
            else:
                payload["tool_choice"] = tc

        if validated_request.stream:
            payload["stream_options"] = {"include_usage": True, "continuous_usage_stats": True}

        # Get circuit breaker
        circuit_breaker = get_circuit_breaker()

        # Make upstream request
        url = f"{CONFIG.baseten_base_url}/chat/completions"
        # Use a separate variable to avoid shadowing the route's `headers` parameter
        # and to prevent upstream auth credentials from leaking into the client response.
        upstream_headers = {
            "Authorization": f"Bearer {CONFIG.baseten_api_key}",
            "Content-Type": "application/json",
        }

        async def make_request():
            client = ConnectionPool.get_client()
            if validated_request.stream:
                return client.stream("POST", url, headers=upstream_headers, json=payload)
            else:
                return await client.post(url, headers=upstream_headers, json=payload)

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
            # Return streaming response.
            # raise_for_status() is called inside the context manager so that
            # upstream 4xx/5xx during streaming are translated to UpstreamError
            # before any response bytes are sent to the client.
            async def stream_generator():
                async with response as resp:
                    try:
                        resp.raise_for_status()
                    except httpx.HTTPStatusError as e:
                        upstream_status = e.response.status_code
                        if upstream_status == 429:
                            retry_after_raw = e.response.headers.get("Retry-After", "60")
                            try:
                                retry_after = int(retry_after_raw)
                            except ValueError:
                                retry_after = 60
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
                    async for event in stream_anthropic_response(resp, request_id, model_id):
                        yield event

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
            )
        else:
            # Return non-streaming response.
            # raise_for_status() must be inside its own try/except so upstream
            # 4xx errors are surfaced as UpstreamError rather than a generic 500.
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                upstream_status = e.response.status_code
                if upstream_status == 429:
                    retry_after_raw = e.response.headers.get("Retry-After", "60")
                    try:
                        retry_after = int(retry_after_raw)
                    except ValueError:
                        retry_after = 60
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

            upstream_data = response.json()

            # Parse with handler
            parsed = handler.parse_response(upstream_data)

            # Build Anthropic response
            anthropic_response = build_anthropic_message(parsed, model_id)

            return JSONResponse(content=anthropic_response)

    except ProxyError:
        raise
    except Exception as e:
        logger.exception("unexpected_error_processing_request")
        raise ProxyError(
            message=f"Request processing failed: {e}",
            status_code=500,
            error_type="internal_error",
        ) from e


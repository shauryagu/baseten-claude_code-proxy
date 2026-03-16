"""
Anthropic-compatible /v1/messages proxy → Baseten OpenAI API (e.g. Kimi K2.5).

Translates tool definitions, messages, and tool calls bidirectionally so that
Claude Code CLI can use Kimi K2.5 as a drop-in replacement for Claude.
"""

import json
import os
import re
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import reduce
from typing import Any

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ---------------------------------------------------------------------------
# Configuration (loaded at module init - immutable after load)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    """Configuration loaded from environment variables."""
    baseten_api_key: str | None
    openai_base_url: str
    target_model: str
    proxy_auth_key: str | None
    log_path: str

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            baseten_api_key=os.getenv("BASETEN_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://inference.baseten.co/v1").rstrip("/"),
            target_model=os.getenv("TARGET_MODEL", "moonshotai/Kimi-K2.5"),
            proxy_auth_key=os.getenv("PROXY_AUTH_KEY"),
            log_path="/Users/shauryagunderia/PersonalProjects/AiExplore/.cursor/debug-7c7eab.log",
        )

CONFIG = Config.from_env()

# ---------------------------------------------------------------------------
# Pure Data Transformations
# ---------------------------------------------------------------------------

def extract_text(content: Any) -> str:
    """Flatten Anthropic content blocks to a plain string."""
    match content:
        case str():
            return content
        case list():
            return "".join(
                block.get("text", "") if isinstance(block, dict) and block.get("type") == "text" else str(block)
                for block in content
            )
        case _:
            return str(content)


def parse_content_block(block: dict) -> dict | None:
    """Parse a content block into the internal representation."""
    match block.get("type"):
        case "text":
            return {"type": "text", "text": block.get("text", "")}
        case "tool_use":
            return {
                "type": "tool_use",
                "id": block.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                "name": block["name"],
                "input": block.get("input", {}),
            }
        case _:
            return None


def convert_assistant_content(content: Any) -> tuple[str | None, list[dict]]:
    """Extract text and tool calls from an assistant message."""
    if not isinstance(content, list):
        return (extract_text(content), [])

    text_parts: list[str] = []
    tool_uses: list[dict] = []

    for block in content:
        if not isinstance(block, dict):
            text_parts.append(str(block))
            continue

        match block.get("type"):
            case "text":
                text_parts.append(block.get("text", ""))
            case "tool_use":
                tool_uses.append({
                    "id": block.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block.get("input", {})),
                    },
                })

    text = "".join(text_parts).strip()
    return (text if text else None, tool_uses)


def convert_user_content(content: Any) -> tuple[str | None, list[dict]]:
    """Extract text and tool results from a user message."""
    if not isinstance(content, list):
        return (extract_text(content), [])

    text_parts: list[str] = []
    tool_results: list[dict] = []

    for block in content:
        if not isinstance(block, dict):
            text_parts.append(str(block))
            continue

        match block.get("type"):
            case "text":
                text_parts.append(block.get("text", ""))
            case "tool_result":
                rc = block.get("content", "")
                if isinstance(rc, list):
                    rc = "".join(
                        b.get("text", "") for b in rc
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": str(rc),
                })

    text = "".join(text_parts).strip()
    return (text if text else None, tool_results)


# ---------------------------------------------------------------------------
# Function Composition for Message Conversion
# ---------------------------------------------------------------------------

def compose(*functions: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Compose functions right-to-left: compose(f, g)(x) == f(g(x))."""
    def composed(x: Any) -> Any:
        return reduce(lambda v, f: f(v), reversed(functions), x)
    return composed


def filter_empty_messages(messages: list[dict]) -> list[dict]:
    """Remove messages that have no content."""
    return [m for m in messages if m.get("content") or m.get("role") == "tool"]


def convert_single_message(msg: dict) -> list[dict]:
    """Convert one Anthropic message to one or more OpenAI messages."""
    role = msg.get("role")
    content = msg.get("content")

    match role:
        case "assistant":
            text, tool_calls = convert_assistant_content(content)
            result: dict = {"role": "assistant"}
            if text:
                result["content"] = text
            if tool_calls:
                result["tool_calls"] = tool_calls
            return [result]

        case "user":
            text, tool_results = convert_user_content(content)
            messages = []
            # Tool results must come before user text
            messages.extend(tool_results)
            if text:
                messages.append({"role": "user", "content": text})
            return messages

        case _:
            return [{"role": role, "content": extract_text(content)}]


def flatten_messages(nested: list[list[dict]]) -> list[dict]:
    """Flatten a nested list of message lists into a single list."""
    return [msg for sublist in nested for msg in sublist]


# Composed conversion pipeline
anthropic_to_openai_messages = compose(
    flatten_messages,
    lambda msgs: [convert_single_message(m) for m in msgs],
)


# ---------------------------------------------------------------------------
# Tool Conversion
# ---------------------------------------------------------------------------

def convert_tool(tool: dict) -> dict:
    """Convert an Anthropic tool definition to OpenAI format."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
        },
    }


def convert_tools(tools: list[dict] | None) -> list[dict] | None:
    """convert tools list, returns None if empty."""
    if not tools:
        return None
    return [convert_tool(t) for t in tools]


# ---------------------------------------------------------------------------
# System Prompt Extraction
# ---------------------------------------------------------------------------

def extract_system_parts(system: Any) -> list[str]:
    """extract text parts from system prompt."""
    match system:
        case str():
            return [system]
        case list():
            parts = []
            for block in system:
                match block:
                    case {"type": "text", "text": str(t)}:
                        parts.append(t)
                    case str(s):
                        parts.append(s)
            return parts
        case _:
            return []


def build_messages_with_system(body: dict, openai_messages: list[dict]) -> list[dict]:
    """prepend system message if present."""
    system_parts = extract_system_parts(body.get("system"))
    if system_parts:
        return [{"role": "system", "content": "\n\n".join(system_parts)}] + openai_messages
    return openai_messages


# ---------------------------------------------------------------------------
# Payload Construction (Pure)
# ---------------------------------------------------------------------------

def build_payload(body: dict[str, Any], config: Config) -> dict[str, Any]:
    """
    build OpenAI chat.completions payload from Anthropic body.
    All transformations are pure - no side effects.
    """
    # Transform messages through composed pipeline
    anthropic_messages = body.get("messages", [])
    openai_messages = anthropic_to_openai_messages(anthropic_messages)
    messages = build_messages_with_system(body, openai_messages)

    # Transform tools
    tools = convert_tools(body.get("tools"))
    stream = body.get("stream", False)

    # Build payload dict (pure construction)
    payload: dict[str, Any] = {
        "model": config.target_model,
        "messages": messages,
        "stream": stream,
    }

    # Add optional parameters (pure updates)
    optional_params = {
        "max_tokens": body.get("max_tokens"),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
    }
    payload.update({k: v for k, v in optional_params.items() if v is not None})

    if tools:
        payload["tools"] = tools

    if stream:
        payload["stream_options"] = {"include_usage": True, "continuous_usage_stats": True}

    return payload


# ---------------------------------------------------------------------------
# Streaming with Functional Approach
# ---------------------------------------------------------------------------

def create_sse_event(event: str, data: dict[str, Any]) -> str:
    """format SSE event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def map_finish_reason(finish: str | None) -> str:
    """map OpenAI finish_reason to Anthropic stop_reason."""
    mapping = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}
    return mapping.get(finish or "", finish or "end_turn")


# Tool call parsing - pure functions
_KIMI_CALL_RE = re.compile(
    r'<\|tool_call_begin\|>(?:functions\.)?(\w+)(?::\d+)?'
    r'<\|tool_call_argument_begin\|>(.*?)<\|tool_call_end\|>',
    re.DOTALL,
)


def parse_kimi_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """
    extract tool calls from Kimi's text-embedded syntax.
    Returns (clean_text, tool_uses).
    """
    # Extract text outside tool sections
    clean_parts = text.split("<|tool_calls_section_begin|>")
    before = clean_parts[0] if clean_parts else ""
    after_parts = text.split("<|tool_calls_section_end|>")
    after = after_parts[-1] if len(after_parts) > 1 else ""
    clean = (before + after).strip()

    # Parse tool calls
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


# ---------------------------------------------------------------------------
# Streaming State Machine (Immutable State Transitions)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StreamState:
    """Immutable streaming state container."""
    content_index: int
    text_block_open: bool
    accumulated_text: tuple[str, ...]  # immutable tuple
    structured_tcs: tuple[tuple[int, dict[str, str]], ...]  # immutable
    kimi_buf: tuple[str, ...]
    in_kimi_section: bool
    output_tokens: int
    finish_reason: str | None


@dataclass(frozen=True)
class DeltaUpdate:
    """Immutable delta update from OpenAI stream."""
    content: str | None
    tool_calls: list[dict] | None
    finish_reason: str | None


def parse_delta(obj: dict[str, Any]) -> DeltaUpdate | None:
    """extract delta from OpenAI stream chunk."""
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
    """
    compute next state and SSE events from delta.
    Returns (new_state, events_to_yield).
    """
    events: list[str] = []
    new_state = state

    # Handle tool calls
    if delta.tool_calls:
        # Close text block if open
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

        # Accumulate tool calls
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

    # Handle text content
    chunk = delta.content
    if not chunk:
        # Just update finish reason if present
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

    # Detect Kimi tool section
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

    # In Kimi section - buffer and skip
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

    # Normal text - emit immediately
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


# ---------------------------------------------------------------------------
# Streaming with Immutable State
# ---------------------------------------------------------------------------

async def stream_sse(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    http_client: httpx.AsyncClient,
) -> Iterator[str]:
    """
    Generator pipeline: consume OpenAI stream, yield Anthropic SSE events.
    Uses immutable state transitions instead of mutable buffers.
    """
    msg_id = f"msg_{uuid.uuid4().hex}"

    # Initial state
    initial_state = StreamState(
        content_index=0,
        text_block_open=False,
        accumulated_text=(),
        structured_tcs=(),
        kimi_buf=(),
        in_kimi_section=False,
        output_tokens=0,
        finish_reason=None,
    )

    yield create_sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": CONFIG.target_model,
            "content": [],
            "stop_reason": None,
        },
    })

    state = initial_state

    async with http_client.stream("POST", url, headers=headers, json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
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

                # State transition: pure function
                new_state, events = transition_state(state, delta)
                state = new_state

                # Yield all events
                for event in events:
                    yield event

            except (json.JSONDecodeError, KeyError):
                continue

    # Finalize: close text block, emit tool uses, send stop events
    if state.text_block_open:
        yield create_sse_event("content_block_stop", {"type": "content_block_stop", "index": state.content_index})

    # Convert structured tool calls to tool_use blocks
    tool_blocks = []
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

    # Emit tool_use blocks
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
# Non-streaming Response (Pure)
# ---------------------------------------------------------------------------

def build_anthropic_message(oai_response: dict[str, Any]) -> dict[str, Any]:
    """convert OpenAI response to Anthropic format."""
    choices = oai_response.get("choices") or []
    usage = oai_response.get("usage") or {}

    if not choices:
        return {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "model": CONFIG.target_model,
            "content": [{"type": "text", "text": ""}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    choice = choices[0]
    msg = choice.get("message", {})
    blocks: list[dict[str, Any]] = []

    # Text content
    raw_text = msg.get("content") or ""
    clean_text, kimi_tools = parse_kimi_tool_calls(raw_text) if " <|tool_calls_section_begin|>" in raw_text else (raw_text, [])
    if clean_text:
        blocks.append({"type": "text", "text": clean_text})

    # Structured tool_calls
    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            fn = tc.get("function", {})
            try:
                inp = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                inp = {"raw": fn.get("arguments", "")}
            blocks.append({
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                "name": fn.get("name", ""),
                "input": inp,
            })
    elif kimi_tools:
        blocks.extend(kimi_tools)

    if not blocks:
        blocks.append({"type": "text", "text": ""})

    has_tool = any(b["type"] == "tool_use" for b in blocks)
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": CONFIG.target_model,
        "content": blocks,
        "stop_reason": "tool_use" if has_tool else map_finish_reason(choice.get("finish_reason")),
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# ---------------------------------------------------------------------------
# HTTP Client Lifecycle
# ---------------------------------------------------------------------------

_http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Create a single AsyncClient for the app lifetime."""
    global _http_client
    _http_client = httpx.AsyncClient(timeout=httpx.Timeout(None))
    try:
        yield
    finally:
        await _http_client.aclose()
        _http_client = None


app = FastAPI(title="Anthropic→Baseten Proxy", lifespan=_lifespan)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    """Return proxy config for quick verification."""
    return {"status": "ok", "upstream": CONFIG.openai_base_url, "model": CONFIG.target_model}


def check_proxy_auth(authorization: str | None, x_api_key: str | None) -> bool:
    """validate proxy authentication."""
    if not CONFIG.proxy_auth_key:
        return True
    return (authorization == f"Bearer {CONFIG.proxy_auth_key}") or (x_api_key == CONFIG.proxy_auth_key)


def create_error_response(status_code: int, error_type: str, message: str) -> JSONResponse:
    """create Anthropic-shaped error response."""
    return JSONResponse(
        {"error": {"type": error_type, "message": message}},
        status_code=status_code,
    )


@app.post("/v1/messages")
async def messages(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Anthropic /v1/messages → Baseten OpenAI chat.completions."""
    if not check_proxy_auth(authorization, x_api_key):
        return create_error_response(401, "authentication_error", "invalid api key")
    if not CONFIG.baseten_api_key:
        return create_error_response(500, "api_error", "BASETEN_API_KEY not configured")

    try:
        body = await request.json()
    except Exception:
        return create_error_response(400, "invalid_request_error", "Invalid JSON body")

    # Build payload using pure function
    payload = build_payload(body, CONFIG)
    url = f"{CONFIG.openai_base_url}/chat/completions"
    hdrs = {"Authorization": f"Bearer {CONFIG.baseten_api_key}", "Content-Type": "application/json"}

    try:
        if payload.get("stream"):
            return StreamingResponse(
                stream_sse(url, hdrs, payload, _http_client),
                media_type="text/event-stream",
            )
        resp = await _http_client.post(url, headers=hdrs, json=payload)
        resp.raise_for_status()
        return JSONResponse(build_anthropic_message(resp.json()))
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        try:
            detail = e.response.json()
            err_msg = detail.get("error", {}).get("message", detail.get("message", e.response.text or str(e)))
        except Exception:
            err_msg = e.response.text or str(e)
        return create_error_response(min(status, 502), "api_error", err_msg)

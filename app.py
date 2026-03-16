"""
Anthropic-compatible /v1/messages proxy → Baseten OpenAI API (e.g. Kimi K2.5).

Translates tool definitions, messages, and tool calls bidirectionally so that
Claude Code CLI can use Kimi K2.5 as a drop-in replacement for Claude.
"""

import json
import os
import re
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

BASETEN_API_KEY = os.getenv("BASETEN_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://inference.baseten.co/v1").rstrip("/")
TARGET_MODEL = os.getenv("TARGET_MODEL", "moonshotai/Kimi-K2.5")
PROXY_AUTH_KEY = os.getenv("PROXY_AUTH_KEY")

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


@app.get("/health")
def health() -> dict[str, str]:
    """Return proxy config for quick verification."""
    return {"status": "ok", "upstream": OPENAI_BASE_URL, "model": TARGET_MODEL}


def _proxy_auth_ok(authorization: str | None, x_api_key: str | None) -> bool:
    """True when PROXY_AUTH_KEY is unset or the caller supplies a matching key."""
    if not PROXY_AUTH_KEY:
        return True
    return (authorization == f"Bearer {PROXY_AUTH_KEY}") or (x_api_key == PROXY_AUTH_KEY)


# ---------------------------------------------------------------------------
# Tool definition conversion  (Anthropic → OpenAI)
# ---------------------------------------------------------------------------

def _anthropic_tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic tool definitions to OpenAI function-calling tool objects."""
    oai_tools = []
    for t in tools:
        oai_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return oai_tools


# ---------------------------------------------------------------------------
# Message conversion  (Anthropic ↔ OpenAI)
# ---------------------------------------------------------------------------

def _extract_text_from_content(content: Any) -> str:
    """Flatten Anthropic content (string or list of blocks) into a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def _anthropic_messages_to_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert a list of Anthropic messages to OpenAI messages.

    Key mappings:
      - assistant content blocks with type=tool_use  → assistant.tool_calls[]
      - user content blocks with type=tool_result    → separate role=tool messages
    """
    oai: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        # --- assistant ---
        if role == "assistant":
            if isinstance(content, list):
                text_parts: list[str] = []
                tool_calls: list[dict] = []
                for block in content:
                    if not isinstance(block, dict):
                        text_parts.append(str(block))
                        continue
                    btype = block.get("type")
                    if btype == "text":
                        text_parts.append(block.get("text", ""))
                    elif btype == "tool_use":
                        tool_calls.append({
                            "id": block.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        })
                oai_msg: dict[str, Any] = {"role": "assistant"}
                text = "".join(text_parts).strip()
                oai_msg["content"] = text if text else None
                if tool_calls:
                    oai_msg["tool_calls"] = tool_calls
                oai.append(oai_msg)
            else:
                oai.append({"role": "assistant", "content": _extract_text_from_content(content)})

        # --- user (may contain tool_result blocks) ---
        elif role == "user":
            if isinstance(content, list):
                text_parts_u: list[str] = []
                tool_results: list[dict] = []
                for block in content:
                    if not isinstance(block, dict):
                        text_parts_u.append(str(block))
                        continue
                    btype = block.get("type")
                    if btype == "tool_result":
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
                    elif btype == "text":
                        text_parts_u.append(block.get("text", ""))
                    else:
                        text_parts_u.append(str(block))

                # tool results MUST immediately follow the assistant tool_calls
                oai.extend(tool_results)
                text_u = "".join(text_parts_u).strip()
                if text_u:
                    oai.append({"role": "user", "content": text_u})
            else:
                oai.append({"role": "user", "content": _extract_text_from_content(content)})

        # --- system or other ---
        else:
            oai.append({"role": role, "content": _extract_text_from_content(content)})

    return oai


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------

def to_openai_payload(body: dict[str, Any]) -> dict[str, Any]:
    """Build an OpenAI chat.completions payload from an Anthropic /v1/messages body."""
    messages = _anthropic_messages_to_openai(body.get("messages", []))

    # System prompt
    system_parts: list[str] = []
    system = body.get("system")
    if system:
        if isinstance(system, str):
            system_parts.append(system)
        elif isinstance(system, list):
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    system_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    system_parts.append(block)
    if system_parts:
        messages = [{"role": "system", "content": "\n\n".join(system_parts)}] + messages

    # Tools
    anthropic_tools = body.get("tools", [])
    oai_tools = _anthropic_tools_to_openai(anthropic_tools) if anthropic_tools else None

    stream = body.get("stream", False)
    payload: dict[str, Any] = {
        "model": TARGET_MODEL,
        "messages": messages,
        "stream": stream,
    }
    if body.get("max_tokens"):
        payload["max_tokens"] = body["max_tokens"]
    if body.get("temperature") is not None:
        payload["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        payload["top_p"] = body["top_p"]

    if oai_tools:
        payload["tools"] = oai_tools

    if stream:
        payload["stream_options"] = {"include_usage": True, "continuous_usage_stats": True}

    return payload


# ---------------------------------------------------------------------------
# Error helper
# ---------------------------------------------------------------------------

def _anthropic_error(status_code: int, error_type: str, message: str) -> JSONResponse:
    """Return an Anthropic-shaped error response."""
    return JSONResponse(
        {"error": {"type": error_type, "message": message}},
        status_code=status_code,
    )


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

def _sse_event(event: str, data: dict[str, Any]) -> str:
    """Format one SSE frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _oai_finish_to_stop(finish: str | None) -> str:
    """Map OpenAI finish_reason → Anthropic stop_reason."""
    return {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}.get(finish or "", finish or "end_turn")


# ---------------------------------------------------------------------------
# Parse Kimi text-embedded tool syntax  (fallback when structured tool_calls
# are absent but Kimi embeds its own syntax in the text content)
# ---------------------------------------------------------------------------

_KIMI_CALL_RE = re.compile(
    r'<\|tool_call_begin\|>(?:functions\.)?(\w+)(?::\d+)?'
    r'<\|tool_call_argument_begin\|>(.*?)<\|tool_call_end\|>',
    re.DOTALL,
)


def _parse_kimi_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """
    Extract tool calls from Kimi's text-embedded syntax.
    Returns (clean_text_without_tool_syntax, list_of_tool_use_dicts).
    """
    clean = text
    if "<|tool_calls_section_begin|>" in text:
        before = text.split("<|tool_calls_section_begin|>")[0]
        after_parts = text.split("<|tool_calls_section_end|>")
        after = after_parts[-1] if len(after_parts) > 1 else ""
        clean = (before + after).strip()

    tool_uses: list[dict[str, Any]] = []
    for m in _KIMI_CALL_RE.finditer(text):
        name = m.group(1)
        args_raw = m.group(2).strip()
        try:
            inp = json.loads(args_raw)
        except json.JSONDecodeError:
            inp = {"raw": args_raw}
        tool_uses.append({
            "type": "tool_use",
            "id": f"toolu_{uuid.uuid4().hex[:24]}",
            "name": name,
            "input": inp,
        })

    return clean, tool_uses


# ---------------------------------------------------------------------------
# Streaming  (Baseten OpenAI SSE → Anthropic SSE)
# ---------------------------------------------------------------------------

async def _stream_sse(url: str, headers: dict[str, str], payload: dict[str, Any]):
    """
    Consume OpenAI streaming response and yield Anthropic-format SSE events.

    Handles BOTH:
      A) Structured tool_calls in delta  (OpenAI native)
      B) Kimi's text-embedded <|tool_calls_section_begin|> syntax
    """
    msg_id = f"msg_{uuid.uuid4().hex}"
    yield _sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant",
            "model": TARGET_MODEL, "content": [], "stop_reason": None,
        },
    })

    content_index = 0          # next Anthropic content block index
    text_block_open = False    # whether we've started a text content block
    accumulated_text: list[str] = []

    # (A) Structured tool_calls accumulator  {oai_index: {id, name, arguments}}
    structured_tcs: dict[int, dict[str, str]] = {}

    # (B) Kimi text-embedded tool call buffer
    kimi_buf: list[str] = []
    in_kimi_section = False

    output_tokens = 0
    finish_reason: str | None = None

    async with _http_client.stream("POST", url, headers=headers, json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if raw == "[DONE]":
                break
            try:
                obj = json.loads(raw)
                choices = obj.get("choices")
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                finish_reason = choices[0].get("finish_reason") or finish_reason

                # ---- (A) Structured tool_calls (skip empty arrays) ----
                if delta.get("tool_calls"):
                    if text_block_open:
                        yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": content_index})
                        content_index += 1
                        text_block_open = False

                    for tc in delta["tool_calls"]:
                        idx = tc.get("index", 0)
                        if idx not in structured_tcs:
                            structured_tcs[idx] = {
                                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                                "name": tc.get("function", {}).get("name", ""),
                                "arguments": "",
                            }
                        fn = tc.get("function", {})
                        if fn.get("name"):
                            structured_tcs[idx]["name"] = fn["name"]
                        if fn.get("arguments"):
                            structured_tcs[idx]["arguments"] += fn["arguments"]
                    output_tokens += 1
                    continue

                # ---- (B) Text content (may contain Kimi embedded syntax) ----
                chunk = delta.get("content")
                if not chunk:
                    continue
                output_tokens += 1

                # Detect Kimi tool section start
                if "<|tool_calls_section_begin|>" in chunk:
                    in_kimi_section = True
                    before = chunk.split("<|tool_calls_section_begin|>")[0]
                    if before.strip():
                        if not text_block_open:
                            yield _sse_event("content_block_start", {
                                "type": "content_block_start", "index": content_index,
                                "content_block": {"type": "text", "text": ""},
                            })
                            text_block_open = True
                        yield _sse_event("content_block_delta", {
                            "type": "content_block_delta", "index": content_index,
                            "delta": {"type": "text_delta", "text": before},
                        })
                        accumulated_text.append(before)
                    kimi_buf.append(chunk)
                    continue

                if in_kimi_section:
                    kimi_buf.append(chunk)
                    if "<|tool_calls_section_end|>" in chunk:
                        in_kimi_section = False
                    continue

                # Normal text – emit immediately
                if not text_block_open:
                    yield _sse_event("content_block_start", {
                        "type": "content_block_start", "index": content_index,
                        "content_block": {"type": "text", "text": ""},
                    })
                    text_block_open = True
                yield _sse_event("content_block_delta", {
                    "type": "content_block_delta", "index": content_index,
                    "delta": {"type": "text_delta", "text": chunk},
                })
                accumulated_text.append(chunk)

            except (json.JSONDecodeError, KeyError):
                continue

    # Close open text block
    if text_block_open:
        yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": content_index})
        content_index += 1

    # --- Collect tool calls to emit as Anthropic tool_use blocks ---
    tool_blocks: list[dict[str, Any]] = []

    # Prefer structured
    if structured_tcs:
        for idx in sorted(structured_tcs):
            tc = structured_tcs[idx]
            try:
                inp = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                inp = {"raw": tc["arguments"]}
            tool_blocks.append({"id": tc["id"], "name": tc["name"], "input": inp})

    # Fallback: parse Kimi text-embedded syntax
    elif kimi_buf:
        _, parsed = _parse_kimi_tool_calls("".join(kimi_buf))
        for p in parsed:
            tool_blocks.append({"id": p["id"], "name": p["name"], "input": p["input"]})

    # Emit each tool_use as its own content block
    for tb in tool_blocks:
        yield _sse_event("content_block_start", {
            "type": "content_block_start", "index": content_index,
            "content_block": {"type": "tool_use", "id": tb["id"], "name": tb["name"], "input": {}},
        })
        yield _sse_event("content_block_delta", {
            "type": "content_block_delta", "index": content_index,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps(tb["input"])},
        })
        yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": content_index})
        content_index += 1

    stop = "tool_use" if tool_blocks else _oai_finish_to_stop(finish_reason)

    yield _sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop},
        "usage": {"output_tokens": output_tokens},
    })
    yield _sse_event("message_stop", {"type": "message_stop"})


# ---------------------------------------------------------------------------
# Non-streaming response
# ---------------------------------------------------------------------------

def _anthropic_message_nonstream(oai: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI chat completion (non-streaming) to an Anthropic message."""
    choices = oai.get("choices") or []
    usage = oai.get("usage") or {}

    if not choices:
        return {
            "id": f"msg_{uuid.uuid4().hex}", "type": "message", "role": "assistant",
            "model": TARGET_MODEL, "content": [{"type": "text", "text": ""}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    choice = choices[0]
    msg = choice.get("message", {})
    blocks: list[dict[str, Any]] = []

    # Text (possibly containing Kimi tool syntax)
    raw_text = msg.get("content") or ""
    clean_text, kimi_tools = _parse_kimi_tool_calls(raw_text) if "<|tool_calls_section_begin|>" in raw_text else (raw_text, [])
    if clean_text:
        blocks.append({"type": "text", "text": clean_text})

    # Structured tool_calls from OpenAI
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
        "id": f"msg_{uuid.uuid4().hex}", "type": "message", "role": "assistant",
        "model": TARGET_MODEL, "content": blocks,
        "stop_reason": "tool_use" if has_tool else _oai_finish_to_stop(choice.get("finish_reason")),
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# ---------------------------------------------------------------------------
# Main route
# ---------------------------------------------------------------------------

@app.post("/v1/messages")
async def messages(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Anthropic /v1/messages → Baseten OpenAI chat.completions with full tool bridging."""
    if not _proxy_auth_ok(authorization, x_api_key):
        return _anthropic_error(401, "authentication_error", "invalid api key")
    if not BASETEN_API_KEY:
        return _anthropic_error(500, "api_error", "BASETEN_API_KEY not configured")

    try:
        body = await request.json()
    except Exception:
        return _anthropic_error(400, "invalid_request_error", "Invalid JSON body")

    payload = to_openai_payload(body)
    url = f"{OPENAI_BASE_URL}/chat/completions"
    hdrs = {"Authorization": f"Bearer {BASETEN_API_KEY}", "Content-Type": "application/json"}

    try:
        if payload.get("stream"):
            return StreamingResponse(_stream_sse(url, hdrs, payload), media_type="text/event-stream")
        resp = await _http_client.post(url, headers=hdrs, json=payload)
        resp.raise_for_status()
        return JSONResponse(_anthropic_message_nonstream(resp.json()))
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        try:
            detail = e.response.json()
            err_msg = detail.get("error", {}).get("message", detail.get("message", e.response.text or str(e)))
        except Exception:
            err_msg = e.response.text or str(e)
        return _anthropic_error(min(status, 502), "api_error", err_msg)

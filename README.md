# Anthropic Proxy

A production-ready, model-agnostic proxy that translates Anthropic's Messages API to OpenAI-compatible requests. Supports Kimi K2.5 via Baseten, OpenAI GPT models, and other OpenAI-compatible providers.

## Features

- **Model-Agnostic Architecture**: Supports multiple backends through a unified handler system
- **Streaming & Non-Streaming**: Full support for both request types
- **Tool Calling**: Bidirectional tool call translation for Claude Code compatibility
- **Rate Limiting**: Token bucket algorithm with per-key and per-model limits
- **Circuit Breaker**: Automatic failover for failing upstream providers
- **Health Checks**: Multi-layer health monitoring with upstream validation
- **Structured Logging**: JSON/text logging with request correlation IDs
- **Security**: Input validation, content sanitization, and request size limits

## Architecture

The proxy uses a modular architecture with clear separation of concerns:

```
Anthropic Request
  → Validation (security.py)
  → Rate Limiting (rate_limit.py)
  → Model Routing (models/registry.py)
  → Model Handler (models/)
  → Tool Adapter (tools/)
  → OpenAI-compatible Request
  → Upstream Provider
  → Response Parsing
  → Streaming / Non-streaming Translation
  → Anthropic Response
```

### Key Modules

| Module | Responsibility |
|--------|--------------|
| `config.py` | Pydantic-based configuration with env var overrides |
| `models.yaml` | Model capability definitions |
| `errors.py` | Exception hierarchy with structured error responses |
| `retry.py` | Exponential backoff, circuit breaker pattern |
| `security.py` | Input validation with Pydantic models |
| `rate_limit.py` | Token bucket algorithm, per-key/model limits |
| `logging_config.py` | Structured JSON/text logging |
| `middleware.py` | Request logging, metrics, timing, tracing |
| `performance.py` | Connection pooling, caching |
| `lifecycle.py` | Start/stop/restart with graceful shutdown |
| `health.py` | Multi-layer health checking |
| `main.py` | FastAPI application entry point |

## Setup

### 1. Configure the Proxy

Create `.env` in the project root:

```bash
BASETEN_API_KEY=your_baseten_api_key
```

Optional environment variables:
- `OPENAI_BASE_URL` - Defaults to `https://inference.baseten.co/v1`
- `DEFAULT_MODEL` - Defaults to `kimi-k2.5`
- `PROXY_AUTH_KEY` - Optional auth key for the proxy itself
- `LOG_LEVEL` - Defaults to `INFO`
- `LOG_FORMAT` - Defaults to `json`

### 2. Start the Proxy

#### Using Python directly:

```bash
cd anthropic-proxy
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

#### Using Docker:

```bash
docker build -t anthropic-proxy .
docker run -p 8000:8000 --env-file .env anthropic-proxy
```

#### Using Docker Compose:

```bash
docker-compose up -d
```

Verify it's running:
```bash
curl -s http://localhost:8000/health
# Expected: {"status": "healthy", "checks": {...}, "latency_ms": ...}
```

### 3. Configure Claude Code

Create or edit `~/.claude/settings.json`:

```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "apiKeyHelper": "echo proxy",
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8000"
  }
}
```

The `apiKeyHelper` provides a dummy API key since the proxy validates via `BASETEN_API_KEY`. If you set `PROXY_AUTH_KEY` in `.env`, change `apiKeyHelper` to `echo your_key`.

### 4. Run Claude Code

```bash
claude
```

If you see a login screen on first run:
1. Choose **Anthropic Console (option 2)** and complete OAuth
2. Type `/logout` in Claude Code
3. Restart `claude` - it will now use `apiKeyHelper` only

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Basic health check |
| `/health/detailed` | GET | Detailed health with upstream validation |
| `/v1/messages` | POST | Anthropic-compatible messages endpoint |

## Troubleshooting

**"Auth conflict" warning**: Run `/logout` in Claude Code and restart.

**"Model is Sonner 4.6" in UI**: This is just Claude Code's UI label. The proxy routes to the configured model - verify with the `/health` endpoint.

**No streaming response**: Check that the proxy is running and `ANTHROPIC_BASE_URL` is set correctly in settings.json.

**Import errors**: Ensure you're using Python 3.9+ and all dependencies are installed.

## Limitations

- **Extended thinking**: Not supported by Kimi K2.5
- **Prompt caching**: Baseten may not support Anthropic's caching headers
- **Vision/multimodal**: Image inputs are not translated
- **Tool use**: The proxy handles Claude Code's built-in tools, but LLM-initiated function calling has limited support

## Development

### Running Tests

```bash
pytest tests/unit/ -v
pytest tests/integration/ -v
```

### Code Quality

```bash
black .
ruff check .
mypy .
```

### Architecture Decisions

- **Immutable streaming state**: Uses dataclasses with pure transition functions
- **Function composition**: Message conversion uses `compose()` to chain transformations
- **Pattern-based registry**: Model handlers registered with glob patterns
- **Adapter pattern**: Tool formats abstracted through `ToolFormatAdapter`

## License

MIT License - See LICENSE file for details.

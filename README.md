# Anthropic Proxy

A production-ready, model-agnostic proxy that translates Anthropic's Messages API to OpenAI-compatible requests. This enables Claude tooling (like Claude Code) to use alternative backends such as Kimi K2.5 via Baseten, OpenAI GPT models, and other OpenAI-compatible providers.

## 🎯 Purpose

This proxy solves a specific problem: **Claude's official API only works with Anthropic's models**. If you want to use Claude Code or other Claude tooling with alternative LLM providers (like Kimi, GPT-4o, or any OpenAI-compatible API), you need a translation layer.

This proxy provides that translation layer by:
- Accepting Anthropic's Messages API format
- Converting requests to OpenAI-compatible format
- Routing to your chosen backend provider
- Translating responses back to Anthropic format
- Supporting streaming, tool calls, and other advanced features

## ✨ Key Features

- **Model-Agnostic Architecture**: Support multiple backends through a unified handler system
- **Streaming & Non-Streaming**: Full support for both request types
- **Tool Calling**: Bidirectional tool call translation for Claude Code compatibility
- **Rate Limiting**: Token bucket algorithm with per-key and per-model limits
- **Circuit Breaker**: Automatic failover for failing upstream providers
- **Health Checks**: Multi-layer health monitoring with upstream validation
- **Structured Logging**: JSON/text logging with request correlation IDs
- **Security**: Input validation, content sanitization, and request size limits
- **Production-Ready**: Type hints, error handling, tests, and Docker deployment

## 🏗️ Architecture

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

## 🚀 Quick Start

### 1. Configure the Proxy

Create `.env` in the project root:

```bash
# Required: Your backend provider API key
BASETEN_API_KEY=your_baseten_api_key

# Optional: OpenAI API key (for GPT models)
OPENAI_API_KEY=your_openai_api_key

# Optional: Auth key for the proxy itself
PROXY_AUTH_KEY=optional_proxy_key

# Optional: Default model to use
DEFAULT_MODEL=kimi-k2.5

# Optional: Logging configuration
LOG_LEVEL=INFO
LOG_FORMAT=json
```

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

## 📚 Usage Examples

### Basic Request

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: proxy" \
  -d '{
    "model": "kimi-k2.5",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Hello, world!"}
    ]
  }'
```

### Streaming Request

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: proxy" \
  -d '{
    "model": "kimi-k2.5",
    "max_tokens": 1024,
    "stream": true,
    "messages": [
      {"role": "user", "content": "Tell me a short story"}
    ]
  }'
```

### Tool Calling

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: proxy" \
  -d '{
    "model": "kimi-k2.5",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "What is the weather in San Francisco?"}
    ],
    "tools": [
      {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "input_schema": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state"
            }
          },
          "required": ["location"]
        }
      }
    ]
  }'
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BASETEN_API_KEY` | Baseten API key for Kimi models | Required |
| `OPENAI_API_KEY` | OpenAI API key for GPT models | Optional |
| `PROXY_AUTH_KEY` | Auth key for proxy access | Optional |
| `DEFAULT_MODEL` | Default model to use | `kimi-k2.5` |
| `OPENAI_BASE_URL` | OpenAI-compatible API base URL | `https://inference.baseten.co/v1` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `LOG_FORMAT` | Log format (`json` or `text`) | `json` |

### Model Configuration

Models are configured in `models.yaml`. Each model defines:
- `capabilities`: What the model supports (streaming, tools, vision, etc.)
- `handler`: Which handler to use for translation
- `adapter`: Which tool adapter to use

To add a new model:
1. Add it to `models.yaml`
2. Reuse an existing handler if possible
3. Otherwise implement a new `ModelHandler`
4. Register it in `models/registry.py`

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Basic health check |
| `/health/detailed` | GET | Detailed health with upstream validation |
| `/v1/messages` | POST | Anthropic-compatible messages endpoint |

## 🧪 Testing

```bash
# Run unit tests
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v

# Run with coverage
pytest --cov=. tests/

# Run specific test files
pytest tests/unit/test_config.py -v
pytest tests/unit/test_tools.py -v
```

## 🔍 Troubleshooting

### "Auth conflict" warning
Run `/logout` in Claude Code and restart.

### "Model is Sonnet 4.6" in UI
This is just Claude Code's UI label. The proxy routes to the configured model - verify with the `/health` endpoint.

### No streaming response
Check that the proxy is running and `ANTHROPIC_BASE_URL` is set correctly in settings.json.

### Import errors
Ensure you're using Python 3.9+ and all dependencies are installed.

### Upstream errors
Check the proxy logs for detailed error information. Enable debug logging:
```bash
LOG_LEVEL=DEBUG uvicorn main:app --host 0.0.0.0 --port 8000
```

## ⚠️ Limitations

- **Extended thinking**: Not supported by Kimi K2.5
- **Prompt caching**: Baseten may not support Anthropic's caching headers
- **Vision/multimodal**: Image inputs are not translated
- **Tool use**: The proxy handles Claude Code's built-in tools, but LLM-initiated function calling has limited support

## 🏗️ Development

### Code Quality

```bash
# Format code
black .

# Lint code
ruff check .

# Type checking
mypy .
```

### Architecture Decisions

- **Immutable streaming state**: Uses dataclasses with pure transition functions
- **Function composition**: Message conversion uses `compose()` to chain transformations
- **Pattern-based registry**: Model handlers registered with glob patterns
- **Adapter pattern**: Tool formats abstracted through `ToolFormatAdapter`

### Adding a New Provider

1. Define model capabilities in `models.yaml`
2. Implement a handler (extend `ModelHandler`)
3. Implement a tool adapter if needed (extend `ToolFormatAdapter`)
4. Validate Anthropic-compatible request and response behavior
5. Add tests for streaming and non-streaming cases

## 📖 Documentation

- `CLAUDE.md` - Development guidelines and architecture rules
- `OPTIMIZATIONS.md` - Functional programming optimizations
- `IMPLEMENTATION_PLAN.md` - Implementation status and roadmap

## 🤝 Contributing

Contributions are welcome! Please read `CLAUDE.md` for development guidelines before contributing.

## 📄 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

Built to enable flexible use of Claude tooling with alternative LLM providers. Inspired by the need for model-agnostic AI infrastructure.

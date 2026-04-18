# Anthropic Proxy

A production-ready, model-agnostic proxy that translates Anthropic's Messages API to OpenAI-compatible requests. Enables Claude tooling (like Claude Code) to use alternative backends such as Kimi K2.5 via Baseten, OpenAI GPT models, and other OpenAI-compatible providers.

## Purpose

Claude's official API only works with Anthropic's models. This proxy provides a translation layer that lets you use Claude Code and other Claude tooling with alternative LLM providers while maintaining full compatibility with Anthropic's Messages API format.

## Architecture

The proxy uses a clean, modular architecture designed for simplicity and performance:

```
Anthropic Request → Validation → Model Routing → OpenAI Request → Provider → Response Translation → Anthropic Response
```

**Key Design Principles:**
- **Horizontal Scaling**: Configurable worker processes for parallel request handling
- **Vertical Scaling**: Optimized connection pooling with HTTP/2 multiplexing
- **Simplified Logic**: Leverages provider infrastructure (rate limiting, circuit breaking) instead of duplicating it
- **Production-Ready**: Health checks, structured logging, graceful error handling

## Quick Start

### 1. Configure

Create `.env`:

```bash
# Required: Your backend provider API key
BASETEN_API_KEY=your_baseten_api_key
DEFAULT_MODEL=zai-org/GLM-4.7

# Performance: Horizontal scaling
WORKERS=4

# Performance: Vertical scaling  
MAX_CONNECTIONS=200
KEEPALIVE_CONNECTIONS=100

# Simplification: Let provider handle these
RATE_LIMIT_ENABLED=false
CIRCUIT_BREAKER_ENABLED=false
```

### 2. Start

```bash
python -m cli start --workers 4
```

### 3. Configure Claude Code

Edit `~/.claude/settings.json`:

```json
{
  "apiKeyHelper": "echo proxy",
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8000"
  }
}
```

### 4. Verify

```bash
curl http://localhost:8000/health/connections
# Expected: {"status": "healthy", "http2_enabled": true, "pool_active": true}
```

## Configuration

### Scaling

| Setting | Description | Default | Impact |
|----------|-------------|---------|--------|
| `WORKERS` | Worker processes for parallel request handling | `4` | Horizontal scaling - more workers = more concurrent requests |
| `MAX_CONNECTIONS` | Connection pool limit | `200` | Vertical scaling - higher limit = more concurrent connections |
| `KEEPALIVE_CONNECTIONS` | Reusable connections | `100` | Connection reuse - reduces connection overhead |

### Simplification

| Setting | Description | Default | Rationale |
|----------|-------------|---------|-----------|
| `RATE_LIMIT_ENABLED` | Enable local rate limiting | `false` | Provider handles rate limiting per API key |
| `CIRCUIT_BREAKER_ENABLED` | Enable local circuit breaking | `false` | Provider handles model availability and failover |

### Provider

| Setting | Description | Default |
|----------|-------------|---------|
| `BASETEN_API_KEY` | Baseten API key | Required |
| `DEFAULT_MODEL` | Default model to use | `zai-org/GLM-4.7` |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Basic health check |
| `/health/connections` | GET | Connection pool status |
| `/v1/messages` | POST | Anthropic-compatible messages endpoint |

## Usage Examples

### Basic Request

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zai-org/GLM-4.7",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Streaming Request

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zai-org/GLM-4.7",
    "stream": true,
    "messages": [{"role": "user", "content": "Tell me a story"}]
  }'
```

### Tool Calling

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zai-org/GLM-4.7",
    "messages": [{"role": "user", "content": "What is the weather in San Francisco?"}],
    "tools": [{
      "name": "get_weather",
      "description": "Get current weather",
      "input_schema": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"]
      }
    }]
  }'
```

## CLI Commands

```bash
# Start with custom workers
python -m cli start --workers 8

# Check status
python -m cli status

# Health check
python -m cli health --verbose

# Validate configuration
python -m cli config validate

# Show configuration
python -m cli config show
```

## Testing

```bash
# Validate configuration
python -m cli config validate

# Run tests
pytest tests/unit/ -v

# Test concurrent load
for i in {1..20}; do
  curl -X POST http://localhost:8000/v1/messages \
    -H "Content-Type: application/json" \
    -d '{"model":"zai-org/GLM-4.7","messages":[{"role":"user","content":"Hello"}],"stream":false}' &
done
wait
```

## Troubleshooting

**"Auth conflict" warning**: Run `/logout` in Claude Code and restart.

**No streaming response**: Check proxy is running and `ANTHROPIC_BASE_URL` is set correctly.

**Connection issues**: Verify health status: `curl http://localhost:8000/health/connections`

**High latency**: Consider increasing `WORKERS` or `MAX_CONNECTIONS` in `.env`

## Development

```bash
# Format code
black .

# Lint code
ruff check .

# Type checking
mypy .
```

## Documentation

- `CLAUDE.md` - Development guidelines and architecture rules
- `models.yaml` - Model capability definitions

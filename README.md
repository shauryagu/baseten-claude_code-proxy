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

You can run the proxy directly with Python or in a container with Docker/Podman.

#### Option A: Python (local)

```bash
python -m cli start --workers 4
```

#### Option B: Docker

```bash
# Build the image once
docker build -t anthropic-proxy .

# Run it (loads variables from .env and exposes port 8000)
docker run -d \
  --name anthropic-proxy \
  --restart unless-stopped \
  -p 8000:8000 \
  --env-file .env \
  anthropic-proxy
```

Or with Compose for the full stack:

```bash
docker compose up -d              # proxy only
docker compose --profile with-redis up -d     # with Redis
docker compose --profile with-metrics up -d   # with Prometheus + Grafana
```

#### Option C: Podman

Podman is CLI-compatible with Docker, so the same commands work with `podman` substituted in:

```bash
# Build
podman build -t anthropic-proxy .

# Run (rootless by default; --userns=keep-id keeps file ownership sane on bind mounts)
podman run -d \
  --name anthropic-proxy \
  --restart unless-stopped \
  -p 8000:8000 \
  --env-file .env \
  anthropic-proxy
```

For Compose with Podman, either use `podman compose ...` (Podman 4+) or install `podman-compose` and run `podman-compose up -d`.

To make the container start automatically on login (macOS / Linux with systemd-user), generate a Podman service unit:

```bash
# Linux (systemd user services)
podman generate systemd --new --name anthropic-proxy \
  > ~/.config/systemd/user/anthropic-proxy.service
systemctl --user daemon-reload
systemctl --user enable --now anthropic-proxy.service

# macOS: use 'podman machine start' on boot, then rely on --restart unless-stopped
podman machine set --rootful=false
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

## Run From Any Directory (Persistent Alias)

Once the image is built and a `.env` exists in the repo, you can wrap the container lifecycle in shell aliases / functions so you can start it from anywhere.

### Recommended: shell functions in `~/.zshrc`

Functions are preferable to aliases here because they can resolve absolute paths reliably.

```bash
# ~/.zshrc  (or ~/.bashrc)
# Anthropic proxy container helpers — works from any directory

# Absolute path to this repo (edit if you move it)
export ANTHROPIC_PROXY_DIR="$HOME/PersonalProjects/AiExplore/anthropic-proxy"

# Container runtime: switch to "docker" if you prefer Docker
export ANTHROPIC_PROXY_RUNTIME="podman"

proxy-build() {
  $ANTHROPIC_PROXY_RUNTIME build -t anthropic-proxy "$ANTHROPIC_PROXY_DIR"
}

proxy-start() {
  $ANTHROPIC_PROXY_RUNTIME run -d \
    --name anthropic-proxy \
    --restart unless-stopped \
    -p 8000:8000 \
    --env-file "$ANTHROPIC_PROXY_DIR/.env" \
    anthropic-proxy
}

proxy-stop() {
  $ANTHROPIC_PROXY_RUNTIME stop anthropic-proxy 2>/dev/null
  $ANTHROPIC_PROXY_RUNTIME rm   anthropic-proxy 2>/dev/null
}

proxy-restart() { proxy-stop; proxy-start; }
proxy-logs()    { $ANTHROPIC_PROXY_RUNTIME logs -f anthropic-proxy; }
proxy-status()  { $ANTHROPIC_PROXY_RUNTIME ps --filter name=anthropic-proxy; }
```

Reload your shell once:

```bash
source ~/.zshrc
```

Then from anywhere on your system you can run:

```bash
proxy-start     # launch the container
proxy-logs      # tail logs
proxy-restart   # stop + start
proxy-stop      # stop and remove
```

### Alternative: pure aliases

If you prefer one-liners, put these in `~/.zshrc`:

```bash
alias proxy-build='podman build -t anthropic-proxy "$HOME/PersonalProjects/AiExplore/anthropic-proxy"'
alias proxy-start='podman run -d --name anthropic-proxy --restart unless-stopped -p 8000:8000 --env-file "$HOME/PersonalProjects/AiExplore/anthropic-proxy/.env" anthropic-proxy'
alias proxy-stop='podman stop anthropic-proxy && podman rm anthropic-proxy'
alias proxy-logs='podman logs -f anthropic-proxy'
```

### Auto-start on boot/login

- **Linux (systemd user)**: see the `podman generate systemd` snippet in the Quick Start. Enabling that unit makes the container start on every login automatically.
- **macOS with Podman**: ensure the Podman VM starts on login with `podman machine start` (you can add this to a LaunchAgent or to your `~/.zshrc`), then `--restart unless-stopped` will bring the container back up.
- **Docker Desktop**: enable "Start Docker Desktop when you log in" in settings; `--restart unless-stopped` then handles the container itself.

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

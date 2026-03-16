# Anthropic → Baseten Proxy

A protocol translation layer that exposes Anthropic's `/v1/messages` API and forwards to Baseten's OpenAI-compatible endpoint (Kimi K2.5). This lets you use Claude Code CLI with Kimi as the backend model.

## Setup

### 1. Configure the Proxy

Create `.env` in the project root:

```bash
BASETEN_API_KEY=your_baseten_api_key
```

Optional environment variables:
- `OPENAI_BASE_URL` - Defaults to `https://inference.baseten.co/v1`
- `TARGET_MODEL` - Defaults to `moonshotai/Kimi-K2.5`
- `PROXY_AUTH_KEY` - Optional auth key for the proxy itself

### 2. Start the Proxy

```bash
cd anthropic-proxy
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
```

Verify it's running:
```bash
curl -s http://localhost:8000/health
# Expected: {"status":"ok","upstream":"https://inference.baseten.co/v1","model":"moonshotai/Kimi-K2.5"}
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

## Troubleshooting

**"Auth conflict" warning**: Run `/logout` in Claude Code and restart.

**"Model is Sonnet 4.6" in UI**: This is just Claude Code's UI label. The proxy always calls Kimi K2.5 - verify with the `/health` endpoint.

**No streaming response**: Check that the proxy is running and `ANTHROPIC_BASE_URL` is set correctly in settings.json.

## Limitations

- **Extended thinking**: Not supported by Kimi K2.5
- **Prompt caching**: Baseten may not support Anthropic's caching headers
- **Vision/multimodal**: Image inputs are not translated
- **Tool use**: The proxy handles Claude Code's built-in tools, but LLM-initiated function calling has limited support

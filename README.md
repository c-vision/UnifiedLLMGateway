# Unified Gateway

[![Go Version](https://img.shields.io/badge/go-1.26%2B-00ADD8?logo=go)](https://go.dev)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS-lightgrey)](#requirements)

A lightweight Go proxy that lets **Anthropic-native** tools (like [Claude Code](https://github.com/anthropics/claude-code)) and **OpenAI-native** tools (like [OpenCode](https://github.com/sst/opencode)) share the same local LLM backend — with zero client-side changes beyond a base URL.

Point Claude Code at Unified Gateway's Anthropic adapter, point your OpenAI-compatible tools at its OpenAI adapter, and swap the model actually running underneath with a single command. The gateway also owns the lifecycle of the local model process itself, so there's no separate launcher script to keep in sync.

## Why

Local inference servers (`llama.cpp`-based tools, [`ollama`](https://ollama.com), MLX runners, etc.) generally speak one dialect — usually OpenAI's Chat Completions format. Claude Code, however, speaks Anthropic's Messages API: different request shape, different streaming event format, different tool-calling contract. Getting Claude Code to work against a local model means translating between the two — correctly, including edge cases most minimal proxies miss (system prompt placement, tool call streaming, conversation-compaction requests).

Unified Gateway does that translation, and adds model lifecycle management on top, so `some-model-tool serve` + a hand-rolled proxy script becomes one binary and one config file.

## Features

- **Dual protocol adapters** — a dedicated Anthropic Messages API port and a dedicated OpenAI Chat Completions port, both backed by the same underlying model
- **Correct Anthropic↔OpenAI translation** — `system` prompt (including prompts Claude Code sends interleaved mid-conversation, which most naive proxies miss), `tools`/`tool_use`/`tool_result`, and full SSE streaming translation in both directions
- **Conversation-compaction awareness** — recognizes Claude Code's summarization and auto-continue requests so they're observable instead of looking like malformed traffic
- **Model lifecycle management** — one command to hot-swap the active local model, whether it's served by a locally-spawned process or an already-running daemon like `ollama`, without restarting the gateway
- **No port collisions, by design** — the gateway's own adapters, the model backend, and third-party tools (e.g. `ollama`) are always kept on distinct, non-overlapping ports
- **Optional background service** — install as a macOS `launchd` agent that starts at login and restarts on crash

## Architecture

```
                 ┌─────────────────────────┐
 Claude Code ──▶ │ Anthropic Adapter :8083 │──┐
                 └─────────────────────────┘  │
                                               ├──▶  active backend
                 ┌─────────────────────────┐  │      (rapid-mlx / ds4-server / ollama)
 OpenCode etc ─▶ │  OpenAI Adapter  :8082  │──┘
                 └─────────────────────────┘
```

The gateway process is stateless with respect to which model is loaded — a small state file records the currently active backend (port + upstream model name), written whenever you run `unified-gateway load <model>`. This means switching between a locally-spawned inference process and an independent daemon like Ollama never requires restarting the gateway itself.

## Requirements

- macOS (the model-launching and background-service tooling shell out to `lsof`, `launchctl`, and POSIX process groups)
- Go 1.26+
- At least one local inference backend: [`ollama`](https://ollama.com), an MLX-based server, or a `llama.cpp`-compatible server that exposes an OpenAI-compatible `/v1/chat/completions` endpoint

## Installation

```bash
git clone https://github.com/<your-username>/unified-gateway.git
cd unified-gateway
go build -o unified-gateway .
```

Copy the binary and your `models.json` (see [Configuration](#configuration)) into a directory on your `PATH`, e.g.:

```bash
mkdir -p ~/.local/bin
cp unified-gateway models.json ~/.local/bin/
```

`models.json` must always live next to the binary — that's how the gateway finds its model configuration and where it writes its runtime state.

## Configuration

Define your models in `models.json`:

```json
{
  "backend_port": 11435,
  "ollama_port": 11434,
  "venv_dir": "~/path/to/your/mlx/venv",
  "ds4_dir": "~/path/to/ds4-server",
  "models": {
    "my-mlx-model": {
      "path": "~/models/some-model-4bit",
      "label": "My MLX Model",
      "backend": "mlx",
      "model_type": "qwen3_5",
      "has_vision": true
    },
    "my-ollama-model": {
      "label": "My Ollama Model",
      "backend": "ollama",
      "ollama_model": "gemma3:27b"
    }
  }
}
```

| Field | Meaning |
|---|---|
| `backend_port` | Port that locally-spawned backends (`mlx`, `ds4`) listen on |
| `ollama_port` | Ollama's own port (default `11434`), independent of `backend_port` |
| `venv_dir` | Python virtualenv containing your MLX-based server binary |
| `ds4_dir` | Directory containing a `ds4-server` binary, for GGUF-based models |
| `models.<name>.backend` | `"mlx"`, `"ds4"`, or `"ollama"` |
| `models.<name>.ollama_model` | For `"ollama"` entries: Ollama's own model tag, if it differs from the shortname key |

## Usage

Start the gateway (both adapters run in the same process):

```bash
unified-gateway
```

In another terminal, load a model:

```bash
unified-gateway load my-mlx-model      # spawns/replaces the local inference process
unified-gateway load my-ollama-model   # no process spawned — just warms up the model in Ollama
```

`load` also starts the gateway itself if it isn't already running, so in practice you rarely need to invoke `unified-gateway` directly — `unified-gateway load <name>` is the one command you need day to day.

Point your tools at it:

| Client | Base URL |
|---|---|
| Claude Code | `http://localhost:8083` |
| OpenAI-compatible tools (OpenCode, etc.) | `http://localhost:8082` |

Example Claude Code settings:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8083",
    "ANTHROPIC_AUTH_TOKEN": "local",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "my-mlx-model",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "my-mlx-model"
  },
  "model": "sonnet"
}
```

## Running as a background service (macOS)

Two small standalone commands manage a `launchd` LaunchAgent that starts the gateway at login and restarts it if it crashes:

```bash
go build -o install-service ./cmd/install-service
go build -o uninstall-service ./cmd/uninstall-service

./install-service     # registers + starts now, idempotent
./uninstall-service   # stops + removes
```

`install-service` requires `unified-gateway` and `models.json` to already be present next to it. It writes `~/Library/LaunchAgents/local.unified-gateway.plist` and logs to `~/Library/Logs/unified-gateway/`. It manages the gateway process only — load a model separately with `unified-gateway load <name>`.

## Ports

| Interface | Port | Purpose |
|---|---|---|
| Anthropic Adapter | `8083` | Claude Code |
| OpenAI Adapter | `8082` | OpenCode and other OpenAI-compatible clients |
| Local backend (`mlx`/`ds4`) | `11435` (configurable) | The actively loaded model process |
| Ollama | `11434` (its own default) | Independent, always-on daemon |

## Known limitations

This gateway implements the Anthropic↔OpenAI translation needed for real-world Claude Code usage (including tool calling and streaming), but does not yet implement every edge case of Claude Code's wire protocol — notably prompt caching (`cache_control`), some request-shape optimizations around tool results, and IDE-specific tool sanitization. Contributions welcome.

## Contributing

Issues and pull requests are welcome. If you're adding support for a new backend type or a new Claude Code protocol edge case, please include a description of how you verified it (a `curl` transcript or a note on which client you tested against).

## License

[MIT](LICENSE)

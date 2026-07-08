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
- **Optional menu bar controller** — a small native tray app to start/stop the gateway, the model backend, and Ollama, and switch models, without a terminal

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

## Model discovery API

`GET /v1/models` on the OpenAI adapter (`:8082`) returns the full `models.json` catalog in the standard OpenAI shape (`{"object":"list","data":[...]}`), not just whatever the currently active backend happens to report — each entry is tagged `"active": true/false`. This is what lets an external client (a WebUI, say) discover which models exist and which one is loaded right now, without reading `models.json` off disk itself.

```bash
curl http://localhost:8082/v1/models
```

`POST /v1/models/<name>/load` triggers loading that model in the background and returns immediately (`202 Accepted`) — it can take minutes, so this doesn't hold the connection open waiting, the same way Ollama/LM Studio handle model loads. Works the same regardless of which backend `<name>` maps to in `models.json` (`mlx`, `ds4`, or `ollama`) — it's the same underlying `loadModel` the CLI and the auto-recovery path already use, so there's nothing backend-specific about this endpoint. Poll `GET /v1/models` afterward (or just retry your `/v1/chat/completions` call — it already self-heals on an unreachable backend) to see when it's ready.

```bash
curl -X POST http://localhost:8082/v1/models/my-model/load
```

Every `loadModel` call — this endpoint, `unified-gateway load` on the command line, and the automatic recovery when a request hits an unreachable backend — is serialized through a cross-process file lock (`load.lock`, next to the binary), so two overlapping load requests from different sources queue up and run cleanly one after another instead of racing on the backend port.

## Running as a background service (macOS)

Two small standalone commands manage a `launchd` LaunchAgent that starts the gateway at login and restarts it if it crashes:

```bash
go build -o install-service ./cmd/install-service
go build -o uninstall-service ./cmd/uninstall-service

./install-service     # registers + starts now, idempotent
./uninstall-service   # stops + removes
```

`install-service` requires `unified-gateway` and `models.json` to already be present next to it. It writes `~/Library/LaunchAgents/local.unified-gateway.plist` and logs to `~/Library/Logs/unified-gateway/`. It manages the gateway process only — load a model separately with `unified-gateway load <name>`.

## Menu bar controller (macOS)

`cmd/menubar` is a standalone tray app (using [`getlantern/systray`](https://github.com/getlantern/systray)) for controlling everything without a terminal:

<p align="center"><img src="docs/menubar-screenshot.png" alt="Unified Gateway menu bar menu" width="260"></p>

Reading it top to bottom:

- **`🟡 no model`** (or `🟢 <model-name>` / `🔴 stopped`) — what the gateway is *actually routing to right now*, asked directly of the gateway itself (`GET /v1/models` on its own OpenAI adapter). Green means it's up and serving a model; yellow means the adapters are up but nothing is loaded; red means they're stopped. Below it, **Start/Stop Gateway Adapters** are mutually exclusive — whichever doesn't apply right now is grayed out.
- **`rapid-mlx`** / **`ds4`** — one entry per local backend, each a submenu with its own **Start**, **Stop**, and a list of the models configured for that backend (`models.json`'s `"backend": "mlx"` / `"ds4"` entries); clicking a model loads it directly. The 🟢/🔴 dot and model name are read straight from the real process's own command line (which one owns the backend port, and its `--served-model-name`/`--model` argument) — not from a state file this tool wrote earlier, so it's still correct even if that process was started or replaced by something else.
- **`Ollama`** — same shape, but for the independently-running Ollama daemon: **Start Ollama**/**Stop Ollama** plus the list of `"backend": "ollama"` models to warm up. Whether it's running comes from checking the process itself, and which model it has loaded comes from Ollama's own `/api/ps` — so stopping Ollama by hand, outside this tool, is reflected immediately too.
- **Start All** / **Stop All** — bulk versions of the above.
- **Reload Settings** — the model list is only read from `models.json` once, at startup, and there's no clean way to rebuild an existing systray submenu tree in place — so this relaunches a fresh copy of the app (which reads `models.json` fresh) and quits the current one. Use it after editing `models.json` by hand, or after `unified-gateway` itself changes it.
- **Quit Menu Bar** — closes only this tray app. It never touches the gateway, the model backend, or Ollama; see [below](#why-two-separate-launchagents) for why.

**Port conflicts are handled at click time, not as a status you have to interpret.** Clicking any "Start" item first checks whether its target port is already occupied by something else; if so, a native confirmation dialog asks whether to stop it and proceed. Switching between models *within* the same backend (clicking a different model in `rapid-mlx`'s own list, say) never prompts — that's the everyday case and stays as frictionless as running `unified-gateway load <name>` directly.

```bash
go build -o unified-gateway-menubar ./cmd/menubar
cp unified-gateway-menubar ~/.local/bin/
```

It's registered as its **own, separate** LaunchAgent from the gateway's:

```bash
go build -o install-menubar ./cmd/install-menubar
go build -o uninstall-menubar ./cmd/uninstall-menubar

./install-menubar     # registers + starts now, idempotent
./uninstall-menubar   # stops + removes
```

#### Why two separate LaunchAgents

This is deliberate, not an oversight: `local.unified-gateway` (the actual API service) and `local.unified-gateway-menubar` (this tray app) are independent LaunchAgents. The gateway's `KeepAlive` respawns it on crash because other things — Claude Code, OpenCode, a future WebUI — depend on it being up continuously. The menu bar app has no `KeepAlive`: it's a convenience layer, so quitting or crashing it does not restart it automatically, and critically, does **not** affect the gateway, which keeps serving requests either way.

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

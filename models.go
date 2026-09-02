package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"
)

// ============================================================================
// Model configuration (models.json, read from the executable's own directory)
// ============================================================================

const omlxGatewayAPIKey = "unified-gateway-local"

// ModelConfig.Backend is one of:
//   - "mlx"    — served via `rapid-mlx`, spawned/killed by us on Config.BackendPort
//   - "ds4"    — served via `ds4-server`, same lifecycle as "mlx"
//   - "omlx"   — served via the installed oMLX runtime; used for checkpoint
//     formats (notably DFlash2) that Rapid-MLX/mlx-vlm cannot load
//   - "ollama" — served by an already-running `ollama serve` on Config.OllamaPort;
//     we never spawn/kill it, only warm up the requested model and point the
//     gateway's routing at Ollama's own OpenAI-compatible endpoint. OllamaModel
//     is Ollama's own model tag (e.g. "gemma4:31b-mlx"), which is usually not
//     the same as the shortname key used to select it.
//
// ModelConfig.Kind distinguishes ordinary chat models from special-purpose
// ones that aren't meant to show up in a chat model picker -- OCR (backend
// "mlx", has_vision:true, but its only use is "read this image", not
// conversation) and image-generation models like FLUX (backend "mflux")
// both qualify. Empty/omitted means "chat" -- the vast majority of
// entries, and the only value that existed before this field was added.
// "media" entries are filtered out of handleListModels (so opencode/pi/
// Claude Code never see them as a chat choice) and out of the menu bar's
// normal per-backend model lists, each getting its own section instead.
//
// Media-kind entries form two SHARED pools, mirroring exactly how chat
// models already work: switching within a pool is exclusive (loading one
// entry kills whatever else was on that pool's port), but the two pools
// -- and the chat pool -- are fully independent of each other. OCR-like
// entries (any backend except "mflux") share Config.MediaBackendPort;
// FLUX-family entries (backend "mflux") share Config.FluxBackendPort.
// Requesting/loading OCR never touches a FLUX model or the active chat
// model, and vice versa -- but two OCR-like models (or two FLUX models)
// can't run at once any more than two chat models can, since rapid-mlx/
// ds4-server/the flux server each only serve one model per process.
type ModelConfig struct {
	Path        string `json:"path,omitempty"`
	Label       string `json:"label"`
	Backend     string `json:"backend"`
	ModelType   string `json:"model_type,omitempty"`
	HasVision   bool   `json:"has_vision,omitempty"`
	Ctx         int    `json:"ctx,omitempty"`
	QuantBits   int    `json:"quant_bits,omitempty"`
	OllamaModel string `json:"ollama_model,omitempty"`
	Kind        string `json:"kind,omitempty"`
	// Disabled marks a model that must not be loaded or served (e.g. one that
	// is unstable/unsupported on the available backend). Disabled models stay
	// visible in GET /v1/models with active=false + disabled=true (so a client
	// like unicorn or the menu bar can render them greyed/non-selectable), but
	// every load/request path refuses them with a clear error instead of
	// attempting a (crashing) load.
	Disabled bool `json:"disabled,omitempty"`
	// ToolCallParser overrides rapid-mlx's own tool-call-format
	// auto-detection (vllm_mlx/model_auto_config.py), which matches
	// regexes against the model PATH STRING, not model_type -- e.g.
	// "qwen3" only fires if that substring literally appears in the
	// directory/repo name. Rebranded checkpoints (bonsai4b:
	// "Ternary-Bonsai-4B-mlx-2bit", base model Qwen3-4B per its own
	// README, but no "qwen3" anywhere in the path) fall through to no
	// parser at all -- tool calls silently don't work, not a loud
	// failure. Set explicitly here for any model whose path won't
	// self-match; passed as "--tool-call-parser <value>
	// --enable-auto-tool-choice" in launchMLX.
	ToolCallParser string `json:"tool_call_parser,omitempty"`
	// Ds4SsdStreaming passes --ssd-streaming to ds4-server (launchDS4)
	// instead of ds4's default full-residency load. Full residency has
	// no memory headroom check anywhere in ds4 before prefill -- it just
	// mmaps the whole model and hopes the Metal driver can still find
	// room for prefill's transient command-buffer allocations, which
	// fails with a raw kIOGPUCommandBufferCallbackErrorOutOfMemory once
	// there isn't much RAM left over after the model itself. --ssd-streaming
	// is ds4's own opt-in path with real headroom accounting (reserves an
	// explicit prefill-headroom budget up front) -- verified directly on
	// the antirez/ds4flash.gguf model (90.88 GiB) on a 128 GiB M5 Max:
	// full residency failed prefill with that exact OOM error, the same
	// model with this flag set planned 50.20 GiB total and completed a
	// real chat completion successfully. Only needed for models large
	// enough that full residency doesn't leave comfortable headroom.
	Ds4SsdStreaming bool `json:"ds4_ssd_streaming,omitempty"`
	// DFlash support (rapid-mlx speculative decoding, DFlash2 for Qwen3.8).
	// When set, launchMLX passes --speculative-config '{"method":"dflash","model":...}'
	// instead of --no-spec-decode, enabling block-diffusion draft verification.
	// Requires rapid-mlx built-in alias supports_dflash=true (e.g. qwen3.8-27b-8bit
	// added in aliases.json) and draft model at DflashDraftModel (HF ID or local path).
	// For quantized targets, dflash needs block_size <=5 (z-lab/dflash README).
	DflashDraftModel     string  `json:"dflash_draft_model,omitempty"`
	DflashBlockSize      int     `json:"dflash_block_size,omitempty"`
	MTPEnabled           bool    `json:"mtp_enabled,omitempty"`
	WorkingSetMultiplier float64 `json:"working_set_multiplier,omitempty"`
}

type Config struct {
	BackendPort                    int                    `json:"backend_port"`
	OllamaPort                     int                    `json:"ollama_port,omitempty"`
	MediaBackendPort               int                    `json:"media_backend_port,omitempty"`
	FluxBackendPort                int                    `json:"flux_backend_port,omitempty"`
	SmallBackendPort               int                    `json:"small_backend_port,omitempty"`
	VenvDir                        string                 `json:"venv_dir"`
	DS4Dir                         string                 `json:"ds4_dir"`
	MfluxVenvDir                   string                 `json:"mflux_venv_dir,omitempty"`
	FluxServerScript               string                 `json:"flux_server_script,omitempty"`
	OmlxBin                        string                 `json:"omlx_bin,omitempty"`
	MemoryWatchdogThresholdPercent float64                `json:"memory_watchdog_threshold_percent,omitempty"`
	MemoryWatchdogIntervalSeconds  int                    `json:"memory_watchdog_interval_seconds,omitempty"`
	StallWatchdogThresholdSeconds  int                    `json:"stall_watchdog_threshold_seconds,omitempty"`
	Models                         map[string]ModelConfig `json:"models"`
}

func expandHome(path string) string {
	if strings.HasPrefix(path, "~/") {
		home, _ := os.UserHomeDir()
		return filepath.Join(home, path[2:])
	}
	return path
}

func gatewayDir() string {
	if exe, err := os.Executable(); err == nil {
		return filepath.Dir(exe)
	}
	return "."
}

func loadConfig() (*Config, error) {
	path := filepath.Join(gatewayDir(), "models.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("cannot read %s: %w", path, err)
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("invalid models.json: %w", err)
	}
	cfg.VenvDir = expandHome(cfg.VenvDir)
	cfg.DS4Dir = expandHome(cfg.DS4Dir)
	cfg.MfluxVenvDir = expandHome(cfg.MfluxVenvDir)
	cfg.OmlxBin = expandHome(cfg.OmlxBin)
	if cfg.OmlxBin == "" {
		cfg.OmlxBin = "/opt/homebrew/bin/omlx"
	}
	cfg.FluxServerScript = expandHome(cfg.FluxServerScript)
	if cfg.OllamaPort == 0 {
		cfg.OllamaPort = 11434
	}
	if cfg.MediaBackendPort == 0 {
		cfg.MediaBackendPort = cfg.BackendPort + 1
	}
	if cfg.FluxBackendPort == 0 {
		cfg.FluxBackendPort = cfg.MediaBackendPort + 1
	}
	if cfg.SmallBackendPort == 0 {
		cfg.SmallBackendPort = cfg.FluxBackendPort + 1
	}
	// 0 (absent from models.json) means "use the default, watchdog on."
	// A negative value is the explicit opt-out (free% can never go
	// negative, so the check in checkMemoryPressure never fires).
	if cfg.MemoryWatchdogThresholdPercent == 0 {
		cfg.MemoryWatchdogThresholdPercent = defaultWatchdogThresholdPercent
	}
	if cfg.MemoryWatchdogIntervalSeconds == 0 {
		cfg.MemoryWatchdogIntervalSeconds = defaultWatchdogIntervalSeconds
	}
	if cfg.StallWatchdogThresholdSeconds == 0 {
		cfg.StallWatchdogThresholdSeconds = defaultStallWatchdogThresholdSeconds
	}
	for name, m := range cfg.Models {
		m.Path = expandHome(m.Path)
		m.DflashDraftModel = expandHome(m.DflashDraftModel)
		cfg.Models[name] = m
	}
	return &cfg, nil
}

// modelPoolPort returns which shared port an entry belongs to, by kind:
//   - chat (Kind empty): Config.BackendPort
//   - "media" + backend "mflux": Config.FluxBackendPort
//   - "media" (OCR-like): Config.MediaBackendPort
//   - "small": Config.SmallBackendPort
//
// All are genuine pools, not per-model ports -- loading one entry within a
// pool is exclusive within that pool, exactly like the chat pool, but the
// pools never affect each other. The "small" pool exists so a cheap
// secondary model (e.g. opencode's small_model for conversation titles)
// stays resident on its own port without killing the main chat model on
// every request that names it.
func modelPoolPort(cfg *Config, m ModelConfig) int {
	if m.Kind == "media" {
		if m.Backend == "mflux" {
			return cfg.FluxBackendPort
		}
		return cfg.MediaBackendPort
	}
	if m.Kind == "small" {
		return cfg.SmallBackendPort
	}
	return cfg.BackendPort
}

// mediaPoolPort is kept as a thin alias for call sites that semantically
// mean "a media-kind pool port" -- see modelPoolPort for the general case.
func mediaPoolPort(cfg *Config, m ModelConfig) int {
	return modelPoolPort(cfg, m)
}

// ============================================================================
// Active backend state — determined live, every time, by inspecting the
// real processes and Ollama's own API. There is deliberately no state
// file: one was tried (active-backend.json, written by loadModel on every
// switch) and it went stale the instant a backend process died on its own
// — crash, manual kill, anything not funneled through loadModel — leaving
// the gateway confidently routing to (and reporting as "active") a model
// that no longer existed. Nothing here is cached across calls.
// ============================================================================

type activeBackend struct {
	Port          int    // 0 if nothing is actually live right now
	UpstreamModel string // set only for Ollama: its own tag for the loaded model
	Model         string // shortname key in models.json, "" if nothing live matches one
}

// portOwnerCommand returns the full command line of whichever process is
// listening on port, or "" if nothing is listening. Restricted to LISTEN
// sockets only (-sTCP:LISTEN) — without it, lsof also matches outbound
// client connections (e.g. the gateway's own keep-alive HTTP connection
// to the backend while proxying), which showed up as a lower PID and got
// mistaken for the actual owner.
func portOwnerCommand(port int) string {
	out, err := exec.Command("lsof", "-ti", fmt.Sprintf(":%d", port), "-sTCP:LISTEN").Output()
	if err != nil {
		return ""
	}
	pids := strings.Fields(string(out))
	if len(pids) == 0 {
		return ""
	}
	psOut, err := exec.Command("ps", "-p", pids[0], "-o", "command=").Output()
	if err != nil {
		return ""
	}
	return string(psOut)
}

// commandFlagValue returns the value following flag in a space-separated
// command line, or "" if flag isn't present.
func commandFlagValue(cmd, flag string) string {
	fields := strings.Fields(cmd)
	for i, f := range fields {
		if f == flag && i+1 < len(fields) {
			return fields[i+1]
		}
	}
	return ""
}

// runningBackendModel inspects whichever real process owns port right now
// and returns the models.json shortname it's serving. Works for any of the
// three locally-spawned backend types, on whichever port they happen to be
// asked about (the chat pool's BackendPort, the OCR-like pool's
// MediaBackendPort, or the FLUX pool's FluxBackendPort -- all three share
// this same introspection, never a cached value):
//   - rapid-mlx: reads its own --served-model-name argument directly
//   - ds4-server: has no equivalent flag, so its --model path is matched
//     back against configured "ds4" entries
//   - flux-server/server.py (mflux): same idea as ds4 -- no --served-model-name
//     equivalent, so its --model-path is matched back against configured
//     "mflux" entries
//
// "" if nothing is listening, or it doesn't look like any of the three.
func runningBackendModel(cfg *Config, port int) string {
	cmd := portOwnerCommand(port)
	if cmd == "" {
		return ""
	}
	if strings.Contains(cmd, "rapid-mlx") {
		return commandFlagValue(cmd, "--served-model-name")
	}
	if strings.Contains(cmd, "omlx") {
		basePath := commandFlagValue(cmd, "--base-path")
		if basePath != "" {
			return filepath.Base(basePath)
		}
		// omlx-server clears its own argv (ps shows just "omlx-server",
		// never the --base-path), so the command line can't tell us which
		// model it's serving. Ask the server directly which model is loaded.
		return runningOmlxModel(cfg, port)
	}
	if strings.Contains(cmd, "ds4-server") {
		path := commandFlagValue(cmd, "--model")
		for name, m := range cfg.Models {
			if m.Backend == "ds4" && m.Path == path {
				return name
			}
		}
	}
	if strings.Contains(cmd, "flux-server") {
		path := commandFlagValue(cmd, "--model-path")
		for name, m := range cfg.Models {
			if m.Backend == "mflux" && m.Path == path {
				return name
			}
		}
	}
	return ""
}

// runningOmlxModel asks the oMLX server (using the gateway's shared API
// key) which model it currently has loaded, and returns the gateway
// shortname for it. oMLX exposes /v1/models/status which carries each
// model's loaded state plus the profile model_alias (= the gateway
// shortname); if the alias is absent we match the model id (the model
// directory basename) back against configured omlx entries.
func runningOmlxModel(cfg *Config, port int) string {
	req, err := http.NewRequest(http.MethodGet, fmt.Sprintf("http://127.0.0.1:%d/v1/models/status", port), nil)
	if err != nil {
		return ""
	}
	req.Header.Set("Authorization", "Bearer "+omlxGatewayAPIKey)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return ""
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return ""
	}
	var parsed struct {
		Models []struct {
			ID         string `json:"id"`
			Loaded     bool   `json:"loaded"`
			ModelAlias string `json:"model_alias"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return ""
	}
	for _, m := range parsed.Models {
		if !m.Loaded {
			continue
		}
		if m.ModelAlias != "" {
			return m.ModelAlias
		}
		// Fall back: match the model directory basename against omlx entries.
		for name, mc := range cfg.Models {
			if mc.Backend == "omlx" && filepath.Base(mc.Path) == m.ID {
				return name
			}
		}
	}
	return ""
}

// has loaded in memory (its own tag, e.g. "gemma4:31b-mlx"), "" if none.
func queryOllamaLoadedModel(port int) string {
	resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/api/ps", port))
	if err != nil {
		return ""
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return ""
	}
	var parsed struct {
		Models []struct {
			Name  string `json:"name"`
			Model string `json:"model"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil || len(parsed.Models) == 0 {
		return ""
	}
	if parsed.Models[0].Name != "" {
		return parsed.Models[0].Name
	}
	return parsed.Models[0].Model
}

// resolveActiveBackend determines what's actually running right now, live:
// whichever of rapid-mlx/ds4-server owns cfg.BackendPort (they're mutually
// exclusive, sharing one port), or failing that, whatever Ollama reports
// as loaded via its own API. Falls back to Port: fallbackPort with
// everything else empty if neither is detected, so a caller can still
// form a URL even though nothing will answer it.
func resolveActiveBackend(cfg *Config, fallbackPort int) activeBackend {
	if cfg != nil {
		if name := runningBackendModel(cfg, cfg.BackendPort); name != "" {
			return activeBackend{Port: cfg.BackendPort, Model: name}
		}
		if tag := queryOllamaLoadedModel(cfg.OllamaPort); tag != "" {
			for name, m := range cfg.Models {
				if m.Backend == "ollama" {
					wantTag := m.OllamaModel
					if wantTag == "" {
						wantTag = name
					}
					if wantTag == tag {
						return activeBackend{Port: cfg.OllamaPort, UpstreamModel: tag, Model: name}
					}
				}
			}
			// Ollama has something loaded, but it doesn't match any
			// configured alias — still route requests there, just
			// without a models.json shortname to report.
			return activeBackend{Port: cfg.OllamaPort, UpstreamModel: tag}
		}
	}
	return activeBackend{Port: fallbackPort}
}

// resolveActiveMediaPool is resolveActiveBackend's counterpart for a media
// pool's shared port (Config.MediaBackendPort for OCR-like entries,
// Config.FluxBackendPort for FLUX-family ones) -- exclusive within that
// pool exactly like the chat pool is within itself, but fully independent
// of the chat pool and of the OTHER media pool. No Ollama case here --
// media models are always locally-spawned processes, never an Ollama tag.
func resolveActiveMediaPool(cfg *Config, poolPort int) activeBackend {
	if cfg == nil || poolPort == 0 {
		return activeBackend{Port: poolPort}
	}
	if name := runningBackendModel(cfg, poolPort); name != "" {
		return activeBackend{Port: poolPort, Model: name}
	}
	return activeBackend{Port: poolPort}
}

// isMediaModelActive reports whether shortName (a "kind":"media" entry) is
// specifically the one active on its pool's shared port right now.
func isMediaModelActive(cfg *Config, shortName string) bool {
	if cfg == nil {
		return false
	}
	m, ok := cfg.Models[shortName]
	if !ok || m.Kind != "media" {
		return false
	}
	return resolveActiveMediaPool(cfg, mediaPoolPort(cfg, m)).Model == shortName
}

// isSmallModelActive reports whether shortName (a "kind":"small" entry) is
// specifically the one active on the small pool's shared port right now --
// the same live introspection media models use, just against
// Config.SmallBackendPort.
func isSmallModelActive(cfg *Config, shortName string) bool {
	if cfg == nil {
		return false
	}
	m, ok := cfg.Models[shortName]
	if !ok || m.Kind != "small" {
		return false
	}
	return resolveActiveMediaPool(cfg, cfg.SmallBackendPort).Model == shortName
}

// warmOllamaModel triggers Ollama to load the model into memory now, via its
// native /api/generate endpoint with an empty prompt, instead of waiting for
// the first real chat request to pay that cost.
func warmOllamaModel(port int, model string) error {
	payload := map[string]interface{}{"model": model, "prompt": "", "stream": false}
	body, _ := json.Marshal(payload)
	resp, err := http.Post(fmt.Sprintf("http://127.0.0.1:%d/api/generate", port), "application/json", bytes.NewReader(body))
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return fmt.Errorf("ollama returned status %d", resp.StatusCode)
	}
	return nil
}

// ============================================================================
// Backend process management (replaces llm-launch)
// ============================================================================

// Both functions restrict lsof to LISTEN sockets only (-sTCP:LISTEN).
// Without it, lsof also matches the gateway's own outbound client
// connections to the backend port (e.g. its keep-alive HTTP connection
// while proxying requests) — killPort(cfg.BackendPort) could then kill
// the gateway process itself instead of (or in addition to) the actual
// backend, since the gateway is always a "user" of that port too.

func portInUse(port int) bool {
	out, err := exec.Command("lsof", "-ti", fmt.Sprintf(":%d", port), "-sTCP:LISTEN").Output()
	return err == nil && strings.TrimSpace(string(out)) != ""
}

func killPort(port int) {
	out, err := exec.Command("lsof", "-ti", fmt.Sprintf(":%d", port), "-sTCP:LISTEN").Output()
	if err != nil {
		return
	}
	for _, pidStr := range strings.Fields(string(out)) {
		exec.Command("kill", "-9", pidStr).Run()
	}
}

// clearDiskKVCheckpoints removes rapid-mlx's disk-backed KV checkpoints
// (~/.cache/rapid-mlx/kv_checkpoints/). Called automatically on every model
// switch (loadModelLocked): with --kv-disk-checkpoint-interval 0 (the
// launcher default) no NEW checkpoints are written, but the ones accumulated
// before that fix — or by a previous model — are keyed by the OLD model's
// request hash (request_hash includes the model name), so a switched model
// can never reuse them. Deleting them here keeps the disk from silently
// filling with stale snapshots across model switches. The persisted
// prefix_cache/ (per-model subdirs, saved at shutdown / loaded at startup)
// is deliberately NOT touched: those are still useful for the model being
// loaded.
func clearDiskKVCheckpoints() {
	home, err := os.UserHomeDir()
	if err != nil {
		return
	}
	root := filepath.Join(home, ".cache", "rapid-mlx", "kv_checkpoints")
	entries, err := os.ReadDir(root)
	if err != nil {
		return // missing or unreadable -- nothing to clear
	}
	removed := 0
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		if os.RemoveAll(filepath.Join(root, e.Name())) == nil {
			removed++
		}
	}
	if removed > 0 {
		fmt.Printf("[unified-gateway] cleared %d stale KV checkpoint dir(s) from %s\n", removed, root)
	}
}

func waitForPort(port int, timeout time.Duration, apiKey string) bool {
	deadline := time.Now().Add(timeout)
	url := fmt.Sprintf("http://127.0.0.1:%d/v1/models", port)
	for time.Now().Before(deadline) {
		req, _ := http.NewRequest(http.MethodGet, url, nil)
		if apiKey != "" {
			req.Header.Set("Authorization", "Bearer "+apiKey)
		}
		resp, err := http.DefaultClient.Do(req)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				return true
			}
		}
		time.Sleep(2 * time.Second)
	}
	return false
}

// omlxAPIKey returns the Bearer token to send when polling a backend's
// readiness on the given backend, or "" for backends that don't require
// auth on /v1/models. oMLX gates every route (including /v1/models) behind
// an API key, so the gateway must send its shared key or the readiness poll
// sees a 401 and would never pass -- and preloadOMLX would never run.
func omlxAPIKey(backend string) string {
	if backend == "omlx" {
		return omlxGatewayAPIKey
	}
	return ""
}

func launchMLX(cfg *Config, shortName string, m ModelConfig, port int) (*exec.Cmd, error) {
	rapidBin := filepath.Join(cfg.VenvDir, "bin", "rapid-mlx")
	args := []string{
		"serve", m.Path,
		"--host", "127.0.0.1",
		"--port", fmt.Sprintf("%d", port),
		"--served-model-name", shortName,
	}
	if (m.ModelType == "qwen3" || m.ModelType == "qwen3_5" || m.ModelType == "qwen2_moe") && m.HasVision {
		args = append(args, "--text-only", "--no-mllm")
	}
	// DFlash takes precedence over --no-spec-decode: when a draft is configured,
	// speculative decoding must stay enabled (enable_dflash branch). Gate on
	// DflashDraftModel so only qw38-dflash (or future DFlash2 models) enable it.
	if m.DflashDraftModel != "" {
		draftPath := expandHome(m.DflashDraftModel)
		// For DFlash, speculative-config only supports method+model (no num_speculative_tokens)
		// Block size for quantized targets (<=5) is handled internally by the DFlash server.
		specCfg := fmt.Sprintf(`{"method":"dflash","model":%q}`, draftPath)
		args = append(args, "--speculative-config", specCfg)
	} else if m.ModelType == "qwen3" || m.ModelType == "qwen3_5" || m.ModelType == "qwen2_moe" {
		args = append(args, "--no-spec-decode")
	}
	if m.ToolCallParser != "" {
		args = append(args, "--tool-call-parser", m.ToolCallParser, "--enable-auto-tool-choice")
	}
	cacheMB := mlxCacheReserveMBFor(estimateModelSizeGB(m))
	args = append(args, "--cache-memory-mb", fmt.Sprintf("%d", cacheMB))
	// TurboQuant (KV-cache quantization) re-enabled: the three rapid-mlx
	// crashes (2026-07-08, 07-09, 07-10) all trace to a documented upstream
	// MLX bug (rapid-mlx engine_core.py, issue #353 / mlx-lm#1015) --
	// mlx::core::gpu::check_error throws from Metal's async completion
	// queue, which can't propagate through Python and aborts the whole
	// process. TurboQuant isn't the cause (2 of 3 crashes predate it) and
	// mlx 0.32.0 (PR #3523, "catch error in CommandBuffer and poison the
	// events") ships the actual fix. TurboQuant matters for exactly the
	// workload PFlash was supposed to help: it shrinks the KV cache's
	// per-token footprint, letting more conversation history stay
	// cache-resident before capacity eviction forces a recompute.
	// k8v4 (K=8-bit Walsh-Hadamard, V=4-bit Lloyd-Max) instead of the bare
	// flag's v4 default (V-only, K stays fp16). Measured 2026-07-14 on
	// qw3635 with matched cold prompts at two context sizes: at ~19k
	// tokens v4 and k8v4 are a wash (75 vs 77 tok/s decode), but at ~80k
	// tokens k8v4 wins clearly (48.8 vs 63.6 tok/s decode, +30%) because
	// decode on Apple Silicon is bandwidth-bound on the KV-cache read as
	// context grows, and k8v4 roughly halves what has to be read per
	// token vs v4. No quality regression observed in spot checks. This
	// directly targets the "slows down on long prompts" symptom that
	// PFlash was originally meant to help with (see below) -- unlike
	// PFlash, it doesn't touch the prefix-cache path at all.
	//
	// EXCEPT for 8-bit weight-quantized models: measured 2026-07-14 on
	// qw3coder8bit, same prompt/config -- v4 gave ~79 tok/s, k8v4 gave
	// ~5.9 tok/s. A 13x regression, not a wash, reproduced twice. Only
	// ever benchmarked k8v4 against 4-bit/5-bit weights (qw3635,
	// qwopuscoder) before making it the default; never against 8-bit.
	// Whatever's happening (K-cache's own 8-bit quantization colliding
	// with the model weights' 8-bit quantization on the same Metal
	// kernel path is the leading guess) it's model-weight-precision
	// specific, not something PFlash/prefix-cache/context-length
	// explains -- so gate strictly on QuantBits, not on model size or type.
	turboQuant := "k8v4"
	if m.QuantBits == 8 {
		turboQuant = "v4"
	}
	args = append(args, "--kv-cache-turboquant", turboQuant)
	// Disk-backed KV checkpointing (R15 #296) DISABLED: rapid-mlx snapshots
	// the whole KV state to ~/.cache/rapid-mlx/kv_checkpoints/ every
	// --kv-disk-checkpoint-interval (default 256) tokens, and each snapshot
	// is the full live KV (~5 GiB at 76k tokens, measured 2026-08-18). With
	// loads=0 for the entire process lifetime (the load path only runs at
	// startup via scan_checkpoints, and a restart reloads from the in-memory
	// prefix-cache save instead), these writes are pure I/O churn on the
	// generation hot path: 191 writes + 194 disk-cap evictions in one
	// session, ~20 GiB written for zero reuse. Disabling it removes that
	// cost entirely; if disk persistence across restarts is ever needed it
	// can be re-enabled for the specific workload that needs it.
	args = append(args, "--kv-disk-checkpoint-interval", "0")
	// Hybrid (Mamba/GatedDeltaNet) models: rapid-mlx drops prefix-cache
	// entries by default (hybrid_reuse_max_entries=0, #1025/#1058), so every
	// turn of a hybrid model like Qwen3-Coder-Next re-prefills the whole
	// context. #1103 added a safe, bounded within-conversation reuse (exact /
	// prefix-extension fetch, LRU-evicted). Opt in here so hybrid models reuse
	// their prefix too; non-hybrid models are unaffected by the gate.
	args = append(args, "--hybrid-cache-entries", "8")
	// PFlash OFF by default: rapid-mlx's own metrics.py documents that
	// "when PFlash compression engages, the prompt skips the prefix-cache
	// fetch + store paths entirely" -- and qw3627/qw27/etc are
	// pflash_tier=verified aliases, defaulting to --pflash always. Measured
	// directly on a real OpenCode session: rapid_mlx_prefix_cache_hits_total
	// stayed at 0 across 4 consecutive requests (misses=4) once PFlash was
	// enabled, including trivial one-line follow-ups -- because the
	// accumulated conversation crosses PFlash's ~32k-token auto-threshold
	// once, and every request after that bypasses the cache and reprocesses
	// close to the full history instead of just the new turn's tokens. For
	// the long, growing, multi-turn agentic sessions this gateway actually
	// serves (OpenCode/Claude Code), prefix-cache reuse is worth far more
	// than PFlash's one-shot compression -- so PFlash stays off here, and
	// is left available as a manual `--pflash auto` opt-in (not wired into
	// this launcher) for genuine single-shot huge-paste-and-analyze use
	// without follow-up turns.

	cmd := exec.Command(rapidBin, args...)
	cmd.Env = append(os.Environ(),
		"VIRTUAL_ENV="+cfg.VenvDir,
		"PATH="+filepath.Join(cfg.VenvDir, "bin")+":"+os.Getenv("PATH"),
	)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	// rapid-mlx's own stdout/stderr were previously discarded entirely
	// (unset Cmd.Stdout/Stderr defaults to os.DevNull) -- every internal
	// log line (scheduler decisions, cache admission, warnings) was
	// invisible. Captured here instead of mixed into the gateway's own
	// log so the two don't interleave.
	if logFile, err := os.OpenFile(filepath.Join(gatewayDir(), "rapid-mlx.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644); err == nil {
		cmd.Stdout = logFile
		cmd.Stderr = logFile
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return cmd, nil
}

func launchDS4(cfg *Config, m ModelConfig, port int) (*exec.Cmd, error) {
	ds4Server := filepath.Join(cfg.DS4Dir, "ds4-server")
	ctx := m.Ctx
	if ctx == 0 {
		ctx = 32768
	}
	args := []string{
		"--model", m.Path,
		"--metal", "--ctx", fmt.Sprintf("%d", ctx),
		"--power", "100",
		"--host", "127.0.0.1", "--port", fmt.Sprintf("%d", port),
	}
	if m.Ds4SsdStreaming {
		args = append(args, "--ssd-streaming")
	}
	cmd := exec.Command(ds4Server, args...)
	cmd.Dir = cfg.DS4Dir
	cmd.Env = append(os.Environ(),
		"DS4_METAL_FLASH_ATTN_SOURCE="+filepath.Join(cfg.DS4Dir, "metal", "flash_attn.metal"),
	)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return cmd, nil
}

// launchOMLX creates an isolated local profile for one gateway shortname.
// The user's oMLX GUI settings remain untouched, and --base-path doubles as
// live process identity for runningBackendModel.
func launchOMLX(cfg *Config, shortName string, m ModelConfig, port int) (*exec.Cmd, error) {
	basePath := filepath.Join(gatewayDir(), "omlx", shortName)
	if err := writeOMLXProfile(basePath, shortName, m); err != nil {
		return nil, err
	}
	args := []string{
		"serve", "--model-dir", filepath.Dir(m.Path),
		"--host", "127.0.0.1", "--port", fmt.Sprintf("%d", port),
		"--base-path", basePath, "--memory-guard", "balanced",
		"--max-concurrent-requests", "1", "--no-cache",
		"--api-key", omlxGatewayAPIKey,
	}
	cmd := exec.Command(cfg.OmlxBin, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	if logFile, err := os.OpenFile(filepath.Join(gatewayDir(), "omlx.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644); err == nil {
		cmd.Stdout = logFile
		cmd.Stderr = logFile
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return cmd, nil
}

func writeOMLXProfile(basePath, shortName string, m ModelConfig) error {
	if err := os.MkdirAll(basePath, 0755); err != nil {
		return fmt.Errorf("create oMLX base path: %w", err)
	}
	settings := map[string]any{
		"model_alias":    shortName,
		"is_hidden":      false,
		"dflash_enabled": m.DflashDraftModel != "",
		"mtp_enabled":    m.MTPEnabled,
	}
	if m.DflashDraftModel != "" {
		settings["dflash_draft_model"] = m.DflashDraftModel
		settings["dflash_draft_quant_enabled"] = false
		settings["dflash_block_size"] = m.DflashBlockSize
		settings["dflash_max_ctx"] = m.Ctx
	}
	profile := map[string]any{
		"version": 1,
		"models":  map[string]any{filepath.Base(m.Path): settings},
	}
	data, err := json.MarshalIndent(profile, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(basePath, "model_settings.json"), append(data, '\n'), 0644); err != nil {
		return fmt.Errorf("write oMLX model profile: %w", err)
	}
	return nil
}

func preloadOMLX(port int, m ModelConfig) error {
	modelID := url.PathEscape(filepath.Base(m.Path))
	req, err := http.NewRequest(http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/admin/api/models/%s/load", port, modelID), nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+omlxGatewayAPIKey)
	resp, err := (&http.Client{Timeout: 15 * time.Minute}).Do(req)
	if err != nil {
		return fmt.Errorf("oMLX preload request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("oMLX preload returned HTTP %d", resp.StatusCode)
	}
	return nil
}

func launchConfiguredBackend(cfg *Config, shortName string, m ModelConfig, port int) (*exec.Cmd, error) {
	switch m.Backend {
	case "ds4":
		return launchDS4(cfg, m, port)
	case "mflux":
		return launchMflux(cfg, m, port)
	case "omlx":
		return launchOMLX(cfg, shortName, m, port)
	default:
		return launchMLX(cfg, shortName, m, port)
	}
}

// launchMflux spawns the persistent Python server (cfg.FluxServerScript,
// run from cfg.MfluxVenvDir) that wraps mflux's Python API -- mflux itself
// has no server mode, only one-shot CLI commands that reload the model
// from disk on every invocation, which would make "start once, generate
// many times" and auto-load-on-request impossible. m.ModelType carries
// mflux's own config-name alias (e.g. "dev" for FLUX.1-dev, or
// "flux2-klein-4b"), which the script uses to pick the right model
// architecture defaults while still loading actual weights from m.Path
// (a local directory) instead of triggering a HuggingFace download.
func launchMflux(cfg *Config, m ModelConfig, port int) (*exec.Cmd, error) {
	python := filepath.Join(cfg.MfluxVenvDir, "bin", "python")
	args := []string{
		cfg.FluxServerScript,
		"--model-path", m.Path,
		"--config-name", m.ModelType,
		"--port", fmt.Sprintf("%d", port),
	}
	cmd := exec.Command(python, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	if logFile, err := os.OpenFile(filepath.Join(gatewayDir(), "flux-server.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644); err == nil {
		cmd.Stdout = logFile
		cmd.Stderr = logFile
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return cmd, nil
}

// ensureGatewayRunning starts a detached copy of this same binary (with no
// args, i.e. the persistent Anthropic/OpenAI adapter servers) if the adapter
// ports aren't already listening.
func ensureGatewayRunning() {
	if portInUse(8082) || portInUse(8083) {
		return
	}
	exe, err := os.Executable()
	if err != nil {
		return
	}
	logPath := filepath.Join(gatewayDir(), "gateway.log")
	logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		logFile = nil
	}
	cmd := exec.Command(exe)
	if logFile != nil {
		cmd.Stdout = logFile
		cmd.Stderr = logFile
	}
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	if err := cmd.Start(); err != nil {
		fmt.Printf("[unified-gateway] WARNING: failed to start gateway adapters: %v\n", err)
		return
	}
	waitForPort(8082, 15*time.Second, "")
}

var (
	autoLoadMu       sync.Mutex
	autoLoadInFlight bool
)

// crashTracker catches a specific failure mode: rapid-mlx crashing (SIGABRT,
// a Metal GPU command-buffer error -- see mlxCacheReserveMBFor's doc comment
// neighbor above) on a growing/heavy prompt. Without this, the sequence is
// invisible from outside a single request but devastating in aggregate:
// crash -> connection error -> ensureBackendLoading reloads the model (cold,
// tens of seconds) -> client retries the same prompt -> crashes again on the
// same GPU condition -> repeat. Each cycle looks like an ordinary retry;
// several cycles is the "tens of minutes" a user actually experiences.
// Tracking lets ensureBackendLoading refuse to keep feeding that loop and
// surface a real error instead.
var crashTracker = &crashState{
	loadedAt:    make(map[string]time.Time),
	recentCrash: make(map[string][]time.Time),
}

type crashState struct {
	mu          sync.Mutex
	loadedAt    map[string]time.Time
	recentCrash map[string][]time.Time
}

// quickCrashWindow: dying this soon after becoming ready looks like the
// GPU-crash pattern, not an unrelated/deliberate shutdown (e.g. a manual
// `load` of a different model, which also kills this port).
const quickCrashWindow = 5 * time.Minute

// crashLoopWindow/crashLoopThreshold: this many quick crashes inside this
// long a lookback means "reloading isn't helping, stop trying automatically."
const crashLoopWindow = 10 * time.Minute
const crashLoopThreshold = 2

func (c *crashState) recordLoad(shortName string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.loadedAt[shortName] = time.Now()
}

func (c *crashState) recordExit(shortName string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	loadedAt, ok := c.loadedAt[shortName]
	if !ok || time.Since(loadedAt) > quickCrashWindow {
		return
	}
	now := time.Now()
	cutoff := now.Add(-crashLoopWindow)
	kept := c.recentCrash[shortName][:0]
	for _, t := range c.recentCrash[shortName] {
		if t.After(cutoff) {
			kept = append(kept, t)
		}
	}
	c.recentCrash[shortName] = append(kept, now)
}

func (c *crashState) isLooping(shortName string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	cutoff := time.Now().Add(-crashLoopWindow)
	count := 0
	for _, t := range c.recentCrash[shortName] {
		if t.After(cutoff) {
			count++
		}
	}
	return count >= crashLoopThreshold
}

// ensureBackendLoading is called by the request handlers when the backend
// port turns out to be unreachable — it self-heals by loading the model
// the client actually asked for, in the background, so a client that never
// ran `unified-gateway load` (or whose backend crashed) recovers on its
// own instead of failing forever. It never blocks the caller: Claude Code
// and other clients already retry on 5xx responses (observed retrying
// automatically after the "Local LLM Backend unreachable" error), so the
// retry is what picks up the now-loaded model — we don't need to hold the
// original HTTP request open for however long a model load takes.
// autoLoadInFlight prevents piling up redundant loads if several requests
// arrive (or retry) while one is already in progress.
//
// Returns false (and does NOT reload) if this model has crashed repeatedly
// right after loading — see crashTracker's doc comment. Reloading again
// would just feed the same crash-reload-retry-crash loop; the caller should
// surface a real error instead of implying a retry will help.
func ensureBackendLoading(shortName string) bool {
	if shortName == "" {
		return false
	}
	if crashTracker.isLooping(shortName) {
		fmt.Printf("[unified-gateway] %q crashed repeatedly right after loading -- not auto-reloading again, needs manual attention\n", shortName)
		return false
	}
	autoLoadMu.Lock()
	if autoLoadInFlight {
		autoLoadMu.Unlock()
		return true
	}
	autoLoadInFlight = true
	autoLoadMu.Unlock()

	go func() {
		defer func() {
			autoLoadMu.Lock()
			autoLoadInFlight = false
			autoLoadMu.Unlock()
		}()
		fmt.Printf("[unified-gateway] backend unreachable, auto-loading %q...\n", shortName)
		if err := loadModel(shortName); err != nil {
			fmt.Printf("[unified-gateway] auto-load of %q failed: %v\n", shortName, err)
		}
	}()
	return true
}

// loadLockPath is a plain empty file used only as an flock() target — its
// contents are never read or written.
func loadLockPath() string {
	return filepath.Join(gatewayDir(), "load.lock")
}

// withLoadLock serializes every loadModel call across ALL processes, not
// just goroutines in this one — loadModel can be triggered independently
// by a direct `unified-gateway load <name>` CLI invocation, the HTTP
// /v1/models/:id/load endpoint, and the auto-load-on-request path, each
// potentially a different OS process. Without a cross-process lock, two
// concurrent loads race on killPort/spawn/writeActiveBackend: observed
// live (2026-07-08) as active-backend.json reporting "laguna" while the
// real process on the backend port was actually qw27, because a second
// load's status write landed after the first one's, even though the
// first one's process is what actually survived.
//
// flock (not a "lockfile exists" check) is what makes this safe rather
// than just less-likely-to-race: if the process holding the lock dies —
// crash, kill -9, whatever — the kernel releases it the moment the file
// descriptor closes. There's no stale lock to detect or clean up by hand.
// LOCK_EX blocks until the previous load finishes, so a second request
// queues and runs cleanly afterward instead of failing fast or racing.
func withLoadLock(fn func() error) error {
	f, err := os.OpenFile(loadLockPath(), os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		return fmt.Errorf("cannot open lock file: %w", err)
	}
	defer f.Close()

	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX); err != nil {
		return fmt.Errorf("cannot acquire load lock: %w", err)
	}
	defer syscall.Flock(int(f.Fd()), syscall.LOCK_UN)

	return fn()
}

// loadModel kills whatever is on the backend port and launches the requested
// model, then makes sure the gateway adapters are up. This is the single
// entry point that replaces `llm-launch openai/anthropic <shortname>`.
// Serialized across processes by withLoadLock — see its doc comment.
func loadModel(shortName string) error {
	var err error
	lockErr := withLoadLock(func() error {
		err = loadModelLocked(shortName)
		return nil
	})
	if lockErr != nil {
		return lockErr
	}
	return err
}

// modelDisabled reports whether a configured model is marked disabled
// (must not be loaded/served). Reads config fresh, like every other handler.
func modelDisabled(shortName string) bool {
	cfg, err := loadConfig()
	if err != nil {
		return false
	}
	m, ok := cfg.Models[shortName]
	return ok && m.Disabled
}

func loadModelLocked(shortName string) error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}
	m, ok := cfg.Models[shortName]
	if !ok {
		names := make([]string, 0, len(cfg.Models))
		for n := range cfg.Models {
			names = append(names, n)
		}
		return fmt.Errorf("unknown model %q — available: %s", shortName, strings.Join(names, ", "))
	}
	if m.Disabled {
		return fmt.Errorf("model %q is disabled (unstable/unsupported on this backend) and cannot be loaded", shortName)
	}
	if m.Backend != "mlx" && m.Backend != "ds4" && m.Backend != "ollama" && m.Backend != "mflux" && m.Backend != "omlx" {
		return fmt.Errorf("model %q has unsupported backend %q", shortName, m.Backend)
	}
	if m.Backend != "ollama" {
		info, statErr := os.Stat(m.Path)
		if statErr != nil || (m.Backend != "ds4" && !info.IsDir()) {
			return fmt.Errorf("model %q path is not usable: %s", shortName, m.Path)
		}
	}
	if m.DflashDraftModel != "" {
		// DFlash2 is supported by both backends: omlx (native dflash_mlx) and
		// mlx/rapid-mlx (vendored drafter). A model with a draft may use either.
		if m.Backend != "omlx" && m.Backend != "mlx" {
			return fmt.Errorf("model %q uses a DFlash2 checkpoint and must use backend omlx or mlx", shortName)
		}
		if _, statErr := os.Stat(m.DflashDraftModel); statErr != nil {
			return fmt.Errorf("model %q DFlash draft path is not usable: %s", shortName, m.DflashDraftModel)
		}
		if m.DflashBlockSize <= 0 {
			return fmt.Errorf("model %q requires a positive dflash_block_size", shortName)
		}
	}

	if m.Backend == "ollama" {
		upstreamModel := m.OllamaModel
		if upstreamModel == "" {
			upstreamModel = shortName
		}
		fmt.Printf("[unified-gateway] using Ollama model %s (%s) on :%d...\n", upstreamModel, m.Label, cfg.OllamaPort)
		if !portInUse(cfg.OllamaPort) {
			return fmt.Errorf("ollama does not seem to be running on :%d — start it first (e.g. `ollama serve`)", cfg.OllamaPort)
		}
		if err := warmOllamaModel(cfg.OllamaPort, upstreamModel); err != nil {
			fmt.Printf("[unified-gateway] WARNING: could not warm up %s: %v\n", upstreamModel, err)
		} else {
			fmt.Printf("[unified-gateway] %s warmed up on :%d\n", m.Label, cfg.OllamaPort)
		}
	} else {
		// Media-kind models share one of two POOLS (OCR-like on
		// MediaBackendPort, FLUX-family on FluxBackendPort) and "small"
		// models a third (SmallBackendPort) -- switching within a pool is
		// exclusive, exactly like the chat pool, but loading/switching a
		// pooled model never touches the chat backend or the OTHER pools.
		// This is the same non-exclusive relationship Ollama already has
		// with the chat backend, just generalized to independent pools
		// instead of one daemon. The memory check below only ever
		// credits/kills whatever was previously on THIS pool's port, never
		// the chat backend's or another pool's.
		port := cfg.BackendPort
		if m.Kind == "media" || m.Kind == "small" {
			port = modelPoolPort(cfg, m)
		}

		if required := estimateModelSizeGB(m); required > 0 {
			if m.Backend == "mlx" {
				// rapid-mlx's own prefix-cache reservation, capped via
				// --cache-memory-mb above (same size-scaled value) -- on
				// top of the model's weights, not covered by the on-disk
				// size estimate.
				required += float64(mlxCacheReserveMBFor(required)) / 1024.0
			}
			if m.WorkingSetMultiplier > 1 {
				required *= m.WorkingSetMultiplier
			}
			freeing := runningRSSGB(port)
			if ok, msg := checkMemory(required, freeing); !ok {
				return fmt.Errorf("%s", msg)
			}
		}

		previousName := runningBackendModel(cfg, port)
		previous, hadPrevious := cfg.Models[previousName]
		fmt.Printf("[unified-gateway] loading %s (%s) on :%d...\n", shortName, m.Label, port)
		killPort(port)

		// The model switch invalidates every disk KV checkpoint accumulated
		// by the previous model: they're keyed by that model's request hash
		// (request_hash includes the model name), so a switched model can
		// never reuse them, and with --kv-disk-checkpoint-interval 0 no new
		// ones are written either. Clear them now, before the new backend
		// starts, so its startup scan doesn't load stale snapshots.
		clearDiskKVCheckpoints()

		cmd, err := launchConfiguredBackend(cfg, shortName, m, port)
		if err != nil {
			return fmt.Errorf("failed to launch backend: %w", err)
		}

		fmt.Printf("[unified-gateway] waiting for backend on :%d...\n", port)
		// FLUX models load their weights AND compile the first time a
		// generation runs, which can meaningfully exceed rapid-mlx/ds4's
		// own cold-start time -- give mflux more room before giving up.
		readyTimeout := 180 * time.Second
		if m.Backend == "mflux" {
			readyTimeout = 300 * time.Second
		}
		ready := waitForPort(port, readyTimeout, omlxAPIKey(m.Backend))
		if ready && m.Backend == "omlx" {
			err = preloadOMLX(port, m)
			ready = err == nil
		}
		if !ready {
			failure := fmt.Errorf("backend did not become ready on :%d", port)
			if err != nil {
				failure = err
			}
			killPort(port)
			if hadPrevious && previousName != shortName {
				fmt.Printf("[unified-gateway] restoring previous backend %s after failed load...\n", previousName)
				if restored, restoreErr := launchConfiguredBackend(cfg, previousName, previous, port); restoreErr == nil && waitForPort(port, readyTimeout, omlxAPIKey(previous.Backend)) {
					go restored.Wait()
					return fmt.Errorf("%v; previous backend %s restored", failure, previousName)
				}
			}
			return failure
		}
		fmt.Printf("[unified-gateway] %s ready on :%d\n", m.Label, port)
		crashTracker.recordLoad(shortName)
		go func() {
			cmd.Wait()
			crashTracker.recordExit(shortName)
		}()
	}

	ensureGatewayRunning()
	fmt.Println("[unified-gateway] adapters active on :8082 (OpenAI) and :8083 (Anthropic)")
	return nil
}

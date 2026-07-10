package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
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

// ModelConfig.Backend is one of:
//   - "mlx"    — served via `rapid-mlx`, spawned/killed by us on Config.BackendPort
//   - "ds4"    — served via `ds4-server`, same lifecycle as "mlx"
//   - "ollama" — served by an already-running `ollama serve` on Config.OllamaPort;
//     we never spawn/kill it, only warm up the requested model and point the
//     gateway's routing at Ollama's own OpenAI-compatible endpoint. OllamaModel
//     is Ollama's own model tag (e.g. "gemma4:31b-mlx"), which is usually not
//     the same as the shortname key used to select it.
//
// ModelConfig.Kind distinguishes ordinary chat models from special-purpose
// ones that happen to share the same "mlx"/rapid-mlx serving path but
// aren't meant to show up in a chat model picker -- OCR being the first
// example (has_vision:true, but its only use is "read this image", not
// conversation). Empty/omitted means "chat" -- the vast majority of
// entries, and the only value that existed before this field was added.
// "media" entries are still fully loadable/servable exactly like any
// other mlx/ds4/ollama backend; they're just filtered out of
// handleListModels (so opencode/pi/Claude Code never see them as a chat
// choice) and out of the menu bar's normal per-backend model lists,
// getting their own section instead. This is NOT where image-generation
// models like FLUX belong -- those need mflux, a completely different
// runtime the gateway doesn't spawn or route to at all, so they're
// documented only in models.txt, never added here.
type ModelConfig struct {
	Path        string `json:"path,omitempty"`
	Label       string `json:"label"`
	Backend     string `json:"backend"`
	ModelType   string `json:"model_type,omitempty"`
	HasVision   bool   `json:"has_vision,omitempty"`
	Ctx         int    `json:"ctx,omitempty"`
	OllamaModel string `json:"ollama_model,omitempty"`
	Kind        string `json:"kind,omitempty"`
}

type Config struct {
	BackendPort int                    `json:"backend_port"`
	OllamaPort  int                    `json:"ollama_port,omitempty"`
	VenvDir     string                 `json:"venv_dir"`
	DS4Dir      string                 `json:"ds4_dir"`
	Models      map[string]ModelConfig `json:"models"`
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
	if cfg.OllamaPort == 0 {
		cfg.OllamaPort = 11434
	}
	for name, m := range cfg.Models {
		m.Path = expandHome(m.Path)
		cfg.Models[name] = m
	}
	return &cfg, nil
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

// runningBackendModel inspects whichever real process owns cfg.BackendPort
// right now and returns the models.json shortname it's serving — read
// from rapid-mlx's own --served-model-name argument, or ds4-server's
// --model path matched back against configured ds4 entries. "" if
// nothing is listening, or it doesn't look like either.
func runningBackendModel(cfg *Config, port int) string {
	cmd := portOwnerCommand(port)
	if cmd == "" {
		return ""
	}
	if strings.Contains(cmd, "rapid-mlx") {
		return commandFlagValue(cmd, "--served-model-name")
	}
	if strings.Contains(cmd, "ds4-server") {
		path := commandFlagValue(cmd, "--model")
		for name, m := range cfg.Models {
			if m.Backend == "ds4" && m.Path == path {
				return name
			}
		}
	}
	return ""
}

// queryOllamaLoadedModel asks Ollama's own API which model it currently
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

func waitForPort(port int, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	url := fmt.Sprintf("http://127.0.0.1:%d/v1/models", port)
	for time.Now().Before(deadline) {
		resp, err := http.Get(url)
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

func launchMLX(cfg *Config, shortName string, m ModelConfig) (*exec.Cmd, error) {
	rapidBin := filepath.Join(cfg.VenvDir, "bin", "rapid-mlx")
	args := []string{
		"serve", m.Path,
		"--host", "127.0.0.1",
		"--port", fmt.Sprintf("%d", cfg.BackendPort),
		"--served-model-name", shortName,
	}
	if (m.ModelType == "qwen3" || m.ModelType == "qwen3_5" || m.ModelType == "qwen2_moe") && m.HasVision {
		args = append(args, "--text-only", "--no-mllm")
	}
	if m.ModelType == "qwen3" || m.ModelType == "qwen3_5" || m.ModelType == "qwen2_moe" {
		args = append(args, "--no-spec-decode")
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
	args = append(args, "--kv-cache-turboquant")
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

func launchDS4(cfg *Config, m ModelConfig) (*exec.Cmd, error) {
	ds4Server := filepath.Join(cfg.DS4Dir, "ds4-server")
	ctx := m.Ctx
	if ctx == 0 {
		ctx = 32768
	}
	args := []string{
		"--model", m.Path,
		"--metal", "--ctx", fmt.Sprintf("%d", ctx),
		"--power", "100",
		"--host", "127.0.0.1", "--port", fmt.Sprintf("%d", cfg.BackendPort),
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
	waitForPort(8082, 15*time.Second)
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
		if required := estimateModelSizeGB(m); required > 0 {
			if m.Backend == "mlx" {
				// rapid-mlx's own prefix-cache reservation, capped via
				// --cache-memory-mb above (same size-scaled value) -- on
				// top of the model's weights, not covered by the on-disk
				// size estimate.
				required += float64(mlxCacheReserveMBFor(required)) / 1024.0
			}
			freeing := runningRSSGB(cfg.BackendPort)
			if ok, msg := checkMemory(required, freeing); !ok {
				return fmt.Errorf("%s", msg)
			}
		}

		fmt.Printf("[unified-gateway] loading %s (%s)...\n", shortName, m.Label)
		killPort(cfg.BackendPort)

		var cmd *exec.Cmd
		if m.Backend == "ds4" {
			cmd, err = launchDS4(cfg, m)
		} else {
			cmd, err = launchMLX(cfg, shortName, m)
		}
		if err != nil {
			return fmt.Errorf("failed to launch backend: %w", err)
		}

		fmt.Printf("[unified-gateway] waiting for backend on :%d...\n", cfg.BackendPort)
		if !waitForPort(cfg.BackendPort, 180*time.Second) {
			return fmt.Errorf("backend did not become ready on :%d", cfg.BackendPort)
		}
		fmt.Printf("[unified-gateway] %s ready on :%d\n", m.Label, cfg.BackendPort)
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

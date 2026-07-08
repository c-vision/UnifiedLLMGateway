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
type ModelConfig struct {
	Path        string `json:"path,omitempty"`
	Label       string `json:"label"`
	Backend     string `json:"backend"`
	ModelType   string `json:"model_type,omitempty"`
	HasVision   bool   `json:"has_vision,omitempty"`
	Ctx         int    `json:"ctx,omitempty"`
	OllamaModel string `json:"ollama_model,omitempty"`
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
// Active backend state — which port (and, for Ollama, which upstream model
// name) the running gateway should currently forward to. `loadModel` writes
// this file every time it switches models; the gateway's HTTP handlers read
// it on every request instead of relying on a value fixed at startup, since
// Ollama and rapid-mlx/ds4-server live on different, independent ports.
// ============================================================================

type activeBackend struct {
	Port          int    `json:"port"`
	UpstreamModel string `json:"upstream_model,omitempty"`
	Model         string `json:"model,omitempty"`
}

func activeBackendPath() string {
	return filepath.Join(gatewayDir(), "active-backend.json")
}

func writeActiveBackend(ab activeBackend) error {
	data, err := json.Marshal(ab)
	if err != nil {
		return err
	}
	return os.WriteFile(activeBackendPath(), data, 0644)
}

// readActiveBackend falls back to fallbackPort (no model-name rewrite) if the
// state file is missing or invalid, e.g. before the first `load` ever ran.
func readActiveBackend(fallbackPort int) activeBackend {
	data, err := os.ReadFile(activeBackendPath())
	if err != nil {
		return activeBackend{Port: fallbackPort}
	}
	var ab activeBackend
	if err := json.Unmarshal(data, &ab); err != nil || ab.Port == 0 {
		return activeBackend{Port: fallbackPort}
	}
	return ab
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

	cmd := exec.Command(rapidBin, args...)
	cmd.Env = append(os.Environ(),
		"VIRTUAL_ENV="+cfg.VenvDir,
		"PATH="+filepath.Join(cfg.VenvDir, "bin")+":"+os.Getenv("PATH"),
	)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
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
func ensureBackendLoading(shortName string) {
	if shortName == "" {
		return
	}
	autoLoadMu.Lock()
	if autoLoadInFlight {
		autoLoadMu.Unlock()
		return
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
		if err := writeActiveBackend(activeBackend{Port: cfg.OllamaPort, UpstreamModel: upstreamModel, Model: shortName}); err != nil {
			return fmt.Errorf("failed to record active backend: %w", err)
		}
	} else {
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
		_ = cmd

		fmt.Printf("[unified-gateway] waiting for backend on :%d...\n", cfg.BackendPort)
		if !waitForPort(cfg.BackendPort, 180*time.Second) {
			return fmt.Errorf("backend did not become ready on :%d", cfg.BackendPort)
		}
		fmt.Printf("[unified-gateway] %s ready on :%d\n", m.Label, cfg.BackendPort)

		if err := writeActiveBackend(activeBackend{Port: cfg.BackendPort, Model: shortName}); err != nil {
			return fmt.Errorf("failed to record active backend: %w", err)
		}
	}

	ensureGatewayRunning()
	fmt.Println("[unified-gateway] adapters active on :8082 (OpenAI) and :8083 (Anthropic)")
	return nil
}

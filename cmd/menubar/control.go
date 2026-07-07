package main

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

// modelConfig/gwConfig mirror the subset of the root unified-gateway's
// models.json this app needs. They live in their own "main" package (Go
// won't let one main package import another), so the fields are kept in
// sync by hand with models.go's Config/ModelConfig.
type modelConfig struct {
	Path        string `json:"path,omitempty"`
	Label       string `json:"label"`
	Backend     string `json:"backend"`
	OllamaModel string `json:"ollama_model,omitempty"`
}

type gwConfig struct {
	BackendPort int                    `json:"backend_port"`
	OllamaPort  int                    `json:"ollama_port"`
	Models      map[string]modelConfig `json:"models"`
}

func expandHome(path string) string {
	if strings.HasPrefix(path, "~/") {
		home, _ := os.UserHomeDir()
		return filepath.Join(home, path[2:])
	}
	return path
}

func binDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return "."
	}
	return filepath.Join(home, ".local", "bin")
}

func gatewayBinary() string  { return filepath.Join(binDir(), "unified-gateway") }
func modelsJSONPath() string { return filepath.Join(binDir(), "models.json") }

func loadGWConfig() (*gwConfig, error) {
	data, err := os.ReadFile(modelsJSONPath())
	if err != nil {
		return nil, err
	}
	var cfg gwConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	if cfg.OllamaPort == 0 {
		cfg.OllamaPort = 11434
	}
	for name, m := range cfg.Models {
		m.Path = expandHome(m.Path)
		cfg.Models[name] = m
	}
	return &cfg, nil
}

func portOpen(port int) bool {
	conn, err := net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", port), 500*time.Millisecond)
	if err != nil {
		return false
	}
	conn.Close()
	return true
}

// killPort and portOwnerCommand both restrict lsof to LISTEN sockets only
// (-sTCP:LISTEN). Without it, lsof also matches the gateway's own
// outbound client connections to the backend port (its keep-alive HTTP
// connection while proxying requests) — which showed up as a *lower* PID
// than the actual backend process and got picked first, so the menu bar
// misidentified the gateway itself as "not rapid-mlx, not ds4" and kept
// showing 🔴 even while a model was loaded and serving requests fine.

func killPort(port int) {
	out, err := exec.Command("lsof", "-ti", fmt.Sprintf(":%d", port), "-sTCP:LISTEN").Output()
	if err != nil {
		return
	}
	for _, pidStr := range strings.Fields(string(out)) {
		exec.Command("kill", "-9", pidStr).Run()
	}
}

// portOwnerCommand returns the full command line of whichever process is
// listening on port, or "" if nothing is listening (or it can't be
// determined). This is the ground truth for "what's actually running" —
// nothing here is cached or read from a file we wrote earlier, since
// another client could have replaced the process, or the user could have
// stopped it, entirely outside this tool. rapid-mlx runs as a Python
// script (ps -o comm= would just say "python3.12"), so we match against
// the full command line ("command="), which still contains "rapid-mlx"
// as the invoked script path; ds4-server is a real binary and shows up
// under its own name either way.
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

// runningMLXModel reports whether rapid-mlx is the process on port, and
// if so, which model it's serving — read directly from its own
// --served-model-name argument, not from any bookkeeping file.
func runningMLXModel(port int) (shortName string, active bool) {
	cmd := portOwnerCommand(port)
	if !strings.Contains(cmd, "rapid-mlx") {
		return "", false
	}
	return commandFlagValue(cmd, "--served-model-name"), true
}

// runningDS4Model reports whether ds4-server is the process on port, and
// if so, which model it's serving. ds4-server has no --served-model-name
// equivalent, only --model <path>, so the path is matched back against
// models.json to recover the shortname; if no configured model matches
// (e.g. it was started manually with an unlisted file), the raw path is
// returned instead so there's still something meaningful to show.
func runningDS4Model(cfg *gwConfig, port int) (label string, active bool) {
	cmd := portOwnerCommand(port)
	if !strings.Contains(cmd, "ds4-server") {
		return "", false
	}
	path := commandFlagValue(cmd, "--model")
	if path == "" {
		return "", true
	}
	if cfg != nil {
		for name, m := range cfg.Models {
			if m.Backend == "ds4" && m.Path == path {
				return name, true
			}
		}
	}
	return path, true
}

// runningOllamaModel asks Ollama's own API which model it currently has
// loaded in memory (empty if none), rather than assuming based on what
// this tool last told it to warm up.
func runningOllamaModel(port int) string {
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

// gatewayCurrentModel asks the gateway's own OpenAI adapter which model
// it's routing to right now (GET /v1/models) — this reflects the
// gateway's own live backend-selection logic directly, so there's
// nothing here for the menu bar to cache or get out of sync with.
func gatewayCurrentModel() string {
	resp, err := http.Get("http://127.0.0.1:8082/v1/models")
	if err != nil {
		return ""
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return ""
	}
	var parsed struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil || len(parsed.Data) == 0 {
		return ""
	}
	return parsed.Data[0].ID
}

func ollamaRunning() bool {
	return exec.Command("pgrep", "-x", "Ollama").Run() == nil
}

func startOllama() { exec.Command("open", "-a", "Ollama").Start() }
func stopOllama()  { exec.Command("killall", "Ollama").Run() }

// confirmDialog shows a native macOS confirmation dialog and returns true
// only if the user clicked the affirmative button. AppleScript treats a
// button literally named "Cancel" as user cancellation, which osascript
// surfaces as a non-zero exit code — that's the signal we check.
func confirmDialog(message string) bool {
	script := fmt.Sprintf(
		`display dialog %s with title "Unified Gateway" buttons {"Cancel", "Stop & Start"} default button "Stop & Start" with icon caution`,
		appleScriptQuote(message),
	)
	return exec.Command("osascript", "-e", script).Run() == nil
}

func appleScriptQuote(s string) string {
	return `"` + strings.ReplaceAll(s, `"`, `\"`) + `"`
}

// confirmPortFree checks whether port is occupied; if so, it asks the
// user (via confirmDialog) whether to stop whatever's there. Returns true
// if it's safe to proceed — the port was already free, or the user
// confirmed and it has now been killed — false if the user cancelled.
func confirmPortFree(port int, serviceLabel string) bool {
	if !portOpen(port) {
		return true
	}
	if !confirmDialog(fmt.Sprintf("Port %d is in use.\nStop it and start %s?", port, serviceLabel)) {
		return false
	}
	killPort(port)
	return true
}

// startGateway launches a detached copy of the unified-gateway binary (no
// args), i.e. just the persistent OpenAI/Anthropic adapter servers.
func startGateway() error {
	if portOpen(8082) && portOpen(8083) {
		return nil
	}
	if _, err := os.Stat(gatewayBinary()); err != nil {
		return fmt.Errorf("unified-gateway binary not found at %s", gatewayBinary())
	}
	cmd := exec.Command(gatewayBinary())
	cmd.Dir = binDir()
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	return cmd.Start()
}

func stopGateway() {
	killPort(8082)
	killPort(8083)
}

func stopBackend(cfg *gwConfig) {
	if cfg == nil || cfg.BackendPort == 0 {
		return
	}
	killPort(cfg.BackendPort)
}

// loadModelAsync runs `unified-gateway load <shortName>` out-of-process
// (it can block for up to several minutes while a model warms up) without
// waiting for it — refreshLoop's polling picks up the result (or lack of
// one) on its own, so there's no completion callback to wire up here.
func loadModelAsync(shortName string) {
	go func() {
		cmd := exec.Command(gatewayBinary(), "load", shortName)
		cmd.Dir = binDir()
		cmd.Run()
	}()
}

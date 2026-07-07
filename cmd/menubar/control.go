package main

import (
	"encoding/json"
	"fmt"
	"net"
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
	Label   string `json:"label"`
	Backend string `json:"backend"`
}

type gwConfig struct {
	BackendPort int                    `json:"backend_port"`
	OllamaPort  int                    `json:"ollama_port"`
	Models      map[string]modelConfig `json:"models"`
}

type activeBackend struct {
	Port          int    `json:"port"`
	UpstreamModel string `json:"upstream_model,omitempty"`
	Model         string `json:"model,omitempty"`
}

func binDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return "."
	}
	return filepath.Join(home, ".local", "bin")
}

func gatewayBinary() string     { return filepath.Join(binDir(), "unified-gateway") }
func modelsJSONPath() string    { return filepath.Join(binDir(), "models.json") }
func activeBackendPath() string { return filepath.Join(binDir(), "active-backend.json") }

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
	return &cfg, nil
}

func readActiveBackend() activeBackend {
	data, err := os.ReadFile(activeBackendPath())
	if err != nil {
		return activeBackend{}
	}
	var ab activeBackend
	_ = json.Unmarshal(data, &ab)
	return ab
}

func portOpen(port int) bool {
	conn, err := net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", port), 500*time.Millisecond)
	if err != nil {
		return false
	}
	conn.Close()
	return true
}

func killPort(port int) {
	out, err := exec.Command("lsof", "-ti", fmt.Sprintf(":%d", port)).Output()
	if err != nil {
		return
	}
	for _, pidStr := range strings.Fields(string(out)) {
		exec.Command("kill", "-9", pidStr).Run()
	}
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
// (it can block for up to several minutes while a model warms up) and
// reports completion via onDone, called from the spawned goroutine.
func loadModelAsync(shortName string, onDone func(error)) {
	go func() {
		cmd := exec.Command(gatewayBinary(), "load", shortName)
		cmd.Dir = binDir()
		err := cmd.Run()
		onDone(err)
	}()
}

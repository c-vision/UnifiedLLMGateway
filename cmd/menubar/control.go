package main

import (
	"bytes"
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
	Kind        string `json:"kind,omitempty"`
}

// MediaBackendPort/FluxBackendPort are the two media POOLS -- OCR-like
// entries (any "kind":"media" model except backend "mflux") share
// MediaBackendPort, FLUX-family entries (backend "mflux") share
// FluxBackendPort. Both are exclusive within themselves exactly like
// BackendPort is for chat models, but fully independent of the chat pool
// and of each other -- mirrors ModelConfig.Kind's doc comment in the root
// models.go.
type gwConfig struct {
	BackendPort      int                    `json:"backend_port"`
	OllamaPort       int                    `json:"ollama_port"`
	MediaBackendPort int                    `json:"media_backend_port,omitempty"`
	FluxBackendPort  int                    `json:"flux_backend_port,omitempty"`
	Models           map[string]modelConfig `json:"models"`
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

// relaunchSelf starts a fresh copy of this same menu bar binary — used to
// pick up models.json changes made externally (e.g. by hand, or by
// unified-gateway itself), since the model list is only read once at
// startup and systray has no clean way to rebuild an existing submenu
// tree in place. The caller is expected to systray.Quit() right after.
func relaunchSelf() error {
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	cmd := exec.Command(exe)
	cmd.Dir = binDir()
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	return cmd.Start()
}

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
	if cfg.MediaBackendPort == 0 {
		cfg.MediaBackendPort = cfg.BackendPort + 1
	}
	if cfg.FluxBackendPort == 0 {
		cfg.FluxBackendPort = cfg.MediaBackendPort + 1
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

// runningFluxModel reports whether the persistent flux_server.py process is
// on port, and if so, which model it's serving -- same idea as
// runningDS4Model, since mflux's server has no --served-model-name
// equivalent either: its --model-path is matched back against configured
// "mflux" entries.
func runningFluxModel(cfg *gwConfig, port int) (shortName string, active bool) {
	cmd := portOwnerCommand(port)
	if !strings.Contains(cmd, "flux_server") {
		return "", false
	}
	path := commandFlagValue(cmd, "--model-path")
	if cfg != nil {
		for name, m := range cfg.Models {
			if m.Backend == "mflux" && m.Path == path {
				return name, true
			}
		}
	}
	return "", true
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
//
// /v1/models now returns the FULL models.json catalog (every configured
// model, each tagged "active": true/false), not just a single active
// one — this must find the entry actually marked active, never just
// take the first item: Go map iteration order (used to build that list
// server-side) isn't stable across calls, so blindly taking data[0]
// picks a different, arbitrary model on every poll.
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
			ID     string `json:"id"`
			Active bool   `json:"active"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return ""
	}
	for _, m := range parsed.Data {
		if m.Active {
			return m.ID
		}
	}
	return ""
}

// compressionState mirrors the gateway's GET /v1/compression response.
type compressionState struct {
	Enabled            bool  `json:"enabled"`
	RequestsCompressed int64 `json:"requests_compressed"`
	CharsSaved         int64 `json:"chars_saved"`
}

// getCompressionState asks the gateway for prompt-compression's current
// live state (toggle + cumulative savings). ok is false when the gateway
// itself isn't reachable, so the caller can distinguish "off" from
// "unknown" rather than defaulting the checkbox to unchecked either way.
func getCompressionState() (state compressionState, ok bool) {
	resp, err := http.Get("http://127.0.0.1:8082/v1/compression")
	if err != nil {
		return compressionState{}, false
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return compressionState{}, false
	}
	if json.NewDecoder(resp.Body).Decode(&state) != nil {
		return compressionState{}, false
	}
	return state, true
}

// setCompressionEnabled flips the gateway's live prompt-compression flag
// (POST /v1/compression) -- takes effect on the very next request, no
// model reload or gateway restart needed, unlike almost every other knob
// in this menu.
func setCompressionEnabled(enabled bool) error {
	body, _ := json.Marshal(map[string]bool{"enabled": enabled})
	req, err := http.NewRequest("POST", "http://127.0.0.1:8082/v1/compression", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("gateway returned %d", resp.StatusCode)
	}
	return nil
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

// stopAllMediaBackends kills every "kind":"media" model's own dedicated
// port (OCR, each FLUX model, ...) -- used only by "Stop All". Individual
// Stop clicks (see addIndividualMediaItems in main.go) kill just the one
// port for that specific model instead; this never touches the chat
// backend's port, since media models always have their own.
// stopAllMediaBackends kills both media pools -- MediaBackendPort (OCR-
// like) and FluxBackendPort (FLUX-family) -- used only by "Stop All".
// Individual Stop clicks on the OCR/FLUX sections kill just that one
// pool's port instead; neither ever touches the chat backend's port.
func stopAllMediaBackends(cfg *gwConfig) {
	if cfg == nil {
		return
	}
	if cfg.MediaBackendPort != 0 {
		killPort(cfg.MediaBackendPort)
	}
	if cfg.FluxBackendPort != 0 {
		killPort(cfg.FluxBackendPort)
	}
}

// loadModelAsync runs `unified-gateway load <shortName>` out-of-process
// (it can block for up to several minutes while a model warms up) without
// waiting for it — refreshLoop's polling picks up the resulting status
// (checkmark, port label) on its own. Clicking a menu item closes the
// menu immediately (that's macOS, not something we control), which used
// to look exactly like success even when the load was still running or
// had failed — these three notifications (start/success/failure) are
// what tell the user which one it actually was, without needing to leave
// the menu open to watch it.
func loadModelAsync(shortName string) {
	go func() {
		notify("Unified Gateway", fmt.Sprintf("Loading %s…", shortName))

		cmd := exec.Command(gatewayBinary(), "load", shortName)
		cmd.Dir = binDir()
		logPath := filepath.Join(binDir(), "menubar-load.log")
		if logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644); err == nil {
			cmd.Stdout = logFile
			cmd.Stderr = logFile
			defer logFile.Close()
		}
		if err := cmd.Run(); err != nil {
			// Previously silent — a failure here (including the process
			// never starting at all, e.g. fork/exec denied) left no trace
			// anywhere: no crash report (nothing to crash), no gateway log
			// line (the child never got that far). Surface it visibly
			// instead of leaving the menu bar looking like it just did
			// nothing.
			notify("Unified Gateway", fmt.Sprintf("Failed to load %s — check menubar-load.log", shortName))
		} else {
			notify("Unified Gateway", fmt.Sprintf("%s ready", shortName))
		}
	}()
}

// notify shows a native macOS banner notification — used for failures that
// happen in the background (a menu click has no request/response cycle to
// report back through), so they're not silently invisible.
func notify(title, message string) {
	script := fmt.Sprintf(`display notification %s with title %s`, appleScriptQuote(message), appleScriptQuote(title))
	exec.Command("osascript", "-e", script).Run()
}

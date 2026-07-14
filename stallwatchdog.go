package main

import (
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Stall watchdog: catches the failure mode memwatchdog.go doesn't -- a
// request sits with rapid_mlx_requests_running > 0 and never finishes,
// the process still burning CPU (not deadlocked, just never finishing).
//
// Detection is metrics-based, not log-based: poll rapid-mlx's own
// /metrics on the chat backend port, and track how long
// rapid_mlx_steps_executed_total has gone unchanged WHILE
// requests_running > 0. requests_running == 0 is excluded deliberately --
// an idle backend with a static step count is healthy, not stalled.
//
// steps_executed_total, not completion_tokens_total (2026-07-14 fix): the
// original version watched completion tokens, which only move during
// decode. A legitimate compaction call summarizing a large chunk of
// history spends most of its wall-clock time in PREFILL -- zero
// completion tokens by definition until the first output token, which on
// a big-enough chunk can take longer than any restart threshold. Watching
// completion tokens alone can't tell "genuinely hung forever" from "slow
// but still working through prefill", and a false-positive restart there
// is actively harmful: it kills the compaction mid-flight, the client
// retries the same call, and it never gets far enough to finish --
// an infinite kill/retry loop instead of the one-time recovery this is
// meant to provide. steps_executed_total increments on every scheduler
// step, prefill chunks included (confirmed live: it climbs in lockstep
// with the periodic "[Metal memory] ... step=" log line during a request
// that's still deep in prefill with zero completion tokens so far) -- so
// it's live exactly when the engine is doing anything at all, and frozen
// only when it truly isn't.
//
// Threshold matters: a legitimate cold prefill on a huge context can sit at
// zero completion tokens for tens of seconds before the first token comes
// out (measured directly this session: ~55s for an ~80k-token prompt), and
// a compaction-sized prefill could run longer still. 5 minutes (operator
// choice, 2026-07-14) is checked against step progress now, not token
// progress, so it only fires when nothing is happening at all.
//
// Only the chat pool (cfg.BackendPort) is covered: OCR and FLUX run
// different servers (PaddleOCR-VL / mflux) that don't expose this same
// counter, so there's no equivalent progress signal to poll there.

const defaultStallWatchdogThresholdSeconds = 300 // 5 minutes

var (
	stallMu            sync.Mutex
	lastSteps          = -1.0 // -1 = not yet observed
	lastStepChangeTime time.Time
	lastStallRestart   time.Time
)

// rapidMLXProgress fetches /metrics on port and extracts the two counters
// the stall check needs. ok is false if the backend isn't up or doesn't
// expose rapid-mlx-shaped metrics (e.g. nothing loaded, or a non-rapid-mlx
// backend) -- callers must treat that as "nothing to check", not a stall.
func rapidMLXProgress(port int) (running int, stepsExecuted float64, ok bool) {
	resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/metrics", port))
	if err != nil {
		return 0, 0, false
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, 0, false
	}

	var sawRunning, sawSteps bool
	for _, line := range strings.Split(string(body), "\n") {
		switch {
		case strings.HasPrefix(line, "rapid_mlx_requests_running "):
			v, err := strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(line, "rapid_mlx_requests_running ")))
			if err != nil {
				return 0, 0, false
			}
			running = v
			sawRunning = true
		case strings.HasPrefix(line, "rapid_mlx_steps_executed_total "):
			v, err := strconv.ParseFloat(strings.TrimSpace(strings.TrimPrefix(line, "rapid_mlx_steps_executed_total ")), 64)
			if err != nil {
				return 0, 0, false
			}
			stepsExecuted = v
			sawSteps = true
		}
	}
	return running, stepsExecuted, sawRunning && sawSteps
}

// notifyMacOS fires a native notification via osascript -- works from a
// headless process with no window/tray of its own, unlike systray.Notify
// which needs a running systray context (the menu bar app, not this
// gateway process).
func notifyMacOS(title, message string) {
	script := fmt.Sprintf("display notification %q with title %q", message, title)
	exec.Command("osascript", "-e", script).Run()
}

// checkRequestStall is called from the same ticker as checkMemoryPressure
// (see startMemoryWatchdog) -- same polling cadence, no separate goroutine
// needed.
func checkRequestStall(cfg *Config) {
	if cfg.StallWatchdogThresholdSeconds < 0 {
		return // explicit opt-out
	}

	running, steps, ok := rapidMLXProgress(cfg.BackendPort)
	if !ok {
		return // backend down, or not rapid-mlx-shaped metrics -- nothing to check
	}

	stallMu.Lock()
	defer stallMu.Unlock()

	if steps != lastSteps {
		lastSteps = steps
		lastStepChangeTime = time.Now()
		return
	}
	if running == 0 {
		// No request in flight -- an unchanged count is expected, not a stall.
		return
	}
	if lastStepChangeTime.IsZero() {
		// First tick ever for this process; nothing to compare against yet.
		lastStepChangeTime = time.Now()
		return
	}

	stalledFor := time.Since(lastStepChangeTime)
	threshold := time.Duration(cfg.StallWatchdogThresholdSeconds) * time.Second
	if stalledFor < threshold {
		return
	}
	if since := time.Since(lastStallRestart); since < watchdogCooldown {
		fmt.Printf(
			"[unified-gateway] stall watchdog: stuck for %s but last restart was %s ago (cooldown %s) -- waiting\n",
			stalledFor.Round(time.Second), since.Round(time.Second), watchdogCooldown,
		)
		return
	}

	name := runningBackendModel(cfg, cfg.BackendPort)
	if name == "" {
		return
	}

	fmt.Printf(
		"[unified-gateway] stall watchdog: %q on :%d stuck for %s with a request in flight (requests_running=%d, scheduler steps stuck at %.0f) -- restarting\n",
		name, cfg.BackendPort, stalledFor.Round(time.Second), running, steps,
	)
	notifyMacOS("Unified Gateway", fmt.Sprintf("%s sembrava bloccato (%s senza progressi) — riavviato automaticamente", name, stalledFor.Round(time.Second)))

	// Claim the same in-flight flag ensureBackendLoading uses: a client
	// request arriving while THIS restart is still mid-flight (killPort
	// through the port coming back up) hits an unreachable backend and
	// would otherwise trigger ensureBackendLoading's own auto-load path
	// right behind this one -- two full reloads back to back for one
	// stall, observed directly (2026-07-14). Sharing the flag makes that
	// second trigger see "already loading" and skip its own.
	autoLoadMu.Lock()
	alreadyLoading := autoLoadInFlight
	if !alreadyLoading {
		autoLoadInFlight = true
	}
	autoLoadMu.Unlock()
	if alreadyLoading {
		fmt.Println("[unified-gateway] stall watchdog: a load is already in flight -- skipping, it'll cover this restart too")
		return
	}
	defer func() {
		autoLoadMu.Lock()
		autoLoadInFlight = false
		autoLoadMu.Unlock()
	}()

	if err := loadModel(name); err != nil {
		fmt.Printf("[unified-gateway] stall watchdog: failed to restart %q: %v\n", name, err)
		return
	}

	lastStallRestart = time.Now()
	// Fresh process, fresh counters -- reset so the new instance gets a
	// full observation window before it can be judged stalled again.
	lastSteps = -1
	lastStepChangeTime = time.Time{}
}

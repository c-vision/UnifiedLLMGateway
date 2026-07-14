package main

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Background memory watchdog. memcheck.go's checkMemory only runs at
// switch time, right before killPort/launch -- it can't help with the
// failure mode observed directly on this machine: a model (or several,
// across the chat/OCR/FLUX pools) sits loaded for a long session and free
// RAM keeps falling anyway (rapid-mlx's own prefix cache growing, KV-cache
// growth on a long conversation, Metal buffer fragmentation that never
// gets reclaimed) until the whole system locks up or macOS force-restarts
// it under memory pressure -- with nothing having "switched" to trigger
// the existing check. This runs continuously instead, independent of any
// load/switch request.
//
// "Compaction" isn't something reachable from outside these opaque
// backend processes (rapid-mlx/ds4-server/flux-server.py) -- there's no
// exposed cache-trim endpoint to call. A restart (kill + fresh launch of
// the SAME model) is the only form of compaction actually available: it
// forces macOS to reclaim 100% of that process's resident memory,
// including whatever accumulated cache or fragmentation caused the
// pressure, and the backend reloads clean. This does drop whatever
// request that backend was mid-handling at the moment of restart --
// accepted deliberately, since the alternative observed in practice is
// the whole machine locking up or forcing a reboot, which loses far more.

const (
	defaultWatchdogThresholdPercent = 15.0
	defaultWatchdogIntervalSeconds  = 30
	watchdogCooldown                = 5 * time.Minute
)

var (
	watchdogMu          sync.Mutex
	lastWatchdogRestart time.Time
)

// totalRAMGB returns total physical memory in GB via sysctl (Apple Silicon
// only, same assumption freeRAMGB already makes).
func totalRAMGB() float64 {
	out, err := exec.Command("sysctl", "-n", "hw.memsize").Output()
	if err != nil {
		return 0
	}
	bytes, err := strconv.ParseFloat(strings.TrimSpace(string(out)), 64)
	if err != nil {
		return 0
	}
	return bytes / (1024 * 1024 * 1024)
}

// startMemoryWatchdog runs for the lifetime of the process, checking free
// RAM every Config.MemoryWatchdogIntervalSeconds and force-restarting
// whatever's currently loaded if free RAM falls below
// Config.MemoryWatchdogThresholdPercent (both defaulted in loadConfig).
// Never blocks the caller -- runs in its own goroutine.
func startMemoryWatchdog() {
	total := totalRAMGB()
	if total <= 0 {
		fmt.Println("[unified-gateway] memory watchdog: could not read total RAM (sysctl hw.memsize failed) -- disabled")
		return
	}

	cfg, err := loadConfig()
	if err != nil {
		fmt.Printf("[unified-gateway] memory watchdog: could not read models.json (%v) -- disabled\n", err)
		return
	}
	if cfg.MemoryWatchdogThresholdPercent < 0 {
		fmt.Println("[unified-gateway] memory watchdog: disabled (negative threshold in models.json)")
		return
	}

	interval := time.Duration(cfg.MemoryWatchdogIntervalSeconds) * time.Second
	fmt.Printf(
		"[unified-gateway] memory watchdog active: restart loaded backends if free RAM drops below %.0f%% of %.0f GB total (checked every %s)\n",
		cfg.MemoryWatchdogThresholdPercent, total, interval,
	)
	if cfg.StallWatchdogThresholdSeconds >= 0 {
		fmt.Printf(
			"[unified-gateway] stall watchdog active: restart the chat backend if a request sits with no token progress for %ds (checked every %s)\n",
			cfg.StallWatchdogThresholdSeconds, interval,
		)
	} else {
		fmt.Println("[unified-gateway] stall watchdog: disabled (negative threshold in models.json)")
	}

	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for range ticker.C {
			checkMemoryPressure(total)
			if liveCfg, err := loadConfig(); err == nil {
				checkRequestStall(liveCfg)
			}
		}
	}()
}

func checkMemoryPressure(totalGB float64) {
	// Re-read models.json each tick rather than capturing the threshold
	// once at startup -- same live-config philosophy as the rest of the
	// gateway (loadConfig is cheap, called fresh per-request everywhere
	// else too), so editing the threshold takes effect without a restart.
	cfg, err := loadConfig()
	if err != nil {
		return
	}
	if cfg.MemoryWatchdogThresholdPercent < 0 {
		return
	}

	free := freeRAMGB()
	freePercent := free / totalGB * 100
	if freePercent >= cfg.MemoryWatchdogThresholdPercent {
		return
	}

	watchdogMu.Lock()
	sinceLastRestart := time.Since(lastWatchdogRestart)
	if sinceLastRestart < watchdogCooldown {
		watchdogMu.Unlock()
		fmt.Printf(
			"[unified-gateway] memory watchdog: %.1f%% free (< %.0f%% threshold) but last restart was %s ago (cooldown %s) -- waiting\n",
			freePercent, cfg.MemoryWatchdogThresholdPercent, sinceLastRestart.Round(time.Second), watchdogCooldown,
		)
		return
	}
	watchdogMu.Unlock()

	restarted := restartActiveBackends(cfg)

	watchdogMu.Lock()
	lastWatchdogRestart = time.Now()
	watchdogMu.Unlock()

	if len(restarted) > 0 {
		fmt.Printf(
			"[unified-gateway] memory watchdog: %.1f%% free RAM (< %.0f%% threshold) -- restarted: %s\n",
			freePercent, cfg.MemoryWatchdogThresholdPercent, strings.Join(restarted, ", "),
		)
	} else {
		fmt.Printf(
			"[unified-gateway] memory watchdog: %.1f%% free RAM (< %.0f%% threshold) but nothing gateway-managed is currently loaded to restart (Ollama isn't spawned/killed by the gateway, so it's never touched here)\n",
			freePercent, cfg.MemoryWatchdogThresholdPercent,
		)
	}
}

// restartActiveBackends force-restarts whatever model is currently active
// on each gateway-spawned pool (chat, OCR, FLUX), one at a time -- Ollama
// is deliberately skipped, see the package doc comment above. Each
// restart goes through loadModel (the same cross-process-locked entry
// point `unified-gateway load` uses), so it can't race a concurrent
// user-triggered load, and reuses the exact same memory-safety check
// (which passes here in practice: freeing an already-loaded model's RSS
// almost always covers relaunching that same model, since RSS includes
// its KV-cache/prefix-cache on top of raw weight size, not just the
// weights the check estimates as "required").
func restartActiveBackends(cfg *Config) []string {
	var restarted []string
	for _, port := range []int{cfg.BackendPort, cfg.MediaBackendPort, cfg.FluxBackendPort} {
		name := runningBackendModel(cfg, port)
		if name == "" {
			continue
		}
		fmt.Printf("[unified-gateway] memory watchdog: restarting %q on :%d...\n", name, port)
		if err := loadModel(name); err != nil {
			fmt.Printf("[unified-gateway] memory watchdog: failed to restart %q: %v\n", name, err)
			continue
		}
		restarted = append(restarted, name)
	}
	return restarted
}

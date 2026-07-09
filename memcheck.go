package main

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

// Memory safety net for backend switches on Apple Silicon. Loading a new
// rapid-mlx/ds4-server model without checking free RAM first can push the
// machine into heavy swap or trigger a jetsam-driven forced restart under
// memory pressure (observed directly: a large model plus an already-loaded
// one pushed the system hard enough to force a reboot) -- so this runs
// BEFORE killPort/launch in loadModelLocked, and refuses the switch rather
// than leaving the user with a dead backend and a still-struggling machine.
// This used to live in unicorn-server's Python downloader; it moved here
// because this is the one place that actually spawns/kills these
// processes now.

// mlxCacheReserveMB caps rapid-mlx's memory-aware prefix cache
// (--cache-memory-mb) instead of leaving it at its default of ~20% of
// free RAM (--cache-memory-percent 0.20). Observed directly: a persisted
// on-disk prefix cache (~/.cache/rapid-mlx/prefix_cache/) gets reloaded
// into that reservation on every model start -- 9.4GB for a single 27B
// model in one test -- on top of the model's own weights. That's on
// top of, not instead of, the weight-size estimate this file already
// checks, and it scales with however much free RAM happens to be lying
// around at that moment, making it invisible to a size-based estimate.
// A fixed cap keeps the caching benefit (repeated/shared prompts still
// hit) without letting it balloon into the real cause of a "the model
// froze" report that isn't actually about the model at all.
const mlxCacheReserveMB = 4096
const memoryMarginFraction = 0.10 // 10% headroom beyond the raw model size

// freeRAMGB returns free+inactive pages from vm_stat, in GB. Apple Silicon
// Macs use a 16KB page size (vs 4KB on Intel) -- this assumes Apple
// Silicon, which is what rapid-mlx/MLX requires anyway.
func freeRAMGB() float64 {
	out, err := exec.Command("vm_stat").Output()
	if err != nil {
		return 0
	}
	var free, inactive float64
	for _, line := range strings.Split(string(out), "\n") {
		if strings.Contains(line, "Pages free") {
			free = parseVMStatPages(line)
		} else if strings.Contains(line, "Pages inactive") {
			inactive = parseVMStatPages(line)
		}
	}
	const pageSize = 16384.0
	return (free + inactive) * pageSize / (1024 * 1024 * 1024)
}

func parseVMStatPages(line string) float64 {
	parts := strings.SplitN(line, ":", 2)
	if len(parts) != 2 {
		return 0
	}
	v, err := strconv.ParseFloat(strings.TrimSuffix(strings.TrimSpace(parts[1]), "."), 64)
	if err != nil {
		return 0
	}
	return v
}

// estimateModelSizeGB estimates a model's resident memory footprint from
// its on-disk weight size -- the same proxy the old Python downloader used
// (sum of .safetensors shards, or the raw file size for ds4's single
// .gguf), since actual RSS isn't knowable before launch.
func estimateModelSizeGB(m ModelConfig) float64 {
	if m.Path == "" {
		return 0
	}
	info, err := os.Stat(m.Path)
	if err != nil {
		return 0
	}
	if !info.IsDir() {
		return float64(info.Size()) / (1024 * 1024 * 1024)
	}
	entries, err := os.ReadDir(m.Path)
	if err != nil {
		return 0
	}
	var total int64
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".safetensors") {
			continue
		}
		if fi, err := e.Info(); err == nil {
			total += fi.Size()
		}
	}
	return float64(total) / (1024 * 1024 * 1024)
}

// runningRSSGB returns the RSS, in GB, of whatever process currently owns
// port -- the memory that killPort(port) is about to free up.
func runningRSSGB(port int) float64 {
	out, err := exec.Command("lsof", "-ti", fmt.Sprintf(":%d", port), "-sTCP:LISTEN").Output()
	if err != nil {
		return 0
	}
	pids := strings.Fields(string(out))
	if len(pids) == 0 {
		return 0
	}
	psOut, err := exec.Command("ps", "-p", pids[0], "-o", "rss=").Output()
	if err != nil {
		return 0
	}
	kb, err := strconv.ParseFloat(strings.TrimSpace(string(psOut)), 64)
	if err != nil {
		return 0
	}
	return kb / (1024 * 1024)
}

// checkMemory refuses a switch if free + freeing RAM doesn't cover the new
// model plus a margin, rather than launching into a machine that's about
// to start swapping or get jetsam-killed.
func checkMemory(requiredGB, freeingGB float64) (bool, string) {
	free := freeRAMGB()
	available := free + freeingGB
	needed := requiredGB * (1.0 + memoryMarginFraction)
	if available < needed {
		return false, fmt.Sprintf(
			"insufficient memory: %.0f GB available (%.0f GB free + %.0f GB to be freed by killing the current backend), need ~%.0f GB (%.0f GB model + %.0f%% margin) — aborting, current backend left running",
			available, free, freeingGB, needed, requiredGB, memoryMarginFraction*100,
		)
	}
	return true, ""
}

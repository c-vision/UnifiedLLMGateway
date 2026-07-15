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

// mlxCacheReserveMBFor caps rapid-mlx's memory-aware prefix cache
// (--cache-memory-mb) instead of leaving it at its default of ~20% of
// free RAM (--cache-memory-percent 0.20). Observed directly: a persisted
// on-disk prefix cache (~/.cache/rapid-mlx/prefix_cache/) gets reloaded
// into that reservation on every model start -- 9.4GB for a single 27B
// model in one test -- on top of the model's own weights. That's on
// top of, not instead of, the weight-size estimate this file already
// checks, and it scales with however much free RAM happens to be lying
// around at that moment, making it invisible to a size-based estimate.
//
// The cap scales inversely with model size instead of one fixed value for
// everything: on a 128GB machine with ~100GB typically free, a 15GB model
// has plenty of headroom for a much bigger cache (real benefit: more
// repeated/shared-prompt hits), while a 70GB+ model already uses most of
// that headroom on weights alone and should stay conservative. Thresholds
// are tuned against this catalog's actual on-disk sizes (11-74GB).
func mlxCacheReserveMBFor(modelSizeGB float64) int {
	// Bumped 2026-07-10: observed directly on a real growing OpenCode
	// conversation -- a 41k-token request MISSED entirely because a
	// prefix-pressure eviction (cache_max=15.5GB, the previous <20GB
	// tier) had just evicted the very entry this conversation needed.
	// TurboQuant (KV-cache quantization, re-enabled the same day) cuts
	// the per-token footprint, and this machine typically has 30-47GB
	// free even with heavy IDEs open (checkMemory still refuses a load
	// outright if that's not true at load time) -- so there's real
	// headroom to trade for fewer growing-conversation evictions.
	switch {
	case modelSizeGB < 20:
		return 28672
	case modelSizeGB < 45:
		return 16384
	default:
		return 8192
	}
}

const memoryMarginFraction = 0.10 // 10% headroom beyond the raw model size

// freeRAMGB returns available RAM in GB as total physical memory minus the
// sum of every live process's RSS (2026-07-15 rewrite -- replaced a
// vm_stat "Pages free + Pages inactive" calculation that measurably
// undercounted what was actually available and was blocking loads of
// larger models the machine had real room for). vm_stat's free+inactive
// only counts pages already sitting idle; it ignores "active"/
// "speculative"/file-backed cache pages that aren't part of any live
// process's resident set -- disk cache and prefetch, not real
// application memory pressure, and macOS reclaims exactly this kind of
// page first under pressure. Direct comparison on this machine: vm_stat
// free+inactive read ~64GB while total-RSS read ~91GB (37GB used across
// ~900 processes) with 128GB installed -- the gap is precisely that
// reclaimable-but-not-idle memory. total-RSS matches what mactop (the
// user's reference terminal monitor) reports, confirmed directly by the
// user (ps -axo rss sum ~36-37GB used, matching mactop's own reading).
func freeRAMGB() float64 {
	total := totalRAMGB()
	if total <= 0 {
		return 0
	}
	used, err := usedRAMGB()
	if err != nil {
		return 0
	}
	avail := total - used
	if avail < 0 {
		avail = 0
	}
	return avail
}

// usedRAMGB sums the RSS of every live process via `ps -axo rss=` --
// deliberately not adjusted for shared pages (multiple processes mapping
// the same dyld shared cache, frameworks, etc. each count their share of
// it), which in principle can overcount true physical usage. Kept simple
// anyway because it's what was verified against mactop's own number on
// this machine, and the alternative (vm_stat free+inactive) was
// confirmed wrong in the other direction, blocking loads that had real
// headroom.
func usedRAMGB() (float64, error) {
	out, err := exec.Command("ps", "-axo", "rss=").Output()
	if err != nil {
		return 0, err
	}
	var sumKB float64
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		v, err := strconv.ParseFloat(line, 64)
		if err != nil {
			continue
		}
		sumKB += v
	}
	return sumKB / (1024 * 1024), nil
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

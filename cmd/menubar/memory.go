package main

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// Simplified live memory readout for the menu bar -- the user's own request
// after seeing mactop (a terminal system monitor) and asking for something
// similar, even if minimal, inside the menu bar itself. Scoped to memory
// only (not CPU/GPU/ANE/power like mactop): those need `powermetrics`,
// which requires sudo and would mean this app silently prompting for admin
// rights on every launch -- memory is the one stat directly relevant to the
// gateway's own memory-watchdog feature (see the root package's
// memwatchdog.go) and is readable with plain vm_stat/sysctl, no privilege
// escalation. Same vm_stat/sysctl parsing as the root package's
// memcheck.go/memwatchdog.go -- duplicated rather than shared, see
// gwConfig's doc comment above for why (two separate "main" packages).

// freeRAMGB returns available RAM in GB as total physical memory minus the
// sum of every live process's RSS (2026-07-15 rewrite, same reasoning as
// the root package's memcheck.go -- kept in sync by hand, see this
// package's doc comment above for why it's duplicated rather than
// shared). Replaced a vm_stat "Pages free + Pages inactive" calculation
// that measurably undercounted real availability: vm_stat's free+inactive
// ignores active/speculative/file-backed cache pages that aren't part of
// any live process's resident set but are reclaimed first under real
// pressure. Verified directly against mactop's own reading on this
// machine (ps -axo rss sum ~36-37GB used, matching mactop).
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

// usedRAMGB sums the RSS of every live process via `ps -axo rss=`. Not
// adjusted for shared pages (see memcheck.go's copy for the full
// rationale) -- kept simple because it's what was verified against
// mactop, and the vm_stat alternative was confirmed wrong in the other
// direction.
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

// totalRAMGB returns total physical memory in GB via sysctl.
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

// memoryWarnPercent/memoryCritPercent thresholds are purely cosmetic here
// (🟡/🔴 in the menu item) -- memoryCritPercent matches the watchdog's own
// default restart threshold in memwatchdog.go (kept as a literal, not read
// from models.json, since this is a simplified/best-effort display, not
// something that needs to track a live-editable config value).
const (
	memoryWarnPercent = 25.0
	memoryCritPercent = 15.0
)

// memoryStatusLine renders a one-line summary for the menu item title:
// "🟢 RAM: 92.1/128 GB free (72%)". Falls back to a plain dash if either
// reading comes back zero (vm_stat/sysctl unavailable or parse failure).
func memoryStatusLine() string {
	total := totalRAMGB()
	free := freeRAMGB()
	if total <= 0 {
		return "RAM: —"
	}
	percent := free / total * 100
	icon := "🟢"
	switch {
	case percent < memoryCritPercent:
		icon = "🔴"
	case percent < memoryWarnPercent:
		icon = "🟡"
	}
	return fmt.Sprintf("%s RAM: %.1f/%.0f GB free (%.0f%%)", icon, free, total, percent)
}

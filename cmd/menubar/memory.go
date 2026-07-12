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

// freeRAMGB returns free+inactive pages from vm_stat, in GB. Apple Silicon
// only (16KB page size).
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

// uninstall-service stops and removes the unified-gateway LaunchAgent
// installed by install-service.
package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

const label = "local.unified-gateway"

func main() {
	home, err := os.UserHomeDir()
	if err != nil {
		fmt.Fprintf(os.Stderr, "cannot resolve home directory: %v\n", err)
		os.Exit(1)
	}

	plistPath := filepath.Join(home, "Library", "LaunchAgents", label+".plist")
	target := fmt.Sprintf("gui/%d", os.Getuid())

	if out, err := exec.Command("launchctl", "bootout", target, plistPath).CombinedOutput(); err != nil {
		fmt.Printf("launchctl bootout: %v (%s) — continuing, it may already be stopped\n", err, string(out))
	} else {
		fmt.Println("🛑 unified-gateway LaunchAgent stopped")
	}

	if _, err := os.Stat(plistPath); err == nil {
		if err := os.Remove(plistPath); err != nil {
			fmt.Fprintf(os.Stderr, "cannot remove plist %s: %v\n", plistPath, err)
			os.Exit(1)
		}
		fmt.Printf("🗑️  removed %s\n", plistPath)
	} else {
		fmt.Println("no plist found — nothing to remove")
	}

	fmt.Println("✅ unified-gateway LaunchAgent uninstalled (the binary and models.json are untouched)")
}

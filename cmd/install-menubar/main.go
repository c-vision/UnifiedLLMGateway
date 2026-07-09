// install-menubar registers the unified-gateway menu bar controller as its
// own LaunchAgent, separate from local.unified-gateway. It's a convenience
// UI, not a critical service, so a deliberate Quit (clean exit(0) from
// systray.Quit()) stays down until the next login rather than respawning —
// the gateway itself keeps running either way, since the two are
// deliberately decoupled. KeepAlive's SuccessfulExit=false only covers the
// other case: getting killed out from under itself (observed in the wild
// — jetsam SIGTERM'ing it moments after RunAtLoad during a memory-pressure
// reboot, leaving the tray icon gone until manually restarted), which
// isn't a deliberate quit and should self-heal.
package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"text/template"
)

const label = "local.unified-gateway-menubar"

const plistTemplate = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>{{.Label}}</string>
	<key>ProgramArguments</key>
	<array>
		<string>{{.BinaryPath}}</string>
	</array>
	<key>WorkingDirectory</key>
	<string>{{.WorkingDir}}</string>
	<key>RunAtLoad</key>
	<true/>
	<key>KeepAlive</key>
	<dict>
		<key>SuccessfulExit</key>
		<false/>
	</dict>
	<key>ThrottleInterval</key>
	<integer>10</integer>
	<key>StandardOutPath</key>
	<string>{{.StdoutLog}}</string>
	<key>StandardErrorPath</key>
	<string>{{.StderrLog}}</string>
</dict>
</plist>
`

func main() {
	home, err := os.UserHomeDir()
	if err != nil {
		fmt.Fprintf(os.Stderr, "cannot resolve home directory: %v\n", err)
		os.Exit(1)
	}

	binDir := filepath.Join(home, ".local", "bin")
	binaryPath := filepath.Join(binDir, "unified-gateway-menubar")
	if _, err := os.Stat(binaryPath); err != nil {
		fmt.Fprintf(os.Stderr, "unified-gateway-menubar binary not found at %s — build and install it there first\n", binaryPath)
		os.Exit(1)
	}

	logDir := filepath.Join(home, "Library", "Logs", "unified-gateway")
	if err := os.MkdirAll(logDir, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "cannot create log directory: %v\n", err)
		os.Exit(1)
	}

	launchAgentsDir := filepath.Join(home, "Library", "LaunchAgents")
	if err := os.MkdirAll(launchAgentsDir, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "cannot create LaunchAgents directory: %v\n", err)
		os.Exit(1)
	}
	plistPath := filepath.Join(launchAgentsDir, label+".plist")

	data := struct {
		Label      string
		BinaryPath string
		WorkingDir string
		StdoutLog  string
		StderrLog  string
	}{
		Label:      label,
		BinaryPath: binaryPath,
		WorkingDir: binDir,
		StdoutLog:  filepath.Join(logDir, "menubar-stdout.log"),
		StderrLog:  filepath.Join(logDir, "menubar-stderr.log"),
	}

	f, err := os.Create(plistPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cannot write plist: %v\n", err)
		os.Exit(1)
	}
	tmpl := template.Must(template.New("plist").Parse(plistTemplate))
	if err := tmpl.Execute(f, data); err != nil {
		f.Close()
		fmt.Fprintf(os.Stderr, "cannot render plist: %v\n", err)
		os.Exit(1)
	}
	f.Close()

	target := fmt.Sprintf("gui/%d", os.Getuid())

	// Idempotent: tear down any previous instance before (re)installing.
	exec.Command("launchctl", "bootout", target, plistPath).Run()

	if out, err := exec.Command("launchctl", "bootstrap", target, plistPath).CombinedOutput(); err != nil {
		fmt.Fprintf(os.Stderr, "launchctl bootstrap failed: %v\n%s\n", err, out)
		os.Exit(1)
	}
	if out, err := exec.Command("launchctl", "enable", target+"/"+label).CombinedOutput(); err != nil {
		fmt.Fprintf(os.Stderr, "warning: launchctl enable failed: %v\n%s\n", err, out)
	}

	fmt.Printf("✅ unified-gateway-menubar installed as a LaunchAgent (%s)\n", label)
	fmt.Printf("   binary:  %s\n", binaryPath)
	fmt.Printf("   plist:   %s\n", plistPath)
	fmt.Printf("   logs:    %s\n", logDir)
	fmt.Println("   Starts at login, restarts if killed unexpectedly (e.g. jetsam during a")
	fmt.Println("   memory-pressure reboot) — but a deliberate Quit stays down until next login.")
	fmt.Println("   Does not affect local.unified-gateway, which is a separate, independent")
	fmt.Println("   LaunchAgent.")
}

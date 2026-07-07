// install-service registers unified-gateway as a macOS LaunchAgent so it
// starts automatically at login and restarts if it crashes.
package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"text/template"
)

const label = "local.unified-gateway"

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
	<true/>
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
	binaryPath := filepath.Join(binDir, "unified-gateway")
	if _, err := os.Stat(binaryPath); err != nil {
		fmt.Fprintf(os.Stderr, "unified-gateway binary not found at %s — build and install it there first\n", binaryPath)
		os.Exit(1)
	}
	if _, err := os.Stat(filepath.Join(binDir, "models.json")); err != nil {
		fmt.Fprintf(os.Stderr, "models.json not found next to %s — copy it there first\n", binaryPath)
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
		StdoutLog:  filepath.Join(logDir, "stdout.log"),
		StderrLog:  filepath.Join(logDir, "stderr.log"),
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

	fmt.Printf("✅ unified-gateway installed as a LaunchAgent (%s)\n", label)
	fmt.Printf("   binary:  %s\n", binaryPath)
	fmt.Printf("   plist:   %s\n", plistPath)
	fmt.Printf("   logs:    %s\n", logDir)
	fmt.Println("   Starts automatically at login, restarts on crash (KeepAlive).")
	fmt.Println("   Note: it only starts the gateway adapters — load a model with 'unified-gateway load <shortname>' separately.")
}

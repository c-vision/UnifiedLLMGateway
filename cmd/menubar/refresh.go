package main

import (
	"fmt"
	"time"

	"github.com/getlantern/systray"
)

// refreshRefs bundles every menu item refreshLoop needs to keep in sync
// with live process/port state. mMLX, mDS4, mStartMLX, mStartDS4, mStopMLX
// and mStopDS4 are nil when models.json failed to load.
type refreshRefs struct {
	mStatus       *systray.MenuItem
	mStartGateway *systray.MenuItem
	mStopGateway  *systray.MenuItem
	mMLX          *systray.MenuItem
	mDS4          *systray.MenuItem
	mOllamaGroup  *systray.MenuItem
	mStartMLX     *systray.MenuItem // permanently disabled by addStartItem if mlxDefault == ""
	mStartDS4     *systray.MenuItem // permanently disabled by addStartItem if ds4Default == ""
	mlxDefault    string            // model Start rapid-mlx loads; "" if none configured
	ds4Default    string            // model Start ds4 loads; "" if none configured
	mStopMLX      *systray.MenuItem
	mStopDS4      *systray.MenuItem
	mOllamaStart  *systray.MenuItem
	mOllamaStop   *systray.MenuItem
	cfg           *gwConfig
	modelItems    map[string]*systray.MenuItem
}

// setEnabled is a small helper since MenuItem only exposes Enable/Disable,
// not a single "SetEnabled(bool)" call.
func setEnabled(item *systray.MenuItem, enabled bool) {
	if item == nil {
		return
	}
	if enabled {
		item.Enable()
	} else {
		item.Disable()
	}
}

// refreshLoop polls process/port state every few seconds and reflects it
// across the whole menu: gateway/model status line, Start/Stop items
// disabled to match reality (never both enabled, never both disabled),
// which of rapid-mlx/ds4/Ollama is confirmed active, and the checkmark on
// the active model. The tray icon itself stays a fixed black "U" — status
// is conveyed only by the 🟢/🟡/🔴 text in the menu items. Each backend
// only ever shows two states here — running (🟢) or not (🔴) — port
// conflicts with some other, unrelated process are handled at Start time
// instead (see confirmPortFree in control.go), not as a third status here.
func refreshLoop(r refreshRefs) {
	ticker := time.NewTicker(4 * time.Second)
	defer ticker.Stop()

	for {
		gwUp := portOpen(8082) && portOpen(8083)
		ab := readActiveBackend()
		ollUp := ollamaRunning()

		setEnabled(r.mStartGateway, !gwUp)
		setEnabled(r.mStopGateway, gwUp)
		setEnabled(r.mOllamaStart, !ollUp)
		setEnabled(r.mOllamaStop, ollUp)

		statusLine := "🔴 stopped"
		switch {
		case gwUp && ab.Model != "":
			statusLine = fmt.Sprintf("🟢 %s", ab.Model)
		case gwUp:
			statusLine = "🟡 no model"
		}
		r.mStatus.SetTitle(statusLine)

		activeType := "" // "mlx", "ds4", "ollama", or "" if unknown/idle
		if r.cfg != nil && ab.Model != "" {
			if m, ok := r.cfg.Models[ab.Model]; ok {
				activeType = m.Backend
			}
		}

		if r.cfg != nil {
			backendUp := portOpen(r.cfg.BackendPort)
			mlxActive := backendUp && activeType == "mlx"
			ds4Active := backendUp && activeType == "ds4"

			if mlxActive {
				r.mMLX.SetTitle(fmt.Sprintf("🟢 rapid-mlx %s (port %d)", ab.Model, r.cfg.BackendPort))
			} else {
				r.mMLX.SetTitle("🔴 rapid-mlx")
			}
			if r.mlxDefault != "" {
				setEnabled(r.mStartMLX, !mlxActive)
			}
			setEnabled(r.mStopMLX, mlxActive)

			if ds4Active {
				r.mDS4.SetTitle(fmt.Sprintf("🟢 ds4 %s (port %d)", ab.Model, r.cfg.BackendPort))
			} else {
				r.mDS4.SetTitle("🔴 ds4")
			}
			if r.ds4Default != "" {
				setEnabled(r.mStartDS4, !ds4Active)
			}
			setEnabled(r.mStopDS4, ds4Active)
		}

		ollamaPort := 11434
		if r.cfg != nil && r.cfg.OllamaPort != 0 {
			ollamaPort = r.cfg.OllamaPort
		}
		switch {
		case ollUp && activeType == "ollama":
			r.mOllamaGroup.SetTitle(fmt.Sprintf("🟢 Ollama %s (port %d)", ab.Model, ollamaPort))
		case ollUp:
			r.mOllamaGroup.SetTitle(fmt.Sprintf("🟢 Ollama (port %d)", ollamaPort))
		default:
			r.mOllamaGroup.SetTitle("🔴 Ollama")
		}

		for name, item := range r.modelItems {
			if gwUp && ab.Model == name {
				item.Check()
			} else {
				item.Uncheck()
			}
		}

		<-ticker.C
	}
}

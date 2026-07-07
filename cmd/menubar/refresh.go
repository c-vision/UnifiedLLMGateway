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

// refreshLoop polls live state every few seconds and reflects it across
// the whole menu. Nothing here is read from a file this tool wrote
// earlier — a different client could have replaced rapid-mlx or ds4's
// process, or the user could have stopped Ollama by hand, so every
// status is re-derived from the real, current state each tick:
//   - rapid-mlx/ds4: which process actually owns the backend port, and
//     which model it's serving, read from that process's own command
//     line (see runningMLXModel/runningDS4Model in control.go)
//   - Ollama: whether the app process is alive (pgrep), and which model
//     it has loaded, via Ollama's own /api/ps
//   - the top status line: what the gateway itself is routing to right
//     now, via its own GET /v1/models — the gateway's live answer, not a
//     cached guess
func refreshLoop(r refreshRefs) {
	ticker := time.NewTicker(4 * time.Second)
	defer ticker.Stop()

	for {
		gwUp := portOpen(8082) && portOpen(8083)
		ollUp := ollamaRunning()

		setEnabled(r.mStartGateway, !gwUp)
		setEnabled(r.mStopGateway, gwUp)
		setEnabled(r.mOllamaStart, !ollUp)
		setEnabled(r.mOllamaStop, ollUp)

		statusLine := "🔴 stopped"
		if gwUp {
			if m := gatewayCurrentModel(); m != "" {
				statusLine = fmt.Sprintf("🟢 %s", m)
			} else {
				statusLine = "🟡 no model"
			}
		}
		r.mStatus.SetTitle(statusLine)

		var mlxModel, ds4Model string
		var mlxActive, ds4Active bool
		if r.cfg != nil {
			mlxModel, mlxActive = runningMLXModel(r.cfg.BackendPort)
			ds4Model, ds4Active = runningDS4Model(r.cfg, r.cfg.BackendPort)

			switch {
			case mlxActive && mlxModel != "":
				r.mMLX.SetTitle(fmt.Sprintf("🟢 rapid-mlx %s (port %d)", mlxModel, r.cfg.BackendPort))
			case mlxActive:
				r.mMLX.SetTitle(fmt.Sprintf("🟢 rapid-mlx (port %d)", r.cfg.BackendPort))
			default:
				r.mMLX.SetTitle("🔴 rapid-mlx")
			}
			if r.mlxDefault != "" {
				setEnabled(r.mStartMLX, !mlxActive)
			}
			setEnabled(r.mStopMLX, mlxActive)

			switch {
			case ds4Active && ds4Model != "":
				r.mDS4.SetTitle(fmt.Sprintf("🟢 ds4 %s (port %d)", ds4Model, r.cfg.BackendPort))
			case ds4Active:
				r.mDS4.SetTitle(fmt.Sprintf("🟢 ds4 (port %d)", r.cfg.BackendPort))
			default:
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
		var ollamaModel string
		if ollUp {
			ollamaModel = runningOllamaModel(ollamaPort)
		}
		switch {
		case ollUp && ollamaModel != "":
			r.mOllamaGroup.SetTitle(fmt.Sprintf("🟢 Ollama %s (port %d)", ollamaModel, ollamaPort))
		case ollUp:
			r.mOllamaGroup.SetTitle(fmt.Sprintf("🟢 Ollama (port %d)", ollamaPort))
		default:
			r.mOllamaGroup.SetTitle("🔴 Ollama")
		}

		for name, item := range r.modelItems {
			checked := (mlxActive && name == mlxModel) || (ds4Active && name == ds4Model)
			if !checked && ollUp && ollamaModel != "" && r.cfg != nil {
				if m, ok := r.cfg.Models[name]; ok && m.Backend == "ollama" {
					tag := m.OllamaModel
					if tag == "" {
						tag = name
					}
					checked = tag == ollamaModel
				}
			}
			if checked {
				item.Check()
			} else {
				item.Uncheck()
			}
		}

		<-ticker.C
	}
}

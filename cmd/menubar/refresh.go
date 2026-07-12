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
	mMemory       *systray.MenuItem // simplified live RAM readout, see memory.go
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
	mCompression  *systray.MenuItem
	mOCR          *systray.MenuItem // nil if no OCR-like "kind":"media" entries configured
	mStartOCR     *systray.MenuItem // permanently disabled by addStartItem if ocrDefault == ""
	mStopOCR      *systray.MenuItem
	ocrDefault    string            // model Start OCR loads; "" if none configured
	mFlux         *systray.MenuItem // nil if no "backend":"mflux" entries configured
	mStartFlux    *systray.MenuItem // permanently disabled by addStartItem if fluxDefault == ""
	mStopFlux     *systray.MenuItem
	fluxDefault   string // model Start FLUX loads; "" if none configured
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

		if r.mCompression != nil {
			setEnabled(r.mCompression, gwUp)
			if gwUp {
				if state, ok := getCompressionState(); ok {
					if state.Enabled {
						r.mCompression.Check()
					} else {
						r.mCompression.Uncheck()
					}
					r.mCompression.SetTooltip(fmt.Sprintf(
						"Trim stale/duplicate old-message content before it reaches the model — takes effect instantly, no restart\n%d requests compressed, %d chars saved this session",
						state.RequestsCompressed, state.CharsSaved,
					))
				}
			}
		}

		if r.mMemory != nil {
			r.mMemory.SetTitle(memoryStatusLine())
		}

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

		// OCR-like and FLUX-family media models are each their own POOL,
		// same live-detection pattern as rapid-mlx/ds4 above -- exclusive
		// switching within the pool, but the two pools (and the chat pool)
		// never affect each other. runningFluxModel mirrors runningDS4Model:
		// mflux's server has no --served-model-name equivalent, so its
		// --model-path is matched back against configured "mflux" entries.
		var ocrModel string
		var ocrActive bool
		if r.cfg != nil && r.mOCR != nil {
			ocrModel, ocrActive = runningMLXModel(r.cfg.MediaBackendPort)
			if !ocrActive {
				if m, active := runningDS4Model(r.cfg, r.cfg.MediaBackendPort); active {
					ocrModel, ocrActive = m, true
				}
			}
			switch {
			case ocrActive && ocrModel != "":
				r.mOCR.SetTitle(fmt.Sprintf("🟢 OCR %s (port %d)", ocrModel, r.cfg.MediaBackendPort))
			case ocrActive:
				r.mOCR.SetTitle(fmt.Sprintf("🟢 OCR (port %d)", r.cfg.MediaBackendPort))
			default:
				r.mOCR.SetTitle("🔴 OCR")
			}
			if r.ocrDefault != "" {
				setEnabled(r.mStartOCR, !ocrActive)
			}
			setEnabled(r.mStopOCR, ocrActive)
		}

		var fluxModel string
		var fluxActive bool
		if r.cfg != nil && r.mFlux != nil {
			fluxModel, fluxActive = runningFluxModel(r.cfg, r.cfg.FluxBackendPort)
			switch {
			case fluxActive && fluxModel != "":
				r.mFlux.SetTitle(fmt.Sprintf("🟢 FLUX %s (port %d)", fluxModel, r.cfg.FluxBackendPort))
			case fluxActive:
				r.mFlux.SetTitle(fmt.Sprintf("🟢 FLUX (port %d)", r.cfg.FluxBackendPort))
			default:
				r.mFlux.SetTitle("🔴 FLUX (Image Generation)")
			}
			if r.fluxDefault != "" {
				setEnabled(r.mStartFlux, !fluxActive)
			}
			setEnabled(r.mStopFlux, fluxActive)
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
			checked := (mlxActive && name == mlxModel) || (ds4Active && name == ds4Model) ||
				(ocrActive && name == ocrModel) || (fluxActive && name == fluxModel)
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

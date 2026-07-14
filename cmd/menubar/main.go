// menubar is a macOS menu bar controller for the unified-gateway stack:
// start/stop the gateway adapters, the currently-loaded model backend
// (rapid-mlx/ds4), and Ollama, plus switch the active model — all
// independently of the headless unified-gateway LaunchAgent, which keeps
// running whether or not this app is open.
package main

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/getlantern/systray"
)

func main() {
	systray.Run(onReady, func() {})
}

// addModelItems adds one clickable submenu item per model name (sorted),
// wired to load that model on click, and records each in modelItems so
// refreshLoop can checkmark whichever one is currently active. Titles
// never change on click — refreshLoop's periodic polling is the only
// source of truth for what's running, so there's no transient
// "loading…"/"failed" text that could get stuck if a load is aborted or
// fails. No port confirmation here either: switching between models of
// the same backend is the everyday case and should stay frictionless,
// exactly like `unified-gateway load <name>` already behaves on the
// command line.
func addModelItems(parent *systray.MenuItem, cfg *gwConfig, names []string, modelItems map[string]*systray.MenuItem) {
	// Sorted by on-disk size (smallest first), not alphabetically -- that's
	// the number that actually decides whether a model coexists with
	// whatever else is running, so it's the more useful ordering in a menu
	// you're scanning to pick one. Note this reads live off disk: a model
	// that's still downloading will sort/display by its current partial
	// size until the download finishes, not its final size. Entries with
	// no computable disk size (ollama backend: weights aren't at a local
	// path this app can see) sort last, grouped together, rather than
	// being guessed into the middle.
	sort.Strings(names) // stable alphabetical tie-break before the real sort
	sort.SliceStable(names, func(i, j int) bool {
		di, dj := estimateDiskGB(cfg.Models[names[i]].Path), estimateDiskGB(cfg.Models[names[j]].Path)
		if di == 0 {
			di = math.Inf(1)
		}
		if dj == 0 {
			dj = math.Inf(1)
		}
		return di < dj
	})
	for _, n := range names {
		m := cfg.Models[n]
		var extra []string
		if m.Ctx > 0 {
			extra = append(extra, formatCtx(m.Ctx))
		}
		if disk := estimateDiskGB(m.Path); disk > 0 {
			extra = append(extra, fmt.Sprintf("%.0fGB", disk))
		}
		// Insert ctx/disk into m.Label's own trailing "(...)" group (the
		// quantization info) rather than the "(shortname)" group added
		// below -- every label in models.json ends in ")", so this is a
		// straight strip-and-reclose rather than real parsing.
		displayLabel := m.Label
		if len(extra) > 0 && strings.HasSuffix(displayLabel, ")") {
			displayLabel = strings.TrimSuffix(displayLabel, ")") + ", " + strings.Join(extra, ", ") + ")"
		}
		label := fmt.Sprintf("%s (%s)", displayLabel, n)
		if icon, ok := codingQualityIcon[n]; ok {
			label = icon + " " + label
		}
		item := parent.AddSubMenuItem(label, "Load "+n)
		modelItems[n] = item
		go func(shortName string, item *systray.MenuItem) {
			for range item.ClickedCh {
				loadModelAsync(shortName)
			}
		}(n, item)
	}
}

// addStartItem adds a "Start <backend>" item that loads targetModel (the
// first model configured for that backend, alphabetically) on click —
// the rapid-mlx/ds4 equivalent of Ollama's plain "Start Ollama", which
// doesn't need a model choice up front. It's only enabled (see
// refreshLoop) when this backend isn't already the active one, so a port
// conflict here always means a genuinely different service is in the
// way — worth confirming before killing it. Like addModelItems, its title
// never changes on click; refreshLoop's polling drives enabled/disabled
// state, so there's nothing that can get stuck showing "failed".
// Permanently disabled if the backend has no models configured at all
// (targetModel == "").
func addStartItem(parent *systray.MenuItem, label, targetModel string, port int) *systray.MenuItem {
	tooltip := fmt.Sprintf("Load %s on port %d", targetModel, port)
	item := parent.AddSubMenuItem(label, tooltip)
	if targetModel == "" {
		item.Disable()
		return item
	}
	go func() {
		for range item.ClickedCh {
			if !confirmPortFree(port, label) {
				continue
			}
			loadModelAsync(targetModel)
		}
	}()
	return item
}

func onReady() {
	icon := buildIcon()
	systray.SetTemplateIcon(icon, icon)
	systray.SetTooltip("Unified Gateway")

	mStatus := systray.AddMenuItem("Checking status...", "")
	mStatus.Disable()

	// Simplified live memory readout, refreshed on the same 4s tick as
	// everything else (see memory.go). Deliberately left enabled rather
	// than Disable()'d like mStatus above it -- macOS renders disabled
	// items dimmed/grey, which made this hard to read. Left clickable
	// with a no-op drain instead (same shape as the model items' own
	// per-item goroutine below) so it renders at full contrast.
	mMemory := systray.AddMenuItem("RAM: checking...", "Free RAM (vm_stat free+inactive) vs total physical memory")
	go func() {
		for range mMemory.ClickedCh {
			// no-op -- informational only, click just needs somewhere to go
		}
	}()
	systray.AddSeparator()

	mStartGateway := systray.AddMenuItem("Start Gateway Adapters", "Start OpenAI (port 8082) and Anthropic (port 8083) adapters")
	mStopGateway := systray.AddMenuItem("Stop Gateway Adapters", "Stop OpenAI/Anthropic adapters")
	systray.AddSeparator()

	cfg, cfgErr := loadGWConfig()
	modelItems := map[string]*systray.MenuItem{}
	var mMLX, mDS4, mStartMLX, mStartDS4, mStopMLX, mStopDS4 *systray.MenuItem
	var mOCR, mStartOCR, mStopOCR *systray.MenuItem
	var mFlux, mStartFlux, mStopFlux *systray.MenuItem
	var mlxDefault, ds4Default, ocrDefault, fluxDefault string

	if cfgErr != nil {
		mMissing := systray.AddMenuItem("Backends unavailable (models.json not found)", "")
		mMissing.Disable()
	} else {
		// Media-kind entries split into two POOLS, not lumped together --
		// OCR-like ones (any backend except "mflux") share
		// cfg.MediaBackendPort, FLUX-family ones (backend "mflux") share
		// cfg.FluxBackendPort. Each pool behaves exactly like rapid-mlx/ds4
		// above: one shared Start/Stop, exclusive switching within the
		// pool (loading flux2-klein-4b kills flux1-dev if it was running,
		// same as loading a different chat model kills the previous one)
		// -- but the two pools and the chat pool never touch each other.
		var mlxNames, ds4Names, ocrNames, fluxNames []string
		for n, m := range cfg.Models {
			if m.Kind == "media" {
				if m.Backend == "mflux" {
					fluxNames = append(fluxNames, n)
				} else {
					ocrNames = append(ocrNames, n)
				}
				continue
			}
			switch m.Backend {
			case "mlx":
				mlxNames = append(mlxNames, n)
			case "ds4":
				ds4Names = append(ds4Names, n)
			}
		}
		sort.Strings(mlxNames)
		sort.Strings(ds4Names)
		sort.Strings(ocrNames)
		sort.Strings(fluxNames)

		if len(mlxNames) > 0 {
			mlxDefault = mlxNames[0]
		}
		if len(ds4Names) > 0 {
			ds4Default = ds4Names[0]
		}
		if len(ocrNames) > 0 {
			ocrDefault = ocrNames[0]
		}
		if len(fluxNames) > 0 {
			fluxDefault = fluxNames[0]
		}

		mMLX = systray.AddMenuItem("rapid-mlx", "")
		mStartMLX = addStartItem(mMLX, "Start rapid-mlx", mlxDefault, cfg.BackendPort)
		mStopMLX = mMLX.AddSubMenuItem("Stop rapid-mlx", fmt.Sprintf("Stop the backend on port %d", cfg.BackendPort))
		addModelItems(mMLX, cfg, mlxNames, modelItems)

		mDS4 = systray.AddMenuItem("ds4", "")
		mStartDS4 = addStartItem(mDS4, "Start ds4", ds4Default, cfg.BackendPort)
		mStopDS4 = mDS4.AddSubMenuItem("Stop ds4", fmt.Sprintf("Stop the backend on port %d", cfg.BackendPort))
		addModelItems(mDS4, cfg, ds4Names, modelItems)

		if len(ocrNames) > 0 || len(fluxNames) > 0 {
			systray.AddSeparator()
		}
		if len(ocrNames) > 0 {
			mOCR = systray.AddMenuItem("OCR", "Non-chat models kept out of the chat pickers -- own pool, own port, independent of the chat backend and of FLUX")
			mStartOCR = addStartItem(mOCR, "Start OCR", ocrDefault, cfg.MediaBackendPort)
			mStopOCR = mOCR.AddSubMenuItem("Stop OCR", fmt.Sprintf("Stop the backend on port %d", cfg.MediaBackendPort))
			addModelItems(mOCR, cfg, ocrNames, modelItems)
		}
		if len(fluxNames) > 0 {
			mFlux = systray.AddMenuItem("FLUX (Image Generation)", "Image-generation models via mflux -- own pool, own port, independent of the chat backend and of OCR")
			mStartFlux = addStartItem(mFlux, "Start FLUX", fluxDefault, cfg.FluxBackendPort)
			mStopFlux = mFlux.AddSubMenuItem("Stop FLUX", fmt.Sprintf("Stop the backend on port %d", cfg.FluxBackendPort))
			addModelItems(mFlux, cfg, fluxNames, modelItems)
		}
	}
	systray.AddSeparator()

	mOllamaGroup := systray.AddMenuItem("Ollama", "")
	mOllamaStart := mOllamaGroup.AddSubMenuItem("Start Ollama", "")
	mOllamaStop := mOllamaGroup.AddSubMenuItem("Stop Ollama", "")
	if cfgErr == nil {
		var ollamaNames []string
		for n, m := range cfg.Models {
			if m.Backend == "ollama" {
				ollamaNames = append(ollamaNames, n)
			}
		}
		addModelItems(mOllamaGroup, cfg, ollamaNames, modelItems)
	}
	systray.AddSeparator()

	mStartAll := systray.AddMenuItem("Start All", "Start gateway adapters + Ollama")
	mStopAll := systray.AddMenuItem("Stop All", "Stop gateway adapters, backend, and Ollama")
	systray.AddSeparator()

	mCompression := systray.AddMenuItem("Prompt Compression", "Trim stale/duplicate old-message content before it reaches the model — takes effect instantly, no restart")
	systray.AddSeparator()

	mReload := systray.AddMenuItem("Reload Settings", "Restart this menu bar to pick up changes to models.json")
	systray.AddSeparator()

	mQuit := systray.AddMenuItem("Quit Menu Bar", "Quits only this menu bar — services keep running")

	go refreshLoop(refreshRefs{
		mStatus:       mStatus,
		mMemory:       mMemory,
		mStartGateway: mStartGateway,
		mStopGateway:  mStopGateway,
		mMLX:          mMLX,
		mDS4:          mDS4,
		mOllamaGroup:  mOllamaGroup,
		mStartMLX:     mStartMLX,
		mStartDS4:     mStartDS4,
		mlxDefault:    mlxDefault,
		ds4Default:    ds4Default,
		mStopMLX:      mStopMLX,
		mStopDS4:      mStopDS4,
		mOllamaStart:  mOllamaStart,
		mOllamaStop:   mOllamaStop,
		cfg:           cfg,
		modelItems:    modelItems,
		mCompression:  mCompression,
		mOCR:          mOCR,
		mStartOCR:     mStartOCR,
		mStopOCR:      mStopOCR,
		ocrDefault:    ocrDefault,
		mFlux:         mFlux,
		mStartFlux:    mStartFlux,
		mStopFlux:     mStopFlux,
		fluxDefault:   fluxDefault,
	})

	ollamaPort := 11434
	if cfg != nil && cfg.OllamaPort != 0 {
		ollamaPort = cfg.OllamaPort
	}

	go func() {
		for {
			select {
			case <-mStartGateway.ClickedCh:
				startGateway()
			case <-mStopGateway.ClickedCh:
				stopGateway()
			case <-clickedOrNil(mStopMLX):
				stopBackend(cfg)
			case <-clickedOrNil(mStopDS4):
				stopBackend(cfg)
			case <-clickedOrNil(mStopOCR):
				killPort(cfg.MediaBackendPort)
			case <-clickedOrNil(mStopFlux):
				killPort(cfg.FluxBackendPort)
			case <-mOllamaStart.ClickedCh:
				go func() {
					if confirmPortFree(ollamaPort, "Ollama") {
						startOllama()
					}
				}()
			case <-mOllamaStop.ClickedCh:
				stopOllama()
			case <-mStartAll.ClickedCh:
				startGateway()
				startOllama()
			case <-mStopAll.ClickedCh:
				stopGateway()
				stopBackend(cfg)
				stopAllMediaBackends(cfg)
				stopOllama()
			case <-mCompression.ClickedCh:
				state, ok := getCompressionState()
				if !ok {
					notify("Unified Gateway", "Gateway unreachable — start it first")
					continue
				}
				if err := setCompressionEnabled(!state.Enabled); err != nil {
					notify("Unified Gateway", fmt.Sprintf("Failed to toggle prompt compression: %v", err))
					continue
				}
				if state.Enabled {
					notify("Unified Gateway", "Prompt compression disabled")
				} else {
					notify("Unified Gateway", "Prompt compression enabled")
				}
			case <-mReload.ClickedCh:
				if err := relaunchSelf(); err != nil {
					continue // couldn't spawn the replacement — stay running rather than quit into nothing
				}
				systray.Quit()
				return
			case <-mQuit.ClickedCh:
				systray.Quit()
				return
			}
		}
	}()
}

// clickedOrNil returns item's ClickedCh, or nil (a channel that never
// fires) if item itself is nil — lets optional menu items (absent when
// models.json failed to load) sit safely in the same select statement.
func clickedOrNil(item *systray.MenuItem) chan struct{} {
	if item == nil {
		return nil
	}
	return item.ClickedCh
}

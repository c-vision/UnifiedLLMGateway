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
	"time"

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
		// qw38dflash2 shares same Path as qw38, but total with draft is +1GB
		if names[i] == "qw38dflash2" {
			di += 1
		}
		if names[j] == "qw38dflash2" {
			dj += 1
		}
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
		disk := estimateDiskGB(m.Path)
		// qw38 shows 27GB via .safetensors sum, but du is 28G — and dflash adds draft
		if n == "qw38dflash2" && disk > 0 {
			disk += 1 // draft 975M ≈1G
		}
		if disk > 0 {
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
		// Disabled models are shown greyed out and not clickable (their load
		// would fail anyway — the gateway refuses them).
		if m.Disabled {
			item.Disable()
		}
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
	var mSmall, mStartSmall, mStopSmall *systray.MenuItem
	var mlxDefault, ds4Default, ocrDefault, fluxDefault, smallDefault string

	if cfgErr != nil {
		mMissing := systray.AddMenuItem("Backends unavailable (models.json not found)", "")
		mMissing.Disable()
	} else {
		// Media-kind entries split into two POOLS, not lumped together --
		// OCR-like ones (any backend except "mflux") share
		// cfg.MediaBackendPort, FLUX-family ones (backend "mflux") share
		// cfg.FluxBackendPort. "small" entries get their own dedicated
		// pool on cfg.SmallBackendPort. Each pool behaves exactly like
		// rapid-mlx/ds4 above: one shared Start/Stop, exclusive switching
		// within the pool (loading flux2-klein-4b kills flux1-dev if it
		// was running, same as loading a different chat model kills the
		// previous one) -- but the pools never touch each other.
		var mlxNames, ds4Names, ocrNames, fluxNames, smallNames, omlxNames []string
		for n, m := range cfg.Models {
			if m.Kind == "media" {
				if m.Backend == "mflux" {
					fluxNames = append(fluxNames, n)
				} else {
					ocrNames = append(ocrNames, n)
				}
				continue
			}
			if m.Kind == "small" {
				smallNames = append(smallNames, n)
				continue
			}
			switch m.Backend {
			case "mlx":
				mlxNames = append(mlxNames, n)
			case "omlx":
				omlxNames = append(omlxNames, n)
			case "ds4":
				ds4Names = append(ds4Names, n)
			}
		}
		sort.Strings(mlxNames)
		sort.Strings(omlxNames)
		sort.Strings(ds4Names)
		sort.Strings(ocrNames)
		sort.Strings(fluxNames)
		sort.Strings(smallNames)

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
		if len(smallNames) > 0 {
			smallDefault = smallNames[0]
		}

		mMLX = systray.AddMenuItem("rapid-mlx", "")
		mStartMLX = addStartItem(mMLX, "Start rapid-mlx", mlxDefault, cfg.BackendPort)
		mStopMLX = mMLX.AddSubMenuItem("Stop rapid-mlx", fmt.Sprintf("Stop the backend on port %d", cfg.BackendPort))
		addModelItems(mMLX, cfg, mlxNames, modelItems)

		if len(smallNames) > 0 {
			mSmall = systray.AddMenuItem("small models", "Cheap secondary models (bonsai4b, llama3.2-1B; e.g. opencode's small_model for conversation titles) -- own pool, own port, never kills the main chat model")
			mStartSmall = addStartItem(mSmall, "Start small", smallDefault, cfg.SmallBackendPort)
			mStopSmall = mSmall.AddSubMenuItem("Stop small", fmt.Sprintf("Stop the backend on port %d", cfg.SmallBackendPort))
			addModelItems(mSmall, cfg, smallNames, modelItems)
		}

		mDS4 = systray.AddMenuItem("ds4", "")
		mStartDS4 = addStartItem(mDS4, "Start ds4", ds4Default, cfg.BackendPort)
		mStopDS4 = mDS4.AddSubMenuItem("Stop ds4", fmt.Sprintf("Stop the backend on port %d", cfg.BackendPort))
		addModelItems(mDS4, cfg, ds4Names, modelItems)

		// oMLX-backend models get their own section (they're served by the
		// isolated oMLX server, not rapid-mlx). No Start/Stop here: oMLX shares
		// the gateway backend port handling, and today its only entry is a
		// disabled/unusable model — so this is purely a category so the engine
		// is unambiguous. Disabled entries are greyed/non-clickable by
		// addModelItems.
		if len(omlxNames) > 0 {
			// Static red status dot: today oMLX is never running (its only entry,
			// qw38flash, is disabled and unsupported by omlx 0.6.4). Keep it red
			// so it reads like the other backend groups (🔴 = stopped). If a
			// working omlx model is ever added, wire this into refreshLoop.
			mOmlx := systray.AddMenuItem("🔴 omlx", "Models served by the isolated oMLX server (backend \"omlx\") — currently no loadable omlx model")
			addModelItems(mOmlx, cfg, omlxNames, modelItems)
		}

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
			mFlux = systray.AddMenuItem("IMAGES", "Image-generation models (FLUX, Qwen-Image, etc.) via mflux -- own pool, own port, independent of the chat backend and of OCR")
			mStartFlux = addStartItem(mFlux, "Start IMAGES", fluxDefault, cfg.FluxBackendPort)
			mStopFlux = mFlux.AddSubMenuItem("Stop IMAGES", fmt.Sprintf("Stop the backend on port %d", cfg.FluxBackendPort))
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
	mClearCaches := systray.AddMenuItem("Clear Disk Caches", "Delete rapid-mlx orphaned KV checkpoints (~/.cache/rapid-mlx/kv_checkpoints) left over from before disk checkpointing was disabled")
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
		mSmall:        mSmall,
		mStartSmall:   mStartSmall,
		mStopSmall:    mStopSmall,
		smallDefault:  smallDefault,
	})

	// Avvia SEMPRE il pool "small models" (bonsai) al lancio: è il modello
	// leggero che gateway/opencode usano per titoli/warmup e deve restare
	// sempre attivo. Gira sul pool dedicato SmallBackendPort (porta separata,
	// concorrente e indipendente dal backend chat). NON avvia automaticamente
	// né il backend chat rapid-mlx né ds4 né IMAGES (FLUX) né Ollama: questi
	// restano spenti finché non vengono avviati esplicitamente dal menu.
	// Ollama, se è già in esecuzione a livello di sistema, NON viene toccato
	// (resta attivo): qui non lo si avvia e non lo si ferma mai.
	if cfg != nil && smallDefault != "" {
		go func() {
			time.Sleep(3 * time.Second)
			if _, active := runningMLXModel(cfg.SmallBackendPort); !active {
				loadModelAsync(smallDefault)
			}
		}()
	}

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
			case <-clickedOrNil(mStopSmall):
				killPort(cfg.SmallBackendPort)
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
				killPort(cfg.SmallBackendPort)
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
			case <-mClearCaches.ClickedCh:
				bytesFreed, files, err := clearDiskKVCheckpoints()
				if err != nil {
					notify("Unified Gateway", fmt.Sprintf("Failed to clear disk caches: %v", err))
					continue
				}
				notify("Unified Gateway", fmt.Sprintf("Cleared %d orphaned KV checkpoint files (%.1f MB)", files, float64(bytesFreed)/(1024*1024)))
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

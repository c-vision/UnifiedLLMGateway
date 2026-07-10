// menubar is a macOS menu bar controller for the unified-gateway stack:
// start/stop the gateway adapters, the currently-loaded model backend
// (rapid-mlx/ds4), and Ollama, plus switch the active model — all
// independently of the headless unified-gateway LaunchAgent, which keeps
// running whether or not this app is open.
package main

import (
	"fmt"
	"sort"

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
	sort.Strings(names)
	for _, n := range names {
		m := cfg.Models[n]
		label := fmt.Sprintf("%s (%s)", m.Label, n)
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

// mediaItemRefs bundles what refreshLoop needs to keep one individually-
// controlled media model's menu entry in sync -- port is fixed at
// creation time (each media model gets its own dedicated port, unlike
// the shared chat/backend pool), so refreshLoop only ever needs to poll
// that one port to know this specific entry's live state.
type mediaItemRefs struct {
	item  *systray.MenuItem
	start *systray.MenuItem
	stop  *systray.MenuItem
	port  int
}

// addIndividualMediaItems adds one top-level menu entry PER model in
// names, each with its own Start/Stop pair -- unlike addModelItems (which
// assumes every model in the list shares one pooled port, so only needs a
// single click-to-load per item) or addStartItem (one Start/Stop pair for
// an entire section). Every "kind":"media" entry gets its own dedicated
// port (see ModelConfig.Port in the root models.go), so OCR and each FLUX
// model must be startable, stoppable, and status-reflected (🟢/🔴 in its
// own title, same convention as rapid-mlx/ds4/Ollama above) completely
// independently of one another and of whatever chat model is active --
// this is what makes that possible in the menu itself, not just in the
// gateway's own routing.
func addIndividualMediaItems(cfg *gwConfig, names []string) map[string]mediaItemRefs {
	sort.Strings(names)
	refs := make(map[string]mediaItemRefs, len(names))
	for _, n := range names {
		m := cfg.Models[n]
		item := systray.AddMenuItem(fmt.Sprintf("%s (%s)", m.Label, n), "")
		start := item.AddSubMenuItem("Start", fmt.Sprintf("Load %s on its own port %d", n, m.Port))
		stop := item.AddSubMenuItem("Stop", fmt.Sprintf("Stop the backend on port %d", m.Port))

		go func(shortName string, s *systray.MenuItem) {
			for range s.ClickedCh {
				loadModelAsync(shortName)
			}
		}(n, start)
		go func(port int, s *systray.MenuItem) {
			for range s.ClickedCh {
				killPort(port)
			}
		}(m.Port, stop)

		refs[n] = mediaItemRefs{item: item, start: start, stop: stop, port: m.Port}
	}
	return refs
}

func onReady() {
	icon := buildIcon()
	systray.SetTemplateIcon(icon, icon)
	systray.SetTooltip("Unified Gateway")

	mStatus := systray.AddMenuItem("Checking status...", "")
	mStatus.Disable()
	systray.AddSeparator()

	mStartGateway := systray.AddMenuItem("Start Gateway Adapters", "Start OpenAI (port 8082) and Anthropic (port 8083) adapters")
	mStopGateway := systray.AddMenuItem("Stop Gateway Adapters", "Stop OpenAI/Anthropic adapters")
	systray.AddSeparator()

	cfg, cfgErr := loadGWConfig()
	modelItems := map[string]*systray.MenuItem{}
	var mMLX, mDS4, mStartMLX, mStartDS4, mStopMLX, mStopDS4 *systray.MenuItem
	var mlxDefault, ds4Default string
	var mediaRefs map[string]mediaItemRefs

	if cfgErr != nil {
		mMissing := systray.AddMenuItem("Backends unavailable (models.json not found)", "")
		mMissing.Disable()
	} else {
		// Media-kind entries are split in two, not lumped into one section:
		// OCR-like ones (mlx/ds4-backed, the gateway spawns/kills them
		// itself, same as rapid-mlx/ds4 above) vs FLUX-family ones (backend
		// "mflux", a completely different runtime -- see ModelConfig.Kind's
		// doc comment in the root models.go). Every entry in EITHER group
		// gets its own dedicated port and its own individual Start/Stop --
		// see addIndividualMediaItems -- so starting/stopping one never
		// touches any other media model or the active chat model.
		var mlxNames, ds4Names, mediaOtherNames, mediaFluxNames []string
		for n, m := range cfg.Models {
			if m.Kind == "media" {
				if m.Backend == "mflux" {
					mediaFluxNames = append(mediaFluxNames, n)
				} else {
					mediaOtherNames = append(mediaOtherNames, n)
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

		if len(mlxNames) > 0 {
			mlxDefault = mlxNames[0]
		}
		if len(ds4Names) > 0 {
			ds4Default = ds4Names[0]
		}

		mMLX = systray.AddMenuItem("rapid-mlx", "")
		mStartMLX = addStartItem(mMLX, "Start rapid-mlx", mlxDefault, cfg.BackendPort)
		mStopMLX = mMLX.AddSubMenuItem("Stop rapid-mlx", fmt.Sprintf("Stop the backend on port %d", cfg.BackendPort))
		addModelItems(mMLX, cfg, mlxNames, modelItems)

		mDS4 = systray.AddMenuItem("ds4", "")
		mStartDS4 = addStartItem(mDS4, "Start ds4", ds4Default, cfg.BackendPort)
		mStopDS4 = mDS4.AddSubMenuItem("Stop ds4", fmt.Sprintf("Stop the backend on port %d", cfg.BackendPort))
		addModelItems(mDS4, cfg, ds4Names, modelItems)

		mediaRefs = map[string]mediaItemRefs{}
		if len(mediaOtherNames) > 0 || len(mediaFluxNames) > 0 {
			systray.AddSeparator()
		}
		for n, r := range addIndividualMediaItems(cfg, mediaOtherNames) {
			mediaRefs[n] = r
		}
		for n, r := range addIndividualMediaItems(cfg, mediaFluxNames) {
			mediaRefs[n] = r
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
		mediaRefs:     mediaRefs,
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

package main

// direct.go — percorso AGGIUNTIVO "direttamente a rapid-mlx".
//
// Non modifica, non migliora e non cancella il path storico: quando
// directMode() è true (DEFAULT) le richieste generation dell'adapter OpenAI
// vengono inoltrate al backend con il body del client pressoché INTATTO —
// nessuna riscrittura dei messaggi (compressMessages), nessuna iniezione di
// reasoning_effort, nessun session-tracking sul body, nessuna serializzazione
// globale inferenceMu. L'obiettivo è che i messaggi restino byte-identici
// turno-dopo-turno, così la prefix-cache di rapid-mlx riusa davvero (LCP
// stabile) — è esattamente la ragione per cui "diretto a rapidMLX" è veloce
// mentre "attraverso ug" risultava sempre lento.
//
// Resta solo il minimo indispensabile di routing: determinare il backend/pool
// (chat vs small vs media) e attivare il modello se non è quello già attivo,
// perché un processo rapid-mlx serve un solo modello per porta. Tutto ciò che
// era una "presunta miglioria" resta raggiungibile: basta settare
// UG_DIRECT_RAPIDMLX=0 per tornare al comportamento storico (compressione ecc.).

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/gin-gonic/gin"
)

// directMode is ON by default; set UG_DIRECT_RAPIDMLX=0|false|off|no to run
// the historical transformed path instead.
func directMode() bool {
	switch strings.ToLower(os.Getenv("UG_DIRECT_RAPIDMLX")) {
	case "0", "false", "off", "no":
		return false
	default:
		return true
	}
}

// handleDirectOpenAI is the new direct-to-backend path. bodyBytes is the
// client's original HTTP payload, forwarded essentially unmodified.
func (g *Gateway) handleDirectOpenAI(c *gin.Context, bodyBytes []byte) {
	isStream := false
	originalModel := ""
	if len(bodyBytes) > 0 {
		var payload map[string]interface{}
		if json.Unmarshal(bodyBytes, &payload) == nil {
			isStream = payload["stream"] == true
			if m, ok := payload["model"].(string); ok {
				originalModel = m
			}
		}
	}

	if modelDisabled(originalModel) {
		c.JSON(400, gin.H{"error": fmt.Sprintf("model %q is disabled and cannot be served", originalModel)})
		return
	}

	backend := g.resolveBackend()

	// Pool routing identico al path storico: media/small su porta dedicata,
	// perché loading/switching di uno non deve mai toccare il backend chat.
	if cfg, err := loadConfig(); err == nil && originalModel != "" {
		if mc, ok := cfg.Models[originalModel]; ok && (mc.Kind == "media" || mc.Kind == "small") {
			active := false
			if mc.Kind == "media" {
				active = isMediaModelActive(cfg, originalModel)
			} else {
				active = isSmallModelActive(cfg, originalModel)
			}
			if !active {
				if ensureBackendLoading(originalModel) {
					c.JSON(503, gin.H{"error": fmt.Sprintf("model %q is not active yet -- load triggered, retry shortly", originalModel)})
				} else {
					c.JSON(503, gin.H{"error": fmt.Sprintf("model %q crashed repeatedly right after loading -- not retrying automatically, check gateway logs and restart manually", originalModel)})
				}
				return
			}
			backend = activeBackend{Port: modelPoolPort(cfg, mc), Model: originalModel}
		}
	}

	// Self-heal minimo: se il client chiede un modello diverso da quello
	// attivo sul backend chat, attiva il cambio e chiede retry — non ha senso
	// inoltrare a un backend che serve un altro modello.
	if backend.Port != 0 && originalModel != "" && originalModel != backend.Model && originalModel != backend.UpstreamModel {
		if ensureBackendLoading(originalModel) {
			c.JSON(503, gin.H{"error": fmt.Sprintf("model %q is not active (currently serving %q) -- switch triggered, retry shortly", originalModel, backend.Model)})
		} else {
			c.JSON(503, gin.H{"error": fmt.Sprintf("model %q crashed repeatedly right after loading (rapid-mlx/Metal GPU error) -- not retrying automatically, check gateway logs and restart manually", originalModel)})
		}
		return
	}

	// Model-name: rapid-mlx serve già con --served-model-name == shortname,
	// quindi per rapid-mlx il nome resta quello del client. Solo Ollama ha un
	// tag upstream diverso da riscrivere.
	out := bodyBytes
	if backend.UpstreamModel != "" && originalModel != "" {
		var payload map[string]interface{}
		if json.Unmarshal(bodyBytes, &payload) == nil {
			payload["model"] = backend.UpstreamModel
			if rewritten, err := json.Marshal(payload); err == nil {
				out = rewritten
			}
		}
	}

	targetURL := fmt.Sprintf("http://localhost:%d%s", backend.Port, c.Request.URL.Path)
	if c.Request.URL.RawQuery != "" {
		targetURL += "?" + c.Request.URL.RawQuery
	}
	proxyReq, err := http.NewRequest(c.Request.Method, targetURL, bytes.NewReader(out))
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to build backend request"})
		return
	}
	if ct := c.Request.Header.Get("Content-Type"); ct != "" {
		proxyReq.Header.Set("Content-Type", ct)
	}
	if cfg, cfgErr := loadConfig(); cfgErr == nil && cfg.Models[originalModel].Backend == "omlx" {
		proxyReq.Header.Set("Authorization", "Bearer "+omlxGatewayAPIKey)
	}

	resp, err := http.DefaultClient.Do(proxyReq)
	if err != nil {
		if ensureBackendLoading(originalModel) {
			c.JSON(500, gin.H{"error": fmt.Sprintf("Local LLM Backend unreachable on %d", backend.Port)})
		} else {
			c.JSON(500, gin.H{"error": fmt.Sprintf("model %q crashed repeatedly right after loading (rapid-mlx/Metal GPU error) -- not retrying automatically, check gateway logs and restart manually", originalModel)})
		}
		return
	}
	defer resp.Body.Close()

	if isStream {
		// rapid-mlx parla già OpenAI: passthrough SSE senza traduzione.
		c.Header("Content-Type", "text/event-stream")
		c.Header("Cache-Control", "no-cache")
		io.Copy(c.Writer, resp.Body)
		return
	}
	var result map[string]interface{}
	if json.NewDecoder(resp.Body).Decode(&result) != nil {
		c.Status(resp.StatusCode)
		return
	}
	if originalModel != "" {
		if _, has := result["model"]; has {
			result["model"] = originalModel
		}
	}
	c.JSON(resp.StatusCode, result)
}

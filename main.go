package main

import (
	"bufio"
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
)

func randomID(prefix string) string {
	b := make([]byte, 8)
	rand.Read(b)
	return prefix + "_" + hex.EncodeToString(b)
}

// ============================================================================
// Unified Gateway State
// ============================================================================

type Gateway struct {
	// backendPort is only the startup fallback, used if models.json can't
	// be read at all when resolveBackend runs — otherwise every call
	// re-derives the real backend live (see resolveActiveBackend), never
	// from a state file. A dead backend process used to still get
	// reported as "active" indefinitely by a stale active-backend.json;
	// there is no longer anything cached to go stale.
	backendPort int
}

// inferenceMu serializes chat-completion forwarding to the backend --
// observed directly (2026-07-10): rapid-mlx's own metrics showed
// requests_running=3, all for the SAME growing OpenCode conversation
// (message counts climbing 165→167→169... across overlapping [REQUEST]
// log lines), meaning OpenCode fires the next turn before the model has
// finished the previous one. rapid-mlx has no admission control of its
// own -- it just batches whatever arrives -- so those overlapping
// generations compete for the same single GPU instead of one finishing
// before the next starts, which only makes an already-large prompt slower
// for everyone. A single local backend serving one user has nothing to
// gain from concurrent generations the way a real multi-tenant server
// does; queuing them one at a time is strictly better here. This does not
// touch /v1/models, /v1/models/*/load, or /v1/compression -- those stay
// fast and responsive even while a completion is in flight.
var inferenceMu sync.Mutex

func NewGateway() *Gateway {
	port := 11435
	if cfg, err := loadConfig(); err == nil && cfg.BackendPort != 0 {
		port = cfg.BackendPort
	} else {
		log.Printf("⚠️  could not read models.json (%v), defaulting backend port to %d", err, port)
	}
	return &Gateway{backendPort: port}
}

// resolveBackend determines which backend to route to right now, live —
// see resolveActiveBackend's doc comment for exactly how.
func (g *Gateway) resolveBackend() activeBackend {
	cfg, err := loadConfig()
	if err != nil {
		return activeBackend{Port: g.backendPort}
	}
	return resolveActiveBackend(cfg, g.backendPort)
}

// Middleware to log requests in detail
func requestLogger() gin.HandlerFunc {
	return func(c *gin.Context) {
		startTime := time.Now()
		path := c.Request.URL.Path
		method := c.Request.Method

		// Read body for logging
		var bodyBytes []byte
		if c.Request.Body != nil {
			bodyBytes, _ = io.ReadAll(c.Request.Body)
		}
		// Restore body for subsequent handlers
		c.Request.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))

		log.Printf("[%s] 📩 Incoming %s %s", time.Now().Format("15:04:05"), method, path)
		if len(bodyBytes) > 0 {
			content := string(bodyBytes)
			if len(content) > 200 {
				content = content[:200] + "... [truncated]"
			}
			log.Printf("📦 Payload: %s", content)
		}

		c.Next()

		latency := time.Since(startTime)
		log.Printf("[%s] ✅ Response %d | Latency: %v", time.Now().Format("15:04:05"), c.Writer.Status(), latency)
	}
}

// ============================================================================
// Anthropic Adapter (Claude Code Support)
// ============================================================================

type AnthropicRequest struct {
	Model       string                   `json:"model"`
	System      interface{}              `json:"system,omitempty"`
	Messages    []Message                `json:"messages"`
	Tools       []map[string]interface{} `json:"tools,omitempty"`
	MaxTokens   int                      `json:"max_tokens"`
	Temperature float64                  `json:"temperature"`
	Stream      bool                     `json:"stream"`
}

type Message struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

func (g *Gateway) handleAnthropicMessages(c *gin.Context) {
	var req AnthropicRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": "Invalid request format"})
		return
	}

	if compactType := detectCompactType(req.System, req.Messages); compactType != compactNone {
		log.Printf("🗜️  %s", compactTypeLabel(compactType))
	}

	backend := g.resolveBackend()

	// The active backend serves whatever it was loaded with, regardless of
	// what model name a request asks for -- rapid-mlx itself rejects a
	// mismatched name with its own 404 rather than silently switching. Left
	// unhandled, a stale/different model being active (left over from
	// another client, a manual test, anything) gets forwarded to anyway,
	// and on this endpoint the response-translation below turned that 404
	// into a bare {"error":"no choices found"} with a 200 status -- from
	// Claude Code's side that reads as no response at all, not an error.
	// Treat "reachable but serving something else" the same way
	// "unreachable" is already treated below: trigger a load and tell the
	// client to retry, instead of forwarding to a backend that can't serve
	// this request.
	if backend.Port != 0 && req.Model != "" && req.Model != backend.Model && req.Model != backend.UpstreamModel {
		if ensureBackendLoading(req.Model) {
			c.JSON(503, gin.H{"error": fmt.Sprintf("model %q is not active (currently serving %q) -- switch triggered, retry shortly", req.Model, backend.Model)})
		} else {
			c.JSON(503, gin.H{"error": fmt.Sprintf("model %q crashed repeatedly right after loading (rapid-mlx/Metal GPU error) -- not retrying automatically, check gateway logs and restart manually", req.Model)})
		}
		return
	}

	inferenceMu.Lock()
	defer inferenceMu.Unlock()

	upstreamModel := req.Model
	if backend.UpstreamModel != "" {
		upstreamModel = backend.UpstreamModel
	}

	openaiPayload := map[string]interface{}{
		"model":       upstreamModel,
		"messages":    translateMessagesToOpenAI(req.System, req.Messages),
		"max_tokens":  req.MaxTokens,
		"temperature": req.Temperature,
		"stream":      req.Stream,
	}
	if tools := translateToolsToOpenAI(req.Tools); tools != nil {
		openaiPayload["tools"] = tools
	}

	body, _ := json.Marshal(openaiPayload)
	backendURL := fmt.Sprintf("http://localhost:%d", backend.Port)
	resp, err := http.Post(backendURL+"/v1/chat/completions", "application/json", bytes.NewBuffer(body))
	if err != nil {
		if ensureBackendLoading(req.Model) {
			c.JSON(500, gin.H{"error": fmt.Sprintf("Local LLM Backend unreachable on %d", backend.Port)})
		} else {
			c.JSON(500, gin.H{"error": fmt.Sprintf("model %q crashed repeatedly right after loading (rapid-mlx/Metal GPU error) -- not retrying automatically, check gateway logs and restart manually", req.Model)})
		}
		return
	}
	defer resp.Body.Close()

	if req.Stream {
		streamOpenAIToAnthropic(c, resp.Body, req.Model)
	} else {
		var openaiResp map[string]interface{}
		json.NewDecoder(resp.Body).Decode(&openaiResp)
		// Propagate the backend's real status instead of always answering
		// 200 -- an error body (wrong model, bad request, backend fault)
		// has no "choices" either, and used to get silently repackaged as
		// {"error":"no choices found"} with a 200, indistinguishable from
		// no response at all on the client side.
		if resp.StatusCode != 200 {
			c.JSON(resp.StatusCode, gin.H{"error": openaiResp})
			return
		}
		anthroResp := translateOpenAIResponseToAnthropic(openaiResp, req.Model)
		c.JSON(200, anthroResp)
	}
}

// streamOpenAIToAnthropic reads an OpenAI-style SSE stream from the backend
// and re-emits it as Anthropic Messages API SSE events, since Claude Code
// expects message_start/content_block_delta/message_stop, not raw OpenAI
// chat.completion.chunk events.
func streamOpenAIToAnthropic(c *gin.Context, body io.Reader, model string) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")

	flusher, _ := c.Writer.(http.Flusher)
	sseSend := func(event string, data interface{}) {
		payload, _ := json.Marshal(data)
		fmt.Fprintf(c.Writer, "event: %s\ndata: %s\n\n", event, payload)
		if flusher != nil {
			flusher.Flush()
		}
	}

	msgID := randomID("msg")
	sseSend("message_start", gin.H{
		"type": "message_start",
		"message": gin.H{
			"id": msgID, "type": "message", "role": "assistant",
			"content": []interface{}{}, "model": model,
			"stop_reason": nil, "stop_sequence": nil,
			"usage": gin.H{"input_tokens": 0, "output_tokens": 0},
		},
	})

	nextIndex := 0
	textIndex := -1
	textOpen := false
	toolBlockIndex := map[float64]int{}
	var toolOpenOrder []int
	outputTokens := 0
	finishReason := "stop"

	ensureTextOpen := func() {
		if !textOpen {
			textIndex = nextIndex
			nextIndex++
			sseSend("content_block_start", gin.H{
				"type": "content_block_start", "index": textIndex,
				"content_block": gin.H{"type": "text", "text": ""},
			})
			textOpen = true
		}
	}
	closeTextIfOpen := func() {
		if textOpen {
			sseSend("content_block_stop", gin.H{"type": "content_block_stop", "index": textIndex})
			textOpen = false
		}
	}

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
			continue
		}
		var chunk map[string]interface{}
		if err := json.Unmarshal([]byte(line[6:]), &chunk); err != nil {
			continue
		}
		choices, _ := chunk["choices"].([]interface{})
		if len(choices) == 0 {
			continue
		}
		choice, _ := choices[0].(map[string]interface{})
		if fr, ok := choice["finish_reason"].(string); ok && fr != "" {
			finishReason = fr
		}
		delta, _ := choice["delta"].(map[string]interface{})
		if delta == nil {
			continue
		}
		if text, ok := delta["content"].(string); ok && text != "" {
			ensureTextOpen()
			outputTokens++
			sseSend("content_block_delta", gin.H{
				"type": "content_block_delta", "index": textIndex,
				"delta": gin.H{"type": "text_delta", "text": text},
			})
		}
		if rawTCs, ok := delta["tool_calls"].([]interface{}); ok {
			closeTextIfOpen()
			for _, rawTC := range rawTCs {
				tc, ok := rawTC.(map[string]interface{})
				if !ok {
					continue
				}
				oaiIdx, _ := tc["index"].(float64)
				fn, _ := tc["function"].(map[string]interface{})
				blockIdx, seen := toolBlockIndex[oaiIdx]
				if !seen {
					blockIdx = nextIndex
					nextIndex++
					toolBlockIndex[oaiIdx] = blockIdx
					toolOpenOrder = append(toolOpenOrder, blockIdx)
					name := ""
					if fn != nil {
						if n, ok := fn["name"].(string); ok {
							name = n
						}
					}
					id, _ := tc["id"].(string)
					sseSend("content_block_start", gin.H{
						"type": "content_block_start", "index": blockIdx,
						"content_block": gin.H{"type": "tool_use", "id": id, "name": name, "input": gin.H{}},
					})
				}
				if fn != nil {
					if args, ok := fn["arguments"].(string); ok && args != "" {
						outputTokens++
						sseSend("content_block_delta", gin.H{
							"type": "content_block_delta", "index": blockIdx,
							"delta": gin.H{"type": "input_json_delta", "partial_json": args},
						})
					}
				}
			}
		}
	}

	closeTextIfOpen()
	for _, idx := range toolOpenOrder {
		sseSend("content_block_stop", gin.H{"type": "content_block_stop", "index": idx})
	}

	stopReason := "end_turn"
	switch finishReason {
	case "tool_calls":
		stopReason = "tool_use"
	case "length":
		stopReason = "max_tokens"
	}

	sseSend("message_delta", gin.H{
		"type":  "message_delta",
		"delta": gin.H{"stop_reason": stopReason, "stop_sequence": nil},
		"usage": gin.H{"output_tokens": outputTokens},
	})
	sseSend("message_stop", gin.H{"type": "message_stop"})
}

func (g *Gateway) handleCountTokens(c *gin.Context) {
	var req map[string]interface{}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": "Invalid request"})
		return
	}

	c.JSON(200, gin.H{
		"usage": gin.H{
			"input_tokens": 100,
		},
	})
}

// extractBlockText concatenates the text of any {"type":"text","text":"..."}
// blocks found in an Anthropic content value (string or block array).
func extractBlockText(content interface{}) string {
	switch v := content.(type) {
	case string:
		return v
	case []interface{}:
		var sb strings.Builder
		for _, item := range v {
			if b, ok := item.(map[string]interface{}); ok && b["type"] == "text" {
				sb.WriteString(fmt.Sprintf("%v", b["text"]))
			}
		}
		return sb.String()
	}
	return ""
}

// translateMessagesToOpenAI converts Anthropic system + messages (including
// tool_use / tool_result content blocks) into an OpenAI-style message list.
// translateMessagesToOpenAI converts Anthropic system + messages into an
// OpenAI-style message list. Any role:"system" message embedded inside the
// messages array (Claude Code sends these interleaved, e.g. the available
// agents/skills list) is pulled out and merged into a single leading system
// message — most chat templates expect exactly one system turn at position
// 0, and leaving it in its original (later) position can make local models
// silently produce empty completions.
func translateMessagesToOpenAI(system interface{}, anthroMsgs []Message) []map[string]interface{} {
	var out []map[string]interface{}

	var systemParts []string
	if sysText := extractBlockText(system); sysText != "" {
		systemParts = append(systemParts, sysText)
	}
	var restMsgs []Message
	for _, m := range anthroMsgs {
		if m.Role == "system" {
			if text := extractBlockText(m.Content); text != "" {
				systemParts = append(systemParts, text)
			}
			continue
		}
		restMsgs = append(restMsgs, m)
	}
	if len(systemParts) > 0 {
		out = append(out, map[string]interface{}{"role": "system", "content": strings.Join(systemParts, "\n\n")})
	}

	for _, m := range restMsgs {
		switch content := m.Content.(type) {
		case string:
			if content != "" || m.Role == "assistant" {
				out = append(out, map[string]interface{}{"role": m.Role, "content": content})
			}
		case []interface{}:
			var textParts []string
			var toolCalls []map[string]interface{}
			for _, item := range content {
				block, ok := item.(map[string]interface{})
				if !ok {
					continue
				}
				switch block["type"] {
				case "text":
					textParts = append(textParts, fmt.Sprintf("%v", block["text"]))
				case "tool_use":
					inputJSON, _ := json.Marshal(block["input"])
					toolCalls = append(toolCalls, map[string]interface{}{
						"id":   block["id"],
						"type": "function",
						"function": map[string]interface{}{
							"name":      block["name"],
							"arguments": string(inputJSON),
						},
					})
				case "tool_result":
					out = append(out, map[string]interface{}{
						"role":         "tool",
						"tool_call_id": block["tool_use_id"],
						"content":      extractBlockText(block["content"]),
					})
				}
			}
			text := strings.TrimSpace(strings.Join(textParts, ""))
			if len(toolCalls) > 0 {
				out = append(out, map[string]interface{}{
					"role": m.Role, "content": text, "tool_calls": toolCalls,
				})
			} else if text != "" || m.Role == "assistant" {
				out = append(out, map[string]interface{}{"role": m.Role, "content": text})
			}
		}
	}
	return out
}

// translateToolsToOpenAI converts Anthropic tool definitions
// ({"name","description","input_schema"}) to OpenAI's function-tool shape.
func translateToolsToOpenAI(tools []map[string]interface{}) []map[string]interface{} {
	if len(tools) == 0 {
		return nil
	}
	var out []map[string]interface{}
	for _, t := range tools {
		schema := t["input_schema"]
		if schema == nil {
			schema = map[string]interface{}{}
		}
		out = append(out, map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name":        t["name"],
				"description": t["description"],
				"parameters":  schema,
			},
		})
	}
	return out
}

// translateOpenAIResponseToAnthropic builds the Anthropic response. It uses
// originalModel (what the client asked for) rather than openaiResp["model"]
// so that a backend-side rename (e.g. Ollama's own "gemma4:31b-mlx" tag) is
// never leaked back to the client.
func translateOpenAIResponseToAnthropic(openaiResp map[string]interface{}, originalModel string) map[string]interface{} {
	choices, ok := openaiResp["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return map[string]interface{}{"error": "no choices found"}
	}
	firstChoice := choices[0].(map[string]interface{})
	msg, _ := firstChoice["message"].(map[string]interface{})
	usage, _ := openaiResp["usage"].(map[string]interface{})

	var content []map[string]interface{}
	if text, ok := msg["content"].(string); ok && text != "" {
		content = append(content, map[string]interface{}{"type": "text", "text": text})
	}
	if toolCalls, ok := msg["tool_calls"].([]interface{}); ok {
		for _, rawTC := range toolCalls {
			tc, ok := rawTC.(map[string]interface{})
			if !ok {
				continue
			}
			fn, _ := tc["function"].(map[string]interface{})
			var input map[string]interface{}
			if args, ok := fn["arguments"].(string); ok {
				json.Unmarshal([]byte(args), &input)
			}
			content = append(content, map[string]interface{}{
				"type": "tool_use", "id": tc["id"], "name": fn["name"], "input": input,
			})
		}
	}

	stopReason := "end_turn"
	switch fmt.Sprintf("%v", firstChoice["finish_reason"]) {
	case "tool_calls":
		stopReason = "tool_use"
	case "length":
		stopReason = "max_tokens"
	}

	return map[string]interface{}{
		"id":            openaiResp["id"],
		"type":          "message",
		"role":          "assistant",
		"model":         originalModel,
		"content":       content,
		"stop_reason":   stopReason,
		"stop_sequence": nil,
		"usage": map[string]interface{}{
			"input_tokens":  usage["prompt_tokens"],
			"output_tokens": usage["completion_tokens"],
		},
	}
}

// ============================================================================
// OpenAI Adapter (General Purpose)
// ============================================================================

// handleListModels returns the full models.json catalog in the standard
// OpenAI /v1/models shape, unlike a plain passthrough to the active
// backend (which only ever knows about whichever single model it has
// loaded). This is what lets a WebUI populate a model picker without
// reading models.json off disk itself — the same reason the menu bar
// needs the file directly today, a remote/separate WebUI process
// couldn't. "active": true marks whichever one resolveBackend's live
// detection (real process on the backend port, or Ollama's own API)
// actually found running right now — never a cached/stale value.
func (g *Gateway) handleListModels(c *gin.Context) {
	cfg, err := loadConfig()
	if err != nil {
		c.JSON(500, gin.H{"error": fmt.Sprintf("cannot read models.json: %v", err)})
		return
	}
	active := g.resolveBackend()

	data := make([]gin.H, 0, len(cfg.Models))
	for name, m := range cfg.Models {
		// Media-kind entries (OCR, etc.) are deliberately excluded here --
		// see the ModelConfig.Kind doc comment in models.go. They're not a
		// chat choice; opencode/pi/Claude Code all read this endpoint (or
		// the same models.json) to populate a model picker, and a "read
		// this image" model mixed into that list is just noise/confusion.
		// Use GET /v1/media-models for those instead.
		if m.Kind == "media" {
			continue
		}
		data = append(data, gin.H{
			"id":         name,
			"object":     "model",
			"created":    0,
			"owned_by":   "unified-gateway",
			"label":      m.Label,
			"backend":    m.Backend,
			"has_vision": m.HasVision,
			"active":     name == active.Model,
		})
	}
	c.JSON(200, gin.H{"object": "list", "data": data})
}

// handleListMediaModels is /v1/media-models' sibling to handleListModels --
// same shape, but only the entries handleListModels filters OUT (Kind ==
// "media"). Its own endpoint rather than a query param on /v1/models so a
// client has to deliberately ask for this list; nothing currently forwards
// chat completions to a media model, this is discovery-only for now (e.g.
// the menu bar's separate media section reads it the same way it reads
// /v1/models for chat entries).
func (g *Gateway) handleListMediaModels(c *gin.Context) {
	cfg, err := loadConfig()
	if err != nil {
		c.JSON(500, gin.H{"error": fmt.Sprintf("cannot read models.json: %v", err)})
		return
	}
	active := g.resolveBackend()

	data := make([]gin.H, 0)
	for name, m := range cfg.Models {
		if m.Kind != "media" {
			continue
		}
		data = append(data, gin.H{
			"id":         name,
			"object":     "model",
			"created":    0,
			"owned_by":   "unified-gateway",
			"label":      m.Label,
			"backend":    m.Backend,
			"has_vision": m.HasVision,
			"active":     name == active.Model,
		})
	}
	c.JSON(200, gin.H{"object": "list", "data": data})
}

// handleLoadModel triggers loading the named model in the background and
// returns immediately (202) — mirrors how Ollama/LM Studio handle model
// loads that can take minutes, rather than holding the HTTP connection
// open for however long that takes. Reuses ensureBackendLoading, the
// same dedup-guarded loader the auto-load-on-request path already uses,
// so an explicit load click and an implicit "backend unreachable" retry
// never race each other into a double load. shortName comes from the
// path (see handleOpenAIProxy's dispatch — Gin can't mix a static
// /v1/models/:id/load route with the top-level /*path catch-all, so
// this is parsed out by hand rather than bound via c.Param).
func (g *Gateway) handleLoadModel(c *gin.Context, shortName string) {
	cfg, err := loadConfig()
	if err != nil {
		c.JSON(500, gin.H{"error": fmt.Sprintf("cannot read models.json: %v", err)})
		return
	}
	if _, ok := cfg.Models[shortName]; !ok {
		c.JSON(404, gin.H{"error": fmt.Sprintf("unknown model %q", shortName)})
		return
	}
	ensureBackendLoading(shortName)
	c.JSON(202, gin.H{"status": "loading", "model": shortName})
}

// handleOpenAIProxy forwards any method/path to the backend as-is — GET
// /v1/models (no body) is just as valid as a POST /v1/chat/completions, and
// forcing JSON binding on every request broke plain GETs that OpenCode and
// other OpenAI-compatible clients use to list models / check connectivity.
//
// /v1/models (catalog) and /v1/models/:id/load are handled here, ahead of
// the passthrough, rather than as separate Gin routes: Gin's router
// rejects a static route coexisting with a top-level /*path catch-all at
// the same level ("catch-all wildcard conflicts with existing path
// segment"), so the dispatch has to happen inside the one handler that's
// actually registered.
func (g *Gateway) handleOpenAIProxy(c *gin.Context) {
	path := c.Request.URL.Path
	if c.Request.Method == "GET" && path == "/v1/models" {
		g.handleListModels(c)
		return
	}
	if c.Request.Method == "GET" && path == "/v1/media-models" {
		g.handleListMediaModels(c)
		return
	}
	if c.Request.Method == "POST" && strings.HasPrefix(path, "/v1/models/") && strings.HasSuffix(path, "/load") {
		shortName := strings.TrimSuffix(strings.TrimPrefix(path, "/v1/models/"), "/load")
		g.handleLoadModel(c, shortName)
		return
	}
	if path == "/v1/compression" {
		if c.Request.Method == "POST" {
			var body struct {
				Enabled bool `json:"enabled"`
			}
			if err := c.ShouldBindJSON(&body); err != nil {
				c.JSON(400, gin.H{"error": "expected JSON body {\"enabled\": bool}"})
				return
			}
			setPromptCompressionEnabled(body.Enabled)
		}
		c.JSON(200, gin.H{
			"enabled":             promptCompressionEnabled(),
			"requests_compressed": compressionStats.requestsCompressed.Load(),
			"chars_saved":         compressionStats.charsSaved.Load(),
		})
		return
	}

	var bodyBytes []byte
	if c.Request.Body != nil {
		bodyBytes, _ = io.ReadAll(c.Request.Body)
	}

	backend := g.resolveBackend()

	// If the currently active model is served via Ollama under its own tag
	// (e.g. "gemma4:31b-mlx"), rewrite whatever model name the client sent
	// to that tag before forwarding — mirrors what --served-model-name does
	// for rapid-mlx, since Ollama has no equivalent per-request alias.
	isStream := false
	originalModel := ""
	if len(bodyBytes) > 0 {
		var payload map[string]interface{}
		if json.Unmarshal(bodyBytes, &payload) == nil {
			isStream = payload["stream"] == true
			if m, ok := payload["model"].(string); ok {
				originalModel = m
			}
			modified := false
			if backend.UpstreamModel != "" && originalModel != "" {
				payload["model"] = backend.UpstreamModel
				modified = true
			}
			if msgs, ok := payload["messages"].([]interface{}); ok {
				if compressed, saved := compressMessages(msgs); saved > 0 {
					payload["messages"] = compressed
					modified = true
					compressionStats.requestsCompressed.Add(1)
					compressionStats.charsSaved.Add(int64(saved))
					log.Printf("🗜️  prompt compression saved ~%d chars", saved)
				}
			}
			if modified {
				if rewritten, err := json.Marshal(payload); err == nil {
					bodyBytes = rewritten
				}
			}
		}
	}

	// Same self-heal as the Anthropic adapter: the active backend answers
	// with whatever it was loaded with regardless of what model a request
	// names, so a stale/different model already running would otherwise
	// get forwarded to anyway. This endpoint already propagates the
	// backend's real status code below, so a mismatch here at least
	// wasn't silent like the Anthropic side was -- but it still never
	// self-healed, just kept returning the backend's 404 forever.
	if backend.Port != 0 && originalModel != "" && originalModel != backend.Model && originalModel != backend.UpstreamModel {
		if ensureBackendLoading(originalModel) {
			c.JSON(503, gin.H{"error": fmt.Sprintf("model %q is not active (currently serving %q) -- switch triggered, retry shortly", originalModel, backend.Model)})
		} else {
			c.JSON(503, gin.H{"error": fmt.Sprintf("model %q crashed repeatedly right after loading (rapid-mlx/Metal GPU error) -- not retrying automatically, check gateway logs and restart manually", originalModel)})
		}
		return
	}

	inferenceMu.Lock()
	defer inferenceMu.Unlock()

	backendURL := fmt.Sprintf("http://localhost:%d", backend.Port)
	targetURL := backendURL + c.Request.URL.Path
	if c.Request.URL.RawQuery != "" {
		targetURL += "?" + c.Request.URL.RawQuery
	}

	proxyReq, err := http.NewRequest(c.Request.Method, targetURL, bytes.NewReader(bodyBytes))
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to build backend request"})
		return
	}
	if ct := c.Request.Header.Get("Content-Type"); ct != "" {
		proxyReq.Header.Set("Content-Type", ct)
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
		c.Header("Content-Type", "text/event-stream")
		io.Copy(c.Writer, resp.Body)
	} else {
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
}

// ============================================================================
// Main Entry Point
// ============================================================================

func main() {
	if len(os.Args) >= 3 && os.Args[1] == "load" {
		if err := loadModel(os.Args[2]); err != nil {
			log.Fatalf("[unified-gateway] load failed: %v", err)
		}
		return
	}

	g := NewGateway()
	gin.SetMode(gin.ReleaseMode)

	anthroSrv := gin.Default()
	anthroSrv.Use(requestLogger()) // Enable logging for Anthropic API
	anthroSrv.POST("/v1/messages", g.handleAnthropicMessages)
	anthroSrv.POST("/v1/messages/count_tokens", g.handleCountTokens)

	openAiSrv := gin.Default()
	openAiSrv.Use(requestLogger()) // Enable logging for OpenAI API
	openAiSrv.Any("/*path", g.handleOpenAIProxy)

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		log.Println("🚀 Unified Gateway: Anthropic Interface active on :8083")
		if err := anthroSrv.Run(":8083"); err != nil {
			log.Fatalf("Anthropic server failed: %v", err)
		}
	}()

	go func() {
		defer wg.Done()
		log.Println("🚀 Unified Gateway: OpenAI Interface active on :8082")
		if err := openAiSrv.Run(":8082"); err != nil {
			log.Fatalf("OpenAI server failed: %v", err)
		}
	}()

	wg.Wait()
}

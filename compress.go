package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"sync/atomic"
)

// Prompt compression: opt-in (PROMPT_COMPRESSION=1) reduction of stale or
// redundant content in OLDER messages before forwarding to the backend.
// Motivated directly by a real OpenCode session (2026-07-10): a single
// growing conversation reached 41,180 prompt tokens, almost entirely from
// re-sent tool-call output accumulated across many turns -- not from the
// user's actual questions, which stay small. Every extra token there costs
// twice: slower prefill AND more KV-cache pressure (see mlxCacheReserveMBFor
// in memcheck.go), so trimming it helps both problems from the same fix.
//
// Two orthogonal techniques, both scoped to leave the system prompt, tool
// schemas, and the most recent messages completely untouched -- the model
// always sees full, uncompressed content for whatever's actually relevant
// to the current turn:
//
//  1. Duplicate collapse: the exact same tool-result content re-appearing
//     verbatim (e.g. the same file read twice across turns) keeps only the
//     LAST occurrence; earlier ones become a short placeholder.
//  2. Middle truncation: any remaining message body over a size threshold
//     keeps a head and tail slice with a marker for what was cut -- the
//     same sink+tail idea PFlash uses inside rapid-mlx, but applied here in
//     Go before the prompt ever reaches the backend, so it helps every
//     model/backend, not just PFlash-capable non-vision ones.
//
// Never touches: system messages, the protected recent-message window, or
// any message whose content isn't a plain string (multimodal/array content
// is left alone rather than risk mangling it). Tool messages are never
// removed or reordered -- only their content field is replaced -- so
// tool_call_id pairing with the preceding assistant message stays intact.
const (
	compressProtectedWindow   = 6    // last N messages are always left untouched
	compressDedupMinChars     = 200  // shorter tool content isn't worth deduping
	compressTruncateThreshold = 4000 // chars (~1000 tokens); longer bodies get middle-cut
	compressHeadKeepChars     = 800
	compressTailKeepChars     = 800
)

// compressionEnabled holds live on/off state, toggleable at runtime via
// GET/POST /v1/compression (see handleOpenAIProxy) -- no restart needed,
// unlike almost every other knob in this gateway, which all require a
// process restart to change. Defaults to ON (the whole point of the
// feature is to shrink the growing-conversation token/KV-cache cost every
// long OpenCode/Claude-Code session pays -- that only helps if it's
// actually running). Set PROMPT_COMPRESSION=0 to start with it off
// instead; the menu bar toggle (cmd/menubar) always works live regardless
// of how it started.
var compressionEnabled atomic.Bool

func init() {
	compressionEnabled.Store(defaultCompressionEnabled(os.Getenv("PROMPT_COMPRESSION")))
}

// defaultCompressionEnabled computes the startup default from
// PROMPT_COMPRESSION's raw value -- factored out so the default itself is
// directly unit-testable without depending on init()/global-state timing.
func defaultCompressionEnabled(envVal string) bool {
	return envVal != "0" && envVal != "false"
}

func promptCompressionEnabled() bool {
	return compressionEnabled.Load()
}

func setPromptCompressionEnabled(v bool) {
	compressionEnabled.Store(v)
}

// compressionStats is cumulative, process-lifetime savings -- surfaced at
// GET /v1/compression so the menu bar (or curl) can show it's actually
// doing something, not just report an on/off flag.
type compressionStatsT struct {
	requestsCompressed atomic.Int64
	charsSaved         atomic.Int64
}

var compressionStats compressionStatsT

// contentSignature hashes long content instead of using the raw string as a
// map key, so a handful of huge duplicated tool outputs don't themselves
// bloat the compressor's own memory use.
func contentSignature(content string) string {
	if len(content) < compressDedupMinChars {
		return content
	}
	sum := sha256.Sum256([]byte(content))
	return hex.EncodeToString(sum[:])
}

func cloneMessage(m map[string]interface{}) map[string]interface{} {
	clone := make(map[string]interface{}, len(m))
	for k, v := range m {
		clone[k] = v
	}
	return clone
}

// compressMessages returns a possibly-modified copy of messages plus the
// approximate number of characters removed. Returns the original slice
// unmodified (savedChars == 0) when compression is disabled, the
// conversation is too short to have anything outside the protected window,
// or nothing in the old portion qualifies.
func compressMessages(messages []interface{}) ([]interface{}, int) {
	if !promptCompressionEnabled() || len(messages) <= compressProtectedWindow {
		return messages, 0
	}
	cutoff := len(messages) - compressProtectedWindow

	// Pass 1: for each duplicated tool-result signature, record the LAST
	// index it appears at across the WHOLE conversation (including the
	// protected window) -- that occurrence is what's kept intact; an
	// earlier one within the old range becomes a placeholder in pass 2.
	// Scanning past the cutoff matters: a file re-read in the current
	// (protected) turn is exactly the common case an earlier stale copy
	// should be collapsed against.
	lastIndexOf := make(map[string]int)
	for i := 0; i < len(messages); i++ {
		msg, ok := messages[i].(map[string]interface{})
		if !ok || msg["role"] != "tool" {
			continue
		}
		content, ok := msg["content"].(string)
		if !ok || len(content) < compressDedupMinChars {
			continue
		}
		lastIndexOf[contentSignature(content)] = i
	}

	savedChars := 0
	out := make([]interface{}, len(messages))
	copy(out, messages)

	for i := 0; i < cutoff; i++ {
		msg, ok := out[i].(map[string]interface{})
		if !ok || msg["role"] == "system" {
			continue
		}
		content, ok := msg["content"].(string)
		if !ok {
			continue
		}

		if msg["role"] == "tool" && len(content) >= compressDedupMinChars {
			if lastIdx, dup := lastIndexOf[contentSignature(content)]; dup && lastIdx != i {
				clone := cloneMessage(msg)
				placeholder := fmt.Sprintf(
					"[contenuto duplicato (%d caratteri) -- la versione più recente compare più avanti nella conversazione]",
					len(content),
				)
				clone["content"] = placeholder
				savedChars += len(content) - len(placeholder)
				out[i] = clone
				continue
			}
		}

		if len(content) > compressTruncateThreshold {
			clone := cloneMessage(msg)
			omitted := len(content) - compressHeadKeepChars - compressTailKeepChars
			clone["content"] = content[:compressHeadKeepChars] +
				fmt.Sprintf("\n\n[...%d caratteri omessi...]\n\n", omitted) +
				content[len(content)-compressTailKeepChars:]
			savedChars += omitted
			out[i] = clone
		}
	}

	return out, savedChars
}

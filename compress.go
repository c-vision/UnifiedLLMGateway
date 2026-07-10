package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"sort"
	"strings"
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
//     LAST occurrence; earlier ones become a short placeholder. Also
//     catches NEAR-duplicates -- the same content re-read with only
//     numbers/timestamps/whitespace differing (a line-number changed, a
//     counter incremented) -- via a normalized shape signature, since the
//     exact-hash check alone misses these; see normalizedShape.
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

	// Sparse-block selection: a second, more aggressive pass that only
	// engages for genuinely large prompts -- see the doc comment above
	// sparsifyContent for why a size gate instead of always running it.
	sparseGateChars     = 150000 // ~37k tokens; below this, plain head+tail truncation is enough
	sparseKeepRatio     = 0.35   // beyond sink+tail, keep enough top-scoring blocks to hit ~35% of original size
	sparseMinBlocks     = 4      // fewer blocks than this isn't worth block-scoring; falls back to head+tail
	sparseKeywordMinLen = 4      // query words shorter than this are too generic to score on
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

// normalizedShape strips digits and collapses whitespace before hashing --
// catches the same content re-read with only a line number, a timestamp, or
// an incrementing counter differing (e.g. "example_function_1" vs
// "example_function_2"), which the exact-hash contentSignature above
// necessarily misses since even one differing byte changes the hash.
// Digits are replaced with a single placeholder rune rather than dropped
// entirely, so "v1" and "v12" don't collapse into the same shape as "v".
func normalizedShape(content string) string {
	var b strings.Builder
	b.Grow(len(content))
	lastWasDigit := false
	lastWasSpace := false
	for _, r := range content {
		switch {
		case r >= '0' && r <= '9':
			if !lastWasDigit {
				b.WriteRune('#')
			}
			lastWasDigit = true
			lastWasSpace = false
		case r == ' ' || r == '\t' || r == '\n' || r == '\r':
			if !lastWasSpace {
				b.WriteRune(' ')
			}
			lastWasDigit = false
			lastWasSpace = true
		default:
			b.WriteRune(r)
			lastWasDigit = false
			lastWasSpace = false
		}
	}
	sum := sha256.Sum256([]byte(strings.ToLower(b.String())))
	return hex.EncodeToString(sum[:])
}

func cloneMessage(m map[string]interface{}) map[string]interface{} {
	clone := make(map[string]interface{}, len(m))
	for k, v := range m {
		clone[k] = v
	}
	return clone
}

// totalContentChars sums every string-content message's length -- the
// size-gate check for sparsifyContent, computed once per request rather
// than tracked incrementally, since compression only runs once per
// request anyway.
func totalContentChars(messages []interface{}) int {
	total := 0
	for _, m := range messages {
		msg, ok := m.(map[string]interface{})
		if !ok {
			continue
		}
		if content, ok := msg["content"].(string); ok {
			total += len(content)
		}
	}
	return total
}

// queryKeywords pulls a simple keyword set from the most recent user
// message -- the "query" that sparsifyContent ranks middle blocks
// against, mirroring PFlash's own "tail-query overlap" scoring
// criterion (see rapid-mlx's pflash.py compress_tokens) at the text
// level instead of the token level.
func queryKeywords(messages []interface{}) map[string]bool {
	keywords := make(map[string]bool)
	for i := len(messages) - 1; i >= 0; i-- {
		msg, ok := messages[i].(map[string]interface{})
		if !ok || msg["role"] != "user" {
			continue
		}
		content, ok := msg["content"].(string)
		if !ok {
			return keywords
		}
		for _, word := range strings.Fields(content) {
			word = strings.ToLower(strings.Trim(word, ".,;:!?()[]{}\"'`"))
			if len(word) >= sparseKeywordMinLen {
				keywords[word] = true
			}
		}
		return keywords // only the single most recent user message
	}
	return keywords
}

// sparsifyContent is the "sparse token" idea from the MSA paper analysis,
// adapted to something buildable in Go at the text level: instead of a
// learned sparse-attention mechanism inside the model (which needs the
// model itself retrained -- not something a gateway can retrofit), split
// old, oversized content into blocks and keep only the ones that look
// relevant, the same sink+tail+query-overlap shape PFlash already uses
// token-side inside rapid-mlx.
//
// Only ever called above sparseGateChars (see compressMessages) -- for
// anything smaller, plain head+tail truncation already does the job with
// less risk of dropping something that mattered. Blocks are split on
// blank lines (works reasonably across code and prose without needing a
// language-aware parser); the first and last block are always kept
// (sink+tail), and the rest are ranked by how many query keywords they
// contain, keeping enough top-scoring blocks to hit keepRatio of the
// original size while preserving original order in the output.
func sparsifyContent(content string, keywords map[string]bool, keepRatio float64) (string, int) {
	blocks := strings.Split(content, "\n\n")
	if len(blocks) < sparseMinBlocks {
		return content, 0
	}

	sink := blocks[0]
	tail := blocks[len(blocks)-1]
	middle := blocks[1 : len(blocks)-1]

	type scoredBlock struct {
		text  string
		score int
		index int
	}
	scored := make([]scoredBlock, len(middle))
	for i, b := range middle {
		lower := strings.ToLower(b)
		score := 0
		for kw := range keywords {
			if strings.Contains(lower, kw) {
				score++
			}
		}
		scored[i] = scoredBlock{text: b, score: score, index: i}
	}
	sort.SliceStable(scored, func(i, j int) bool { return scored[i].score > scored[j].score })

	targetChars := int(float64(len(content)) * keepRatio)
	usedChars := len(sink) + len(tail)
	kept := make(map[int]bool, len(middle))
	for _, sb := range scored {
		if usedChars >= targetChars {
			break
		}
		kept[sb.index] = true
		usedChars += len(sb.text)
	}

	var out strings.Builder
	out.WriteString(sink)
	out.WriteString("\n\n")
	droppedBlocks, droppedChars := 0, 0
	for i, b := range middle {
		if kept[i] {
			out.WriteString(b)
			out.WriteString("\n\n")
		} else {
			droppedBlocks++
			droppedChars += len(b)
		}
	}
	if droppedBlocks > 0 {
		fmt.Fprintf(&out, "[...%d blocchi meno rilevanti omessi (%d caratteri)...]\n\n", droppedBlocks, droppedChars)
	}
	out.WriteString(tail)

	return out.String(), droppedChars
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
	// lastShapeIndexOf tracks the NORMALIZED shape (digits/whitespace
	// collapsed) separately from the exact hash -- a near-duplicate re-read
	// (same file, one line-number or timestamp different) won't match
	// lastIndexOf at all, since a single differing byte changes the exact
	// hash completely.
	lastIndexOf := make(map[string]int)
	lastShapeIndexOf := make(map[string]int)
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
		lastShapeIndexOf[normalizedShape(content)] = i
	}

	// Size gate for the sparse-block pass: below sparseGateChars, plain
	// head+tail truncation (below) already does the job. Computed once,
	// against the real incoming size, not the already-shrunk size after
	// dedup -- a prompt right at the boundary shouldn't get a different
	// truncation strategy per-message depending on how much dedup already
	// happened to remove earlier in this same pass.
	sparseGated := totalContentChars(messages) >= sparseGateChars
	var keywords map[string]bool
	if sparseGated {
		keywords = queryKeywords(messages)
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
			// Not byte-identical, but the same shape (digits/whitespace
			// aside) recurs later -- e.g. the same file re-read with one
			// line number or timestamp changed. Exact dedup above can't
			// catch this; the normalized signature can.
			if lastIdx, dup := lastShapeIndexOf[normalizedShape(content)]; dup && lastIdx != i {
				clone := cloneMessage(msg)
				placeholder := fmt.Sprintf(
					"[contenuto simile (%d caratteri, differenze minori come numeri o timestamp) -- la versione più recente compare più avanti nella conversazione]",
					len(content),
				)
				clone["content"] = placeholder
				savedChars += len(content) - len(placeholder)
				out[i] = clone
				continue
			}
		}

		if len(content) > compressTruncateThreshold {
			if sparseGated {
				if sparsified, dropped := sparsifyContent(content, keywords, sparseKeepRatio); dropped > 0 {
					clone := cloneMessage(msg)
					clone["content"] = sparsified
					savedChars += dropped
					out[i] = clone
					continue
				}
				// Too few blocks to sparsify usefully -- fall through to
				// plain head+tail truncation below instead.
			}
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

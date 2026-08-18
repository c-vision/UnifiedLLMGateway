package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
	"sync"
	"time"
)

// Session tracking (2026-08-18, WS-A): unified-gateway is stateless toward
// the backend -- rapid-mlx can only recognize a previously-seen prefix by
// exact content-addressed token match (no session id in main.go). That means
// the gateway's own byte stream must be STABLE-or-growing turn after turn,
// or every change to an old message forces a near-full re-prefill.
//
// What this file adds is observability + a stable session identity for the
// gateway itself, so we can (a) detect the moment a client's history
// actually diverges from what we sent before (a "session break" -- an edit
// to an old message, a /compact, an auto-continue), and (b) prove that our
// own compression layer does NOT introduce breaks (WS-C: frozen per-message
// forms). The session key is derived from the content itself, never from a
// client-supplied id -- the same content-addressed philosophy rapid-mlx
// uses, so a break here predicts a prefix-cache miss there.
//
// No tokenization is done here: tokenizing happens inside rapid-mlx on every
// request, and reusing it would require keeping the full token stream for
// every live session in memory. What IS computed is a cheap SHA-256 of the
// canonical serialization of (model, messages[:k]) -- the longest prefix of
// this request that existed identically in the previous request of the same
// conversation. If that prefix length grows monotonically, the backend's
// radix tree stays warm; a drop is a session break.
//
// Conversations are distinguished by an "anchor": the hash of the first up
// to 2 messages (system + first user turn, when present). Interleaved
// conversations on the same gateway must not cross-contaminate each other's
// tracking, and there is no stable client-supplied conversation id in the
// OpenAI/Anthropic protocols to key on instead.

type sessionState struct {
	// lastKey: the full conversation key of the most recent request seen
	// for this anchor. Encodes every message of that turn.
	lastKey string
	// lastLen: number of messages in the most recent request.
	lastLen int
	// breaks: cumulative session breaks for this conversation.
	breaks int64
}

type sessionTracker struct {
	mu sync.Mutex
	// conversations keyed by anchor hash (model + first 2 messages).
	conversations map[string]*sessionState
	// lastSeen: last time any request was tracked.
	lastSeen time.Time
	// lastLogged: the most recently reported (anchor, stablePrefix, broke).
	lastAnchor string
	lastStable int
	lastBroke  bool
	lastModel  string
}

func newSessionTracker() *sessionTracker {
	return &sessionTracker{conversations: make(map[string]*sessionState)}
}

// canonicalConversation serializes the parts of a request that must stay
// byte-stable for the backend's content-addressed cache to work: the model
// name, and every message up to and including index k. Messages after k
// (the just-appended turn) are excluded -- the "prefix" being measured is
// exactly what must match what we sent before.
func canonicalConversation(model string, messages []interface{}, k int) string {
	var b strings.Builder
	b.WriteString("model:")
	b.WriteString(model)
	b.WriteString("\x00")
	for i := 0; i < k && i < len(messages); i++ {
		m, ok := messages[i].(map[string]interface{})
		if !ok {
			continue
		}
		if s, ok := m["content"].(string); ok {
			b.WriteString(m["role"].(string))
			b.WriteString(":")
			b.WriteString(s)
		} else if arr, ok := m["content"].([]interface{}); ok {
			// Multimodal/structured content: serialize deterministically.
			raw, _ := json.Marshal(arr)
			b.WriteString(m["role"].(string))
			b.WriteString(":")
			b.WriteString(string(raw))
		}
		b.WriteString("\x00")
	}
	return b.String()
}

func sha256Hex(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

// conversationAnchor identifies one conversation: the model plus the first
// up to 2 messages (system + first user turn when present). These are
// stable within a conversation and differ across conversations.
func conversationAnchor(model string, messages []interface{}) string {
	n := 2
	if len(messages) < n {
		n = len(messages)
	}
	return sha256Hex(canonicalConversation(model, messages, n))
}

// fullConversationKey hashes the ENTIRE current request (all messages).
func fullConversationKey(model string, messages []interface{}) string {
	return sha256Hex(canonicalConversation(model, messages, len(messages)))
}

// stablePrefixLen computes how many of the current request's leading
// messages were byte-identical to the previous turn of the SAME
// conversation. When the history is append-only, this equals the previous
// turn's length. An edit to an old message, a /compact rewrite, or an
// auto-continue that restarts from a summary all drop it below that.
func stablePrefixLen(prev *sessionState, model string, messages []interface{}) int {
	if prev == nil || prev.lastKey == "" {
		return 0
	}
	// The previous full key hashes the previous turn's ENTIRE message list.
	// If the first `prev.lastLen` messages of this request are unchanged,
	// their canonical hash equals the previous full key exactly.
	if prev.lastLen <= len(messages) {
		k := canonicalConversation(model, messages, prev.lastLen)
		if sha256Hex(k) == prev.lastKey {
			return prev.lastLen
		}
	}
	// History diverged somewhere before prev.lastLen. Find the longest
	// strict prefix that still matches by comparing hashes length by length.
	// This is O(n) hashes per request -- fine for observability.
	for k := prev.lastLen - 1; k >= 1; k-- {
		ck := sha256Hex(canonicalConversation(model, messages, k))
		// The previous key encodes the full previous conversation; a shorter
		// prefix of it is a proper prefix of the previous turn, meaning the
		// current request still matches up to k.
		if strings.HasPrefix(prev.lastKey, ck) {
			return k
		}
	}
	return 0
}

// trackSession updates per-conversation state and reports whether this
// request broke continuity with the previous turn of the same conversation.
func (t *sessionTracker) trackSession(model string, messages []interface{}) (anchor string, stable int, broke bool) {
	if len(messages) == 0 {
		return "", 0, false
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	t.lastSeen = time.Now()
	anchor = conversationAnchor(model, messages)
	st := t.conversations[anchor]

	stable = stablePrefixLen(st, model, messages)
	fullKey := fullConversationKey(model, messages)

	broke = false
	if st != nil && st.lastLen > 0 {
		// Append-only growth keeps stable == st.lastLen. Anything shorter is
		// a break: an old message changed, history was compacted, or the
		// client rewrote the transcript.
		broke = stable < st.lastLen
		if broke {
			st.breaks++
		}
	}

	// Record this turn's full key + length as the new baseline.
	t.conversations[anchor] = &sessionState{
		lastKey: fullKey,
		lastLen: len(messages),
		breaks: func() int64 {
			if st != nil {
				return st.breaks
			}
			return 0
		}(),
	}

	t.lastAnchor = anchor
	t.lastStable = stable
	t.lastBroke = broke
	t.lastModel = model
	return anchor, stable, broke
}

func (t *sessionTracker) totalBreaks() int64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	total := int64(0)
	for _, st := range t.conversations {
		total += st.breaks
	}
	return total
}

func (t *sessionTracker) stats() map[string]interface{} {
	t.mu.Lock()
	defer t.mu.Unlock()
	totalBreaks := int64(0)
	for _, st := range t.conversations {
		totalBreaks += st.breaks
	}
	return map[string]interface{}{
		"conversations":       len(t.conversations),
		"total_breaks":        totalBreaks,
		"last_anchor":         t.lastAnchor,
		"last_model":          t.lastModel,
		"last_stable_prefix":  t.lastStable,
		"last_broke":          t.lastBroke,
		"last_seen":           t.lastSeen,
		"decision_cache_size": compressionDecisionCacheSize(),
		"requests_compressed": compressionStats.requestsCompressed.Load(),
		"chars_saved":         compressionStats.charsSaved.Load(),
	}
}

var sessionTrack = newSessionTracker()

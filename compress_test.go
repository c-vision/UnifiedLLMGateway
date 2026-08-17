package main

import (
	"strings"
	"testing"
)

// resetCompressionDecisionCache clears the stable per-message compression
// cache (compress.go) -- several tests below reuse the same fixture
// content (e.g. long := strings.Repeat("x", ...)) across different
// expected outcomes, which would otherwise leak a cached decision from
// one test into another now that decisions are permanent by design.
func resetCompressionDecisionCache() {
	compressionDecisionMu.Lock()
	compressionDecisionCache = make(map[string]string)
	compressionDecisionMu.Unlock()
}

func withCompressionEnabled(t *testing.T) {
	t.Helper()
	old := promptCompressionEnabled()
	setPromptCompressionEnabled(true)
	resetCompressionDecisionCache()
	t.Cleanup(func() {
		setPromptCompressionEnabled(old)
		resetCompressionDecisionCache()
	})
}

func TestDefaultCompressionEnabled(t *testing.T) {
	cases := map[string]bool{
		"":      true, // unset -- on by default
		"1":     true,
		"true":  true,
		"0":     false, // explicit opt-out
		"false": false,
	}
	for input, want := range cases {
		if got := defaultCompressionEnabled(input); got != want {
			t.Errorf("defaultCompressionEnabled(%q) = %v, want %v", input, got, want)
		}
	}
}

func msg(role, content string) map[string]interface{} {
	m := map[string]interface{}{"role": role, "content": content}
	return m
}

func toInterfaceSlice(msgs []map[string]interface{}) []interface{} {
	out := make([]interface{}, len(msgs))
	for i, m := range msgs {
		out[i] = m
	}
	return out
}

func TestCompressMessages_NoOpWhenDisabled(t *testing.T) {
	old := promptCompressionEnabled()
	setPromptCompressionEnabled(false)
	t.Cleanup(func() { setPromptCompressionEnabled(old) })
	long := strings.Repeat("x", compressTruncateThreshold+500)
	messages := toInterfaceSlice([]map[string]interface{}{
		msg("system", "you are a helpful assistant"),
		msg("tool", long),
		msg("user", "u1"), msg("assistant", "a1"), msg("user", "u2"),
		msg("assistant", "a2"), msg("user", "u3"), msg("assistant", "a3"),
	})
	out, saved := compressMessages(messages)
	if saved != 0 {
		t.Fatalf("expected no compression when disabled, saved=%d", saved)
	}
	gotContent := out[1].(map[string]interface{})["content"].(string)
	if gotContent != long {
		t.Fatalf("message content changed while compression disabled")
	}
}

func TestCompressMessages_ProtectsRecentWindow(t *testing.T) {
	withCompressionEnabled(t)
	long := strings.Repeat("x", compressTruncateThreshold+500)
	// Exactly compressProtectedWindow (6) messages -- nothing is "old".
	messages := toInterfaceSlice([]map[string]interface{}{
		msg("tool", long), msg("user", "u1"), msg("assistant", "a1"),
		msg("user", "u2"), msg("assistant", "a2"), msg("user", "u3"),
	})
	out, saved := compressMessages(messages)
	if saved != 0 {
		t.Fatalf("expected no compression, everything is in the protected window, saved=%d", saved)
	}
	if out[0].(map[string]interface{})["content"].(string) != long {
		t.Fatalf("protected-window message was modified")
	}
}

func TestCompressMessages_TruncatesOversizedOldToolContent(t *testing.T) {
	withCompressionEnabled(t)
	long := strings.Repeat("x", compressTruncateThreshold+500)
	// Build enough trailing messages so the long tool message (index 1) falls
	// outside the protected window (last 6).
	tail := []map[string]interface{}{
		msg("user", "u1"), msg("assistant", "a1"), msg("user", "u2"),
		msg("assistant", "a2"), msg("user", "u3"), msg("assistant", "a3"),
	}
	full := append([]map[string]interface{}{msg("system", "sys"), msg("tool", long)}, tail...)
	out, saved := compressMessages(toInterfaceSlice(full))
	if saved <= 0 {
		t.Fatalf("expected compression to trim the oversized old tool message, saved=%d", saved)
	}
	got := out[1].(map[string]interface{})["content"].(string)
	if len(got) >= len(long) {
		t.Fatalf("expected truncated content, got length %d (original %d)", len(got), len(long))
	}
	if !strings.HasPrefix(got, long[:compressHeadKeepChars]) {
		t.Fatalf("truncated content lost its head slice")
	}
	if !strings.HasSuffix(got, long[len(long)-compressTailKeepChars:]) {
		t.Fatalf("truncated content lost its tail slice")
	}
}

func TestCompressMessages_CollapsesDuplicateToolContent(t *testing.T) {
	withCompressionEnabled(t)
	dup := strings.Repeat("y", compressDedupMinChars+50)
	full := []map[string]interface{}{
		msg("system", "sys"),
		msg("tool", dup), // stale duplicate, should become a placeholder
		msg("user", "u1"), msg("assistant", "a1"),
		msg("tool", dup), // most recent occurrence -- kept intact if outside protected window, else untouched anyway
		msg("user", "u2"), msg("assistant", "a2"), msg("user", "u3"),
	}
	out, saved := compressMessages(toInterfaceSlice(full))
	if saved <= 0 {
		t.Fatalf("expected duplicate collapse to save characters, saved=%d", saved)
	}
	first := out[1].(map[string]interface{})["content"].(string)
	if first == dup {
		t.Fatalf("expected the earlier duplicate to be replaced with a placeholder")
	}
	if !strings.Contains(first, "duplicato") {
		t.Fatalf("expected placeholder marker text, got: %s", first)
	}
}

func TestCompressMessages_CollapsesNearDuplicateToolContent(t *testing.T) {
	withCompressionEnabled(t)
	base := strings.Repeat("line of file content\n", 20)
	// Same file, re-read with only a line number / timestamp different --
	// byte-for-byte different, so exact dedup alone would miss this.
	first := "// last modified line 42 at 09:12:03\n" + base
	second := "// last modified line 57 at 14:38:19\n" + base
	full := []map[string]interface{}{
		msg("system", "sys"),
		msg("tool", first), // stale near-duplicate, should become a placeholder
		msg("user", "u1"), msg("assistant", "a1"),
		msg("tool", second), // most recent occurrence
		msg("user", "u2"), msg("assistant", "a2"), msg("user", "u3"),
	}
	if first == second {
		t.Fatalf("test setup bug: the two contents must NOT be byte-identical")
	}
	out, saved := compressMessages(toInterfaceSlice(full))
	if saved <= 0 {
		t.Fatalf("expected near-duplicate collapse to save characters, saved=%d", saved)
	}
	got := out[1].(map[string]interface{})["content"].(string)
	if got == first {
		t.Fatalf("expected the earlier near-duplicate to be replaced with a placeholder")
	}
	if !strings.Contains(got, "simile") {
		t.Fatalf("expected near-duplicate placeholder marker text, got: %s", got)
	}
	// The kept (most recent) occurrence must stay fully intact.
	keptContent := out[4].(map[string]interface{})["content"].(string)
	if keptContent != second {
		t.Fatalf("expected the most recent occurrence to remain untouched")
	}
}

func TestCompressMessages_NeverTouchesSystemMessages(t *testing.T) {
	withCompressionEnabled(t)
	longSystem := strings.Repeat("s", compressTruncateThreshold+1000)
	full := []map[string]interface{}{
		msg("system", longSystem),
		msg("user", "u0"), msg("assistant", "a0"),
		msg("user", "u1"), msg("assistant", "a1"), msg("user", "u2"),
		msg("assistant", "a2"), msg("user", "u3"),
	}
	out, _ := compressMessages(toInterfaceSlice(full))
	if out[0].(map[string]interface{})["content"].(string) != longSystem {
		t.Fatalf("system message must never be modified by prompt compression")
	}
}

func TestCompressMessages_LeavesNonStringContentAlone(t *testing.T) {
	withCompressionEnabled(t)
	structured := []interface{}{map[string]interface{}{"type": "text", "text": "hi"}}
	full := []map[string]interface{}{
		{"role": "tool", "content": structured},
		msg("user", "u1"), msg("assistant", "a1"), msg("user", "u2"),
		msg("assistant", "a2"), msg("user", "u3"), msg("assistant", "a3"),
	}
	out, _ := compressMessages(toInterfaceSlice(full))
	got := out[0].(map[string]interface{})["content"]
	if _, ok := got.([]interface{}); !ok {
		t.Fatalf("array/structured content must be left untouched, got %T", got)
	}
}

func TestQueryKeywords_UsesOnlyMostRecentUserMessage(t *testing.T) {
	messages := toInterfaceSlice([]map[string]interface{}{
		msg("user", "stale keyword from an earlier turn"),
		msg("assistant", "a1"),
		msg("user", "fix the authentication bug in login handler"),
	})
	kw := queryKeywords(messages)
	if !kw["authentication"] || !kw["login"] || !kw["handler"] {
		t.Fatalf("expected keywords from the most recent user message, got %v", kw)
	}
	if kw["stale"] || kw["earlier"] {
		t.Fatalf("expected earlier user message to be ignored, got %v", kw)
	}
}

func TestQueryKeywords_SkipsShortWords(t *testing.T) {
	messages := toInterfaceSlice([]map[string]interface{}{
		msg("user", "fix it in the db"),
	})
	kw := queryKeywords(messages)
	for word := range kw {
		if len(word) < sparseKeywordMinLen {
			t.Fatalf("expected short words to be filtered out, found %q", word)
		}
	}
}

func TestSparsifyContent_KeepsSinkAndTailUnconditionally(t *testing.T) {
	blocks := []string{"sink block with setup code", "irrelevant middle one", "irrelevant middle two", "irrelevant middle three", "tail block with final state"}
	content := strings.Join(blocks, "\n\n")
	out, dropped := sparsifyContent(content, map[string]bool{"nomatch": true}, 0.1)
	if !strings.HasPrefix(out, blocks[0]) {
		t.Fatalf("expected sink block to be kept at the start, got: %s", out)
	}
	if !strings.HasSuffix(strings.TrimRight(out, "\n"), blocks[len(blocks)-1]) {
		t.Fatalf("expected tail block to be kept at the end, got: %s", out)
	}
	if dropped <= 0 {
		t.Fatalf("expected some middle blocks to be dropped at a 0.1 keep ratio, dropped=%d", dropped)
	}
}

func TestSparsifyContent_PrefersBlocksMatchingQueryKeywords(t *testing.T) {
	blocks := []string{
		"sink",
		"block about payments and billing logic",
		"block about the authentication handler and login flow",
		"block about unrelated logging utilities",
		"tail",
	}
	content := strings.Join(blocks, "\n\n")
	keywords := map[string]bool{"authentication": true, "login": true}
	out, _ := sparsifyContent(content, keywords, 0.4)
	if !strings.Contains(out, "authentication handler and login flow") {
		t.Fatalf("expected the keyword-matching block to survive, got: %s", out)
	}
	if strings.Contains(out, "unrelated logging utilities") {
		t.Fatalf("expected the non-matching block to be dropped in favor of the matching one, got: %s", out)
	}
}

// TestCompressMessages_SparsifyStableAcrossTurns is the regression test
// for the 2026-08-17 fix: an old, sparsified message must keep producing
// the EXACT SAME bytes on a later "turn" (a later, separate call to
// compressMessages against a conversation that has grown and whose
// latest user message -- and therefore queryKeywords -- has changed),
// or the backend's token-level prefix cache loses everything after that
// point on every single turn. See the doc comment above
// compressionDecisionCache in compress.go for the full mechanism.
func TestCompressMessages_SparsifyStableAcrossTurns(t *testing.T) {
	withCompressionEnabled(t)

	buildOldToolContent := func() string {
		var b strings.Builder
		b.WriteString("sink block\n\n")
		filler := "irrelevant filler line about nothing in particular. "
		relevant := "relevant block mentioning authentication login handler.\n\n"
		for b.Len() < sparseGateChars+10000-len(relevant)-200 {
			b.WriteString(strings.Repeat(filler, 20))
			b.WriteString("\n\n")
		}
		b.WriteString(relevant)
		b.WriteString("tail block")
		return b.String()
	}
	oldContent := buildOldToolContent()

	// Turn 1: the current question is about authentication -- sparsify
	// should keep the "authentication login handler" block.
	turn1 := append([]map[string]interface{}{msg("system", "sys"), msg("tool", oldContent)},
		msg("user", "u1"), msg("assistant", "a1"), msg("user", "u2"),
		msg("assistant", "a2"), msg("user", "u3"),
		msg("user", "fix the authentication login handler bug"),
	)
	out1, saved1 := compressMessages(toInterfaceSlice(turn1))
	if saved1 <= 0 {
		t.Fatalf("expected compression to engage on turn 1, saved=%d", saved1)
	}
	content1 := out1[1].(map[string]interface{})["content"].(string)
	if !strings.Contains(content1, "authentication login handler") {
		t.Fatalf("turn 1: expected the auth block to survive sparsify, got: %s", content1[:min(200, len(content1))])
	}

	// Turn 2: conversation grew (window slid forward by one more pair),
	// and the CURRENT question is now about something unrelated -- a
	// query-driven sparsify would legitimately want to drop the auth
	// block now, which is exactly what must NOT happen: the old message
	// was already compressed once, its form must stay frozen.
	turn2 := append([]map[string]interface{}{msg("system", "sys"), msg("tool", oldContent)},
		msg("user", "u1"), msg("assistant", "a1"), msg("user", "u2"),
		msg("assistant", "a2"), msg("user", "u3"), msg("assistant", "a3"),
		msg("user", "how do I run the database migration script"),
	)
	out2, _ := compressMessages(toInterfaceSlice(turn2))
	content2 := out2[1].(map[string]interface{})["content"].(string)

	if content1 != content2 {
		t.Fatalf("sparsified content changed between turns -- breaks the backend's prefix cache\nturn1: %s\nturn2: %s",
			content1[:min(200, len(content1))], content2[:min(200, len(content2))])
	}
}

func TestSparsifyContent_FallsBackBelowMinBlocks(t *testing.T) {
	content := "just one block, no blank-line separators at all"
	out, dropped := sparsifyContent(content, map[string]bool{}, 0.1)
	if out != content || dropped != 0 {
		t.Fatalf("expected no-op when content has too few blocks to sparsify, got out=%q dropped=%d", out, dropped)
	}
}

func TestCompressMessages_SparseGateEngagesOnlyAboveSizeThreshold(t *testing.T) {
	withCompressionEnabled(t)

	// Build one oversized old tool message per test case, sized either
	// just under or just over sparseGateChars, with distinguishable
	// keyword-matching and non-matching blocks so we can tell which
	// compression strategy actually ran.
	buildOldToolContent := func(totalChars int) string {
		var b strings.Builder
		b.WriteString("sink block\n\n")
		filler := "irrelevant filler line about nothing in particular. "
		relevant := "relevant block mentioning authentication login handler.\n\n"
		for b.Len() < totalChars-len(relevant)-200 {
			b.WriteString(strings.Repeat(filler, 20))
			b.WriteString("\n\n")
		}
		b.WriteString(relevant)
		b.WriteString("tail block")
		return b.String()
	}

	tail := []map[string]interface{}{
		msg("user", "u1"), msg("assistant", "a1"), msg("user", "u2"),
		msg("assistant", "a2"), msg("user", "u3"),
		msg("user", "fix the authentication login handler bug"),
	}

	t.Run("below gate: plain truncation, keyword block may be cut", func(t *testing.T) {
		content := buildOldToolContent(compressTruncateThreshold * 2)
		full := append([]map[string]interface{}{msg("system", "sys"), msg("tool", content)}, tail...)
		out, saved := compressMessages(toInterfaceSlice(full))
		if saved <= 0 {
			t.Fatalf("expected some compression below the gate, saved=%d", saved)
		}
		got := out[1].(map[string]interface{})["content"].(string)
		if strings.Contains(got, "blocchi meno rilevanti omessi") {
			t.Fatalf("expected plain head+tail truncation below sparseGateChars, got sparse-style output: %s", got[:200])
		}
	})

	t.Run("above gate: sparse selection keeps the keyword-matching block", func(t *testing.T) {
		content := buildOldToolContent(sparseGateChars + 10000)
		full := append([]map[string]interface{}{msg("system", "sys"), msg("tool", content)}, tail...)
		out, saved := compressMessages(toInterfaceSlice(full))
		if saved <= 0 {
			t.Fatalf("expected compression above the gate, saved=%d", saved)
		}
		got := out[1].(map[string]interface{})["content"].(string)
		if !strings.Contains(got, "authentication login handler") {
			t.Fatalf("expected sparse selection to keep the query-relevant block, got a content sample: %.300s", got)
		}
	})
}

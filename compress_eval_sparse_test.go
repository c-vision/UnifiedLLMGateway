package main

import (
	"fmt"
	"strings"
	"testing"
)

// TestCompressionEval_SparseAndNearDup is a throwaway evaluation (not a
// regression test) for the two features added on top of the original
// dedup+truncate compressor: sparse-block selection (for genuinely large
// prompts) and near-duplicate detection (for recurring-but-not-identical
// content). It builds a conversation shaped like a real long-running
// OpenCode/Claude-Code session that crosses sparseGateChars, and reports
// what each mechanism actually contributes.
// Run with: go test -run TestCompressionEval_SparseAndNearDup -v
func TestCompressionEval_SparseAndNearDup(t *testing.T) {
	withCompressionEnabled(t)

	// A large log file re-read 3 times with only a run counter/timestamp
	// differing each time -- the near-duplicate case exact-hash dedup misses.
	makeLog := func(run int) string {
		var b strings.Builder
		for i := 0; i < 400; i++ {
			fmt.Fprintf(&b, "[2026-07-10 09:%02d:%02d] run=%d step=%d ok\n", i%60, (i*7)%60, run, i)
		}
		return b.String() // ~19KB
	}

	// A large one-off codebase dump (never repeated), split into many
	// paragraph "blocks" -- only a handful mention the thing the user
	// actually ends up asking about, the rest is unrelated filler.
	makeCodeDump := func() string {
		var b strings.Builder
		for i := 0; i < 120; i++ {
			if i == 40 || i == 90 {
				fmt.Fprintf(&b, "func handleAuthMiddleware_%d() {\n\t// validates session tokens against the auth store\n\tvalidateSessionToken()\n}\n\n", i)
			} else {
				fmt.Fprintf(&b, "func unrelatedHelper_%d() {\n\tfmt.Println(\"noop %d\")\n}\n\n", i, i)
			}
		}
		return b.String() // ~9.5KB, ~120 blocks
	}

	messages := []map[string]interface{}{
		{"role": "system", "content": "You are opencode, an interactive CLI tool..."},

		{"role": "user", "content": "leggi il codebase e mostrami i log"},
		{"role": "assistant", "content": "leggo"},
		{"role": "tool", "content": makeCodeDump()},
		{"role": "tool", "content": makeLog(1)},

		{"role": "user", "content": "controlla di nuovo i log dopo il retry"},
		{"role": "assistant", "content": "controllo"},
		{"role": "tool", "content": makeLog(2)}, // near-dup of run 1: same shape, counters differ

		{"role": "user", "content": "un altro giro di log"},
		{"role": "assistant", "content": "ok"},
		{"role": "tool", "content": makeLog(3)}, // near-dup of run 1 & 2

		{"role": "user", "content": "rileggi il codebase per sicurezza"},
		{"role": "assistant", "content": "rileggo"},
		{"role": "tool", "content": makeCodeDump()}, // exact duplicate of the first dump

		// Padding so the conversation is unambiguously above sparseGateChars
		// (150,000 chars) -- a handful of large one-off tool outputs that
		// don't recur, simulating other file reads earlier in the session.
		{"role": "user", "content": "leggi anche questi altri file"},
		{"role": "assistant", "content": "leggo"},
		{"role": "tool", "content": strings.Repeat("some other file content, line filler here\n", 2500)},   // ~107KB
		{"role": "tool", "content": strings.Repeat("yet another unrelated file, more filler text\n", 400)}, // ~18KB

		// Protected window: the current, real turn -- asking specifically
		// about the auth middleware, which is buried in the codebase dump.
		{"role": "user", "content": "com'è fatto il middleware di autenticazione (auth) che valida i session token?"},
		{"role": "assistant", "content": "controllo"},
		{"role": "tool", "content": "checking..."},
		{"role": "user", "content": "e allora?"},
		{"role": "assistant", "content": "un momento"},
		{"role": "user", "content": "allora?"},
	}

	msgsIface := make([]interface{}, len(messages))
	origTotal := 0
	for i, m := range messages {
		msgsIface[i] = m
		if s, ok := m["content"].(string); ok {
			origTotal += len(s)
		}
	}

	if origTotal < sparseGateChars {
		t.Fatalf("test setup error: conversation (%d chars) must exceed sparseGateChars (%d) to exercise sparse selection", origTotal, sparseGateChars)
	}

	compressed, saved := compressMessages(msgsIface)

	compTotal := 0
	for _, m := range compressed {
		if s, ok := m.(map[string]interface{})["content"].(string); ok {
			compTotal += len(s)
		}
	}

	t.Logf("=== dimensioni ===")
	t.Logf("caratteri PRIMA:  %d (%.0fk token circa)", origTotal, float64(origTotal)/4000)
	t.Logf("caratteri DOPO:   %d (%.0fk token circa)", compTotal, float64(compTotal)/4000)
	t.Logf("risparmiati:      %d", saved)
	t.Logf("riduzione:        %.1f%%", float64(origTotal-compTotal)/float64(origTotal)*100)

	t.Logf("=== messaggi modificati ===")
	for i, m := range compressed {
		mm := m.(map[string]interface{})
		s, ok := mm["content"].(string)
		if !ok {
			continue
		}
		origS, _ := messages[i]["content"].(string)
		if s != origS {
			preview := s
			if len(preview) > 200 {
				preview = preview[:200] + "..."
			}
			t.Logf("[%d] role=%s (%d -> %d char): %q", i, mm["role"], len(origS), len(s), preview)
		}
	}

	// The near-duplicate log reads (run 1 and run 2) should have collapsed
	// to placeholders, since a later occurrence with the same normalized
	// shape exists further down the conversation.
	run1 := compressed[4].(map[string]interface{})["content"].(string)
	run2 := compressed[7].(map[string]interface{})["content"].(string)
	run3 := compressed[10].(map[string]interface{})["content"].(string)
	if !strings.Contains(run1, "contenuto simile") {
		t.Errorf("expected run 1 log to collapse as a near-duplicate placeholder, got: %q", run1[:min(80, len(run1))])
	}
	if !strings.Contains(run2, "contenuto simile") {
		t.Errorf("expected run 2 log to collapse as a near-duplicate placeholder, got: %q", run2[:min(80, len(run2))])
	}
	// Run 3 is the LAST occurrence, so dedup doesn't collapse it to a
	// placeholder -- but it's still an old message (well outside the
	// protected window) and still oversized, so it remains a candidate for
	// the size-based truncation pass below. Being "the authoritative copy"
	// only exempts a message from dedup, not from truncation.
	if strings.Contains(run3, "contenuto simile") || strings.Contains(run3, "contenuto duplicato") {
		t.Errorf("run 3 (last occurrence) should never be dedup-collapsed, got a dedup placeholder instead")
	}

	// The FIRST codebase dump (message 3) is old and duplicated -- should
	// collapse to an exact-duplicate placeholder since message 13 re-reads
	// it byte-for-byte later.
	firstDump := compressed[3].(map[string]interface{})["content"].(string)
	if !strings.Contains(firstDump, "contenuto duplicato") {
		t.Errorf("expected first codebase dump to collapse as an exact-duplicate placeholder, got: %q", firstDump[:min(80, len(firstDump))])
	}

	// The SECOND codebase dump (message 13) is the last occurrence, so dedup
	// leaves it as-is -- but it's still old, oversized content, so the
	// sparse-selection pass should engage on it (conversation is above the
	// gate) and specifically should have KEPT the auth-middleware blocks,
	// since the current user question is about "middleware" / "autenticazione"
	// / "session" / "token".
	secondDump := compressed[13].(map[string]interface{})["content"].(string)
	if !strings.Contains(secondDump, "handleAuthMiddleware") {
		t.Errorf("sparse selection should have preserved the auth-middleware block (matches current query keywords), but it was dropped")
	}
	if !strings.Contains(secondDump, "blocchi meno rilevanti omessi") {
		t.Errorf("expected sparse selection to actually drop some unrelated filler blocks from the second codebase dump")
	}

	// Compare against what plain head+tail truncation (the pre-sparse
	// behavior) would have kept for that same message, to isolate the
	// sparse pass's actual contribution.
	plainHead := messages[13]["content"].(string)[:compressHeadKeepChars]
	plainTail := messages[13]["content"].(string)[len(messages[13]["content"].(string))-compressTailKeepChars:]
	plainWouldKeepAuth := strings.Contains(plainHead, "handleAuthMiddleware") || strings.Contains(plainTail, "handleAuthMiddleware")
	t.Logf("=== confronto con troncamento semplice (pre-sparse) ===")
	t.Logf("il vecchio head+tail (800+800 char) avrebbe conservato il blocco auth-middleware? %v", plainWouldKeepAuth)
	t.Logf("la selezione sparse lo conserva? %v", strings.Contains(secondDump, "handleAuthMiddleware"))

	if saved <= 0 {
		t.Fatalf("expected real savings, saved=%d", saved)
	}
}

package main

import (
	"strings"
	"testing"
)

// TestCompressionEval is a throwaway evaluation (not a regression test) that
// runs compressMessages against a conversation shaped like the real
// OpenCode session that motivated this feature: the same files re-read
// across turns, plus a one-off large tool output that's never repeated.
// Run with: go test -run TestCompressionEval -v
func TestCompressionEval(t *testing.T) {
	withCompressionEnabled(t)

	fileContentA := strings.Repeat("func Example() {\n\tfmt.Println(\"hello\")\n}\n", 150) // ~5.5KB
	fileContentB := strings.Repeat("import (\n\t\"fmt\"\n\t\"os\"\n)\n\n", 300)            // ~7KB
	buildLog := strings.Repeat("ok  \tpackage/subpkg\t0.123s\n", 500)                      // ~15KB, one-off, never repeated

	messages := []map[string]interface{}{
		{"role": "system", "content": "You are opencode, an interactive CLI tool..."},
		// Turn 1: read file A
		{"role": "user", "content": "leggi main.go"},
		{"role": "assistant", "content": "leggo il file"},
		{"role": "tool", "content": fileContentA},
		// Turn 2: read file B, run build (one-off huge log)
		{"role": "user", "content": "leggi imports.go e builda"},
		{"role": "assistant", "content": "leggo e builda"},
		{"role": "tool", "content": fileContentB},
		{"role": "tool", "content": buildLog},
		// Turn 3: re-read file A to verify -- classic duplicate pattern
		{"role": "user", "content": "rileggi main.go per conferma"},
		{"role": "assistant", "content": "rileggo"},
		{"role": "tool", "content": fileContentA}, // exact duplicate of turn 1
		// Turn 4: re-read file B again
		{"role": "user", "content": "rileggi imports.go"},
		{"role": "assistant", "content": "rileggo"},
		{"role": "tool", "content": fileContentB}, // exact duplicate of turn 2
		// Recent turns -- protected window, current real work
		{"role": "user", "content": "aggiungi una funzione di test"},
		{"role": "assistant", "content": "aggiungo il test"},
		{"role": "tool", "content": "test file written"},
		{"role": "user", "content": "esegui i test"},
		{"role": "assistant", "content": "eseguo"},
		{"role": "user", "content": "hai fatto questo?"},
	}

	msgsIface := make([]interface{}, len(messages))
	origTotal := 0
	for i, m := range messages {
		msgsIface[i] = m
		if s, ok := m["content"].(string); ok {
			origTotal += len(s)
		}
	}

	compressed, saved := compressMessages(msgsIface)

	compTotal := 0
	for _, m := range compressed {
		if s, ok := m.(map[string]interface{})["content"].(string); ok {
			compTotal += len(s)
		}
	}

	t.Logf("messaggi totali: %d", len(messages))
	t.Logf("caratteri PRIMA: %d", origTotal)
	t.Logf("caratteri DOPO:  %d", compTotal)
	t.Logf("risparmiati (riportato dalla funzione): %d", saved)
	t.Logf("riduzione: %.1f%%", float64(origTotal-compTotal)/float64(origTotal)*100)

	for i, m := range compressed {
		mm := m.(map[string]interface{})
		s, ok := mm["content"].(string)
		if !ok {
			continue
		}
		origS, _ := messages[i]["content"].(string)
		if s != origS {
			preview := s
			if len(preview) > 150 {
				preview = preview[:150] + "..."
			}
			t.Logf("[%d] role=%s MODIFICATO: %q", i, mm["role"], preview)
		}
	}

	if saved <= 0 {
		t.Fatalf("expected real savings on a realistic duplicate-heavy conversation, saved=%d", saved)
	}
}

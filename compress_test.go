package main

import (
	"strings"
	"testing"
)

func withCompressionEnabled(t *testing.T) {
	t.Helper()
	old := promptCompressionEnabled()
	setPromptCompressionEnabled(true)
	t.Cleanup(func() { setPromptCompressionEnabled(old) })
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

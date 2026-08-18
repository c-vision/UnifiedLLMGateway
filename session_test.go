package main

import (
	"testing"
)

func resetSessionTracker() {
	sessionTrack = newSessionTracker()
}

func sessionMessages(msgs ...map[string]interface{}) []interface{} {
	out := make([]interface{}, len(msgs))
	for i, m := range msgs {
		out[i] = m
	}
	return out
}

// TestTrackSession_GrowsWithoutBreak: consecutive turns of the same
// conversation (append-only history) must NEVER report a session break --
// that is exactly the case rapid-mlx's content-addressed prefix cache is
// designed to serve, and our own compression (WS-C) must not change the
// byte stream of old messages.
func TestTrackSession_GrowsWithoutBreak(t *testing.T) {
	resetSessionTracker()

	// Turn 1: first request, full history is the prefix.
	turn1 := sessionMessages(
		msg("system", "sys"),
		msg("user", "u1"), msg("assistant", "a1"),
		msg("user", "u2"),
	)
	_, _, broke := sessionTrack.trackSession("qw38", turn1)
	if broke {
		t.Fatalf("turn 1 (first request) must not report a break")
	}

	// Turn 2: one new user message appended; everything before unchanged.
	turn2 := sessionMessages(
		msg("system", "sys"),
		msg("user", "u1"), msg("assistant", "a1"),
		msg("user", "u2"), msg("assistant", "a2"),
		msg("user", "u3"),
	)
	_, _, broke = sessionTrack.trackSession("qw38", turn2)
	if broke {
		t.Fatalf("append-only growth must not report a session break")
	}

	// Turn 3: another append.
	turn3 := sessionMessages(
		msg("system", "sys"),
		msg("user", "u1"), msg("assistant", "a1"),
		msg("user", "u2"), msg("assistant", "a2"),
		msg("user", "u3"), msg("assistant", "a3"),
		msg("user", "u4"),
	)
	_, _, broke = sessionTrack.trackSession("qw38", turn3)
	if broke {
		t.Fatalf("second append must not report a session break")
	}

	if got := sessionTrack.totalBreaks(); got != 0 {
		t.Fatalf("expected 0 total breaks across 3 growing turns, got %d", got)
	}
}

// TestTrackSession_EditedOldMessageBreaks: if an OLD message's content
// changes (edited history, /compact, auto-continue), the prefix from that
// point on diverges and the backend's content-addressed cache misses on it.
// The tracker must detect exactly that as a session break.
func TestTrackSession_EditedOldMessageBreaks(t *testing.T) {
	resetSessionTracker()

	turn1 := sessionMessages(
		msg("system", "sys"),
		msg("user", "u1"), msg("assistant", "a1"),
		msg("user", "u2"),
	)
	sessionTrack.trackSession("qw38", turn1)

	// The OLD assistant message changed (simulates a /compact rewrite or a
	// client editing history).
	turn2 := sessionMessages(
		msg("system", "sys"),
		msg("user", "u1"), msg("assistant", "CHANGED"),
		msg("user", "u2"), msg("assistant", "a2"),
		msg("user", "u3"),
	)
	_, _, broke := sessionTrack.trackSession("qw38", turn2)
	if !broke {
		t.Fatalf("edited old message must report a session break")
	}

	if got := sessionTrack.totalBreaks(); got != 1 {
		t.Fatalf("expected exactly 1 break, got %d", got)
	}
}

// TestTrackSession_DifferentModelNoCrossTalk: two conversations on
// different models must not interfere (the model is part of the key).
func TestTrackSession_DifferentModelNoCrossTalk(t *testing.T) {
	resetSessionTracker()

	convA := sessionMessages(msg("system", "sys"), msg("user", "u1"), msg("assistant", "a1"), msg("user", "u2"))
	convB := sessionMessages(msg("system", "sys"), msg("user", "q1"), msg("assistant", "q2"), msg("user", "q3"))

	if _, _, broke := sessionTrack.trackSession("modelA", convA); broke {
		t.Fatalf("first request modelA must not break")
	}
	if _, _, broke := sessionTrack.trackSession("modelB", convB); broke {
		t.Fatalf("first request modelB must not break")
	}
	// Growing modelA again must not break (modelB's request must not have
	// poisoned modelA's key).
	convA2 := sessionMessages(msg("system", "sys"), msg("user", "u1"), msg("assistant", "a1"), msg("user", "u2"), msg("assistant", "a2"), msg("user", "u3"))
	if _, _, broke := sessionTrack.trackSession("modelA", convA2); broke {
		t.Fatalf("append-only growth of modelA after an unrelated modelB request must not break")
	}
	if got := sessionTrack.totalBreaks(); got != 0 {
		t.Fatalf("expected 0 breaks, got %d", got)
	}
}

package main

import (
	"os"
	"path/filepath"
	"testing"
)

// TestClearDiskKVCheckpointsAt removes rapid-mlx's orphaned disk KV
// checkpoints (see clearDiskKVCheckpoints in control.go). Layout mirrors
// disk_kv_checkpoint.py: <root>/<req_hash>/checkpoint-<offset>.safetensors
// + .json sidecar. A missing root is a no-op, not an error.
func TestClearDiskKVCheckpointsAt(t *testing.T) {
	dir := t.TempDir()

	// Build one request-hash subdirectory with a couple of checkpoint pairs.
	reqDir := filepath.Join(dir, "cdd8e191cc575120")
	if err := os.MkdirAll(reqDir, 0o755); err != nil {
		t.Fatal(err)
	}
	files := []struct{ name, content string }{
		{"checkpoint-75776.safetensors", "kv-body-1"},
		{"checkpoint-75776.json", `{"token_offset":75776}`},
		{"checkpoint-76032.safetensors", "kv-body-2"},
		{"checkpoint-76032.json", `{"token_offset":76032}`},
	}
	wantBytes := int64(0)
	for _, f := range files {
		if err := os.WriteFile(filepath.Join(reqDir, f.name), []byte(f.content), 0o644); err != nil {
			t.Fatal(err)
		}
		wantBytes += int64(len(f.content))
	}

	bytesFreed, filesRemoved, err := clearDiskKVCheckpointsAt(dir)
	if err != nil {
		t.Fatalf("clearDiskKVCheckpointsAt returned error: %v", err)
	}
	if filesRemoved != len(files) {
		t.Fatalf("expected %d files removed, got %d", len(files), filesRemoved)
	}
	if bytesFreed != wantBytes {
		t.Fatalf("expected %d bytes freed, got %d", wantBytes, bytesFreed)
	}
	if _, err := os.Stat(reqDir); !os.IsNotExist(err) {
		t.Fatalf("expected request-hash subdirectory to be removed, stat err: %v", err)
	}
}

func TestClearDiskKVCheckpointsAt_MissingRootIsNoOp(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "does-not-exist")
	bytesFreed, filesRemoved, err := clearDiskKVCheckpointsAt(dir)
	if err != nil {
		t.Fatalf("missing root must be a no-op, got error: %v", err)
	}
	if bytesFreed != 0 || filesRemoved != 0 {
		t.Fatalf("expected 0 bytes / 0 files for missing root, got %d / %d", bytesFreed, filesRemoved)
	}
}

func TestClearDiskKVCheckpointsAt_LeavesSiblingDirsAlone(t *testing.T) {
	// Real layout: ~/.cache/rapid-mlx/kv_checkpoints/<req_hash>/... with
	// prefix_cache as a SIBLING of kv_checkpoints, never a child. The
	// cleanup must only touch the kv_checkpoints root passed in.
	root := t.TempDir()
	kvRoot := filepath.Join(root, "kv_checkpoints")
	prefixCache := filepath.Join(root, "prefix_cache")
	if err := os.MkdirAll(filepath.Join(kvRoot, "cdd8e191cc575120"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(kvRoot, "cdd8e191cc575120", "checkpoint-75776.safetensors"), []byte("kv"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(prefixCache, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(prefixCache, "entry.safetensors"), []byte("cache"), 0o644); err != nil {
		t.Fatal(err)
	}

	_, _, err := clearDiskKVCheckpointsAt(kvRoot)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, err := os.Stat(filepath.Join(prefixCache, "entry.safetensors")); err != nil {
		t.Fatalf("sibling prefix_cache must be left untouched, stat err: %v", err)
	}
}

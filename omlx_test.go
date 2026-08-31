package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestWriteOMLXProfileDFlash(t *testing.T) {
	base := t.TempDir()
	m := ModelConfig{
		Path:             "/models/Qwen3.8-27B-8bit",
		DflashDraftModel: "/models/Qwen3.8-27B-DFlash2",
		DflashBlockSize:  5,
		Ctx:              32768,
	}
	if err := writeOMLXProfile(base, "qw38dflash2", m); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(base, "model_settings.json"))
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Models map[string]map[string]any `json:"models"`
	}
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	settings := got.Models["Qwen3.8-27B-8bit"]
	if settings["model_alias"] != "qw38dflash2" || settings["dflash_enabled"] != true {
		t.Fatalf("unexpected profile: %#v", settings)
	}
	if settings["dflash_block_size"] != float64(5) {
		t.Fatalf("block size was not preserved: %#v", settings)
	}
	if settings["dflash_draft_quant_enabled"] != false {
		t.Fatalf("pre-quantized draft must not be quantized again: %#v", settings)
	}
}

func TestEstimateModelSizeIncludesDFlashDraft(t *testing.T) {
	root := t.TempDir()
	target := filepath.Join(root, "target")
	draft := filepath.Join(root, "draft")
	for _, dir := range []string{target, draft} {
		if err := os.Mkdir(dir, 0755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(target, "model.safetensors"), make([]byte, 1024), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(draft, "model.safetensors"), make([]byte, 512), 0644); err != nil {
		t.Fatal(err)
	}
	got := estimateModelSizeGB(ModelConfig{Path: target, DflashDraftModel: draft})
	want := float64(1536) / (1024 * 1024 * 1024)
	if got != want {
		t.Fatalf("estimateModelSizeGB() = %g, want %g", got, want)
	}
}

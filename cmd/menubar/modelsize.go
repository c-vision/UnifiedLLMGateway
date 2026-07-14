package main

import (
	"fmt"
	"math"
	"os"
	"strings"
)

// estimateDiskGB mirrors the root package's estimateModelSizeGB (memcheck.go)
// -- duplicated here for the same reason as modelConfig/gwConfig above: this
// is a separate "main" package and can't import the root one. Sums
// .safetensors shards for a directory-based model, or the raw file size for
// a single-file model (ds4's .gguf). Returns 0 (no path, missing file) for
// backends like ollama that don't store weights at a local path this app
// can see -- callers must treat 0 as "unknown", not "free".
func estimateDiskGB(path string) float64 {
	if path == "" {
		return 0
	}
	info, err := os.Stat(path)
	if err != nil {
		return 0
	}
	if !info.IsDir() {
		return float64(info.Size()) / (1024 * 1024 * 1024)
	}
	entries, err := os.ReadDir(path)
	if err != nil {
		return 0
	}
	var total int64
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".safetensors") {
			continue
		}
		if fi, err := e.Info(); err == nil {
			total += fi.Size()
		}
	}
	return float64(total) / (1024 * 1024 * 1024)
}

// formatCtx renders a token count in the "256K" shorthand models.json's
// "ctx" values are always specified in -- one decimal only when the value
// isn't a clean multiple of 1024 (e.g. a hypothetical 200000 -> "195.3K").
func formatCtx(ctx int) string {
	k := float64(ctx) / 1024
	if k == math.Trunc(k) {
		return fmt.Sprintf("%.0fK", k)
	}
	return fmt.Sprintf("%.1fK", k)
}

package main

// codingQualityIcon is a rough, hand-assigned traffic-light call on each
// model's real-world coding quality, C# 13 included -- NOT a benchmark
// score. No local model has a published C# 13-specific eval; this is a
// best-effort judgment from (a) whether the model is coding-specialized by
// training/branding vs. general-purpose, and (b) this catalog's own
// real-world findings this session (mixtral called out as "praticamente
// inutilizzabile", gptoss120/qw122 as much slower, both far more prone to
// stale/awkward output on anything past mainstream C#). Treat it as a
// starting hint, not a verdict -- update by hand as real usage disagrees.
var codingQualityIcon = map[string]string{
	// Coding-specialized MoE, modern training, strong agentic/tool-call
	// track record in this catalog's own testing.
	"qw3coder":     "🟢",
	"qw3coder6bit": "🟢",
	"qw3coder8bit": "🟢",
	"glm45air":     "🟢",

	// General-purpose, capable, can code competently but isn't
	// coding-specialized -- reasonable but not the first choice for
	// unusual/very recent language features.
	"qw3627":   "🟡",
	"qw3635":   "🟡",
	"qw27":     "🟡",
	"qw27opus": "🟡",
	"qw122":    "🟡",

	// Weaker or not coding-focused: dense/slow (mixtral confirmed
	// "praticamente inutilizzabile" this session), reasoning-tuned rather
	// than code-tuned (ds32), or general chat models never marketed for
	// code (gemma4/gemma4o, gptoss120, antirez).
	"ds32":      "🔴",
	"gemma4":    "🔴",
	"gemma4o":   "🔴",
	"mixtral":   "🔴",
	"gptoss120": "🔴",
	"antirez":   "🔴",
}

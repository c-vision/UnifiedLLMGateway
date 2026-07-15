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

	// Promising on paper, unverified in this catalog's own real-world
	// agentic use yet: gptoss20 reportedly beats its own 120B sibling on
	// HumanEval/MMLU despite being 6x smaller (2026-07-15 research), but
	// that's a benchmark claim, not something tested here the way the 🟢
	// tier above was. Benchmark-good has burned this catalog before on
	// real agentic tool-calling (Qwopus, removed 2026-07-14 for exactly
	// this gap). Bump to 🟢 once actually exercised.
	"gptoss20": "🟡",

	// Untested additions (2026-07-15). glm47flash is an official zai-org
	// release in the same lineage as glm45air (🟢) -- same reasoning/tool
	// format, far fewer active experts (4 routed + 1 shared of 65 vs
	// glm45air's 12B active), so plausibly a fast option for the same
	// large-C#-project use case where glm45air was too slow -- but
	// unverified until actually run. bonsai4b is a 4B dense model at
	// extreme ternary (~2-bit) quantization from a research lab (Prism
	// ML): tiny and fast by construction, but this catalog's own findings
	// this session are that small dense models struggle on large real C#
	// work, and 2-bit is far more aggressive than any other quant in this
	// catalog -- treat with at least as much caution as gptoss20 got.
	"glm47flash": "🟡",
	"bonsai4b":   "🟡",

	// qw3codernext (2026-07-15, still downloading when added): official
	// Qwen-Coder branding, same family as qw3coder/qw3coder6bit/8bit (🟢
	// above) -- stronger prior than the other 🟡 entries, but still
	// unverified in this catalog's own real agentic use, so held to the
	// same bar until actually exercised rather than promoted on lineage
	// alone.
	"qw3codernext": "🟡",

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

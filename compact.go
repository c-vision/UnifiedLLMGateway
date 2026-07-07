package main

import (
	"fmt"
	"strings"
)

// Compact / auto-continue detection — ported from copilot-api-dev's
// src/lib/compact.ts + src/routes/messages/preprocess.ts (getCompactType).
// Claude Code sends two special request shapes when a conversation gets too
// long: a COMPACT_REQUEST asking the model to summarize everything so far,
// and a COMPACT_AUTO_CONTINUE that resumes the conversation right after,
// seeded with that summary. Recognizing them doesn't change how we forward
// the request (the instructions are plain text the model already sees),
// but it matters for observability — these requests have an unusual shape
// (huge system-reminder text, no tools expected) and are otherwise easy to
// mistake for a bug when debugging.

const (
	compactNone         = 0
	compactRequest      = 1
	compactAutoContinue = 2
)

var compactSystemPromptStarts = []string{
	"You are a helpful AI assistant tasked with summarizing conversations",
	"You are an anchored context summarization assistant for coding sessions.",
}

const compactTextOnlyGuard = "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools."
const compactSummaryPromptStart = "Your task is to create a detailed summary of the conversation so far"

var compactMessageSections = []string{"Pending Tasks:", "Current Work:"}

var compactAutoContinuePromptStarts = []string{
	"This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.",
	"Continue if you have next steps, or stop and ask for clarification if you are unsure how to proceed.",
	"The previous request exceeded the provider's size limit due to large media attachments. The conversation was compacted and media files were removed from context.",
}

func compactTypeLabel(t int) string {
	switch t {
	case compactRequest:
		return "compact-request (summarize conversation)"
	case compactAutoContinue:
		return "compact-auto-continue (resuming from summary)"
	default:
		return "none"
	}
}

// lastUserMessageText extracts the plain text of a message, skipping any
// <system-reminder> block, but only if it's a user message — compact
// detection only ever looks at the last turn, and only when it's from the
// user (matches getCompactCandidateText in copilot-api-dev).
func lastUserMessageText(m Message) string {
	if m.Role != "user" {
		return ""
	}
	switch content := m.Content.(type) {
	case string:
		return content
	case []interface{}:
		var parts []string
		for _, item := range content {
			block, ok := item.(map[string]interface{})
			if !ok || block["type"] != "text" {
				continue
			}
			text := fmt.Sprintf("%v", block["text"])
			if strings.HasPrefix(text, "<system-reminder>") {
				continue
			}
			if text != "" {
				parts = append(parts, text)
			}
		}
		return strings.Join(parts, "\n\n")
	}
	return ""
}

func isCompactRequestMessage(text string) bool {
	if text == "" {
		return false
	}
	if !strings.Contains(text, compactTextOnlyGuard) || !strings.Contains(text, compactSummaryPromptStart) {
		return false
	}
	for _, section := range compactMessageSections {
		if strings.Contains(text, section) {
			return true
		}
	}
	return false
}

func isCompactAutoContinueMessage(text string) bool {
	if text == "" {
		return false
	}
	for _, promptStart := range compactAutoContinuePromptStarts {
		if strings.HasPrefix(text, promptStart) {
			return true
		}
	}
	return false
}

// detectCompactType mirrors copilot-api-dev's getCompactType: check the last
// message first, then fall back to the system prompt's opening text.
func detectCompactType(system interface{}, messages []Message) int {
	if len(messages) > 0 {
		lastText := lastUserMessageText(messages[len(messages)-1])
		if isCompactRequestMessage(lastText) {
			return compactRequest
		}
		if isCompactAutoContinueMessage(lastText) {
			return compactAutoContinue
		}
	}

	sysText := extractBlockText(system)
	for _, promptStart := range compactSystemPromptStarts {
		if strings.HasPrefix(sysText, promptStart) {
			return compactRequest
		}
	}
	return compactNone
}

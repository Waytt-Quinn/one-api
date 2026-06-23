package xunfei

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestDeepSeekEncodingCase1 mirrors Python test_case_1: thinking
// mode with a tool call followed by a final assistant turn with
// content. The test slices the assistant turn text out of the
// golden prompt at "<｜Assistant｜><think>" and runs
// parseMessageFromCompletionText on it. Expected results are
// hard-coded from the deepseek-v4 reference test suite.
func TestDeepSeekEncodingCase1(t *testing.T) {
	gold := readTestdataFile(t, "test_output_1.txt")

	// First assistant turn (the one with the get_weather tool call).
	marker := "<｜Assistant｜><think>"
	firstStart := strings.Index(gold, marker)
	if firstStart < 0 {
		t.Fatalf("marker %q not found in test_output_1.txt", marker)
	}
	firstStart += len(marker)
	firstEnd := strings.Index(gold[firstStart:], "<｜User｜>")
	if firstEnd < 0 {
		t.Fatalf("first <｜User｜> not found after first marker")
	}
	firstTurn := gold[firstStart : firstStart+firstEnd]
	parsed, err := parseMessageFromCompletionText(firstTurn, "thinking")
	if err != nil {
		t.Fatalf("parse first turn: %v", err)
	}
	if parsed.ReasoningContent != "The user wants to know the weather in Beijing. I should use the get_weather tool." {
		t.Errorf("reasoning_content mismatch:\nwant %q\ngot  %q",
			"The user wants to know the weather in Beijing. I should use the get_weather tool.",
			parsed.ReasoningContent)
	}
	if parsed.Content != "" {
		t.Errorf("expected empty content, got %q", parsed.Content)
	}
	if len(parsed.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(parsed.ToolCalls))
	}
	if parsed.ToolCalls[0].Function.Name != "get_weather" {
		t.Errorf("tool name mismatch: want %q, got %q",
			"get_weather", parsed.ToolCalls[0].Function.Name)
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(parsed.ToolCalls[0].Function.Arguments), &args); err != nil {
		t.Errorf("tool arguments not valid JSON: %v (raw=%q)", err, parsed.ToolCalls[0].Function.Arguments)
	} else {
		want := map[string]any{"location": "Beijing", "unit": "celsius"}
		if fmt.Sprintf("%v", args) != fmt.Sprintf("%v", want) {
			t.Errorf("args mismatch:\nwant %v\ngot  %v", want, args)
		}
	}

	// Last assistant turn (the one with prose content).
	lastStart := strings.LastIndex(gold, marker)
	if lastStart < 0 {
		t.Fatalf("last marker %q not found", marker)
	}
	lastStart += len(marker)
	parsedFinal, err := parseMessageFromCompletionText(gold[lastStart:], "thinking")
	if err != nil {
		t.Fatalf("parse last turn: %v", err)
	}
	if parsedFinal.ReasoningContent != "Got the weather data. Let me format a nice response." {
		t.Errorf("last reasoning mismatch:\nwant %q\ngot  %q",
			"Got the weather data. Let me format a nice response.",
			parsedFinal.ReasoningContent)
	}
	if !strings.Contains(parsedFinal.Content, "22°C") {
		t.Errorf("expected content to contain '22°C', got %q", parsedFinal.Content)
	}
	if len(parsedFinal.ToolCalls) != 0 {
		t.Errorf("expected no tool calls in final turn, got %d", len(parsedFinal.ToolCalls))
	}
}

// TestDeepSeekEncodingCase2 mirrors Python test_case_2: thinking
// mode without tools, with the second assistant turn producing
// a final answer. The Python test also asserts that the first
// assistant's reasoning was dropped during encoding; we don't
// re-verify the encoder, only the parser on the last turn.
func TestDeepSeekEncodingCase2(t *testing.T) {
	gold := readTestdataFile(t, "test_output_2.txt")
	marker := "<｜Assistant｜><think>"
	lastStart := strings.LastIndex(gold, marker)
	if lastStart < 0 {
		t.Fatalf("marker %q not found", marker)
	}
	lastStart += len(marker)

	parsed, err := parseMessageFromCompletionText(gold[lastStart:], "thinking")
	if err != nil {
		t.Fatalf("parse last turn: %v", err)
	}
	if parsed.ReasoningContent != "The user asks about the capital of France. It is Paris." {
		t.Errorf("reasoning mismatch:\nwant %q\ngot  %q",
			"The user asks about the capital of France. It is Paris.",
			parsed.ReasoningContent)
	}
	if parsed.Content != "The capital of France is Paris." {
		t.Errorf("content mismatch:\nwant %q\ngot  %q",
			"The capital of France is Paris.", parsed.Content)
	}
	if len(parsed.ToolCalls) != 0 {
		t.Errorf("expected no tool calls, got %d", len(parsed.ToolCalls))
	}

	// Sanity: the first turn's reasoning should be absent from the
	// golden (mirrors the Python "drop_thinking" assertion).
	if strings.Contains(gold, "The user said hello") {
		t.Errorf("expected first turn's reasoning to be dropped, but it is present in test_output_2.txt")
	}
}

// TestDeepSeekEncodingCase3 mirrors Python test_case_3: interleaved
// thinking + search. The first assistant turn emits a `search` tool
// call. The second assistant turn produces prose. We parse the
// first turn and verify the tool call.
func TestDeepSeekEncodingCase3(t *testing.T) {
	gold := readTestdataFile(t, "test_output_3.txt")
	marker := "<｜Assistant｜><think>"
	firstStart := strings.Index(gold, marker)
	if firstStart < 0 {
		t.Fatalf("marker %q not found", marker)
	}
	firstStart += len(marker)

	// The first turn ends at the next <｜User｜> marker.
	firstEnd := strings.Index(gold[firstStart:], "<｜User｜>")
	if firstEnd < 0 {
		t.Fatalf("first <｜User｜> not found after first marker")
	}
	firstTurn := gold[firstStart : firstStart+firstEnd]

	parsed, err := parseMessageFromCompletionText(firstTurn, "thinking")
	if err != nil {
		t.Fatalf("parse first turn: %v", err)
	}
	if parsed.ReasoningContent != "用户想知道小柴胡冲剂和布洛芬能否一起服用。" {
		t.Errorf("reasoning mismatch:\nwant %q\ngot  %q",
			"用户想知道小柴胡冲剂和布洛芬能否一起服用。",
			parsed.ReasoningContent)
	}
	if parsed.Content != "" {
		t.Errorf("expected empty content, got %q", parsed.Content)
	}
	if len(parsed.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(parsed.ToolCalls))
	}
	if parsed.ToolCalls[0].Function.Name != "search" {
		t.Errorf("expected tool name 'search', got %q", parsed.ToolCalls[0].Function.Name)
	}
	if !strings.Contains(parsed.ToolCalls[0].Function.Arguments, "小柴胡冲剂") {
		t.Errorf("expected args to contain '小柴胡冲剂', got %q", parsed.ToolCalls[0].Function.Arguments)
	}
}

// TestDeepSeekEncodingCase4 mirrors Python test_case_4: chat mode
// (no <think> block) — the parser should still extract the content
// correctly because parseMessageFromCompletionText treats thinking
// mode as a flag, and in chat mode there is no thinking block to
// extract.
func TestDeepSeekEncodingCase4(t *testing.T) {
	gold := readTestdataFile(t, "test_output_4.txt")
	marker := "<｜Assistant｜></think>"
	firstStart := strings.Index(gold, marker)
	if firstStart < 0 {
		t.Fatalf("marker %q not found (chat mode closes </think> first)", marker)
	}
	firstStart += len(marker)

	firstEnd := strings.Index(gold[firstStart:], "<｜User｜>")
	if firstEnd < 0 {
		t.Fatalf("first <｜User｜> not found after first marker")
	}
	firstTurn := gold[firstStart : firstStart+firstEnd]

	parsed, err := parseMessageFromCompletionText(firstTurn, "chat")
	if err != nil {
		t.Fatalf("parse first turn: %v", err)
	}
	if !strings.Contains(parsed.Content, "热海") {
		t.Errorf("expected content to mention 热海, got %q", parsed.Content)
	}
	if len(parsed.ToolCalls) != 0 {
		t.Errorf("expected no tool calls in chat turn, got %d", len(parsed.ToolCalls))
	}
}

// readTestdataFile reads a file from the testdata directory next to
// this _test.go file. We use os.ReadFile rather than go:embed so the
// test data can be regenerated from the deepseek reference without
// changing this file.
func readTestdataFile(t *testing.T, name string) string {
	t.Helper()
	path := filepath.Join("testdata", name)
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read testdata %s: %v", path, err)
	}
	return string(b)
}

package xunfei

import (
	"fmt"
	"strings"
	"testing"

	"github.com/songquanpeng/one-api/relay/model"
)

// TestStreamBufferDSMLRealWSSChunks exercises streamXMLBuffer.consume
// against the actual WSS frame chunks from the cc-switch log. The
// crucial case is the `<` character at the end of frame 5
// (".\n\n<"), which begins a DSML tool-call block that spans many
// subsequent frames. The buffer must hold that `<` and the following
// partial marker until the closing </｜DSML｜invoke> arrives, instead
// of leaking `<｜DSML｜...` as visible prose.
func TestStreamBufferDSMLRealWSSChunks(t *testing.T) {
	chunks := []string{
		"\n\n",                  // frame 0
		"Let me",                // frame 1
		" read",                 // frame 2
		" the file",             // frame 3
		" first",                // frame 4
		".\n\n<",                // frame 5 — "<" must be held
		"｜DSML｜tool",          // frame 6
		"_calls",                // frame 7
		">\n<",                  // frame 8
		"｜DSML｜inv",           // frame 9
		"oke name",              // frame 10
		"=\"Read",               // frame 11
		"\">\n<",                // frame 12
		"｜DSML｜parameter name", // frame 13
		"=\"",                   // frame 14
		"file_path",             // frame 15
		"\" string",             // frame 16
		"=\"true",               // frame 17
		"\">D",                  // frame 18
		":\\Work",               // frame 19
		"Space\\",               // frame 20
		"中期评审",              // frame 21
		"提示",                  // frame 22
		"词.md",                 // frame 23
		"</｜DSML｜",            // frame 24
		"parameter>\n",          // frame 25
		"</｜DSML｜",            // frame 26
		"invoke",                // frame 27
		">",                     // frame 28
	}

	buf := newStreamXMLBuffer()
	var totalVisible strings.Builder
	var totalTools []model.Tool
	for i, c := range chunks {
		visible, tools := buf.consume(c)
		totalVisible.WriteString(visible)
		totalTools = append(totalTools, tools...)
		if visible != "" {
			fmt.Printf("Frame %2d (chunk=%q): visible=%q\n", i, c, visible)
		}
		if len(tools) > 0 {
			fmt.Printf("Frame %2d: TOOLS=%+v\n", i, tools)
		}
	}
	fmt.Println("---")
	fmt.Println("Final visible:", totalVisible.String())
	fmt.Println("Final tools:")
	for _, tc := range totalTools {
		fmt.Printf("  name=%s args=%v\n", tc.Function.Name, tc.Function.Arguments)
	}

	if strings.Contains(totalVisible.String(), "DSML") {
		t.Errorf("DSML marker leaked into visible text: %q", totalVisible.String())
	}
	if len(totalTools) == 0 {
		t.Fatalf("expected at least one tool call, got 0")
	}
	if got := totalTools[0].Function.Name; got != "Read" {
		t.Errorf("expected tool name %q, got %q", "Read", got)
	}
	if got := totalTools[0].Function.Arguments; got == nil {
		t.Errorf("expected non-nil arguments, got nil")
	} else if argsStr, ok := got.(string); !ok {
		t.Errorf("expected JSON-string args, got %T", got)
	} else if !strings.Contains(argsStr, "file_path") || !strings.Contains(argsStr, "中期评审提示词") {
		t.Errorf("expected args to contain file_path and the filename, got %q", argsStr)
	}
}

// TestStreamBufferClaudeCodeToolCall replays the exact WSS frame
// sequence from oneapi-2026.06.23-10.49.log (seq 10-58) that
// caused Claude Code to show raw DSML markup in the chat and
// fail to invoke the local tool. The model emitted a Bash
// invocation with a malformed wrapper opener
// ("<|DSML|tool_calls|\n") and a malformed invoke opener
// ("<|DSML|invoke name=\"Bash\""), and the parameter close
// arrived with an extra pipe. The streaming buffer must:
//   - Suppress the DSML wrapper so it never reaches the chat.
//   - Parse the Bash tool call with command="ls -la".
//   - Not leak any "<|DSML|", "DSML", "<|", or "<tool" substring
//     into the visible content.
func TestStreamBufferClaudeCodeToolCall(t *testing.T) {
	chunks := []string{
		// Prose before the tool call.
		"好的", "，", "我来查看", "当前工作", "目录的结构", "。请", "稍", "等。\n\n",
		// Tool call: malformed wrapper opener, then invoke + parameters.
		"<|", "DS", "ML", "|tool", "_calls", "|\n",
		"  <", "|DS", "ML|", "invoke", " name=\"", "Bash", "\"", "   ",
		" <|", "DSML", "|parameter", " name=\"", "command",
		"\"", "ls -", "la</", "|", "DSML", "|parameter",
		">\n   ", " <|", "DSML", "|parameter", " name=\"", "description",
		"\"", "List", " files", " in current", " directory</",
		"|", "DSML", "|parameter", ">\n ", " </|", "DSML",
		"|inv", "oke>\n", "</|", "DSML", "|tool", "_calls", "|", ">",
	}

	buf := newStreamXMLBuffer()
	var totalVisible strings.Builder
	var totalTools []model.Tool
	for _, c := range chunks {
		visible, tools := buf.consume(c)
		totalVisible.WriteString(visible)
		totalTools = append(totalTools, tools...)
	}
	for _, tc := range buf.flush() {
		totalTools = append(totalTools, tc)
	}

	t.Logf("visible=%q", totalVisible.String())
	for i, tc := range totalTools {
		t.Logf("tool[%d]: name=%q args=%v", i, tc.Function.Name, tc.Function.Arguments)
	}

	if got := totalVisible.String(); strings.Contains(got, "DSML") {
		t.Errorf("DSML marker leaked into visible text: %q", got)
	}
	if got := totalVisible.String(); strings.Contains(got, "<|") {
		t.Errorf("DSML opener leaked into visible text: %q", got)
	}
	if got := totalVisible.String(); strings.Contains(got, "<tool") {
		t.Errorf("tool wrapper leaked into visible text: %q", got)
	}
	if len(totalTools) == 0 {
		t.Fatalf("expected at least one tool call, got 0; visible=%q", totalVisible.String())
	}
	if got := totalTools[0].Function.Name; got != "Bash" {
		t.Errorf("expected tool name %q, got %q", "Bash", got)
	}
	if got := totalTools[0].Function.Arguments; got == nil {
		t.Errorf("expected non-nil arguments, got nil")
	} else if argsStr, ok := got.(string); !ok {
		t.Errorf("expected JSON-string args, got %T", got)
	} else if !strings.Contains(argsStr, "ls -la") {
		t.Errorf("expected args to contain command=\"ls -la\", got %q", argsStr)
	}
}

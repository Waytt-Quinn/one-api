package xunfei

import (
	"fmt"
	"strings"
	"testing"

	"github.com/songquanpeng/one-api/relay/model"
)

// TestStreamBufferMalformedDSML reproduces the WSS frame sequence
// from oneapi-2026.06.23-10.49.log (seq 10-58). The model emitted a
// tool call with several malformations:
//
//  1. The wrapper open has a missing ">": "<|DSML|tool_calls|\n"
//     instead of "<|DSML|tool_calls>\n".
//  2. The invoke open has a missing ">": "<|DSML|invoke name="Bash""
//     instead of "<|DSML|invoke name="Bash">".
//  3. The <|DSML|parameter> openers have missing ">":
//     "<|DSML|parameter name="command"" instead of "...">".
//  4. The wrapper close has an extra pipe: "</|DSML|tool_calls|>"
//     instead of "</|DSML|tool_calls>".
//
// The streaming buffer is expected to extract the Bash tool call
// and strip the malformed DSML markup. As of the previous fix
// (305cbab) the buffer only handled well-formed DSML; the
// malformed openers/closers caused the buffer to emit the whole
// block as visible prose and no tool_calls to be sent to the
// client.
func TestStreamBufferMalformedDSML(t *testing.T) {
	chunks := []string{
		// Prose before the tool call.
		"好的", "，", "我来查看", "当前工作", "目录的结构", "。请", "稍", "等。\n\n",
		// Tool call starts here.
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
	for _, tc := range buf.flush() {
		totalTools = append(totalTools, tc)
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
	if strings.Contains(totalVisible.String(), "<|") {
		t.Errorf("DSML opener leaked: %q", totalVisible.String())
	}
	if len(totalTools) == 0 {
		t.Fatalf("expected at least one tool call, got 0")
	}
}

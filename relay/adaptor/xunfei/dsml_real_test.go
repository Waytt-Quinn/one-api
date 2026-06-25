package xunfei

import (
	"fmt"
	"strings"
	"testing"

	"github.com/songquanpeng/one-api/relay/model"
)

// TestStreamBufferDSMLRealClaudeTokens replays the actual per-token
// WSS frame chunks captured in cc-switch.txt on 2026-06-10. The
// model tokeniser splits "<|DSML|..." across multiple frames
// ("<|DS", "ML", "|tool", "_calls", ...), so our buffer must hold
// the partial "<|DS..." prefix across frames until the closing
// </|DSML|tool_calls> arrives.
func TestStreamBufferDSMLRealClaudeTokens(t *testing.T) {
	// Per-token chunks exactly as they arrived in the
	// content_block_delta stream (verified from cc-switch.txt
	// on 2026-06-10).
	chunks := []string{
		" I'll start by reading the Launcher design document and exploring the code.\n\n",
		"<|DS", "ML", "|tool", "_calls", ">", "\n", "  ",
		"<|DS", "ML|", "invoke", " name=\"", "Read\"", ">", "\n", "    ",
		"<|DS", "ML|", "parameter name", "=\"", "file_path", "\">",
		"D:\\WorkSpace\\note.wiki\\Doc\\Refactor\\Launcher.md",
		"</|DSML|parameter", ">\n  </|DSML|", "invoke", ">\n  ",
		"<|DS", "ML|", "invoke", " name=\"", "Bash\"", ">", "\n", "    ",
		"<|DS", "ML|", "parameter name", "=\"", "command", "\">",
		"find D:\\WorkSpace\\icomp\\core\\src\\main\\java\\com\\iba\\icomp\\core\\launch -type f -name",
		"\n   \"*.java\" 2>/dev/null", "</|DSML|parameter", ">\n  </|DSML|", "invoke", ">\n  ",
		"<|DS", "ML|", "invoke", " name=\"", "Bash\"", ">", "\n", "    ",
		"<|DS", "ML|", "parameter name", "=\"", "command", "\">",
		"find D:\\WorkSpace\\tcs-core\\tcs-core\\src\\main\\java\\com\\cgn\\tcs\\core -maxdepth 1",
		"\n  -type d 2>/dev/null", "</|DSML|parameter", ">\n  </|DSML|", "invoke", ">\n",
		"</|DSML|tool_calls", ">", "\n",
	}

	buf := newStreamXMLBuffer()
	var totalVisible strings.Builder
	var totalTools []model.Tool
	for i, c := range chunks {
		visible, tools := buf.consume(c)
		totalVisible.WriteString(visible)
		totalTools = append(totalTools, tools...)
		if len(tools) > 0 {
			fmt.Printf("Frame %d: TOOLS=%d\n", i, len(tools))
		}
		if visible != "" {
			fmt.Printf("Frame %d: visible=%q\n", i, visible)
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
		t.Errorf("DSML leaked into visible: %q", totalVisible.String())
	}
	if strings.Contains(totalVisible.String(), "<|") {
		t.Errorf("DSML opener leaked: %q", totalVisible.String())
	}
	if len(totalTools) != 3 {
		t.Fatalf("expected 3 tool calls, got %d", len(totalTools))
	}
	if totalTools[0].Function.Name != "Read" {
		t.Errorf("expected first tool Read, got %q", totalTools[0].Function.Name)
	}
	if totalTools[1].Function.Name != "Bash" {
		t.Errorf("expected second tool Bash, got %q", totalTools[1].Function.Name)
	}
	if totalTools[2].Function.Name != "Bash" {
		t.Errorf("expected third tool Bash, got %q", totalTools[2].Function.Name)
	}
}

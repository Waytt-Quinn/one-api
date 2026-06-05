package xunfei

import (
	"fmt"
	"strings"
	"testing"

	"github.com/songquanpeng/one-api/relay/model"
)

// TestStreamBufferPlainXMLToolCall replays the actual SSE text deltas
// captured in cc-switch.txt on 2026-06-05. The model emitted a
// plain <read_file>...</read_file> tool call (no DSML markers), and
// the streaming layer must convert it into a model.Tool so Claude
// Code can execute the read.
func TestStreamBufferPlainXMLToolCall(t *testing.T) {
	chunks := []string{
		"让我",
		"先读取",
		" La",
		"uncher ",
		"模块的设计",
		"文档和相关",
		"代码。\n\n",
		"<read",
		"_file>\n",
		"<file",
		"_path>",
		"D:\\",
		"WorkSpace",
		"\\note",
		".wiki",
		"\\Doc",
		"\\Ref",
		"actor\\",
		"Launcher",
		".md",
		"</file",
		"_path>\n",
		"</read",
		"_file>",
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

	if strings.Contains(totalVisible.String(), "<read_file") {
		t.Errorf("tool call XML leaked into visible: %q", totalVisible.String())
	}
	if len(totalTools) == 0 {
		t.Fatalf("expected at least one tool call, got 0")
	}
	if got := totalTools[0].Function.Name; got != "read_file" {
		t.Errorf("expected tool name %q, got %q", "read_file", got)
	}
}

package xunfei

import (
	"fmt"
	"strings"
	"testing"

	"github.com/songquanpeng/one-api/relay/model"
)

// TestStreamBufferDSMLBurstChunk replays the actual chunks from
// cc-swtich.txt on 2026-06-05 at 10:43. The model sent the inner
// <｜DSML｜invoke>...</｜DSML｜invoke> block as a single burst, and
// the outer <｜DSML｜tool_calls>...</｜DSML｜tool_calls> wrapper
// was split across four subsequent frames. The streaming layer
// must extract the tool call AND strip the wrapper, leaving no
// DSML markers in the visible text.
func TestStreamBufferDSMLBurstChunk(t *testing.T) {
	chunks := []string{
		"\n\n",
		"<｜DSML｜invoke name=\"Read\"><｜DSML｜parameter name=\"file_path\" string=\"true\">D:\\WorkSpace\\note.wiki\\Doc\\Refactor\\Launcher.md</｜DSML｜parameter>\n</｜DSML｜invoke>\n",
		"</｜DSML｜",
		"tool",
		"_calls",
		">",
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
}

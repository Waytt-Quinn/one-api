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
		fmt.Printf("Frame %2d: buf=%q\n", i, buf.accumulated)
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
	} else if m, ok := got.(map[string]string); !ok {
		t.Errorf("expected map[string]string args, got %T", got)
	} else if m["file_path"] == "" {
		t.Errorf("expected non-empty file_path arg, got empty map: %v", m)
	}
}

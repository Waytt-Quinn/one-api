package xunfei

import (
	"fmt"
	"strings"
	"testing"

	"github.com/songquanpeng/one-api/relay/model"
)

// TestStreamBufferRealLogCase reproduces the WSS frame sequence
// captured in oneapi.log on 2026-06-23 around line 66. The model
// split the `<|DSML|tool_calls>` opening tag across six frames
// (`<`, `|`, `DS`, `ML|`, `tool_c`, `alls>`) and the streaming
// buffer leaked the halfwidth-pipe DSML markup into the visible
// text. The fix in main.go (findTrailingPartialDSML) holds any
// non-empty trailing prefix of `<|DSML|`, so the buffer must now
// keep `<|`, `<|D`, `<|DS` etc. attached to the held tail until
// the next frame disambiguates.
func TestStreamBufferRealLogCase(t *testing.T) {
	chunks := []string{
		"Let", " me look", " at what", "'s", " in your", " working",
		" directory first", ".\n\n<", "|", "DS", "ML|", "tool_c",
		"alls>\n", "  <", "|DS", "ML|", "invoke", " name=\"",
		"Bash", "\">\n    <", "|DS", "ML|", "parameter name",
		"=\"", "command", "\">",
		"ls -", "la D", ":/", "WorkSpace", "/claude", "/</",
		"|", "DSML", "|parameter", ">\n", "  </", "|DS",
		"ML|", "invoke", ">\n</", "|", "DSML", "|tool_c", "alls>", "\n",
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
	if totalTools[0].Function.Name != "Bash" {
		t.Errorf("expected tool name %q, got %q", "Bash", totalTools[0].Function.Name)
	}
}

// TestStreamBufferPartialDSMLPrefixes sanity-checks the buffer's
// behaviour when the trailing tail is a non-empty prefix of
// `<|DSML|`. Each of these partials must be held and merged with
// the next chunk before any visible text is emitted.
func TestStreamBufferPartialDSMLPrefixes(t *testing.T) {
	cases := [][]string{
		// Just "<" held, then "|" completes "<|", then "DS" ->
		// "<|DS", then "ML|" -> "<|DSML|", then the rest.
		{"hello<", "|", "DS", "ML|", "tool_calls>world"},
		// Slightly different split: "<" + "|DS" + "ML|tool_c"
		// + "alls>world".
		{"hi<", "|DS", "ML|tool_c", "alls>world"},
		// "<|D" already partial on first chunk.
		{"hi<|D", "SML|tool_calls>world"},
	}
	for i, chunks := range cases {
		buf := newStreamXMLBuffer()
		var visible strings.Builder
		for _, c := range chunks {
			v, _ := buf.consume(c)
			visible.WriteString(v)
		}
		_ = buf.flush()
		got := visible.String()
		if strings.Contains(got, "DSML") {
			t.Errorf("case %d: DSML marker leaked: %q", i, got)
		}
		if strings.Contains(got, "<|") {
			t.Errorf("case %d: DSML opener leaked: %q", i, got)
		}
	}
}

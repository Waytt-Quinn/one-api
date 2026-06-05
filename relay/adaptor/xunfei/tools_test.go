package xunfei

import (
	"fmt"
	"strings"
	"testing"

	"github.com/songquanpeng/one-api/relay/model"
)

// TestStreamBufferToolsShape replays the actual chunks from
// cc-swtich.txt on 2026-06-05 at 15:49. The model emitted yet
// another tool-call shape:
//
//	<tools>
//	<param name="path">D:\WorkSpace\中期评审提示词.md</param>
//	</tools>
//
// (No matching close was observed before the stream ended, but
// the inner <param>...</param> block is fully closed within the
// single burst frame.)
func TestStreamBufferToolsShape(t *testing.T) {
	chunks := []string{
		"\n\n",
		"Let",
		" me read",
		" the file",
		" first",
		".\n\n",
		"<tools>\n<param name=\"path\">D:\\WorkSpace\\中期评审提示词.md</param>\n",
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

	if strings.Contains(totalVisible.String(), "<tools>") {
		t.Errorf("tool call wrapper leaked into visible: %q", totalVisible.String())
	}
	// Even though the model used an ambiguous <tools><param> shape
	// with no explicit tool name, the streaming buffer should have
	// at least suppressed the wrapper. (We can't reliably extract a
	// tool call without a tool name, but flushing the buffer at
	// end-of-stream ensures the held tail is processed by the
	// extractor in complete-buffer mode.)
	tailTools := buf.flush()
	for _, tc := range tailTools {
		totalTools = append(totalTools, tc)
	}
	if len(totalTools) > 0 {
		t.Logf("end-of-stream flush recovered %d tool(s):", len(totalTools))
		for _, tc := range totalTools {
			t.Logf("  name=%s args=%v", tc.Function.Name, tc.Function.Arguments)
		}
	}
}

package xunfei

// This file is a Go port of the deepseek-v4 reference encoder
// (D:\workspace\DeepSeek-V4-Flash\encoding\encoding_dsv4.py,
// lines 600-744). It implements parse_message_from_completion_text
// and its helpers, which together:
//  1. Split an assistant turn into reasoning_content + content +
//     tool_calls.
//  2. Parse the DSML tool-call wrapper, the per-call <invoke>
//     opener, and each <parameter> child (honouring the
//     string="true|false" attribute).
//  3. Decode parameter values into Go native types (string for
//     string="true", json-decoded for string="false") and
//     serialise the final arguments map as a JSON string for
//     OpenAI Chat Completions compatibility.
//
// The original one-api parser in main.go does streaming
// extraction on partial buffers; this file is the
// end-of-stream parser that produces the final structured
// message.

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/songquanpeng/one-api/common/random"
)

// DeepSeek-V4 special tokens. The halfwidth pipe is the canonical
// form; the model may also emit the fullwidth ｜ variant which
// we normalise at the entry point.
const (
	dsv4BOS         = "<｜begin▁of▁sentence｜>"
	dsv4EOS         = "<｜end▁of▁sentence｜>"
	dsv4ThinkEnd    = "</think>"
	dsv4DSML        = "｜DSML｜"
	dsv4TCBlockName = "tool_calls"
	dsv4TCBlockStart = "\n\n<" + dsv4DSML + dsv4TCBlockName
)

// ParsedMessage is the structured form of a single assistant turn
// produced by parseMessageFromCompletionText. It mirrors the
// Python return dict (role, content, reasoning_content,
// tool_calls).
type ParsedMessage struct {
	Role             string
	Content          string
	ReasoningContent string
	ToolCalls        []OpenAIToolCall
}

// OpenAIToolCall is the wire format for one tool call, matching
// the OpenAI Chat Completions spec:
// https://platform.openai.com/docs/api-reference/chat/object
type OpenAIToolCall struct {
	Type     string                  `json:"type"`
	Function OpenAIToolCallFunction  `json:"function"`
	ID       string                  `json:"id,omitempty"`
}

type OpenAIToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// readUntilStop walks text from `from` (in RUNES, not bytes) and
// returns the first stop token's rune position. The content before
// the stop token and the matched stop token (or "" if none found)
// are returned alongside the new rune index.
//
// This is a Go port of the Python _read_until_stop, which uses
// character (rune) positions. We have to do the conversion because
// the deepseek-v4 reference uses char positions throughout and the
// test data is text-only (no emoji or other complex graphemes), so
// staying in runes avoids subtle byte/char drift in the position
// math. The downstream code that uses the returned index slices
// `text[from:pos]` which works equally well for either unit
// (since text is a UTF-8 string and slicing at any byte position
// always lands on a rune boundary when from was also rune-aligned).
func readUntilStop(from int, text string, stops []string) (int, string, string) {
	runes := []rune(text)
	minPos := len(runes)
	var matched string
	for _, s := range stops {
		stopRunes := []rune(s)
		// Find the first occurrence of stopRunes starting at
		// rune position `from`. Loop in runes so we don't
		// depend on byte length.
		pos := indexRune(runes, from, stopRunes)
		if pos >= 0 && pos < minPos {
			minPos = pos
			matched = s
		}
	}
	if matched != "" {
		return minPos + len([]rune(matched)), string(runes[from:minPos]), matched
	}
	return len(runes), string(runes[from:]), ""
}

// indexRune returns the rune position of needle in haystack
// starting from rune index `from`, or -1 if not found. Mirrors
// `text.find(s, index)` from Python.
func indexRune(haystack []rune, from int, needle []rune) int {
	if len(needle) == 0 {
		return from
	}
	for i := from; i+len(needle) <= len(haystack); i++ {
		match := true
		for j := 0; j < len(needle); j++ {
			if haystack[i+j] != needle[j] {
				match = false
				break
			}
		}
		if match {
			return i
		}
	}
	return -1
}

// dsv4InternalToolCall is the in-memory representation of a parsed
// tool call before conversion to OpenAI format. Fields match
// decode_dsml_to_arguments' return shape.
type dsv4InternalToolCall struct {
	Name      string
	Arguments string
}

// toolCallsToOpenAI converts internal tool calls to OpenAI wire
// format. Mirrors tool_calls_to_openai_format.
func toolCallsToOpenAI(calls []dsv4InternalToolCall) []OpenAIToolCall {
	out := make([]OpenAIToolCall, 0, len(calls))
	for _, c := range calls {
		out = append(out, OpenAIToolCall{
			ID:    fmt.Sprintf("call_%s", random.GetUUID()),
			Type:  "function",
			Function: OpenAIToolCallFunction{
				Name:      c.Name,
				Arguments: c.Arguments,
			},
		})
	}
	return out
}

// dsv4DecodeArguments serialises the (key, value, isString) tuples
// to a JSON object string. The deepseek reference wraps string
// values with json.dumps; non-string values are kept as JSON. We
// follow the same rule: if isString is "true", json.Marshal the
// string (which produces a quoted JSON string); otherwise the
// value is already a Go any and json.Marshal will render it as
// its native JSON form.
func dsv4DecodeArguments(toolName string, args map[string]dsv4Param) dsv4InternalToolCall {
	if len(args) == 0 {
		return dsv4InternalToolCall{
			Name:      toolName,
			Arguments: "{}",
		}
	}
	obj := make(map[string]json.RawMessage, len(args))
	for k, p := range args {
		if p.IsString {
			b, _ := json.Marshal(p.Value)
			obj[k] = b
		} else {
			switch v := p.Value.(type) {
			case json.RawMessage:
				obj[k] = v
			case []byte:
				obj[k] = v
			case string:
				obj[k] = json.RawMessage(v)
			default:
				b, _ := json.Marshal(v)
				obj[k] = b
			}
		}
	}
	b, err := json.Marshal(obj)
	if err != nil {
		return dsv4InternalToolCall{Name: toolName, Arguments: "{}"}
	}
	return dsv4InternalToolCall{Name: toolName, Arguments: string(b)}
}

type dsv4Param struct {
	Value    any
	IsString bool
}

// parseMessageFromCompletionText is the Go port of the Python
// parse_message_from_completion_text. See encoding_dsv4.py lines
// 687-744.
func parseMessageFromCompletionText(text, thinkingMode string) (ParsedMessage, error) {
	// The python ref hardcodes these as global constants; we look
	// them up here.
	dsmlToken := dsv4DSML
	tcBlockName := dsv4TCBlockName
	thinkingEndToken := dsv4ThinkEnd
	eosToken := dsv4EOS
	toolCallsStartToken := dsv4TCBlockStart

	isThinking := thinkingMode == "thinking"
	isToolCalling := false

	summaryContent, reasoningContent, toolCalls := "", "", []dsv4InternalToolCall{}
	index := 0
	var stopToken string

	if isThinking {
		var contentDelta string
		var err error
		index, contentDelta, stopToken, err = readUntilStopOrError(text, index, []string{thinkingEndToken, toolCallsStartToken})
		if err != nil {
			return ParsedMessage{}, err
		}
		reasoningContent = contentDelta
		if stopToken != thinkingEndToken {
			return ParsedMessage{}, fmt.Errorf("invalid thinking format: missing </think>")
		}
	}

	var contentDelta string
	var err error
	index, contentDelta, stopToken, err = readUntilStopOrError(text, index, []string{eosToken, toolCallsStartToken})
	if err != nil {
		return ParsedMessage{}, err
	}
	summaryContent = contentDelta
	if stopToken == toolCallsStartToken {
		isToolCalling = true
	} else if stopToken != eosToken {
		return ParsedMessage{}, fmt.Errorf("invalid format: missing EOS token")
	}

	if isToolCalling {
		var tcs []dsv4InternalToolCall
		index, stopToken, tcs, err = parseDSV4ToolCalls(index, text, dsmlToken, tcBlockName)
		if err != nil {
			return ParsedMessage{}, err
		}
		toolCalls = tcs

		var toolEndsText string
		index, toolEndsText, stopToken, err = readUntilStopOrError(text, index, []string{eosToken})
		if err != nil {
			return ParsedMessage{}, err
		}
		if toolEndsText != "" {
			return ParsedMessage{}, fmt.Errorf("unexpected content after tool calls: %q", toolEndsText)
		}
	}

	// readUntilStop operates in runes, so compare against the rune
	// length of the text to detect "we hit the end".
	if index != len([]rune(text)) || (stopToken != eosToken && stopToken != "") {
		return ParsedMessage{}, fmt.Errorf("unexpected content at end: index=%d len=%d stopToken=%q", index, len([]rune(text)), stopToken)
	}

	// Sanity: the deepseek ref asserts no special tokens leak into
	// summary/reasoning content.
	for _, sp := range []string{dsv4BOS, dsv4EOS, "<think>", dsv4ThinkEnd, dsmlToken} {
		if strings.Contains(summaryContent, sp) || strings.Contains(reasoningContent, sp) {
			return ParsedMessage{}, fmt.Errorf("unexpected special token %q in content", sp)
		}
	}

	return ParsedMessage{
		Role:             "assistant",
		Content:          summaryContent,
		ReasoningContent: reasoningContent,
		ToolCalls:        toolCallsToOpenAI(toolCalls),
	}, nil
}

// readUntilStopOrError is a small wrapper that returns an error
// when no stop token is found. The Python ref uses an assert here.
func readUntilStopOrError(text string, from int, stops []string) (int, string, string, error) {
	newIdx, content, stop := readUntilStop(from, text, stops)
	if stop == "" {
		return 0, "", "", fmt.Errorf("missing one of stop tokens %v at index %d in text", stops, from)
	}
	return newIdx, content, stop, nil
}

// parseDSV4ToolCalls parses a sequence of <｜DSML｜invoke>...</｜DSML｜invoke>
// blocks from text starting at `index`. Returns the new index, the
// last matched stop token, and the parsed tool calls. Mirrors
// parse_tool_calls.
func parseDSV4ToolCalls(index int, text, dsmlToken, tcBlockName string) (int, string, []dsv4InternalToolCall, error) {
	var calls []dsv4InternalToolCall
	stopToken := ""
	// Stop tokens, mirroring the Python parser exactly. The Python
	// parser uses these literal strings in `text.find(stop, idx)`
	// calls:
	//   * `</｜DSML｜tool_calls>` is the OUTER close. The Python
	//     includes the trailing `>`.
	//   * `</｜DSML｜invoke` is the INNER close. The Python omits
	//     the trailing `>` (the closing `>` is consumed as part of
	//     the next stop or by the regex's `>$` anchor).
	//   * `<｜DSML｜invoke` is the inner OPENER. No `>`.
	//   * `<｜DSML｜parameter` is the parameter OPENER. No `>`.
	//   * `/｜DSML｜parameter` is the parameter CLOSER. No `>`.
	// We replicate this exactly so the byte/char positions line up
	// with the Python reference and the golden test data.
	toolCallsEndToken := "</" + dsmlToken + tcBlockName + ">"
	invokeStartToken := "<" + dsmlToken + "invoke"
	invokeEndToken := "</" + dsmlToken + "invoke"
	parameterStartToken := "<" + dsmlToken + "parameter"
	parameterEndToken := "/" + dsmlToken + "parameter"

	// Python uses re.findall; in Go we use regexp.
	nameRegex := regexp.MustCompile(`(?s)^\s*name="(.*?)">\n$`)
	paramRegex := regexp.MustCompile(`(?s)^ name="(.*?)" string="(true|false)">(.*?)<$`)

	for index < len([]rune(text)) {
		newIdx, content, stop := readUntilStop(index, text, []string{invokeStartToken, toolCallsEndToken})
		index = newIdx
		// The python ref requires content == ">\n" (i.e. the chars
		// between the previous stop and the next stop must be ">\n").
		if content != ">\n" {
			return 0, "", nil, fmt.Errorf("tool call format error: expected '>\\n' but got %q", content)
		}
		if stop == toolCallsEndToken {
			stopToken = stop
			break
		}
		if stop == "" {
			return 0, "", nil, fmt.Errorf("missing special token in tool calls at index %d", index)
		}
		stopToken = stop

		// Read up to the first parameter or the end of the invoke.
		newIdx, toolNameContent, stop := readUntilStop(index, text, []string{parameterStartToken, invokeEndToken})
		index = newIdx
		m := nameRegex.FindStringSubmatch(toolNameContent)
		if len(m) != 2 {
			return 0, "", nil, fmt.Errorf("tool name format error: %q", toolNameContent)
		}
		toolName := m[1]

		args := map[string]dsv4Param{}
		for stop == parameterStartToken {
			var paramContent string
			newIdx, paramContent, _ = readUntilStop(index, text, []string{parameterEndToken})
			index = newIdx
			m := paramRegex.FindStringSubmatch(paramContent)
			if len(m) != 4 {
				return 0, "", nil, fmt.Errorf("parameter format error: %q", paramContent)
			}
			paramName, isString, rawValue := m[1], m[2] == "true", m[3]

			if _, dup := args[paramName]; dup {
				return 0, "", nil, fmt.Errorf("duplicate parameter name: %q", paramName)
			}

			// The python ref puts (value, isString) into the args
			// map. decode_dsml_to_arguments wraps string values
			// with json.dumps (so they end up as quoted JSON
			// strings). We do the same: store the raw value and
			// the isString flag, then dsv4DecodeArguments handles
			// the JSON encoding.
			if isString {
				args[paramName] = dsv4Param{Value: rawValue, IsString: true}
			} else {
				// Try to parse as JSON. Fall back to rawValue if
				// it isn't valid JSON.
				var decoded any
				if err := json.Unmarshal([]byte(rawValue), &decoded); err == nil {
					args[paramName] = dsv4Param{Value: decoded, IsString: false}
				} else {
					args[paramName] = dsv4Param{Value: rawValue, IsString: false}
				}
			}

			// After the parameter, the next char must be ">\n" then
			// either another parameter or the end of the invoke.
			var content string
			newIdx, content, stop = readUntilStop(index, text, []string{parameterStartToken, invokeEndToken})
			index = newIdx
			if content != ">\n" {
				return 0, "", nil, fmt.Errorf("parameter format error: expected '>\\n' but got %q", content)
			}
		}

		calls = append(calls, dsv4DecodeArguments(toolName, args))
	}
	return index, stopToken, calls, nil
}

package xunfei

import (
	"crypto/hmac"
	"crypto/sha256"
	"crypto/tls"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"

	"github.com/songquanpeng/one-api/common"
	"github.com/songquanpeng/one-api/common/config"
	"github.com/songquanpeng/one-api/common/helper"
	"github.com/songquanpeng/one-api/common/logger"
	"github.com/songquanpeng/one-api/common/random"
	"github.com/songquanpeng/one-api/relay/adaptor/openai"
	"github.com/songquanpeng/one-api/relay/constant"
	"github.com/songquanpeng/one-api/relay/meta"
	"github.com/songquanpeng/one-api/relay/model"
)

// https://console.xfyun.cn/services/cbm
// https://www.xfyun.cn/doc/spark/Web.html

func requestOpenAI2Xunfei(request model.GeneralOpenAIRequest, xunfeiAppId string, domain string) *ChatRequest {
	messages := make([]Message, 0, len(request.Messages)+1)
	if len(request.Tools) > 0 {
		// Inject the ds2api-style tool-call format instructions as
		// the FIRST system message. ds2api's prompt is the de-facto
		// reference for getting DeepSeek-class models to emit
		// machine-parseable tool calls on gateways that don't do
		// the conversion for us; the Xunfei WSS path needs it
		// because the gateway hands back raw text. The prompt uses
		// halfwidth pipes (|) which is also the form our parser
		// now canonicalises to (normalizeDSMLPipe folds the
		// fullwidth ｜ form some DeepSeek-V3 outputs use).
		messages = append(messages, Message{
			Role:        "system",
			Content:     dsmlToolCallInstructions(toolNamesOf(request.Tools)),
			ContentType: "text",
		})
	}
	for _, message := range request.Messages {
		messages = append(messages, Message{
			Role:        message.Role,
			Content:     message.StringContent(),
			ContentType: "text",
		})
	}
	xunfeiRequest := ChatRequest{}
	xunfeiRequest.Header.AppId = xunfeiAppId
	xunfeiRequest.Header.TraceId = random.GetUUID()
	xunfeiRequest.Header.Mode = 0
	if config.XunfeiDomain != "" {
		xunfeiRequest.Parameter.Chat.Domain = config.XunfeiDomain
	} else {
		xunfeiRequest.Parameter.Chat.Domain = domain
	}
	xunfeiRequest.Parameter.Chat.Temperature = request.Temperature
	// Don't set top_k. With top_k=1 the model becomes deterministic
	// (greedy) regardless of temperature, which causes it to get
	// stuck emitting repeated tokens and to ignore tool schemas.
	// We leave the field at the Go zero value (0) and "omitempty"
	// drops it from the JSON, so the gateway uses its default
	// sampling strategy. The right knob for randomness is
	// temperature, not top_k.
	xunfeiRequest.Parameter.Chat.MaxTokens = request.MaxTokens
	if config.XunfeiContextEnabled != "" {
		b := strings.ToLower(config.XunfeiContextEnabled) == "true"
		xunfeiRequest.Parameter.Chat.ContextEnabled = &b
	}
	xunfeiRequest.Payload.SessionId = ""
	xunfeiRequest.Payload.Message.Text = messages

	if len(request.Tools) > 0 && (config.XunfeiDomain != "" || strings.HasPrefix(xunfeiRequest.Parameter.Chat.Domain, "generalv3") || xunfeiRequest.Parameter.Chat.Domain == "4.0Ultra") {
		tools := make([]model.Tool, len(request.Tools))
		for i, tool := range request.Tools {
			tools[i] = tool
		}
		xunfeiRequest.Payload.Tools = tools
	}

	return &xunfeiRequest
}

func getToolCalls(response *ChatResponse) []model.Tool {
	var toolCalls []model.Tool
	if len(response.Payload.Choices.Text) == 0 {
		return toolCalls
	}
	item := response.Payload.Choices.Text[0]
	if item.FunctionCall != nil {
		toolCall := model.Tool{
			Id:       fmt.Sprintf("call_%s", random.GetUUID()),
			Type:     "function",
			Function: *item.FunctionCall,
		}
		toolCalls = append(toolCalls, toolCall)
		return toolCalls
	}
	return extractXMLToolCalls(item.Content)
}

// knownXMLToolNames are tag names we recognise as tool invocations. The
// model may emit tool tags it invented (e.g. <bash>); we accept any tag
// matching the simple state-machine extractor and emit a function_call
// whose name is the tag itself.
var knownXMLToolNames = map[string]bool{
	"read_file":     true,
	"read":          true,
	"write_file":    true,
	"write":         true,
	"edit_file":     true,
	"edit":          true,
	"bash":          true,
	"shell":         true,
	"grep":          true,
	"glob":          true,
	"webfetch":      true,
	"webfetch_tool": true,
	"websearch":     true,
	"todowrite":     true,
	"task":          true,
	"notebookedit":  true,
	"tool_use":      true,
	"function_call": true,
	"invoke":        true,
	"tool":          true,
	"function":      true,
}

// dsmlMarkers are the literal tokens that bracket DSML tool-call blocks
// emitted by DeepSeek models: <|DSML|invoke name="X"> ... </|DSML|invoke>.
// The "|" character is halfwidth (U+007C) in some model outputs and
// fullwidth (U+FF5C, "｜") in others. We define the halfwidth form
// here as the canonical prefix and pre-normalise incoming text via
// normalizeDSMLPipe so the parser matches either form.
const (
	dsmlOpenPrefix  = "<|DSML|"
	dsmlClosePrefix = "</|DSML|"
)

// normalizeDSMLPipe rewrites the fullwidth vertical bar (U+FF5C,
// used by some DeepSeek-V3 outputs) to the halfwidth pipe (U+007C,
// the form the ds2api DSML prompt instructs the model to emit and
// the form our prompt injection below uses). This is a string-wide
// rewrite because the pipe can appear anywhere in the markup.
func normalizeDSMLPipe(s string) string {
	return strings.ReplaceAll(s, "｜", "|")
}

// normalizeMalformedDSMLTags repairs a few common malformations
// the Xunfei WSS gateway produces when the upstream DeepSeek
// tokeniser drops the closing ">" of a DSML tag. The malformed
// forms seen in production (oneapi-2026.06.23-10.49.log, seq 10-58):
//
//   * <|DSML|tool_calls|\n   →  <|DSML|tool_calls>\n
//     (the trailing "|" is a typo; the model wrote the
//     wrapper open as if it were a close tag — drop the "|" and
//     close the open tag with ">")
//   * <|DSML|invoke name="X"  →  <|DSML|invoke name="X">
//   * <|DSML|parameter name="X" ...value  →
//     <|DSML|parameter name="X">...value
//   * </|DSML|tool_calls|>  →  </|DSML|tool_calls>
//     (the trailing "|" is a typo; the model wrote the
//     close tag with a duplicate "|" — drop the extra "|")
//
// Without this fix the streaming parser sees "<|DSML|tool_calls|"
// and treats it as literal text (no matching ">"), so the
// tool-call opener is never recognised and the entire block is
// emitted as visible prose. The function is a no-op on
// well-formed DSML; only malformed tags are touched.
//
// The repair is intentionally conservative — it only fires when
// the model emits the canonical opener prefix followed by a
// name attribute (or the wrapper open with whitespace/newline/EOF
// after it) and a ">" is NOT already present. This avoids
// touching any user-content strings.
func normalizeMalformedDSMLTags(s string) string {
	if !strings.Contains(s, dsmlOpenPrefix) {
		return s
	}
	// Wrapper open: <|DSML|tool_calls| ... → <|DSML|tool_calls>...
	s = repairDSMLOpenTag(s, "<|DSML|tool_calls")
	// Invoke open: <|DSML|invoke name="X"  →  <|DSML|invoke name="X">
	s = repairDSMLOpenTag(s, "<|DSML|invoke")
	// Parameter open: <|DSML|parameter name="X" ...value →
	//                 <|DSML|parameter name="X">...value
	s = repairDSMLOpenTag(s, "<|DSML|parameter")
	// Wrapper close: </|DSML|tool_calls|>  →  </|DSML|tool_calls>
	// (extra trailing pipe)
	s = strings.ReplaceAll(s, "</|DSML|tool_calls|>", "</|DSML|tool_calls>")
	return s
}

// repairDSMLOpenTag finds the next occurrence of opener in s that
// does NOT already have a closing ">" and inserts one. Returns s
// unchanged if opener is not present, all occurrences are
// well-formed, or the tag opener is incomplete (the model is
// mid-token; inserting ">" now would corrupt the attribute).
//
// The malformed forms seen in production:
//   * "<|DSML|tool_calls|"  →  "<|DSML|tool_calls>"  (trailing |)
//   * "<|DSML|tool_calls|\n"  →  "<|DSML|tool_calls>\n"  (trailing | + newline)
//   * "<|DSML|invoke name=\"X\""  →  "<|DSML|invoke name=\"X\">"
//     (no closing >, the model finished the attribute string)
//   * "<|DSML|parameter name=\"X\" string=\"true\"" (no >)  →
//     "<|DSML|parameter name=\"X\" string=\"true\">"
//   * "<|DSML|parameter name=\"X\"<value>" (no >, no space
//     between attribute and value)  →
//     "<|DSML|parameter name=\"X\"><value>"  (the value is the
//     next thing after the closing quote of the last attribute)
//
// For "in flight" cases (the buffer holds a tag opener that
// hasn't yet seen its closing >), we leave it alone — the
// streaming buffer will accumulate more bytes on the next frame
// and the repair will run again with the complete tag.
//
// "Well-formed" check: a tag with a ">" anywhere inside it is
// left alone (we don't re-pair attributes).
//
// "In-quote" tracking: while inside a double-quoted attribute
// value, whitespace and "|" are part of the value and don't end
// the tag. This prevents "<|DSML|parameter name=\"a b\">" from
// being mis-repaired at the space inside the quoted value.
//
// Insertion-point selection: when the tag is incomplete (no
// closing ">"), we walk forward tracking quote state. The
// insertion point is the FIRST of:
//   - A `|` (typo for `>`): drop the `|`, insert `>` there.
//   - An unquoted whitespace: insert `>` there IF we've seen
//     a closing `"` (i.e., the last attribute is complete).
//   - End of buffer: append `>` at the end IF we've seen a
//     closing `"` (i.e., the last attribute is complete).
//   - Otherwise, the tag opener is incomplete (we haven't seen
//     any attribute values complete yet); leave it for the
//     next frame.
func repairDSMLOpenTag(s, opener string) string {
	for {
		idx := strings.Index(s, opener)
		if idx < 0 {
			return s
		}
		end := idx + len(opener)
		// Well-formed check: if there's a ">" anywhere after the
		// opener BEFORE the next "<", the tag is complete;
		// advance past it and look for the next opener. The
		// "before next <" guard is what makes us correctly
		// identify an opener like
		// "<|DSML|parameter name=\"X\"<value></|DSML|parameter>"
		// as MALFORMED — its `>` belongs to the close tag,
		// not to the open tag, so we need to insert one for
		// the open tag first.
		closeIdx := strings.Index(s[end:], ">")
		openIdx := strings.Index(s[end:], "<")
		if closeIdx >= 0 && (openIdx < 0 || closeIdx < openIdx) {
			s = s[:end+closeIdx+1] + repairDSMLOpenTag(s[end+closeIdx+1:], opener)
			return s
		}
		// No ">". Walk forward tracking quote state, looking for
		// the end of the tag opener.
		//
		// The end is the first unquoted terminator (whitespace or
		// "|") AFTER the last complete attribute (i.e. after
		// we've seen a closing `"`). Terminators BEFORE the first
		// closing `"` are between the opener and the first
		// attribute (or between attributes), and don't end the
		// tag.
		absClose, inQuote, lastQuoteClose := -1, false, -1
		for j := end; j < len(s); j++ {
			c := s[j]
			if c == '"' {
				inQuote = !inQuote
				if !inQuote {
					lastQuoteClose = j
					// Reset absClose: any terminator seen
					// before this closing quote was inside
					// the attribute value (which we now know
					// is complete). The next terminator
					// will be a real tag boundary.
					absClose = -1
				}
				continue
			}
			if inQuote {
				continue
			}
			// Before the first closing `"`, terminators are
			// between the opener and the first attribute, or
			// between attributes — don't treat them as the
			// end of the tag.
			if lastQuoteClose < 0 {
				continue
			}
			if c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '|' {
				absClose = j
				break
			}
		}
		// We need a complete last attribute (closing `"` seen) to
		// know where to insert ">". If we never saw a closing
		// `"`, the tag opener is incomplete — the model is still
		// emitting the attribute value (e.g. we have
		// "<|DSML|parameter name=\"" and the next frame may
		// complete it to "<|DSML|parameter name=\"command\"").
		// In that case, leave the buffer alone.
		if lastQuoteClose < 0 {
			return s
		}
		// Pick the insertion point:
		//   - If we found a `|`, drop it and insert `>` there
		//     (handles "<|DSML|tool_calls|").
		//   - Else if we found an unquoted whitespace, insert `>`
		//     there (handles "<|DSML|invoke name=\"X\" <rest>"
		//     where the whitespace is between attributes).
		//   - Else (no terminator at all, but the last attribute
		//     is closed): append ">" at the end of the buffer
		//     (handles "<|DSML|invoke name=\"X\"" arriving in
		//     a single chunk).
		if absClose < 0 {
			// Last attribute closed, no terminator, no ">".
			// The tag opener is complete-but-missing-">". Append
			// ">" at the end of the buffer.
			s = s + ">"
			return s
		}
		if s[absClose] == '|' {
			// Drop the "|", insert ">".
			s = s[:absClose] + ">" + s[absClose+1:]
		} else {
			s = s[:absClose] + ">" + s[absClose:]
		}
		s = s[:absClose+1] + repairDSMLOpenTag(s[absClose+1:], opener)
		return s
	}
}

// extractXMLToolCalls parses the given content for fully-closed XML/DSML
// blocks that look like a tool invocation, and returns them as
// model.Tool slices. DSML blocks are emitted by DeepSeek models and look
// like <|DSML|invoke name="X"> ... <|DSML|parameter name="k">v</|DSML|parameter> ... </|DSML|invoke>;
// they are checked first because DeepSeek models reliably wrap tool
// invocations in DSML. Plain <tag>...</tag> blocks from other models are
// still picked up as a fallback.
func extractXMLToolCalls(content string) []model.Tool {
	content = normalizeDSMLPipe(content)
	content = normalizeMalformedDSMLTags(content)
	if toolCalls := extractDSMLToolCalls(content); len(toolCalls) > 0 {
		return toolCalls
	}
	_, toolCalls := extractCompleteBuffer(content)
	return toolCalls
}

// extractDSMLToolCalls scans content for one or more fully-closed
// <｜DSML｜invoke name="X"> ... </｜DSML｜invoke> blocks and converts
// each into a model.Tool. The arguments field is a JSON object built
// from the inner <｜DSML｜parameter name="k">value</｜DSML｜parameter>
// children; values are kept as strings since DSML doesn't carry types.
func extractDSMLToolCalls(content string) []model.Tool {
	var toolCalls []model.Tool
	for {
		openIdx := strings.Index(content, dsmlOpenPrefix+"invoke ")
		if openIdx < 0 {
			break
		}
		gt := strings.Index(content[openIdx:], ">")
		if gt < 0 {
			break
		}
		openTag := content[openIdx : openIdx+gt+1]
		toolName := extractDSMLAttr(openTag, "name")
		innerStart := openIdx + gt + 1
		closing := dsmlClosePrefix + "invoke>"
		closeIdx := strings.Index(content[innerStart:], closing)
		if closeIdx < 0 {
			break
		}
		inner := content[innerStart : innerStart+closeIdx]
		args := extractDSMLParameters(inner)
		toolCalls = append(toolCalls, model.Tool{
			Id:   fmt.Sprintf("call_%s", random.GetUUID()),
			Type: "function",
			Function: model.Function{
				Name:      toolName,
				Arguments: argumentsToJSONString(args),
			},
		})
		content = content[innerStart+closeIdx+len(closing):]
	}
	return toolCalls
}

// extractDSMLAttr returns the value of the named attribute in an opening
// DSML tag, e.g. extractDSMLAttr(`<｜DSML｜invoke name="Read">`, "name")
// returns "Read". Returns "" when the attribute is missing.
func extractDSMLAttr(tag, attr string) string {
	needle := attr + "=\""
	idx := strings.Index(tag, needle)
	if idx < 0 {
		return ""
	}
	start := idx + len(needle)
	end := strings.Index(tag[start:], "\"")
	if end < 0 {
		return ""
	}
	return tag[start : start+end]
}

// extractDSMLParameters walks the inner text of a <｜DSML｜invoke> block
// and returns a map of parameter name to its decoded value. The
// `string="true|false"` attribute on each <｜DSML｜parameter> tag
// determines the value type:
//   - string="true": the value is taken as a raw string.
//   - string="false": the value is JSON (number, boolean, array,
//     or object) and is decoded into its native Go type.
//
// This matches the behaviour of decode_dsml_to_arguments in the
// DeepSeek-V4 reference encoder
// (D:/workspace/DeepSeek-V4-Flash/encoding/encoding_dsv4.py).
func extractDSMLParameters(inner string) map[string]any {
	args := map[string]any{}
	for {
		openIdx := strings.Index(inner, dsmlOpenPrefix+"parameter ")
		if openIdx < 0 {
			break
		}
		gt := strings.Index(inner[openIdx:], ">")
		if gt < 0 {
			break
		}
		openTag := inner[openIdx : openIdx+gt+1]
		name := extractDSMLAttr(openTag, "name")
		stringAttr := extractDSMLAttr(openTag, "string")
		innerStart := openIdx + gt + 1
		closing := dsmlClosePrefix + "parameter>"
		closeIdx := strings.Index(inner[innerStart:], closing)
		if closeIdx < 0 {
			break
		}
		rawValue := strings.TrimSpace(inner[innerStart : innerStart+closeIdx])
		if stringAttr == "false" {
			var decoded any
			if err := json.Unmarshal([]byte(rawValue), &decoded); err == nil {
				args[name] = decoded
			} else {
				// Malformed JSON: fall back to the raw text so the
				// tool still receives something usable.
				args[name] = rawValue
			}
		} else {
			// Default to string, matching the deepseek reference
			// which wraps string values with json.dumps.
			args[name] = rawValue
		}
		inner = inner[innerStart+closeIdx+len(closing):]
	}
	return args
}

// argumentsToJSONString serialises a parsed argument map to a JSON
// string in the format expected by the OpenAI Chat Completions
// spec (https://platform.openai.com/docs/api-reference/chat/object):
// `function.arguments` is a JSON-encoded string, not an object. The
// deepseek reference uses json.dumps(...) for the same purpose.
func argumentsToJSONString(args map[string]any) string {
	if len(args) == 0 {
		return "{}"
	}
	b, err := json.Marshal(args)
	if err != nil {
		return "{}"
	}
	return string(b)
}

// extractDSMLFromBuffer walks buf, peels out fully-closed DSML
// <｜DSML｜invoke ...>...</｜DSML｜invoke> blocks as tool calls, and
// returns the prose in between. When holdTail is true an unclosed
// opening tag at the end of the buffer is preserved as the tail so the
// streaming layer can carry it forward into the next chunk.
//
// Both the leading <｜DSML｜tool_calls> wrapper and the trailing
// </｜DSML｜tool_calls> close are stripped from the visible prose.
func extractDSMLFromBuffer(buf string, holdTail bool) (string, []model.Tool, string) {
	var (
		visible   strings.Builder
		toolCalls []model.Tool
		i         = 0
	)
	for i < len(buf) {
		openIdx := strings.Index(buf[i:], dsmlOpenPrefix+"invoke ")
		if openIdx == -1 {
			// No more invoke blocks. The remainder may be a
			// closing wrapper ("</|DSML|tool_calls>" plus a
			// trailing newline) — strip the complete portion and
			// hold any partial prefix as the tail so the next
			// chunk can complete the tag.
			visible.WriteString(stripTrailingDSMLWrapper(buf[i:]))
			i = len(buf)
			break
		}
		absOpen := i + openIdx
		// Drop any trailing "<|DSML|tool_calls>" wrapper (or any
		// other DSML-tagged structural marker that isn't the
		// opening of an invoke) from the visible prose. The
		// wrapper is just a container and has no user-facing
		// content.
		visible.WriteString(stripTrailingDSMLWrapper(buf[i:absOpen]))
		gt := strings.Index(buf[absOpen:], ">")
		if gt < 0 {
			// No closing > yet; hold the tail.
			if holdTail {
				return visible.String(), toolCalls, buf[absOpen:]
			}
			visible.WriteString(buf[absOpen:])
			return visible.String(), toolCalls, ""
		}
		openTag := buf[absOpen : absOpen+gt+1]
		toolName := extractDSMLAttr(openTag, "name")
		innerStart := absOpen + gt + 1
		closing := dsmlClosePrefix + "invoke>"
		closeIdx := strings.Index(buf[innerStart:], closing)
		if closeIdx < 0 {
			if holdTail {
				return visible.String(), toolCalls, buf[absOpen:]
			}
			visible.WriteString(buf[absOpen:])
			return visible.String(), toolCalls, ""
		}
		inner := buf[innerStart : innerStart+closeIdx]
		args := extractDSMLParameters(inner)
		toolCalls = append(toolCalls, model.Tool{
			Id:   fmt.Sprintf("call_%s", random.GetUUID()),
			Type: "function",
			Function: model.Function{
				Name:      toolName,
				Arguments: argumentsToJSONString(args),
			},
		})
		i = innerStart + closeIdx + len(closing)
	}
	// If the tail after the last invoke (i.e. buf[i:]) is a
	// partial closing wrapper ("</|DSML|tool_calls>" arriving in
	// pieces), hold it so the next chunk can complete the tag and
	// the stripTrailingDSMLWrapper call above can drop it from
	// visible prose. Without this, "</" or "</|DSML" would leak
	// into the visible text.
	if holdTail {
		if _, ok := findTrailingPartialDSMLClose(buf[i:]); ok {
			if closeStart := strings.LastIndex(buf[i:], "</"); closeStart >= 0 {
				return visible.String(), toolCalls, buf[i+closeStart:]
			}
		}
	}
	return visible.String(), toolCalls, ""
}

// findTrailingPartialDSMLClose returns (true) when buf ends with a
// non-empty prefix of "</|DSML|tool_calls>" (or any other
// </|DSML|... close) that has not yet seen its closing ">". Used
// to detect when a DSML closing wrapper is still arriving across
// WSS frames.
func findTrailingPartialDSMLClose(buf string) (int, bool) {
	// The longest meaningful close we care about is the wrapper
	// itself, "</|DSML|tool_calls>".
	wrapper := dsmlClosePrefix + "tool_calls>"
	// Walk back from the end, looking for a partial prefix that
	// starts with "</". The minimum is "</" (the partial close
	// start) and the maximum is the full wrapper.
	max := len(wrapper)
	if max > len(buf) {
		max = len(buf)
	}
	for length := 2; length <= max; length++ {
		tail := buf[len(buf)-length:]
		if !strings.HasPrefix(tail, "</") {
			break
		}
		if isDSMLClosePrefixOrPartial(tail) && length < len(wrapper) {
			return length, true
		}
	}
	return 0, false
}

// isDSMLClosePrefixOrPartial returns true if s is a non-empty
// prefix of any </|DSML|... close tag, OR s is the full wrapper
// "</|DSML|tool_calls>" (in which case it should be stripped, not
// held). We use the wrapper specifically because the streaming
// layer otherwise has no way to know which close tag is
// incoming.
func isDSMLClosePrefixOrPartial(s string) bool {
	wrapper := dsmlClosePrefix + "tool_calls>"
	if s == wrapper {
		// Full wrapper; not partial.
		return false
	}
	return strings.HasPrefix(wrapper, s)
}

// looksLikeToolCall applies a heuristic for tags we did not pre-register:
// if the inner content has key=value pairs or looks like JSON, accept it
// as a tool invocation. This catches model-invented tool names without
// being too eager to misclassify prose.
func looksLikeToolCall(tag, inner string) bool {
	innerLower := strings.ToLower(inner)
	if strings.HasPrefix(innerLower, "{") && strings.HasSuffix(innerLower, "}") {
		return true
	}
	if strings.Contains(inner, "=") && strings.Contains(inner, "\n") {
		return true
	}
	return false
}

func responseXunfei2OpenAI(response *ChatResponse) *openai.TextResponse {
	if len(response.Payload.Choices.Text) == 0 {
		response.Payload.Choices.Text = []ChatResponseTextItem{
			{
				Content: "",
			},
		}
	}
	toolCalls := getToolCalls(response)
	content := response.Payload.Choices.Text[0].Content
	if len(toolCalls) > 0 {
		content = stripXMLToolCalls(content)
	}
	finishReason := constant.StopFinishReason
	if len(toolCalls) > 0 {
		toolFinish := "tool_calls"
		finishReason = toolFinish
	}
	choice := openai.TextResponseChoice{
		Index: 0,
		Message: model.Message{
			Role:      "assistant",
			Content:   content,
			ToolCalls: toolCalls,
		},
		FinishReason: finishReason,
	}
	fullTextResponse := openai.TextResponse{
		Id:      fmt.Sprintf("chatcmpl-%s", random.GetUUID()),
		Object:  "chat.completion",
		Created: helper.GetTimestamp(),
		Choices: []openai.TextResponseChoice{choice},
		Usage:   response.Payload.Usage.Text,
	}
	return &fullTextResponse
}

// stripXMLToolCalls removes all fully-closed XML tool blocks from the
// content, leaving only the prose that surrounds them.
func stripXMLToolCalls(content string) string {
	visible, _ := extractCompleteBuffer(content)
	return strings.TrimSpace(visible)
}

func streamResponseXunfei2OpenAI(xunfeiResponse *ChatResponse, buf *streamXMLBuffer) *openai.ChatCompletionsStreamResponse {
	if len(xunfeiResponse.Payload.Choices.Text) == 0 {
		xunfeiResponse.Payload.Choices.Text = []ChatResponseTextItem{
			{
				Content: "",
			},
		}
	}
	var choice openai.ChatCompletionsStreamResponseChoice
	rawContent := xunfeiResponse.Payload.Choices.Text[0].Content
	visibleText, xmlToolCalls := buf.consume(rawContent)
	choice.Delta.Content = visibleText
	structToolCalls := getToolCalls(xunfeiResponse)
	allToolCalls := append(structToolCalls, xmlToolCalls...)
	// On end-of-stream, drain any tool calls the buffer was holding
	// in its tail (e.g. a tool block whose closing tag was cut off
	// when the upstream stopped sending).
	if xunfeiResponse.Payload.Choices.Status == 2 {
		allToolCalls = append(allToolCalls, buf.flush()...)
	}
	if len(allToolCalls) > 0 {
		choice.Delta.ToolCalls = allToolCalls
	}
	if xunfeiResponse.Payload.Choices.Status == 2 {
		finishReason := constant.StopFinishReason
		if len(choice.Delta.ToolCalls) > 0 {
			finishReason = "tool_calls"
		}
		choice.FinishReason = &finishReason
	}
	response := openai.ChatCompletionsStreamResponse{
		Id:      fmt.Sprintf("chatcmpl-%s", random.GetUUID()),
		Object:  "chat.completion.chunk",
		Created: helper.GetTimestamp(),
		Model:   "SparkDesk",
		Choices: []openai.ChatCompletionsStreamResponseChoice{choice},
	}
	return &response
}

func buildXunfeiAuthUrl(hostUrl string, apiKey, apiSecret string) string {
	HmacWithShaToBase64 := func(algorithm, data, key string) string {
		mac := hmac.New(sha256.New, []byte(key))
		mac.Write([]byte(data))
		encodeData := mac.Sum(nil)
		return base64.StdEncoding.EncodeToString(encodeData)
	}
	ul, err := url.Parse(hostUrl)
	if err != nil {
		fmt.Println(err)
	}
	date := time.Now().UTC().Format(time.RFC1123)
	signString := []string{"host: " + ul.Host, "date: " + date, "GET " + ul.Path + " HTTP/1.1"}
	sign := strings.Join(signString, "\n")
	sha := HmacWithShaToBase64("hmac-sha256", sign, apiSecret)
	authUrl := fmt.Sprintf("hmac username=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"", apiKey,
		"hmac-sha256", "host date request-line", sha)
	authorization := base64.StdEncoding.EncodeToString([]byte(authUrl))
	v := url.Values{}
	v.Add("host", ul.Host)
	v.Add("date", date)
	v.Add("authorization", authorization)
	callUrl := hostUrl + "?" + v.Encode()
	return callUrl
}

func StreamHandler(c *gin.Context, meta *meta.Meta, textRequest model.GeneralOpenAIRequest, appId string, apiSecret string, apiKey string) (*model.ErrorWithStatusCode, *model.Usage) {
	domain, authUrl := getXunfeiAuthUrl(meta.Config.APIVersion, apiKey, apiSecret)
	dataChan, stopChan, err := xunfeiMakeRequest(textRequest, domain, authUrl, appId)
	if err != nil {
		return openai.ErrorWrapper(err, "xunfei_request_failed", http.StatusInternalServerError), nil
	}
	common.SetEventStreamHeaders(c)
	var usage model.Usage
	streamBuf := newStreamXMLBuffer()
	c.Stream(func(w io.Writer) bool {
		select {
		case xunfeiResponse := <-dataChan:
			usage.PromptTokens += xunfeiResponse.Payload.Usage.Text.PromptTokens
			usage.CompletionTokens += xunfeiResponse.Payload.Usage.Text.CompletionTokens
			usage.TotalTokens += xunfeiResponse.Payload.Usage.Text.TotalTokens
			response := streamResponseXunfei2OpenAI(&xunfeiResponse, streamBuf)
			jsonResponse, err := json.Marshal(response)
			// DEBUG: log the OpenAI SSE chunk so we can verify the
			// model is being converted to delta.tool_calls. The
			// [one-api VERSION] prefix matches the request/response
			// log format added in e7e43db / d5f5f55.
			if err == nil {
				logger.SysLog(fmt.Sprintf("xunfei openai chunk [one-api %s]: %s", common.Version, string(jsonResponse)))
			}
			if err != nil {
				logger.SysError("error marshalling stream response: " + err.Error())
				return true
			}
			c.Render(-1, common.CustomEvent{Data: "data: " + string(jsonResponse)})
			return true
		case <-stopChan:
			c.Render(-1, common.CustomEvent{Data: "data: [DONE]"})
			return false
		}
	})
	return nil, &usage
}

// streamXMLBuffer holds the "in-flight" content across WSS frames:
// any partial DSML prefix, full DSML block, or full plain XML tool
// block that started arriving but hasn't yet been emitted (either
// because the close tag hasn't arrived, or because the
// boundary-detection pass found a partial prefix at the end of the
// last chunk). The buffer runs extractFromBuffer on the combined
// (held + new chunk) content and keeps whatever the extractor
// returns as the new held tail. This guarantees that:
//   - DSML markup (partial or complete) is never emitted as
//     visible prose — it's always either parsed as a tool call
//     (when the close arrives) or held in the buffer for the next
//     chunk to disambiguate.
//   - Plain XML tool blocks (e.g. <tools>...</tools>) are
//     suppressed the same way DSML is, and parsed at end-of-stream
//     or when the close arrives.
//   - The extractor is the single source of truth for what counts
//     as "in-flight" content, so the buffer does not duplicate
//     that logic.
type streamXMLBuffer struct {
	held string
}

func newStreamXMLBuffer() *streamXMLBuffer {
	return &streamXMLBuffer{}
}

// flush is called when the upstream signals end-of-stream. It runs
// the extractor on the held tail in complete-buffer mode (no
// further chunks are coming) and returns any tool calls the buffer
// was holding. The visible text is discarded because the client has
// already seen everything up to the last frame.
func (b *streamXMLBuffer) flush() []model.Tool {
	if b.held == "" {
		return nil
	}
	// Parse the held tail in complete-buffer mode (holdTail=false)
	// so any unmatched opening tag (e.g. <tools> with no </tools>)
	// is processed in the same call rather than being held again.
	_, toolCalls := extractCompleteBuffer(b.held)
	b.held = ""
	return toolCalls
}

// consume appends the new chunk to the held tail, runs the
// extractor, and returns the visible text and any tool calls the
// extractor found in this chunk. The extractor's holdTail
// (extractFromBuffer with holdTail=true) is the single source of
// truth for what is safe to emit vs what must be held.
//
// We do a partial-DSML-prefix lookahead on the new chunk before
// running the extractor: the model tokeniser splits "<|DSML|..."
// across many frames (e.g. one frame ends with "<", the next is
// "|", then "DS", then "ML|..."). The extractor doesn't know
// about partial prefixes — it only sees the combined
// (held + chunk) content — so without this lookahead, a chunk
// ending in "<" would be emitted as visible text and the user
// would see DSML markup leaking in. The fix: if the chunk ends
// with a non-empty prefix of "<|DSML|", we hold the tail of the
// chunk and run the extractor on held + chunk[:splitIdx] only.
func (b *streamXMLBuffer) consume(chunk string) (visibleText string, toolCalls []model.Tool) {
	if chunk == "" {
		return "", nil
	}
	// Canonicalise fullwidth pipes (some DeepSeek-V3 outputs use ｜)
	// to halfwidth so the rest of the parser matches a single set
	// of constants.
	chunk = normalizeDSMLPipe(chunk)

	// Partial-prefix lookahead. findDSMLBoundary returns the byte
	// index where the trailing partial DSML prefix starts in
	// chunk, or -1 if there is no such prefix.
	if splitIdx := findDSMLBoundary(chunk); splitIdx >= 0 {
		// Combine held with the safe part of chunk, run the
		// extractor on that, and hold the partial prefix.
		combined := b.held + chunk[:splitIdx]
		combined = normalizeMalformedDSMLTags(combined)
		visibleText, toolCalls, b.held = extractFromBuffer(combined)
		// Append the partial prefix to whatever the extractor
		// left in b.held. The prefix is at most len(dsmlOpenPrefix)
		// bytes long, so this is cheap.
		b.held = b.held + chunk[splitIdx:]
		return visibleText, toolCalls
	}

	// No partial prefix in chunk. Run the extractor on the full
	// combined content, but first check if the COMBINED buffer
	// ends with a non-empty prefix of "<|DSML|" — the partial
	// prefix may be split across the boundary between b.held
	// (which ended with "<") and chunk (which started with "|").
	// Without this check, the extractor would see "<|" and emit
	// the "<" as visible prose (since "<" by itself isn't a tag
	// start), leaking DSML markup into the chat.
	combined := b.held + chunk
	combined = normalizeMalformedDSMLTags(combined)
	if partialStart, ok := findTrailingPartialDSML(combined); ok {
		// Run the extractor on everything BEFORE the partial
		// prefix, then hold the partial prefix in the buffer.
		safe := combined[:partialStart]
		visibleText, toolCalls, tail := extractFromBuffer(safe)
		b.held = tail + combined[partialStart:]
		return visibleText, toolCalls
	}
	visibleText, toolCalls, b.held = extractFromBuffer(combined)
	return visibleText, toolCalls
}

// findTrailingPartialDSML checks whether s ends with a non-empty
// prefix of dsmlOpenPrefix ("<|DSML|"). If so, returns the byte
// position of the start of that prefix and true. Otherwise returns
// (0, false). The model splits the marker across many WSS frames
// (e.g. one frame ends with "<", next is "|", next "DS", next
// "ML|"), so the buffer must hold any partial prefix until the
// full opener or a different `<...` tag is seen.
func findTrailingPartialDSML(s string) (int, bool) {
	prefix := dsmlOpenPrefix
	// The longest possible trailing prefix is the full prefix.
	// We look at the last len(prefix) bytes of s and see if any
	// suffix of s matches a prefix of `prefix`.
	max := len(prefix)
	if max > len(s) {
		max = len(s)
	}
	// Walk from longest to shortest trailing substring that
	// could be a partial DSML prefix. We need at least 1 char
	// (the leading "<") to match.
	for length := max; length >= 1; length-- {
		tail := s[len(s)-length:]
		if isDSMLPrefixOrPartial(tail) && length < len(prefix) {
			return len(s) - length, true
		}
		// Stop scanning once the first char doesn't match.
		if !isDSMLPrefixOrPartial(tail) {
			break
		}
	}
	return 0, false
}

// findDSMLBoundary returns the byte index in chunk where a partial
// DSML marker starts (and should be held for the next chunk), or
// -1 if no such partial marker exists. The model splits
// "<|DSML|..." into multiple WSS frames (e.g. "<|DS", "ML|",
// "invoke"), so we need to detect the trailing tail as a prefix of
// the canonical DSML marker — not just a bare "<".
func findDSMLBoundary(chunk string) int {
	prefixRunes := []rune(dsmlOpenPrefix)
	runes := []rune(chunk)
	if len(runes) < 2 {
		return -1
	}
	// Walk back from the end, checking the trailing runes for a
	// partial match against "<|DSML|". The longest possible tail
	// we can hold is the full prefix length (so chunk end at
	// "<|DS" — 4 runes — is held).
	maxLook := len(prefixRunes)
	for offset := 1; offset <= maxLook && offset <= len(runes); offset++ {
		tail := string(runes[len(runes)-offset:])
		if isDSMLPrefixOrPartial(tail) {
			// Make sure the position is rune-aligned, not byte-aligned.
			tailRunes := []rune(tail)
			return len(chunk) - len(string(tailRunes))
		}
	}
	return -1
}

// isDSMLPrefixOrPartial returns true if s is the start of a
// potential DSML marker — i.e. s is a prefix of "<|DSML|" (with
// either halfwidth or fullwidth pipes, though we canonicalise to
// halfwidth before calling). This covers "<", "<|", "<|D",
// "<|DS", ... all the way through "<|DSML|".
func isDSMLPrefixOrPartial(s string) bool {
	if s == "" || s[0] != '<' {
		return false
	}
	prefixRunes := []rune(dsmlOpenPrefix)
	sRunes := []rune(s)
	if len(sRunes) > len(prefixRunes) {
		return false
	}
	for i := 0; i < len(sRunes); i++ {
		if sRunes[i] != prefixRunes[i] {
			return false
		}
	}
	return true
}

// stripTrailingDSMLWrapper returns s with any trailing
// "<|DSML|tool_calls>" or other non-invoke DSML wrapper tag
// removed. The wrapper is a structural marker that the ds2api
// prompt instructs the model to emit and is not user-visible.
func stripTrailingDSMLWrapper(s string) string {
	end := len(s)
	// Trim trailing whitespace so "...<|DSML|tool_calls>\n" drops
	// the whole thing.
	for end > 0 && (s[end-1] == ' ' || s[end-1] == '\n' || s[end-1] == '\t' || s[end-1] == '\r') {
		end--
	}
	if end == 0 || s[end-1] != '>' {
		return s
	}
	gt := strings.LastIndex(s[:end], ">")
	openIdx := strings.LastIndex(s[:gt], "<")
	if openIdx < 0 {
		return s
	}
	tag := s[openIdx : gt+1]
	if !strings.HasPrefix(tag, dsmlOpenPrefix) {
		return s
	}
	// Keep "<|DSML|invoke ..." (the actual tool opener); drop
	// "<|DSML|tool_calls>" and other non-invoke structural tags.
	if strings.Contains(tag, "invoke ") {
		return s
	}
	return s[:openIdx]
}

// extractFromBuffer extracts complete XML tool-call blocks from buf and
// returns the prose in between, the parsed tool calls, and any trailing
// incomplete tail that should be carried into the next chunk.
func extractFromBuffer(buf string) (string, []model.Tool, string) {
	return extractFromBufferImpl(buf, true)
}

// extractCompleteBuffer is like extractFromBuffer but treats any
// unmatched opening tag as prose (since there is no "next chunk" to
// carry it forward into). Use for non-streaming responses where the
// full content is already accumulated.
func extractCompleteBuffer(buf string) (string, []model.Tool) {
	visible, toolCalls, _ := extractFromBufferImpl(buf, false)
	return visible, toolCalls
}

func extractFromBufferImpl(buf string, holdTail bool) (string, []model.Tool, string) {
	buf = normalizeDSMLPipe(buf)
	buf = normalizeMalformedDSMLTags(buf)
	// DSML blocks (DeepSeek tool calls) take priority: they start with
	// "<|DSML|" — a halfwidth pipe after the prompt we inject, but
	// some DeepSeek-V3 outputs use a fullwidth "｜" pipe. The
	// normalizeDSMLPipe call above folds the fullwidth form to
	// halfwidth so the rest of the parser can match either with
	// the same string constants.
	if strings.Contains(buf, dsmlOpenPrefix+"invoke ") {
		dsmlVisible, dsmlToolCalls, dsmlTail := extractDSMLFromBuffer(buf, holdTail)
		if len(dsmlToolCalls) > 0 {
			// Run the generic extractor on the leftover prose to catch
			// any plain <tag> blocks the model may also have emitted.
			extraVisible, extraCalls, extraTail := extractFromBufferImplInner(dsmlVisible, holdTail)
			return extraVisible, append(dsmlToolCalls, extraCalls...), dsmlTail + extraTail
		}
	}
	return extractFromBufferImplInner(buf, holdTail)
}

// extractFromBufferImplInner is the original state-machine extractor,
// factored out so the DSML path can chain into it on the leftover prose.
func extractFromBufferImplInner(buf string, holdTail bool) (string, []model.Tool, string) {
	var (
		visible   strings.Builder
		toolCalls []model.Tool
		i         = 0
	)
	for i < len(buf) {
		openIdx := strings.Index(buf[i:], "<")
		if openIdx == -1 {
			visible.WriteString(buf[i:])
			i = len(buf)
			break
		}
		absOpen := i + openIdx
		visible.WriteString(buf[i:absOpen])
		rest := buf[absOpen:]
		// If this '<' is the start of a DSML opening tag (e.g.
		// <｜DSML｜invoke ...> or <｜DSML｜tool_calls>), hold the
		// tail so the DSML extractor (called by the outer
		// extractFromBufferImpl) gets a chance to consume it on a
		// future iteration.
		if strings.HasPrefix(rest, dsmlOpenPrefix) {
			if holdTail {
				return visible.String(), toolCalls, buf[absOpen:]
			}
			visible.WriteString(rest)
			i = len(buf)
			return visible.String(), toolCalls, ""
		}
		// If this '<' is the start of a DSML closing tag (e.g.
		// </｜DSML｜invoke> or </｜DSML｜tool_calls>), drop the
		// whole tag from the visible text since it's a structural
		// marker, not content. We don't try to track in-flight
		// DSML closes here (the in-flight open check at the top
		// of consume handles that); this branch just cleans up
		// the wrapper close that arrives after the inner block
		// has been extracted.
		if strings.HasPrefix(rest, dsmlClosePrefix) {
			closeEnd := strings.Index(rest, ">")
			if closeEnd < 0 {
				if holdTail {
					return visible.String(), toolCalls, buf[absOpen:]
				}
				visible.WriteString(rest)
				i = len(buf)
				break
			}
			i = absOpen + closeEnd + 1
			continue
		}
		// Partial closing DSML wrapper ("</" or "</|DSML" or
		// "</|DSML|tool_calls" without the closing ">"): hold
		// the tail so the next chunk can complete it. Without
		// this, the </|DSML|tool_calls> wrapper close arriving
		// in pieces would leak into the visible prose.
		if strings.HasPrefix(rest, "</") {
			if isDSMLClosePrefixOrPartial(rest) {
				if holdTail {
					return visible.String(), toolCalls, buf[absOpen:]
				}
				// Complete-buffer mode: the partial close
				// never got completed, so emit it as visible.
				visible.WriteString(rest)
				i = len(buf)
				break
			}
		}
		if !isXMLTagStart(rest) {
			visible.WriteByte('<')
			i = absOpen + 1
			continue
		}
		tagName, tagEnd, ok := readXMLTagName(rest)
		if !ok {
			visible.WriteString(rest)
			i = len(buf)
			break
		}
		if strings.HasPrefix(rest, "</") {
			if tagEnd >= len(rest) || rest[tagEnd] != '>' {
				// Partial closing tag (e.g. "</file"). Hold the
				// tail so the next chunk can complete it.
				if holdTail {
					return visible.String(), toolCalls, buf[absOpen:]
				}
				visible.WriteString(rest)
				i = len(buf)
				break
			}
			visible.WriteString(rest[:tagEnd])
			i = absOpen + tagEnd
			continue
		}
		if tagEnd >= len(rest) || rest[tagEnd] != '>' {
			// No closing '>' for the opening tag yet. In streaming
			// mode hold the tail so the next chunk can complete the
			// tag. In complete-buffer mode emit the partial tag as
			// visible prose (nothing else is coming).
			if holdTail {
				return visible.String(), toolCalls, buf[absOpen:]
			}
			visible.WriteString(rest)
			i = len(buf)
			break
		}
		closing := "</" + tagName + ">"
		searchFrom := absOpen + tagEnd + 1
		closeIdx := strings.Index(buf[searchFrom:], closing)
		if closeIdx == -1 {
			if holdTail {
				return visible.String(), toolCalls, buf[absOpen:]
			}
			// Complete-buffer mode: keep the unmatched opening tag in
			// the visible text so nothing is lost.
			visible.WriteString(buf[absOpen:])
			return visible.String(), toolCalls, ""
		}
		innerStart := searchFrom
		innerEnd := innerStart + closeIdx
		inner := strings.TrimSpace(buf[innerStart:innerEnd])
		if knownXMLToolNames[tagName] || looksLikeToolCall(tagName, inner) {
			toolCalls = append(toolCalls, model.Tool{
				Id:   fmt.Sprintf("call_%s", random.GetUUID()),
				Type: "function",
				Function: model.Function{
					Name:      tagName,
					Arguments: inner,
				},
			})
		} else {
			visible.WriteString(buf[absOpen : innerEnd+len(closing)])
		}
		i = innerEnd + len(closing)
	}
	return visible.String(), toolCalls, ""
}

func isXMLTagStart(s string) bool {
	if len(s) < 2 || s[0] != '<' {
		return false
	}
	c := s[1]
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '/'
}

func readXMLTagName(s string) (name string, end int, ok bool) {
	// s starts with "<" or "</"
	i := 1
	if i < len(s) && s[i] == '/' {
		i++
	}
	start := i
	for i < len(s) {
		c := s[i]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' {
			i++
			continue
		}
		break
	}
	if i == start {
		return "", 0, false
	}
	return s[start:i], i, true
}

func Handler(c *gin.Context, meta *meta.Meta, textRequest model.GeneralOpenAIRequest, appId string, apiSecret string, apiKey string) (*model.ErrorWithStatusCode, *model.Usage) {
	domain, authUrl := getXunfeiAuthUrl(meta.Config.APIVersion, apiKey, apiSecret)
	dataChan, stopChan, err := xunfeiMakeRequest(textRequest, domain, authUrl, appId)
	if err != nil {
		return openai.ErrorWrapper(err, "xunfei_request_failed", http.StatusInternalServerError), nil
	}
	var usage model.Usage
	var content string
	var xunfeiResponse ChatResponse
	stop := false
	for !stop {
		select {
		case xunfeiResponse = <-dataChan:
			if len(xunfeiResponse.Payload.Choices.Text) == 0 {
				continue
			}
			content += xunfeiResponse.Payload.Choices.Text[0].Content
			usage.PromptTokens += xunfeiResponse.Payload.Usage.Text.PromptTokens
			usage.CompletionTokens += xunfeiResponse.Payload.Usage.Text.CompletionTokens
			usage.TotalTokens += xunfeiResponse.Payload.Usage.Text.TotalTokens
		case stop = <-stopChan:
		}
	}
	if len(xunfeiResponse.Payload.Choices.Text) == 0 {
		return openai.ErrorWrapper(errors.New("xunfei empty response detected"), "xunfei_empty_response_detected", http.StatusInternalServerError), nil
	}
	xunfeiResponse.Payload.Choices.Text[0].Content = content

	response := responseXunfei2OpenAI(&xunfeiResponse)
	jsonResponse, err := json.Marshal(response)
	if err != nil {
		return openai.ErrorWrapper(err, "marshal_response_body_failed", http.StatusInternalServerError), nil
	}
	c.Writer.Header().Set("Content-Type", "application/json")
	_, _ = c.Writer.Write(jsonResponse)
	return nil, &usage
}

func xunfeiMakeRequest(textRequest model.GeneralOpenAIRequest, domain, authUrl, appId string) (chan ChatResponse, chan bool, error) {
	d := websocket.Dialer{
		HandshakeTimeout: 5 * time.Second,
		TLSClientConfig:   &tls.Config{InsecureSkipVerify: config.XunfeiInsecureSkipVerify},
	}
	var requestHeader http.Header
	if config.XunfeiCookie != "" {
		requestHeader = http.Header{}
		requestHeader.Set("Cookie", config.XunfeiCookie)
	}
	conn, resp, err := d.Dial(authUrl, requestHeader)
	if err != nil || resp.StatusCode != 101 {
		return nil, nil, err
	}
	data := requestOpenAI2Xunfei(textRequest, appId, domain)
	// DEBUG: log the WSS payload so we can see exactly what the
	// gateway receives. Useful for diagnosing tool-calling
	// behaviour when the model emits unexpected XML shapes.
	if payloadJSON, marshalErr := json.Marshal(data); marshalErr == nil {
		logger.SysLog(fmt.Sprintf("xunfei wss request [one-api %s]: %s", common.Version, string(payloadJSON)))
	}
	err = conn.WriteJSON(data)
	if err != nil {
		return nil, nil, err
	}
	_, msg, err := conn.ReadMessage()
	if err != nil {
		return nil, nil, err
	}

	dataChan := make(chan ChatResponse)
	stopChan := make(chan bool)
	go func() {
		for {
			if msg == nil {
				_, msg, err = conn.ReadMessage()
				if err != nil {
					logger.SysError("error reading stream response: " + err.Error())
					break
				}
			}
			var response ChatResponse
			err = json.Unmarshal(msg, &response)
			if err != nil {
				logger.SysError("error unmarshalling stream response: " + err.Error())
				break
			}
			// DEBUG: log the raw WSS response payload.
			logger.SysLog(fmt.Sprintf("xunfei wss response [one-api %s]: %s", common.Version, string(msg)))
			msg = nil
			dataChan <- response
			if response.Payload.Choices.Status == 2 {
				err := conn.Close()
				if err != nil {
					logger.SysError("error closing websocket connection: " + err.Error())
				}
				break
			}
		}
		stopChan <- true
	}()

	return dataChan, stopChan, nil
}

// toolNamesOf extracts the function name from each model.Tool, in
// the order they were declared. Used to render concrete tool names
// in the DSML prompt's example blocks.
func toolNamesOf(tools []model.Tool) []string {
	names := make([]string, 0, len(tools))
	for _, t := range tools {
		if t.Function.Name != "" {
			names = append(names, t.Function.Name)
		}
	}
	return names
}

// dsmlToolCallInstructions returns a system-prompt block that
// instructs the model to emit tool calls using the DSML format that
// ds2api (DeepSeek-to-OpenAI bridge) and our parser both understand.
// Ported from ds2api's internal/toolcall/tool_prompt.go so the
// Xunfei WSS path gets the same prompt that works there.
//
// Halfwidth pipes (|) are used; the parser normalises the fullwidth
// form (some DeepSeek-V3 outputs) to halfwidth before matching.
func dsmlToolCallInstructions(toolNames []string) string {
	const header = `TOOL CALL FORMAT — FOLLOW EXACTLY:

<|DSML|tool_calls>
  <|DSML|invoke name="TOOL_NAME">
    <|DSML|parameter name="PARAMETER_NAME">VALUE</|DSML|parameter>
  </|DSML|invoke>
</|DSML|tool_calls>

RULES:
1) Use the <|DSML|tool_calls> wrapper format.
2) Put one or more <|DSML|invoke> entries under a single <|DSML|tool_calls> root.
3) Put the tool name in the invoke name attribute: <|DSML|invoke name="TOOL_NAME">.
4) Every top-level argument must be a <|DSML|parameter name="ARG_NAME">VALUE</|DSML|parameter> node.
5) Use only the parameter names in the tool schema. Do not invent fields.
6) Fill parameters with the actual values required for this call.
7) If a required parameter value is unknown, answer normally instead of outputting an empty tool call.
8) Do NOT wrap XML in markdown fences. Do NOT output explanations or internal monologue around the tool block.
9) The first non-whitespace characters of the tool block must be exactly <|DSML|tool_calls>.
10) Never omit the opening <|DSML|tool_calls> tag.
11) Compatibility note: the runtime also accepts the legacy XML tags <tool_calls> / <invoke> / <parameter>, but prefer the DSML-prefixed form above.

When you are NOT calling a tool, just answer normally with prose. Do not emit empty <|DSML|tool_calls></|DSML|tool_calls> wrappers.
`
	return header + dsmlExampleBlock(toolNames)
}

// dsmlExampleBlock renders a single positive example using the
// first non-empty tool name from names. The example shows the
// minimal correct format with one parameter.
func dsmlExampleBlock(names []string) string {
	for _, n := range names {
		if n == "" {
			continue
		}
		var b strings.Builder
		b.WriteString("\nEXAMPLE:\n")
		b.WriteString("<|DSML|tool_calls>\n")
		b.WriteString("  <|DSML|invoke name=\"")
		b.WriteString(n)
		b.WriteString("\">\n")
		b.WriteString("    <|DSML|parameter name=\"ARG_NAME\">VALUE</|DSML|parameter>\n")
		b.WriteString("  </|DSML|invoke>\n")
		b.WriteString("</|DSML|tool_calls>\n")
		return b.String()
	}
	return ""
}

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
	"strconv"
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
	messages := make([]Message, 0, len(request.Messages))
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
	if config.XunfeiTopK != "" {
		if n, err := strconv.Atoi(config.XunfeiTopK); err == nil {
			xunfeiRequest.Parameter.Chat.TopK = n
		}
	} else if request.N != 0 {
		xunfeiRequest.Parameter.Chat.TopK = request.N
	}
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
// emitted by DeepSeek models: <｜DSML｜invoke name="X"> ... </｜DSML｜invoke>.
// The "｜" character is U+FF5C (fullwidth vertical bar).
const (
	dsmlOpenPrefix  = "<｜DSML｜"
	dsmlClosePrefix = "</｜DSML｜"
)

// extractXMLToolCalls parses the given content for fully-closed XML/DSML
// blocks that look like a tool invocation, and returns them as
// model.Tool slices. DSML blocks are emitted by DeepSeek models and look
// like <｜DSML｜invoke name="X"> ... <｜DSML｜parameter name="k">v</｜DSML｜parameter> ... </｜DSML｜invoke>;
// they are checked first because DeepSeek models reliably wrap tool
// invocations in DSML. Plain <tag>...</tag> blocks from other models are
// still picked up as a fallback.
func extractXMLToolCalls(content string) []model.Tool {
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
				Arguments: args,
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
// and returns a map of parameter name to its string value.
func extractDSMLParameters(inner string) map[string]string {
	args := map[string]string{}
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
		innerStart := openIdx + gt + 1
		closing := dsmlClosePrefix + "parameter>"
		closeIdx := strings.Index(inner[innerStart:], closing)
		if closeIdx < 0 {
			break
		}
		value := inner[innerStart : innerStart+closeIdx]
		args[name] = strings.TrimSpace(value)
		inner = inner[innerStart+closeIdx+len(closing):]
	}
	return args
}

// extractDSMLFromBuffer walks buf, peels out fully-closed DSML
// <｜DSML｜invoke ...>...</｜DSML｜invoke> blocks as tool calls, and
// returns the prose in between. When holdTail is true an unclosed
// opening tag at the end of the buffer is preserved as the tail so the
// streaming layer can carry it forward into the next chunk.
func extractDSMLFromBuffer(buf string, holdTail bool) (string, []model.Tool, string) {
	var (
		visible   strings.Builder
		toolCalls []model.Tool
		i         = 0
	)
	for i < len(buf) {
		openIdx := strings.Index(buf[i:], dsmlOpenPrefix+"invoke ")
		if openIdx == -1 {
			visible.WriteString(buf[i:])
			i = len(buf)
			break
		}
		absOpen := i + openIdx
		visible.WriteString(buf[i:absOpen])
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
				Arguments: args,
			},
		})
		i = innerStart + closeIdx + len(closing)
	}
	return visible.String(), toolCalls, ""
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

// streamXMLBuffer tracks the partial state of XML tool-call tags across
// streaming frames so we know when a fully-closed block has been emitted
// and can convert it to a function_call.
type streamXMLBuffer struct {
	accumulated string
}

func newStreamXMLBuffer() *streamXMLBuffer {
	return &streamXMLBuffer{}
}

// consume appends the new chunk and returns:
//   - visibleText: the part of the chunk that is NOT inside a tool-call
//     block, safe to send to the client as content.
//   - toolCalls: tool calls that completed inside this chunk.
//
// XML tag state is tracked by counting the most recent unmatched opening
// tag; once a closing tag of the same name arrives, the block is treated
// as a complete tool call and removed from visibleText.
func (b *streamXMLBuffer) consume(chunk string) (visibleText string, toolCalls []model.Tool) {
	if chunk == "" {
		return "", nil
	}

	// Look ahead: if this chunk's end is part of a DSML marker, hold
	// the tail so the next chunk can complete it. Specifically: walk
	// backwards from the end of `chunk` looking for the start of a
	// potential "<｜DSML｜" sequence. If we find a partial DSML marker
	// (e.g. just "<" or "<｜DSML｜t"), split the chunk and keep the
	// tail for the next call.
	if splitIdx := findDSMLBoundary(chunk); splitIdx >= 0 {
		// The chunk's tail is the start of a potential DSML
		// marker (held for next call). Everything up to splitIdx is
		// safe prose — except that b.accumulated + chunk[:splitIdx]
		// may have just completed a DSML opening tag (wrapper
		// <｜DSML｜tool_calls> or opener <｜DSML｜invoke ...>). If so,
		// move the opener to b.accumulated so it stays attached to
		// the held tail and the DSML extractor can pair it with
		// the matching closing tag on a later call.
		visibleText = b.accumulated + chunk[:splitIdx]
		b.accumulated = chunk[splitIdx:]
		stripped, opener, ok := extractTrailingDSMLOpener(visibleText)
		if ok {
			visibleText = stripped
			b.accumulated = opener + b.accumulated
		}
		return visibleText, nil
	}

	b.accumulated += chunk

	// After accumulation, check the buffer for an in-flight DSML
	// block. If a <｜DSML｜ opening tag is present but the matching
	// </｜DSML｜invoke> close hasn't been seen, hold the tail. We
	// look for the specific </｜DSML｜invoke> close (not any
	// </｜DSML｜...>) so that a complete block ending in
	// </｜DSML｜invoke> gets extracted even though it contains
	// intermediate </｜DSML｜parameter> closes.
	dsmlOpenIdx := strings.Index(b.accumulated, dsmlOpenPrefix)
	dsmlInvokeCloseIdx := strings.Index(b.accumulated, dsmlClosePrefix+"invoke>")
	if dsmlOpenIdx >= 0 && (dsmlInvokeCloseIdx < 0 || dsmlInvokeCloseIdx <= dsmlOpenIdx) {
		// Flush any prose that appeared before the DSML opening tag.
		visibleText = b.accumulated[:dsmlOpenIdx]
		b.accumulated = b.accumulated[dsmlOpenIdx:]
		return visibleText, nil
	}

	// DSML block is complete (or no DSML in flight). Run the
	// regular extractor.
	visibleText, toolCalls, b.accumulated = extractFromBuffer(b.accumulated)
	return visibleText, toolCalls
}

// extractTrailingDSMLOpener finds a complete DSML opening tag at the
// end of s (ignoring trailing whitespace) and returns the visible
// text up to (but not including) the opener, the opener itself, and
// a found flag. Wrappers like <｜DSML｜tool_calls> are stripped from
// visible (they are containers, not content); the opener is
// returned so the caller can prepend it to the next-chunk held tail.
// When no complete DSML opener is at the end, returns (s, "", false).
func extractTrailingDSMLOpener(s string) (string, string, bool) {
	// Find the rightmost '>' in s, ignoring trailing whitespace.
	end := len(s)
	for end > 0 && (s[end-1] == ' ' || s[end-1] == '\n' || s[end-1] == '\t' || s[end-1] == '\r') {
		end--
	}
	if end == 0 || s[end-1] != '>' {
		return s, "", false
	}
	gt := strings.LastIndex(s[:end], ">")
	openIdx := strings.LastIndex(s[:gt], "<")
	if openIdx < 0 {
		return s, "", false
	}
	tag := s[openIdx : gt+1]
	if !strings.HasPrefix(tag, dsmlOpenPrefix) {
		return s, "", false
	}
	// Wrapper tag (e.g. <｜DSML｜tool_calls>): drop entirely, do not
	// preserve as opener. Caller will see visible text without the
	// wrapper and no opener attached.
	if !strings.Contains(tag, "invoke ") {
		return s[:openIdx], "", true
	}
	// Real tool opener (e.g. <｜DSML｜invoke name="Read">): keep in
	// b.accumulated so the DSML extractor can pair it with the
	// matching </｜DSML｜invoke> close.
	return s[:openIdx], tag, true
}

// findDSMLBoundary returns the byte index in chunk where a partial
// DSML marker starts (and should be held for the next chunk), or -1
// if no such partial marker exists. The check only looks at the
// trailing runes of chunk (up to the length of "<｜DSML｜") to avoid
// misclassifying normal prose that happens to contain a "<" character
// (e.g. "x < 3") as a DSML start.
func findDSMLBoundary(chunk string) int {
	prefixRunes := []rune(dsmlOpenPrefix)
	maxLook := len(prefixRunes)
	runes := []rune(chunk)
	if len(runes) < 2 {
		return -1
	}
	for offset := 1; offset <= maxLook && offset <= len(runes); offset++ {
		if runes[len(runes)-offset] != '<' {
			continue
		}
		tail := string(runes[len(runes)-offset:])
		if couldBeDSMLStart(tail) {
			tailRunes := []rune(tail)
			return len(chunk) - len(string(tailRunes))
		}
		return -1
	}
	return -1
}

// couldBeDSMLStart returns true if s begins with "<" and the rest of
// s is a prefix of "<｜DSML｜" (i.e. could be the start of a DSML
// marker that hasn't fully arrived yet, but might be completed by
// the next chunk). The check is rune-based because the "｜" character
// is U+FF5C and occupies more than one byte in UTF-8. A bare "<" is
// treated as a potential DSML start (hold) since we cannot tell until
// the next chunk arrives.
func couldBeDSMLStart(s string) bool {
	if s == "" || s[0] != '<' {
		return false
	}
	sRunes := []rune(s)
	prefixRunes := []rune(dsmlOpenPrefix)
	if len(sRunes) == 1 {
		return true
	}
	if sRunes[1] != prefixRunes[1] {
		return false
	}
	for i := 1; i < len(sRunes) && i < len(prefixRunes); i++ {
		if sRunes[i] != prefixRunes[i] {
			return false
		}
	}
	for _, r := range sRunes {
		if r == '>' {
			return false
		}
	}
	return true
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
	// DSML blocks (DeepSeek tool calls) take priority: they start with
	// "<｜DSML｜" (a fullwidth vertical bar), not "<letter>", so the
	// generic <tag> extractor below would misclassify them. Walk through
	// the buffer, peel out fully-closed <｜DSML｜invoke> blocks, and feed
	// the leftover prose through the regular extractor.
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
		// If this '<' is the start of a DSML block, hold the tail so the
		// DSML extractor (called by the outer extractFromBufferImpl) gets
		// a chance to consume it on a future iteration.
		if strings.HasPrefix(rest, dsmlOpenPrefix) {
			if holdTail {
				return visible.String(), toolCalls, buf[absOpen:]
			}
			visible.WriteString(rest)
			i = len(buf)
			return visible.String(), toolCalls, ""
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
			visible.WriteString(rest[:tagEnd])
			i = absOpen + tagEnd
			continue
		}
		if tagEnd >= len(rest) || rest[tagEnd] != '>' {
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

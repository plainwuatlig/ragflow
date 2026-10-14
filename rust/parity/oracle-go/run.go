package main

// The run path, copied from internal/service/agent.go
// (AgentService.buildRunFunc and its helpers) with every DB / Redis /
// run-tracker / upload / persistence step removed. What remains is what
// decides the event stream: decode, state seeding, compile, invoke, the
// terminal answer, and the message / message_end / workflow_finished
// frames. Keep in step with upstream when rebasing.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"sort"
	"strings"
	"time"

	"gorm.io/gorm"

	"ragflow/internal/agent/canvas"
	"ragflow/internal/agent/runtime"
)

// boundComponent carries its cpn id into the invoke context so the stub
// model can key recordings by it; a deferred Agent stream keeps the id
// when the Message opens it later.
type boundComponent struct {
	inner runtime.Component
	o     *oracle
}

func (b *boundComponent) Invoke(ctx context.Context, db *gorm.DB, inputs map[string]any) (map[string]any, error) {
	id := b.o.whoAmI(ctx)
	out, err := b.inner.Invoke(context.WithValue(ctx, cpnKey{}, id), db, inputs)
	if deferred, ok := out["content"].(*runtime.DeferredStream); ok && deferred != nil && deferred.Open != nil {
		open := deferred.Open
		deferred.Open = func(openCtx context.Context, sink runtime.AgentDeltaSink) (map[string]any, error) {
			return open(context.WithValue(openCtx, cpnKey{}, id), sink)
		}
	}
	return out, err
}

// runFunc is buildRunFunc (internal/service/agent.go:1950-2389) without
// persistence.
func (o *oracle) runFunc(dsl map[string]any) canvas.RunFunc {
	return func(ctx context.Context, root map[string]any) (*canvas.CanvasState, error) {
		events, _ := root["__events__"].(chan canvas.RunEvent)
		messageID, _ := root["__message_id__"].(string)
		sessionID, _ := root["__session_id__"].(string)
		emit := func(typ, data string) {
			if events == nil {
				return
			}
			canvas.PushEvent(ctx, events, canvas.RunEvent{
				Type: typ, Data: data, MessageID: messageID,
				CreatedAt: time.Now().Unix(), SessionID: sessionID,
			})
		}
		startedAt := float64(time.Now().UnixNano()) / 1e9
		userInput := root["user_input"]
		wsData, _ := json.Marshal(map[string]any{"inputs": userInput})
		emit("workflow_started", string(wsData))

		state := canvas.NewCanvasState("parity", sessionID)
		c, err := canvas.DecodeFromDSL(dsl)
		if err != nil {
			return nil, fmt.Errorf("decode agent: %w", err)
		}
		defer c.Close()

		ctx2 := canvas.WithRunMeta(ctx, &canvas.RunMeta{Events: events, MessageID: messageID, SessionID: sessionID})
		ctx2 = runtime.WithDeferredNodeRegistry(ctx2)
		agentEmit, agentFinalize, agentReset := makeAgentMessageDeltaEmitter(emit)
		ctx2 = runtime.WithAgentMessageEmitterControl(ctx2, agentEmit, agentFinalize, agentReset)
		ctx2 = runtime.WithCanvasMessageEmitter(ctx2, func(content string) {
			emitMessage(emit, canvas.MessageEvent{Content: content})
		})
		ctx2 = runtime.WithCanvasMessageEventEmitter(ctx2, func(content string, startToThink, endToThink bool) {
			emitMessage(emit, canvas.MessageEvent{Content: content, StartToThink: startToThink, EndToThink: endToThink})
		})
		ctx2 = canvas.WithComponentFactory(ctx2, o.componentFactory())

		for k, v := range c.Globals {
			switch {
			case strings.HasPrefix(k, "sys."):
				state.Sys[strings.TrimPrefix(k, "sys.")] = v
			case strings.HasPrefix(k, "env."):
				state.Env[strings.TrimPrefix(k, "env.")] = v
			default:
				state.Globals[k] = v
			}
		}
		state.SetHistory(c.History)
		state.SetMemory(c.Memory)
		state.EnsureSysDate()
		query := userInput
		if values, ok := userInput.(map[string]any); ok {
			query = values["query"]
		}
		state.Sys["query"] = query
		state.AppendCurrentUser(query)
		if s, ok := query.(string); ok {
			state.AppendSysHistory("user: " + s)
		} else {
			return nil, errors.New("parity: only string queries are supported")
		}
		state.Sys["canvas_id"] = "parity"
		state.Sys["session_id"] = sessionID
		state.IncrementConversationTurns()
		ctx2 = runtime.WithState(ctx2, state)

		cc, err := canvas.Compile(ctx2, c)
		if err != nil {
			return nil, canvas.NewInternalRunError(fmt.Errorf("canvas compile: %w", err))
		}
		wfInput := map[string]any{"query": userInput}
		runStartedAt := time.Now()
		workflowOutput, err := cc.Workflow.Invoke(ctx2, wfInput)

		var thinking string
		var legacyReference []interface{}
		now := float64(time.Now().UnixNano()) / 1e9
		for _, bucket := range state.Snapshot() {
			if v, ok := bucket["thinking"].(string); ok && v != "" && thinking == "" {
				thinking = v
			}
			if v, ok := bucket["reference"].([]interface{}); ok {
				legacyReference = append(legacyReference, v...)
			}
		}
		referencePayload := agentRunReferencePayload(c, state, legacyReference)
		assistantOutput := terminalCanvasOutput(c, state, workflowOutput, runStartedAt)
		answer, _ := assistantOutput["content"].(string)
		attachment, _ := assistantOutput["attachment"].(map[string]any)
		runtime.CompleteAllDeferredNodes(ctx2)
		runtime.FinalizeAgentMessage(ctx2)
		messageEventsEmitted := runtime.AgentMessageEventsEmittedRun(ctx2)
		messageEventsSuppressed := runtime.AgentMessageEventsSuppressedRun(ctx2)
		shouldEmitMessage := messageEventsEmitted || !messageEventsSuppressed

		if err != nil {
			if canvas.IsInterruptError(err) {
				return state, errors.New("parity: interrupts (UserFillUp) are out of scope")
			}
			if failureText := deferredAgentStreamFailureText(err); failureText != "" {
				if answer == "" && !messageEventsEmitted && shouldEmitMessage {
					emitAgentMessageEvents(emit, failureText, thinking, referencePayload)
				}
				if shouldEmitMessage {
					meData, _ := json.Marshal(canvas.MessageEndEvent{Attachment: attachment, Reference: referencePayload})
					emit("message_end", string(meData))
				}
				return state, nil
			}
			return nil, canvasInvokeError(err)
		}
		if !messageEventsEmitted && shouldEmitMessage {
			emitAgentMessageEvents(emit, answer, thinking, referencePayload)
		}
		if shouldEmitMessage {
			meData, _ := json.Marshal(canvas.MessageEndEvent{Attachment: attachment, Reference: referencePayload})
			emit("message_end", string(meData))
		}
		wfData, _ := json.Marshal(map[string]any{
			"inputs":       map[string]any{"query": userInput},
			"outputs":      workflowOutputsFromTerminal(assistantOutput),
			"elapsed_time": now - startedAt,
			"created_at":   now,
		})
		emit("workflow_finished", string(wfData))
		return state, nil
	}
}

// ---- helpers copied from internal/service/agent.go ---------------------

func emitMessage(emit func(string, string), ev canvas.MessageEvent) {
	data, _ := json.Marshal(ev)
	emit("message", string(data))
}

func emitAgentMessageEvents(emit func(string, string), answer, thinking string, reference any) {
	for _, ev := range buildAgentMessageEvents(answer, thinking, reference) {
		emitMessage(emit, ev)
	}
}

func buildAgentMessageEvents(answer, thinking string, reference any) []canvas.MessageEvent {
	answer, thinking = splitInlineThink(answer, thinking)
	if thinking == "" {
		return []canvas.MessageEvent{{Content: answer, Reference: reference}}
	}
	events := []canvas.MessageEvent{{StartToThink: true}}
	for _, chunk := range splitMessageContent(thinking) {
		events = append(events, canvas.MessageEvent{Content: chunk})
	}
	events = append(events, canvas.MessageEvent{EndToThink: true})
	for _, chunk := range splitMessageContent(answer) {
		events = append(events, canvas.MessageEvent{Content: chunk})
	}
	return events
}

func splitInlineThink(answer, thinking string) (string, string) {
	if thinking != "" {
		return answer, thinking
	}
	start := strings.Index(answer, "<think>")
	if start < 0 {
		return answer, thinking
	}
	afterStart := start + len("<think>")
	endRel := strings.Index(answer[afterStart:], "</think>")
	if endRel < 0 {
		return answer, thinking
	}
	end := afterStart + endRel
	thinking = answer[afterStart:end]
	answer = strings.TrimLeft(answer[:start]+answer[end+len("</think>"):], "\r\n")
	return answer, thinking
}

func splitMessageContent(content string) []string {
	if content == "" {
		return nil
	}
	const maxRunes = 24
	runes := []rune(content)
	var chunks []string
	for len(runes) > 0 {
		n := min(maxRunes, len(runes))
		chunks = append(chunks, string(runes[:n]))
		runes = runes[n:]
	}
	return chunks
}

// makeAgentMessageDeltaEmitter is service.agentMessageDeltaEmitter without
// inline <think> tag splitting (NextThinkDelta); the oracle refuses
// recordings that contain <think>, so the two agree on every input it runs.
func makeAgentMessageDeltaEmitter(emit func(string, string)) (func(string, string), func() bool, func()) {
	var inThinking, explicitReasoning, emitted bool
	send := func(ev canvas.MessageEvent) { emitMessage(emit, ev); emitted = true }
	endThinking := func() {
		if inThinking {
			send(canvas.MessageEvent{EndToThink: true})
			inThinking = false
		}
	}
	emitFn := func(contentDelta, thinkingDelta string) {
		if thinkingDelta != "" {
			if !inThinking {
				send(canvas.MessageEvent{StartToThink: true})
				inThinking = true
			}
			explicitReasoning = true
			send(canvas.MessageEvent{Content: thinkingDelta})
		}
		if contentDelta == "" {
			return
		}
		if explicitReasoning {
			endThinking()
			explicitReasoning = false
		}
		send(canvas.MessageEvent{Content: contentDelta})
	}
	finalize := func() bool {
		before := emitted
		if explicitReasoning || inThinking {
			endThinking()
			explicitReasoning = false
		}
		return emitted && !before
	}
	reset := func() { inThinking, explicitReasoning, emitted = false, false, false }
	return emitFn, finalize, reset
}

func agentRunReferencePayload(c *canvas.Canvas, state *canvas.CanvasState, legacyChunks []interface{}) map[string]interface{} {
	reference := state.GetRetrievalReference()
	if len(reference) == 0 {
		if len(legacyChunks) == 0 {
			return nil
		}
		reference = map[string]interface{}{"chunks": legacyChunks, "doc_aggs": []interface{}{}, "total": len(legacyChunks)}
	}
	snapshot := state.Snapshot()
	hasAgent, cite := false, false
	for cpnID, component := range c.Components {
		if !strings.EqualFold(component.Obj.ComponentName, "Agent") || len(snapshot[cpnID]) == 0 {
			continue
		}
		hasAgent = true
		if enabled, _ := component.Obj.Params["cite"].(bool); enabled {
			cite = true
			break
		}
	}
	if hasAgent && !cite {
		delete(reference, "doc_aggs")
	}
	return reference
}

func terminalCanvasOutput(c *canvas.Canvas, state *canvas.CanvasState, workflowOutput map[string]any, runStartedAt time.Time) map[string]any {
	terminalIDs := make([]string, 0)
	messageIDs := make([]string, 0)
	for cpnID, component := range c.Components {
		if strings.EqualFold(component.Obj.ComponentName, "Message") {
			messageIDs = append(messageIDs, cpnID)
		}
		if c.NodeParents[cpnID] != "" || strings.EqualFold(component.Obj.ComponentName, "ExitLoop") {
			continue
		}
		if _, ok := workflowOutput[cpnID]; ok || len(component.Downstream) == 0 {
			terminalIDs = append(terminalIDs, cpnID)
		}
	}
	sort.Strings(terminalIDs)
	sort.Strings(messageIDs)
	for _, cpnID := range terminalIDs {
		if output, ok := workflowOutput[cpnID].(map[string]any); ok && len(output) > 0 {
			return cloneCanvasOutput(output)
		}
	}
	snapshot := state.Snapshot()
	for _, cpnID := range terminalIDs {
		if output := snapshot[cpnID]; len(output) > 0 {
			created, _ := output["_created_time"].(string)
			createdAt, err := time.Parse(time.RFC3339Nano, created)
			if err == nil && !createdAt.Before(runStartedAt) {
				return cloneCanvasOutput(output)
			}
		}
	}
	var partial map[string]any
	latest := runStartedAt
	for _, cpnID := range messageIDs {
		output := snapshot[cpnID]
		created, _ := output["_created_time"].(string)
		createdAt, err := time.Parse(time.RFC3339Nano, created)
		if err == nil && !createdAt.Before(latest) {
			partial, latest = output, createdAt
		}
	}
	if partial != nil {
		return cloneCanvasOutput(partial)
	}
	return map[string]any{}
}

func cloneCanvasOutput(input map[string]any) map[string]any {
	output := maps.Clone(input)
	for _, key := range []string{"__cpn_id__", "state", "__legacy_noop__"} {
		delete(output, key)
	}
	return output
}

func workflowOutputsFromTerminal(output map[string]any) any {
	if len(output) == 0 {
		return ""
	}
	content, _ := output["content"].(string)
	downloads, attachment := output["downloads"], output["attachment"]
	if emptyValue(downloads) && emptyValue(attachment) {
		return content
	}
	out := map[string]any{"content": content}
	if !emptyValue(downloads) {
		out["downloads"] = downloads
	}
	if !emptyValue(attachment) {
		out["attachment"] = attachment
	}
	return out
}

func emptyValue(value any) bool {
	if value == nil {
		return true
	}
	v := reflect.ValueOf(value)
	switch v.Kind() {
	case reflect.Slice, reflect.Array, reflect.Map:
		return v.Len() == 0
	default:
		return false
	}
}

func deferredAgentStreamFailureText(err error) string {
	if err == nil || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return ""
	}
	var deferred *runtime.DeferredStreamError
	if !errors.As(err, &deferred) {
		return ""
	}
	return strings.TrimSpace(deferred.FailureText())
}

func canvasInvokeError(err error) error {
	msg := err.Error()
	if strings.Contains(msg, "[GraphRunError] no tasks to execute") && strings.Contains(msg, "last completed nodes: [Switch:") {
		return errors.New("agent invoke: Switch routing stopped because no connected branch matched the condition; check the Switch branches")
	}
	return fmt.Errorf("agent invoke: %w", err)
}

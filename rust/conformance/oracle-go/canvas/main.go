// Command oracle-go runs one RAGFlow agent canvas through upstream's Go
// runtime with recorded LLM answers and prints the run's events as JSON
// lines, so the Rust runtime's stream can be diffed against it.
//
// It lives inside the `ragflow` module (under rust/, which upstream never
// touches) so it may import internal/ packages. Nothing upstream is edited:
//
//   - The run path is service.AgentService.buildRunFunc
//     (internal/service/agent.go), copied below as runFunc minus every DB,
//     Redis, run-tracker, file-upload and session-persistence step. Keep it
//     in step with that function when rebasing.
//   - Models: every provider in conf/models keeps its name but its
//     ModelDriver is replaced by stubDriver, so Agent (eino ReAct ->
//     EinoChatModel.Stream) and Categorize (einoChatInvoker ->
//     EinoChatModel.Generate) reach the stub through the production
//     GetPreconfiguredDriver lookup.
//   - Which component made a call: Agent and Categorize factories are
//     wrapped (canvas.WithComponentFactory) so their Invoke context
//     carries the cpn id; a deferred Agent's stream (opened later inside
//     the Message) keeps the Agent's id.
//
// Usage:
//
//	oracle-go -canvas dsl.json -query "hi" -recordings rec.json -conf conf/models
//
// Output: one JSON object per line, {"event": <name>, "data": <payload>},
// in upstream's event names and data shapes, plus an extra
// {"event":"llm_call","data":{component_id, call_index, model, messages,
// options}} line for every chat request, written before the answer is
// returned. A call with no recording fails the run and exits 3, naming
// the recording key and the messages.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"sync"

	"ragflow/internal/agent/canvas"
	_ "ragflow/internal/agent/component"
	dslpkg "ragflow/internal/agent/dsl"
	"ragflow/internal/agent/runtime"
	"ragflow/internal/common"
	"ragflow/internal/entity/models"
)

func main() {
	canvasPath := flag.String("canvas", "", "canvas DSL JSON (the dsl object itself)")
	query := flag.String("query", "", "the visitor's query")
	recPath := flag.String("recordings", "", "recorded answers: {\"<cpn_id>#<n>\": \"answer\"}")
	confDir := flag.String("conf", "conf/models", "upstream's provider catalog directory")
	flag.Parse()
	if *canvasPath == "" || *recPath == "" {
		fmt.Fprintln(os.Stderr, "usage: oracle-go -canvas dsl.json -query q -recordings rec.json [-conf conf/models]")
		os.Exit(2)
	}
	o, err := newOracle(*recPath)
	if err != nil {
		fatal(err)
	}
	if err := o.installModels(*confDir); err != nil {
		fatal(err)
	}
	raw, err := os.ReadFile(*canvasPath)
	if err != nil {
		fatal(err)
	}
	var dsl map[string]any
	if err := json.Unmarshal(raw, &dsl); err != nil {
		fatal(fmt.Errorf("canvas: %w", err))
	}
	dsl = dslpkg.NormalizeForRun(dsl)

	root := map[string]any{"user_input": *query}
	events := canvas.NewRunner().Run(o.context(context.Background()), o.runFunc(dsl), "parity", "parity-session", *query, root)
	for ev := range events {
		var data any
		if ev.Data != "" {
			if err := json.Unmarshal([]byte(ev.Data), &data); err != nil {
				data = ev.Data
			}
		}
		o.print(ev.Type, data)
	}
	if o.missing.Load() {
		os.Exit(3)
	}
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "oracle-go:", err)
	os.Exit(1)
}

// ---- recorded models -------------------------------------------------

type oracle struct {
	answers map[string]string
	out     *json.Encoder
	mu      sync.Mutex
	calls   map[string]int
	missing atomicBool

	tokenMu sync.Mutex
	token   int
	owners  map[int]string
}

type atomicBool struct {
	mu sync.Mutex
	v  bool
}

func (b *atomicBool) Store(v bool) { b.mu.Lock(); b.v = v; b.mu.Unlock() }
func (b *atomicBool) Load() bool   { b.mu.Lock(); defer b.mu.Unlock(); return b.v }

func newOracle(path string) (*oracle, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	answers := map[string]string{}
	if err := json.Unmarshal(raw, &answers); err != nil {
		return nil, fmt.Errorf("recordings: %w", err)
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetEscapeHTML(false)
	return &oracle{answers: answers, out: enc, calls: map[string]int{}, owners: map[int]string{}}, nil
}

func (o *oracle) print(event string, data any) {
	o.mu.Lock()
	defer o.mu.Unlock()
	_ = o.out.Encode(map[string]any{"event": event, "data": data})
}

// context installs the run-level progress-fraction sink the cpn-id probe
// relies on (see whoAmI).
func (o *oracle) context(ctx context.Context) context.Context {
	return runtime.WithProgressFractionCallback(ctx, func(component string, fraction float64) {
		o.tokenMu.Lock()
		o.owners[int(fraction)] = component
		o.tokenMu.Unlock()
	})
}

// whoAmI returns the cpn id the canvas framework bound into ctx
// (realComponentBody -> runtime.BindComponentFraction). The bound reporter
// is only reachable through ReportComponentFraction, so a unique token is
// reported and read back.
func (o *oracle) whoAmI(ctx context.Context) string {
	o.tokenMu.Lock()
	o.token--
	token := o.token
	o.tokenMu.Unlock()
	runtime.ReportComponentFraction(ctx, float64(token))
	o.tokenMu.Lock()
	defer o.tokenMu.Unlock()
	id := o.owners[token]
	delete(o.owners, token)
	return id
}

type cpnKey struct{}

func (o *oracle) installModels(confDir string) error {
	if err := models.InitProviderManager(confDir); err != nil {
		return fmt.Errorf("provider catalog: %w", err)
	}
	pm := models.GetProviderManager()
	for i := range pm.Providers {
		pm.Providers[i].ModelDriver = &stubDriver{DummyModel: models.NewDummyModel(nil, models.URLSuffix{}), provider: pm.Providers[i].Name, o: o}
	}
	return nil
}

type stubDriver struct {
	*models.DummyModel
	provider string
	o        *oracle
}

func (d *stubDriver) NewInstance(map[string]string) models.ModelDriver { return d }
func (d *stubDriver) Name() string                                     { return d.provider }

func (d *stubDriver) ChatWithMessages(ctx context.Context, modelName string, messages []models.Message, _ *models.APIConfig, cfg *models.ChatConfig, _ *common.ModelUsage) (*models.ChatResponse, error) {
	answer, err := d.o.answer(ctx, modelName+"@"+d.provider, messages, cfg)
	if err != nil {
		return nil, err
	}
	return &models.ChatResponse{Answer: &answer}, nil
}

func (d *stubDriver) ChatStreamlyWithSender(ctx context.Context, modelName string, messages []models.Message, _ *models.APIConfig, cfg *models.ChatConfig, _ *common.ModelUsage, sender func(*string, *string) error) error {
	answer, err := d.o.answer(ctx, modelName+"@"+d.provider, messages, cfg)
	if err != nil {
		return err
	}
	if err := sender(&answer, nil); err != nil {
		return err
	}
	done := "[DONE]"
	return sender(&done, nil)
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func (o *oracle) answer(ctx context.Context, model string, messages []models.Message, cfg *models.ChatConfig) (string, error) {
	cpn, _ := ctx.Value(cpnKey{}).(string)
	if cpn == "" {
		cpn = o.whoAmI(ctx)
	}
	o.mu.Lock()
	n := o.calls[cpn]
	o.calls[cpn] = n + 1
	o.mu.Unlock()
	key := fmt.Sprintf("%s#%d", cpn, n)

	msgs := make([]chatMessage, 0, len(messages))
	for _, m := range messages {
		content, ok := m.Content.(string)
		if !ok {
			b, _ := json.Marshal(m.Content)
			content = string(b)
		}
		msgs = append(msgs, chatMessage{Role: m.Role, Content: content})
	}
	options := map[string]any{"temperature": nil, "top_p": nil, "max_tokens": nil}
	if cfg != nil {
		if cfg.Temperature != nil {
			options["temperature"] = *cfg.Temperature
		}
		if cfg.TopP != nil {
			options["top_p"] = *cfg.TopP
		}
		if cfg.MaxTokens != nil {
			options["max_tokens"] = *cfg.MaxTokens
		}
	}
	o.print("llm_call", map[string]any{
		"component_id": cpn, "call_index": n, "key": key, "model": model,
		"messages": msgs, "options": options,
	})
	answer, ok := o.answers[key]
	if !ok {
		o.missing.Store(true)
		b, _ := json.MarshalIndent(msgs, "", "  ")
		fmt.Fprintf(os.Stderr, "oracle-go: no recording for %q (model %s); messages:\n%s\n", key, model, b)
		return "", fmt.Errorf("parity: no recording for %q", key)
	}
	if strings.Contains(answer, "<think>") {
		return "", fmt.Errorf("parity: recording %q contains <think>, which this oracle's message emitter does not split", key)
	}
	return answer, nil
}

// componentFactory wraps the production factory: Agent and Categorize get
// their cpn id bound into the invoke context.
func (o *oracle) componentFactory() runtime.ComponentFactory {
	base := runtime.DefaultFactory()
	return func(name string, params map[string]any) (runtime.Component, error) {
		comp, err := base(name, params)
		if err != nil || comp == nil {
			return comp, err
		}
		if strings.EqualFold(name, "Agent") || strings.EqualFold(name, "Categorize") {
			return &boundComponent{inner: comp, o: o}, nil
		}
		return comp, nil
	}
}

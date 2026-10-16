// Command oracle-go runs upstream's agent Retrieval tool (internal/agent/tool
// RetrievalTool → NLPRetrievalAdapter → nlp.RetrievalService → the ES engine)
// against a live Elasticsearch and MySQL, one case per stdin line, and writes
// one JSON line per case to -out. It records the retrieval goldens when a contract version is cut.
//
// Build it inside the upstream tree it should exercise (it is `package main`
// in the ragflow module, so it may import internal/); see
// rust/docker/conformance-retrieval.Dockerfile.
//
// Read-only by construction: the engine talks to Elasticsearch through an
// in-process proxy that forwards only `GET /` and `POST .../_search`, answers
// everything else (the index-template PUTs NewEngine issues) locally, and
// records each forwarded search body. MySQL is only queried.
//
// Differences from the server, all inert for retrieval:
//   - the RuntimeAdapter wrapper (retrievalbridge, which pulls in the whole
//     service package) is re-implemented below in its ten lines;
//   - the enhancer runs RetrievalByChildren like the server's, returns no
//     rank features (correct for datasets without a tag source), and refuses
//     the LLM-backed features;
//   - the query embedding is replayed from a file instead of calling a model.
//
// Environment: ES_URL, ES_USER, ES_PASSWORD, MYSQL_HOST, MYSQL_PORT,
// MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB, RAGFLOW_DICT_PATH; flags -vectors, -out.
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"sync"
	_ "unsafe" // go:linkname

	"ragflow/internal/agent/runtime"
	agenttool "ragflow/internal/agent/tool"
	"ragflow/internal/common"
	"ragflow/internal/dao"
	"ragflow/internal/engine"
	"ragflow/internal/engine/elasticsearch"
	"ragflow/internal/entity"
	models "ragflow/internal/entity/models"
	"ragflow/internal/server/config"
	"ragflow/internal/service/nlp"
	"ragflow/internal/tokenizer"

	gormmysql "gorm.io/driver/mysql"
	"gorm.io/gorm"
	gormLogger "gorm.io/gorm/logger"
)

// engineType is the doc-engine name engine.InitDocEngine records from the
// server config; nlp reads it through engine.GetEngineType().
//
//go:linkname engineType ragflow/internal/engine.engineType
var engineType string

type recorder struct {
	mu       sync.Mutex
	requests []map[string]any
}

func (r *recorder) take() []map[string]any {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := r.requests
	r.requests = nil
	return out
}

// startProxy serves a read-only view of upstream on 127.0.0.1.
func startProxy(upstream string, rec *recorder) (string, error) {
	target, err := url.Parse(upstream)
	if err != nil {
		return "", err
	}
	rp := httputil.NewSingleHostReverseProxy(target)
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		search := r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/_search")
		info := r.Method == http.MethodGet && r.URL.Path == "/"
		if !search && !info {
			w.Header().Set("X-Elastic-Product", "Elasticsearch")
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"acknowledged":true}`))
			return
		}
		if search {
			body, _ := io.ReadAll(r.Body)
			r.Body = io.NopCloser(bytes.NewReader(body))
			var parsed any
			_ = json.Unmarshal(body, &parsed)
			index := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/"), "/_search")
			rec.mu.Lock()
			rec.requests = append(rec.requests, map[string]any{"index": index, "query": r.URL.RawQuery, "body": parsed})
			rec.mu.Unlock()
		}
		rp.ServeHTTP(w, r)
	})
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return "", err
	}
	go func() { _ = http.Serve(ln, handler) }()
	return "http://" + ln.Addr().String(), nil
}

// embedDriver replays recorded query vectors.
type embedDriver struct {
	*models.DummyModel
	vectors map[string][]float64
}

func (d *embedDriver) NewInstance(map[string]string) models.ModelDriver { return d }
func (d *embedDriver) Name() string                                     { return "parity" }
func (d *embedDriver) Embed(_ context.Context, _ *string, req models.EmbedRequest, _ *models.APIConfig, _ *models.EmbeddingConfig, _ *common.ModelUsage) ([]models.EmbeddingData, error) {
	out := make([]models.EmbeddingData, 0, len(req.Texts))
	for i, t := range req.Texts {
		v, ok := d.vectors[t]
		if !ok {
			return nil, fmt.Errorf("parity: no recorded embedding for %q", t)
		}
		out = append(out, models.EmbeddingData{Embedding: v, Index: i})
	}
	return out, nil
}

// enhancer: see the file comment.
type enhancer struct{ docEngine engine.DocEngine }

func (enhancer) CrossLanguages(context.Context, string, string, []string) (string, error) {
	return "", errors.New("parity: cross_languages not supported")
}
func (enhancer) FilterDocuments(context.Context, map[string]any, string, *models.ChatModel, []string, []string) ([]string, error) {
	return nil, errors.New("parity: meta_data_filter not supported")
}
func (enhancer) LabelQuestion(context.Context, string, []*entity.Knowledgebase) map[string]float64 {
	return nil
}
func (enhancer) EnhanceTOC(context.Context, *models.ChatModel, []string, []string, string, int, []map[string]any) ([]map[string]any, error) {
	return nil, errors.New("parity: toc_enhance not supported")
}
func (e enhancer) RetrieveByChildren(ctx context.Context, chunks []map[string]any, tenantIDs []string) []map[string]any {
	return nlp.RetrievalByChildren(chunks, tenantIDs, e.docEngine, ctx)
}

// runtimeAdapter is retrievalbridge.RuntimeAdapter.Search.
type runtimeAdapter struct{ svc agenttool.RetrievalService }

func (a runtimeAdapter) Search(ctx context.Context, db *gorm.DB, req agenttool.RetrievalRequest) ([]agenttool.RetrievalChunk, error) {
	if db == nil {
		db = dao.DB
	}
	f := false
	req.AllowDenseFallback = &f
	if req.RankFeature != nil && len(*req.RankFeature) == 0 {
		req.RankFeature = nil
	}
	return a.svc.Search(ctx, db, req)
}

type parityCase struct {
	Name string          `json:"name"`
	Args json.RawMessage `json:"args"`
}

func env(k string) string { return os.Getenv(k) }

func main() {
	vectorsPath := flag.String("vectors", "", "JSON object: query text -> embedding")
	outPath := flag.String("out", "", "result file (JSON lines); stdout carries upstream's logs")
	flag.Parse()
	if err := run(*vectorsPath, *outPath); err != nil {
		fmt.Fprintln(os.Stderr, "oracle-go:", err)
		os.Exit(1)
	}
}

func run(vectorsPath, outPath string) error {
	ctx := context.Background()
	// Upstream logs to stdout; results go to outPath.
	if err := common.InitLogger("warn", common.FileOutput{}, "oracle-go"); err != nil {
		return fmt.Errorf("logger: %w", err)
	}
	outFile, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer outFile.Close()
	vectors := map[string][]float64{}
	if vectorsPath != "" {
		b, err := os.ReadFile(vectorsPath)
		if err != nil {
			return err
		}
		if err := json.Unmarshal(b, &vectors); err != nil {
			return err
		}
	}

	dictPath := env("RAGFLOW_DICT_PATH")
	if err := tokenizer.Init(&tokenizer.PoolConfig{DictPath: dictPath, MinSize: 1, MaxSize: 4}); err != nil {
		return fmt.Errorf("tokenizer: %w", err)
	}
	if err := nlp.InitQueryBuilderFromTokenizer(dictPath); err != nil {
		return fmt.Errorf("query builder: %w", err)
	}
	engineType = "elasticsearch"
	tokenizer.SetEngineType("elasticsearch")

	rec := &recorder{}
	proxy, err := startProxy(env("ES_URL"), rec)
	if err != nil {
		return err
	}
	docEngine, err := elasticsearch.NewEngine(ctx, config.ElasticsearchConfig{
		Hosts: proxy, Username: env("ES_USER"), Password: env("ES_PASSWORD"),
	})
	if err != nil {
		return fmt.Errorf("es engine: %w", err)
	}
	rec.take() // NewEngine's own traffic

	port := env("MYSQL_PORT")
	if port == "" {
		port = "3306"
	}
	dsn := fmt.Sprintf("%s:%s@tcp(%s:%s)/%s?charset=utf8mb4&parseTime=True&loc=Local",
		env("MYSQL_USER"), env("MYSQL_PASSWORD"), env("MYSQL_HOST"), port, env("MYSQL_DB"))
	dao.DB, err = gorm.Open(gormmysql.Open(dsn), &gorm.Config{Logger: gormLogger.Default.LogMode(gormLogger.Silent)})
	if err != nil {
		return fmt.Errorf("mysql: %w", err)
	}

	driver := &embedDriver{DummyModel: models.NewDummyModel(nil, models.URLSuffix{}), vectors: vectors}
	resolver := func(_ context.Context, _ string, mt entity.ModelType, _ string) (models.ModelDriver, string, *models.APIConfig, int, error) {
		if mt != entity.ModelTypeEmbedding {
			return nil, "", nil, 0, fmt.Errorf("parity: model type %v not supported", mt)
		}
		return driver, "parity-embedding", &models.APIConfig{}, 8192, nil
	}
	adapter := agenttool.NewNLPRetrievalAdapterFromDeps(docEngine, dao.NewDocumentDAO(), resolver, enhancer{docEngine: docEngine})
	agenttool.SetRetrievalService(runtimeAdapter{svc: adapter})

	out := bufio.NewWriter(outFile)
	defer out.Flush()
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 1<<20), 1<<26)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var c parityCase
		if err := json.Unmarshal([]byte(line), &c); err != nil {
			return fmt.Errorf("case: %w", err)
		}
		state := runtime.NewCanvasState("parity", "parity")
		cctx := runtime.WithState(ctx, state)
		toolOut, toolErr := agenttool.NewRetrievalTool().InvokableRun(cctx, string(c.Args))
		result := map[string]any{"name": c.Name, "requests": rec.take()}
		var envelope any
		if err := json.Unmarshal([]byte(toolOut), &envelope); err == nil {
			result["envelope"] = envelope
		} else {
			result["envelope_raw"] = toolOut
		}
		if toolErr != nil {
			result["error"] = toolErr.Error()
		}
		result["references"] = state.GetRetrievalReference()
		b, err := json.Marshal(result)
		if err != nil {
			return err
		}
		fmt.Fprintln(out, string(b))
		out.Flush()
	}
	return sc.Err()
}

// Command esproxy is a search-only view of an Elasticsearch cluster: it
// forwards `GET /` and `POST <index>/_search`, and answers everything else
// itself with `{"acknowledged":true}`. The end-to-end check points both
// servers at it so neither can write to the real indices (the Go server's
// engine constructor PUTs index templates at start-up).
//
//	ES_UPSTREAM=http://es01:9200 ES_USER=... ES_PASSWORD=... esproxy [-listen :9200]
package main

import (
	"flag"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
)

func main() {
	listen := flag.String("listen", ":9200", "address to listen on")
	flag.Parse()
	target, err := url.Parse(os.Getenv("ES_UPSTREAM"))
	if err != nil || target.Host == "" {
		log.Fatal("ES_UPSTREAM must be the Elasticsearch base URL")
	}
	user, pass := os.Getenv("ES_USER"), os.Getenv("ES_PASSWORD")
	rp := httputil.NewSingleHostReverseProxy(target)
	director := rp.Director
	rp.Director = func(r *http.Request) {
		director(r)
		if user != "" {
			r.SetBasicAuth(user, pass)
		}
	}
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		search := r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/_search")
		info := r.Method == http.MethodGet && r.URL.Path == "/"
		if !search && !info {
			w.Header().Set("X-Elastic-Product", "Elasticsearch")
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"acknowledged":true}`))
			return
		}
		rp.ServeHTTP(w, r)
	})
	log.Fatal(http.ListenAndServe(*listen, handler))
}

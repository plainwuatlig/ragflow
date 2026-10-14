// Command gen writes the fixtures ragflow-server's byte-exact tests read,
// using upstream's own code, so they pin the Rust implementation to what the
// Go server produces. Run it from the repository root:
//
//	GOTOOLCHAIN=go1.27.2 CGO_ENABLED=0 go run ./rust/crates/ragflow-server/fixtures/gen \
//	    > rust/crates/ragflow-server/fixtures/go_tokens.json
//
// The session-DSL fixtures need unexported functions of internal/service and
// come from gen_service_test.go.txt (see the comment there).
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"ragflow/internal/common"
)

type tokenCase struct {
	Name        string `json:"name"`
	Secret      string `json:"secret"`
	AccessToken string `json:"access_token"`
	Token       string `json:"token"`
}

func main() {
	secrets := []string{"0123456789abcdef0123456789abcdef", "infiniflow-token", "pässwörd-with-ünicode"}
	access := []struct{ name, value string }{
		{"hex32", strings.Repeat("a1b2c3d4", 4)},
		{"hex32-other", "00112233445566778899aabbccddeeff"},
		{"invalidated", "INVALID_0123456789abcdef0123456789abcdef"},
		{"long-compressible", strings.Repeat("ab", 400)},
		{"unicode", "令牌-token-ü-0123456789abcdef0123456789"},
		{"quote-and-slash", `with "quotes" and \ slash and </script> & more 0123456789`},
	}
	var cases []tokenCase
	for _, s := range secrets {
		for _, a := range access {
			tok, err := common.DumpAccessToken(a.value, s)
			if err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
			back, err := common.ExtractAccessToken(tok, s)
			if err != nil || back != a.value {
				fmt.Fprintln(os.Stderr, "round trip failed", a.name, err)
				os.Exit(1)
			}
			cases = append(cases, tokenCase{Name: a.name, Secret: s, AccessToken: a.value, Token: tok})
		}
	}
	out := json.NewEncoder(os.Stdout)
	out.SetIndent("", "  ")
	out.SetEscapeHTML(false)
	if err := out.Encode(map[string]any{"cases": cases}); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

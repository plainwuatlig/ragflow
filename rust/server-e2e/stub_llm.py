#!/usr/bin/env python3
"""A deterministic, OpenAI-compatible stand-in for a chat model.

    stub_llm.py [--port 8000] [--log requests.jsonl] [--rules rules.json]

POST /v1/chat/completions   chat, streaming or not, with usage
POST /v1/embeddings         a hash-derived vector per input
GET  /v1/models             liveness

The answers depend only on the request, so two servers that send the same
prompts get the same replies, and any difference in what they send shows up
as a different reply:

* a classification request (a system prompt that lists categories after
  "classify it into ONE of the following categories") is answered with the
  category its last user line hashes to, unless --rules names one: a JSON
  list of [substring, category] pairs, the first whose substring occurs in
  the question wins;
* every other request is answered with "STUB-<sha1 of the messages>: " and a
  prefix of the last user message, streamed in 7-character pieces.

A last user message containing the word SLOW is streamed one piece every
half second, which lets a test cancel a run in flight.

Every request body is appended to --log as one JSON line, in arrival order.
"""
import argparse
import hashlib
import json
import math
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCK = threading.Lock()
COUNTER = [0]
LOG = [None]
RULES = []


def text_of(content):
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def categories(system):
    m = re.search(r"ONE of the following categories:\n((?:- .*\n)+)", system)
    if not m:
        return []
    return [line[2:].strip() for line in m.group(1).splitlines()]


def answer_for(messages):
    system = "\n".join(text_of(m.get("content")) for m in messages if m.get("role") == "system")
    last_user = ""
    for m in messages:
        if m.get("role") == "user":
            last_user = text_of(m.get("content"))
    cats = categories(system)
    if cats:
        quoted = re.findall(r'USER: "((?:[^"\\]|\\.)*)" ->', last_user)
        key = quoted[-1] if quoted else last_user
        for needle, category in RULES:
            if needle in key and category in cats:
                return category
        return cats[int(hashlib.sha1(key.encode()).hexdigest(), 16) % len(cats)]
    # Clock readings in a prompt (sys.date) would make two otherwise equal
    # requests differ; the digest ignores them.
    canonical = re.sub(r"\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d", "<date>",
                       json.dumps(messages, sort_keys=True, ensure_ascii=False))
    digest = hashlib.sha1(canonical.encode()).hexdigest()[:10]
    return f"STUB-{digest}: " + re.sub(r"\s+", " ", last_user).strip()[:160]


def tokens(text):
    return max(1, math.ceil(len(text) / 3))


def usage_for(messages, answer):
    prompt = sum(tokens(text_of(m.get("content"))) for m in messages)
    completion = tokens(answer)
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def reply(self, status, body):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            return self.reply(200, {"object": "list", "data": [{"id": "stub", "object": "model"}]})
        self.reply(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        with LOCK:
            COUNTER[0] += 1
            n = COUNTER[0]
            if LOG[0]:
                with open(LOG[0], "a", encoding="utf-8") as f:
                    f.write(json.dumps({"n": n, "path": self.path, "body": body}, ensure_ascii=False) + "\n")
        if self.path.endswith("/embeddings"):
            inputs = body.get("input") or []
            if isinstance(inputs, str):
                inputs = [inputs]
            data = []
            for i, text in enumerate(inputs):
                h = hashlib.sha256(text.encode()).digest()
                data.append({"object": "embedding", "index": i, "embedding": [b / 255.0 - 0.5 for b in h[:16]]})
            return self.reply(200, {"object": "list", "data": data, "model": body.get("model", "stub"),
                                    "usage": {"prompt_tokens": 1, "total_tokens": 1}})
        if not self.path.endswith("/chat/completions"):
            return self.reply(404, {"error": "not found"})
        messages = body.get("messages") or []
        answer = answer_for(messages)
        usage = usage_for(messages, answer)
        model = body.get("model", "stub")
        if not body.get("stream"):
            return self.reply(200, {
                "id": f"stub-{n}", "object": "chat.completion", "created": 0, "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
                "usage": usage,
            })
        slow = "SLOW" in text_of(messages[-1].get("content")) if messages else False
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def chunk(payload):
            data = b"data: " + json.dumps(payload, ensure_ascii=False).encode() + b"\n\n"
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        def delta(d, finish=None):
            return {"id": f"stub-{n}", "object": "chat.completion.chunk", "created": 0, "model": model,
                    "choices": [{"index": 0, "delta": d, "finish_reason": finish}]}

        try:
            chunk(delta({"role": "assistant", "content": ""}))
            pieces = [answer[i:i + 7] for i in range(0, len(answer), 7)] or [""]
            if slow:
                pieces += ["..."] * 40
            for piece in pieces:
                if slow:
                    time.sleep(0.5)
                chunk(delta({"content": piece}))
            chunk(delta({}, "stop"))
            if (body.get("stream_options") or {}).get("include_usage"):
                chunk({"id": f"stub-{n}", "object": "chat.completion.chunk", "created": 0, "model": model,
                       "choices": [], "usage": usage})
            data = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--log")
    ap.add_argument("--rules")
    args = ap.parse_args()
    LOG[0] = args.log
    if args.rules and os.path.exists(args.rules):
        RULES.extend(json.load(open(args.rules, encoding="utf-8")))
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()

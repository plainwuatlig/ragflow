#!/usr/bin/env python3
"""End-to-end comparison of upstream's Go server and ragflow-server.

Both servers run against the same throwaway copy of a RAGFlow deployment
(stack.sh, go.sh, rust.sh), with the same stub model. This script drives them
over HTTP with the same tokens and questions and compares what comes back and
what they write:

  * the SSE events of every turn, ids and timestamps normalised;
  * the rows of api_4_conversation, _message and _reference each leaves;
  * the stub model's requests, so differences in prompts show up;
  * Go's session endpoints over rows Rust wrote, and Rust's over Go's;
  * authentication on a matrix of good and bad tokens;
  * cancellation between the two processes, both ways;
  * the beta-token binding, where the servers differ on purpose.

    e2e.py --state STATE_DIR --canvas CANVAS_ID --probes probes.json [--out OUT]

Exit status is 0 when nothing failed unexpectedly. `KNOWN` rows are
differences that were ruled on and are listed in KNOWN below.
"""
import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib

GO = "http://11.250.0.21:9380"
RUST = "http://11.250.0.22:9380"
DATE = re.compile(r"\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d")
VOLATILE = {"created_at", "elapsed_time", "_created_time", "_elapsed_time", "message_id",
            "session_id", "task_id", "create_time", "update_time", "create_date", "update_date"}

results = []  # (name, status, detail)
TIMES = {"go": [], "rust": []}


def record(name, status, detail=""):
    results.append((name, status, detail))
    mark = {"PASS": " ok ", "FAIL": "FAIL", "KNOWN": "know", "SKIP": "skip"}[status]
    print(f"[{mark}] {name}" + (f"\n         {detail}" if detail and status != "PASS" else ""), flush=True)


# ---------------------------------------------------------------------------
# The stack

class Stack:
    def __init__(self, state):
        self.state = state
        self.secrets = dict(
            line.strip().split("=", 1) for line in open(os.path.join(state, "secrets.env")) if "=" in line)

    def sql(self, query, check=True):
        cmd = ["docker", "exec", "-i", "-e", "MYSQL_PWD=" + self.secrets["E2E_MYSQL_PASSWORD"], "rfe2e-mysql",
               "mysql", "-uroot", "--batch", "--raw", "-N", "--default-character-set=utf8mb4", "rag_flow"]
        p = subprocess.run(cmd, input=query.encode(), capture_output=True)
        if check and p.returncode != 0:
            raise RuntimeError(p.stderr.decode(errors="replace")[:500])
        return p.stdout.decode()

    def rows(self, query):
        return [json.loads(line) for line in self.sql(query).splitlines() if line.strip()]

    def redis(self, *args):
        cmd = ["docker", "exec", "rfe2e-kvrocks", "redis-cli", "-a", self.secrets["E2E_REDIS_PASSWORD"],
               "--no-auth-warning", "-n", "1", *args]
        return subprocess.run(cmd, capture_output=True).stdout.decode().strip()

    def secret_key(self):
        return self.redis("GET", "ragflow:system:secret_key")


def b64(b):
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def mint(access_token, secret, ts):
    """itsdangerous URLSafeTimedSerializer(secret).dumps(access_token) at time ts."""
    payload = b64(json.dumps(access_token).encode())
    stamp = b64(ts.to_bytes(8, "big").lstrip(b"\0"))
    value = f"{payload}.{stamp}"
    key = hashlib.sha1(b"itsdangerous" + b"signer" + secret.encode()).digest()
    return value + "." + b64(hmac.new(key, value.encode(), hashlib.sha1).digest())


# ---------------------------------------------------------------------------
# HTTP

def call(base, method, path, token=None, body=None, headers=None, timeout=300):
    req = urllib.request.Request(base + path, method=method,
                                 data=None if body is None else json.dumps(body).encode())
    if body is not None:
        req.add_header("Content-Type", "application/json")
    if token is not None:
        req.add_header("Authorization", token)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def jcall(base, method, path, **kw):
    status, headers, raw = call(base, method, path, **kw)
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, raw.decode(errors="replace")


def sse(base, path, token, body, timeout=300):
    status, headers, raw = call(base, "POST", path, token=token, body=body, timeout=timeout)
    text = raw.decode()
    frames = []
    for chunk in text.split("\n\n"):
        if not chunk.startswith("data:"):
            continue
        data = chunk[5:]
        frames.append("[DONE]" if data == "[DONE]" else json.loads(data))
    return status, frames, text


# ---------------------------------------------------------------------------
# Normalisation

def strip(v):
    if isinstance(v, dict):
        return {k: strip(x) for k, x in v.items() if k not in VOLATILE}
    if isinstance(v, list):
        return [strip(x) for x in v]
    return v


def events(frames):
    """(event, data) pairs with consecutive message frames merged."""
    out = []
    for f in frames:
        if f == "[DONE]":
            out.append(("[DONE]", None))
            continue
        ev = f.get("event")
        if ev is None and "code" in f:
            out.append(("error-frame", strip(f)))
            continue
        data = strip(f.get("data"))
        if ev == "message" and out and out[-1][0] == "message":
            prev = out[-1][1]
            prev["content"] = prev.get("content", "") + data.get("content", "")
            continue
        out.append((ev, data))
    return out


DEFERRED_SEEN = {"agent_outputs": 0, "message_inputs": 0}


def deferred_agent_fix(ev):
    """Go reports an unopened deferred Agent stream where Rust reports the
    Agent's answer: `{"content": {"Open": null}}` as the Agent's node_finished
    outputs and as the consuming Message's inputs. Ruled difference; both
    sides are mapped to one placeholder."""
    if ev[0] in ("node_started", "node_finished") and ev[1]:
        data = ev[1]
        if data.get("component_name") == "Agent" and ev[0] == "node_finished":
            DEFERRED_SEEN["agent_outputs"] += 1
            return (ev[0], {**data, "outputs": "<deferred agent outputs>"})
        if data.get("component_name") == "Message" and data.get("inputs") in ({"content": {"Open": None}}, {}):
            if data.get("inputs"):
                DEFERRED_SEEN["message_inputs"] += 1
            return (ev[0], {**data, "inputs": {}})
    return ev


def diff_events(go, rust):
    a = [deferred_agent_fix(e) for e in events(go)]
    b = [deferred_agent_fix(e) for e in events(rust)]
    problems = []
    if [e[0] for e in a] != [e[0] for e in b]:
        problems.append(f"event sequence: go={[e[0] for e in a]} rust={[e[0] for e in b]}")
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            problems.append(f"event {i} ({x[0]}):\n go  : {json.dumps(x[1], ensure_ascii=False)[:600]}\n rust: {json.dumps(y[1], ensure_ascii=False)[:600]}")
    return problems


def answer_of(frames):
    text = ""
    for f in frames:
        if f != "[DONE]" and f.get("event") == "message":
            text += f["data"].get("content", "")
    return text


# ---------------------------------------------------------------------------
# Rows

def session_rows(stack, sid):
    q = f"""SELECT JSON_OBJECT('id',id,'name',name,'dialog_id',dialog_id,'user_id',user_id,'exp_user_id',exp_user_id,
      'tokens',tokens,'source',source,'dsl',dsl,'duration',duration,'round',`round`,'thumb_up',thumb_up,'errors',errors,
      'version_title',version_title,'create_time',create_time,'update_time',update_time) FROM api_4_conversation WHERE id='{sid}'"""
    rows = stack.rows(q)
    if not rows:
        return None
    s = rows[0]
    s["dsl"] = json.loads(s["dsl"]) if s.get("dsl") else None
    s["messages"] = stack.rows(f"""SELECT JSON_OBJECT('position',position,'message_id',message_id,'role',role,'content',content,
      'content_type',content_type,'status',status,'thumb_up',thumb_up,'feedback',feedback,'created_at',created_at,'metadata',metadata)
      FROM api_4_conversation_message WHERE conversation_id='{sid}' ORDER BY position""")
    s["references"] = stack.rows(f"""SELECT JSON_OBJECT('position',position,'message_position',message_position,'reference',reference)
      FROM api_4_conversation_reference WHERE conversation_id='{sid}' ORDER BY position""")
    for r in s["references"]:
        r["reference"] = json.loads(r["reference"])
    return s


def norm_session(s, ids):
    """Replace the run-specific ids; drop times and the frozen sys.date."""
    if s is None:
        return None
    s = json.loads(json.dumps(s))
    text = json.dumps(s, ensure_ascii=False)
    for real, name in ids.items():
        text = text.replace(real, name)
    s = json.loads(text)
    for k in ("create_time", "update_time"):
        s.pop(k, None)
    if s.get("dsl"):
        s["dsl"].get("globals", {}).pop("sys.date", None)
    for m in s["messages"]:
        m.pop("created_at", None)
    return s


def diff_json(a, b, path=""):
    out = []
    if type(a) != type(b):
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float)) and a == b):
            out.append(f"{path}: {json.dumps(a, ensure_ascii=False)[:200]} != {json.dumps(b, ensure_ascii=False)[:200]}")
    elif isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}.{k}: only in rust: {json.dumps(b[k], ensure_ascii=False)[:200]}")
            elif k not in b:
                out.append(f"{path}.{k}: only in go: {json.dumps(a[k], ensure_ascii=False)[:200]}")
            else:
                out += diff_json(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff_json(x, y, f"{path}[{i}]")
    elif a != b:
        out.append(f"{path}: {json.dumps(a, ensure_ascii=False)[:200]} != {json.dumps(b, ensure_ascii=False)[:200]}")
    return out


class StubLog:
    def __init__(self, path):
        self.path = path

    def mark(self):
        try:
            return sum(1 for _ in open(self.path, encoding="utf-8"))
        except FileNotFoundError:
            return 0

    def since(self, mark):
        lines = open(self.path, encoding="utf-8").read().splitlines()[mark:]
        out = []
        for line in lines:
            if line.strip():
                body = json.loads(line)["body"]
                # Clock readings (sys.date) differ between two runs.
                text = DATE.sub("<date>", json.dumps(body, ensure_ascii=False))
                body = json.loads(text)
                out.append({k: body.get(k) for k in ("model", "messages", "stream", "temperature", "top_p", "max_tokens", "stream_options")})
        return out


# ---------------------------------------------------------------------------
# The gate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--canvas", required=True)
    ap.add_argument("--probes", required=True, help="JSON: {conversations: [{name, turns: [{id, msg}]}], rules: [[substring, category]]}")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default="", help="comma list of sections: probes,misc,auth,beta,team,cancel")
    args = ap.parse_args()
    out = args.out or os.path.join(args.state, "out")
    os.makedirs(out, exist_ok=True)
    stack = Stack(args.state)
    stub = StubLog(os.path.join(args.state, "stub", "requests.jsonl"))
    only = set(filter(None, args.only.split(",")))
    want = lambda s: not only or s in only
    plan = json.load(open(args.probes, encoding="utf-8"))
    canvas = args.canvas

    api = "ragflow-e2e-owner"
    secret = stack.secret_key()

    if want("probes") or want("rows") or want("endpoints"):
        probes(stack, stub, plan, canvas, api, out)
    if want("misc"):
        misc(stack, stub, canvas, api, out)
    if want("auth"):
        auth_matrix(stack, secret, canvas, api)
    if want("beta"):
        beta(stack, canvas, out)
    if want("team"):
        team(stack, canvas, api)
    if want("cancel"):
        cancel(stack, canvas, api)

    if TIMES["go"]:
        med = lambda v: sorted(v)[len(v) // 2]
        print(f"\nturn latency over {len(TIMES['go'])} turns (stub model): go median {med(TIMES['go'])*1000:.0f} ms, rust median {med(TIMES['rust'])*1000:.0f} ms")
    if DEFERRED_SEEN["agent_outputs"] or DEFERRED_SEEN["message_inputs"]:
        record("deferred Agent: Go's unopened stream vs Rust's answer (ruled difference)", "KNOWN",
               f"{DEFERRED_SEEN['agent_outputs']} Agent node_finished outputs, {DEFERRED_SEEN['message_inputs']} Message inputs")
    failed = [r for r in results if r[1] == "FAIL"]
    known = [r for r in results if r[1] == "KNOWN"]
    print(f"\n{len(results)} checks: {len([r for r in results if r[1]=='PASS'])} passed, {len(known)} known, {len(failed)} failed")
    json.dump(results, open(os.path.join(out, "results.json"), "w"), ensure_ascii=False, indent=1)
    sys.exit(1 if failed else 0)


def probes(stack, stub, plan, canvas, api, out):
    turns_total = 0
    sessions = {}  # conversation name -> {"go": sid, "rust": sid}
    for conv in plan["conversations"]:
        sid = {"go": None, "rust": None}
        for turn in conv["turns"]:
            turns_total += 1
            name = f"{conv['name']}/{turn['id']}"
            frames, llm = {}, {}
            for side, base in (("go", GO), ("rust", RUST)):
                body = {"agent_id": canvas, "query": turn["msg"], "stream": True}
                if sid[side]:
                    body["session_id"] = sid[side]
                mark = stub.mark()
                t0 = time.time()
                status, fr, raw = sse(base, "/api/v1/agents/chat/completions", api, body)
                TIMES[side].append(time.time() - t0)
                if status != 200:
                    record(f"{name} {side} http", "FAIL", f"HTTP {status}: {raw[:300]}")
                    continue
                frames[side] = fr
                llm[side] = stub.since(mark)
                first = next((f for f in fr if f != "[DONE]" and "session_id" in f), None)
                if first:
                    sid[side] = first["session_id"]
            if len(frames) < 2:
                continue
            json.dump({"go": frames["go"], "rust": frames["rust"]}, open(os.path.join(out, f"{name.replace('/', '_')}.frames.json"), "w"), ensure_ascii=False)
            problems = diff_events(frames["go"], frames["rust"])
            ga, ra = answer_of(frames["go"]), answer_of(frames["rust"])
            if ga != ra:
                problems.append(f"answer: go={ga[:200]!r} rust={ra[:200]!r}")
            record(f"{name}: events", "FAIL" if problems else "PASS", "\n         ".join(problems[:6]))
            lp = diff_json(llm["go"], llm["rust"])
            record(f"{name}: model requests ({len(llm['go'])} go / {len(llm['rust'])} rust)", "FAIL" if lp else "PASS", "\n         ".join(lp[:6]))
        sessions[conv["name"]] = sid
        if True:
            ids = {sid["go"]: "SESSION", sid["rust"]: "SESSION"}
            g = session_rows(stack, sid["go"]) if sid["go"] else None
            r = session_rows(stack, sid["rust"]) if sid["rust"] else None
            if g and r:
                gm = {m["message_id"]: f"MSG{i}" for i, m in enumerate(g["messages"]) if m["role"] == "user"}
                rm = {m["message_id"]: f"MSG{i}" for i, m in enumerate(r["messages"]) if m["role"] == "user"}
                ng = norm_session(g, {**ids, **gm})
                nr = norm_session(r, {**ids, **rm})
                d = diff_json(ng, nr)
                json.dump({"go": ng, "rust": nr}, open(os.path.join(out, f"{conv['name']}.rows.json"), "w"), ensure_ascii=False, indent=1)
                record(f"{conv['name']}: api_4_conversation* rows ({len(g['messages'])} messages, {len(g['references'])} references)", "FAIL" if d else "PASS", "\n         ".join(d[:8]))
    # Session endpoints.
    for conv in plan["conversations"]:
        sid = sessions[conv["name"]]
        if not (sid["go"] and sid["rust"]):
            continue
        # Same rows through both servers' endpoints.
        for label, s in (("go-written", sid["go"]), ("rust-written", sid["rust"])):
            gs, gd = jcall(GO, "GET", f"/api/v1/agents/{canvas}/sessions/{s}", token=api)
            rs, rd = jcall(RUST, "GET", f"/api/v1/agents/{canvas}/sessions/{s}", token=api)
            d = diff_json(strip_resp(gd), strip_resp(rd)) if gs == rs == 200 else [f"status {gs}/{rs}"]
            record(f"{conv['name']}: GET session ({label}), go endpoint vs rust endpoint", "FAIL" if d else "PASS", "\n         ".join(d[:6]))
            gs, gd = jcall(GO, "GET", f"/api/v1/agents/{canvas}/sessions?id={s}", token=api)
            rs, rd = jcall(RUST, "GET", f"/api/v1/agents/{canvas}/sessions?id={s}", token=api)
            d = diff_json(strip_resp(gd), strip_resp(rd)) if gs == rs == 200 else [f"status {gs}/{rs}"]
            record(f"{conv['name']}: list by id ({label}), go endpoint vs rust endpoint", "FAIL" if d else "PASS", "\n         ".join(d[:6]))
        # Go's endpoint over rust-written vs go-written rows.
        _, gd = jcall(GO, "GET", f"/api/v1/agents/{canvas}/sessions/{sid['go']}", token=api)
        _, rd = jcall(GO, "GET", f"/api/v1/agents/{canvas}/sessions/{sid['rust']}", token=api)
        ids = {sid["go"]: "SESSION", sid["rust"]: "SESSION"}
        for d_, m in ((gd, None), (rd, None)):
            pass
        gm = {m["id"]: f"MSG{i}" for i, m in enumerate((gd.get("data") or {}).get("message", [])) if m.get("role") == "user"}
        rm = {m["id"]: f"MSG{i}" for i, m in enumerate((rd.get("data") or {}).get("message", [])) if m.get("role") == "user"}
        a = norm_resp(gd, {**ids, **gm})
        b = norm_resp(rd, {**ids, **rm})
        if "dsl" in (a.get("data") or {}):
            a["data"]["dsl"].get("globals", {}).pop("sys.date", None)
            b["data"]["dsl"].get("globals", {}).pop("sys.date", None)
        d = diff_json(a, b)
        record(f"{conv['name']}: go's GET shows the rust-written session like the go-written one", "FAIL" if d else "PASS", "\n         ".join(d[:8]))
    # Listing.
    gs, gd = jcall(GO, "GET", f"/api/v1/agents/{canvas}/sessions?page_size=100&dsl=false", token=api)
    rs, rd = jcall(RUST, "GET", f"/api/v1/agents/{canvas}/sessions?page_size=100&dsl=false", token=api)
    record("list sessions: total agrees", "PASS" if gd.get("total") == rd.get("total") else "FAIL", f"go={gd.get('total')} rust={rd.get('total')}")
    for q in ("keywords=%E5%81%9C&page_size=3", "orderby=create_time&desc=false&page_size=5", "include_history=false&page_size=4"):
        gs, gd = jcall(GO, "GET", f"/api/v1/agents/{canvas}/sessions?{q}", token=api)
        rs, rd = jcall(RUST, "GET", f"/api/v1/agents/{canvas}/sessions?{q}", token=api)
        d = diff_json(strip_resp(gd), strip_resp(rd))
        record(f"list sessions ?{q}", "FAIL" if d else "PASS", "\n         ".join(d[:6]))


def misc(stack, stub, canvas, api, out):
    both = lambda f: {"go": f(GO), "rust": f(RUST)}

    def same(name, a, b, normalise=lambda x: x):
        d = diff_json(normalise(a), normalise(b))
        record(name, "FAIL" if d else "PASS", "\n         ".join(d[:6]))

    def cleaned(body):
        return strip(body)

    # Non-streaming completions.
    r = both(lambda base: jcall(base, "POST", "/api/v1/agents/chat/completions", token=api,
                                body={"agent_id": canvas, "query": "停車場有充電樁嗎？"}))
    same("non-stream completion: same reply", r["go"][1], r["rust"][1], cleaned)
    # The agentbot stream carries the same events as the chat stream.
    ev = both(lambda base: sse(base, f"/api/v1/agentbots/{canvas}/completions", api, {"query": "附近有什麼好吃的小吃？"})[1])
    problems = diff_events(ev["go"], ev["rust"])
    record("agentbot completions: events", "FAIL" if problems else "PASS", "\n         ".join(problems[:4]))
    # Errors.
    for name, body in (
        ("unknown agent", {"agent_id": "0" * 32, "query": "hi", "stream": True}),
        ("no agent_id", {"query": "hi", "stream": True}),
        ("unknown session id", {"agent_id": canvas, "query": "hi", "session_id": "f" * 32, "stream": True}),
        ("last message not from the user", {"agent_id": canvas, "messages": [{"role": "assistant", "content": "x"}], "stream": True}),
    ):
        r = both(lambda base: jcall(base, "POST", "/api/v1/agents/chat/completions", token=api, body=body))
        same(f"error: {name}", r["go"][1], r["rust"][1])
    r = both(lambda base: jcall(base, "POST", f"/api/v1/agentbots/{'0' * 32}/completions", token=api, body={"query": "hi"}))
    same("error: agentbot, unknown agent", r["go"][1], r["rust"][1])
    r = both(lambda base: jcall(base, "POST", f"/api/v1/agentbots/{canvas}/completions", token=api, body={"query": "hi", "session_id": "f" * 32}))
    same("error: agentbot, unknown session", r["go"][1], r["rust"][1])
    # An empty question.
    counts = {}
    for side, base in (("go", GO), ("rust", RUST)):
        before = stack.sql(f"SELECT COUNT(*) FROM api_4_conversation WHERE dialog_id='{canvas}'").strip()
        st, fr, raw = sse(base, "/api/v1/agents/chat/completions", api, {"agent_id": canvas, "stream": True})
        after = stack.sql(f"SELECT COUNT(*) FROM api_4_conversation WHERE dialog_id='{canvas}'").strip()
        counts[side] = (st, [f.get("event") if f != "[DONE]" else f for f in fr], int(after) - int(before))
    same("empty question: events and rows kept", counts["go"], counts["rust"])
    # Form inputs and sessions API.
    r = both(lambda base: jcall(base, "POST", f"/api/v1/agents/{canvas}/sessions", token=api, body={"name": "from api", "source": "e2e"}))
    ids = {r["go"][1]["data"]["id"]: "SESSION", r["rust"][1]["data"]["id"]: "SESSION"}
    same("create session: reply", norm_resp(r["go"][1], ids), norm_resp(r["rust"][1], ids))
    rows = {s: session_rows(stack, r[s][1]["data"]["id"]) for s in ("go", "rust")}
    same("create session: rows", norm_session(rows["go"], ids), norm_session(rows["rust"], ids))
    # The created session can be run by both servers' ids.
    for side, base in (("go", GO), ("rust", RUST)):
        sid = r[side][1]["data"]["id"]
        st, fr, raw = sse(base, "/api/v1/agents/chat/completions", api, {"agent_id": canvas, "query": "停車場有充電樁嗎？", "session_id": sid, "stream": True})
        ok = any(f != "[DONE]" and f.get("event") == "workflow_finished" for f in fr)
        record(f"a session made by POST runs on {side}", "PASS" if ok else "FAIL", str(fr)[:200])
    other = both(lambda base: jcall(base, "POST", f"/api/v1/agents/{canvas}/sessions", token=api, body={}))
    for side in ("go", "rust"):
        sid = other[side][1]["data"]["id"]
        st, body = jcall(GO if side == "go" else RUST, "DELETE", f"/api/v1/agents/{canvas}/sessions/{sid}", token=api)
        gone = stack.sql(f"SELECT COUNT(*) FROM api_4_conversation WHERE id='{sid}'").strip()
        record(f"delete one session on {side}: {body} rows left {gone}", "PASS" if body.get("data") is True and gone == "0" else "FAIL")
    st1, b1 = jcall(GO, "DELETE", f"/api/v1/agents/{canvas}/sessions/{'f' * 32}", token=api)
    st2, b2 = jcall(RUST, "DELETE", f"/api/v1/agents/{canvas}/sessions/{'f' * 32}", token=api)
    same("delete an unknown session", b1, b2)
    # Batch delete.
    made = [jcall(GO, "POST", f"/api/v1/agents/{canvas}/sessions", token=api, body={"name": f"b{i}"})[1]["data"]["id"] for i in range(3)]
    st1, b1 = jcall(GO, "DELETE", f"/api/v1/agents/{canvas}/sessions?ids={made[0]},{made[0]},{'e' * 32}", token=api)
    made2 = [jcall(RUST, "POST", f"/api/v1/agents/{canvas}/sessions", token=api, body={"name": f"b{i}"})[1]["data"]["id"] for i in range(3)]
    st2, b2 = jcall(RUST, "DELETE", f"/api/v1/agents/{canvas}/sessions?ids={made2[0]},{made2[0]},{'e' * 32}", token=api)
    same("batch delete with a duplicate and an unknown id", b1, b2)
    st1, b1 = jcall(GO, "DELETE", f"/api/v1/agents/{canvas}/sessions?ids=", token=api)
    st2, b2 = jcall(RUST, "DELETE", f"/api/v1/agents/{canvas}/sessions?ids=", token=api)
    same("batch delete with nothing", b1, b2)
    # Files are refused by rust only (ruled: v1 does not parse uploads).
    st, body = jcall(RUST, "POST", "/api/v1/agents/chat/completions", token=api, body={"agent_id": canvas, "query": "x", "files": [{"id": "1"}]})
    record("files in a request are refused by rust with a clear message", "KNOWN" if body.get("code") == 102 and "files" in body.get("message", "") else "FAIL", str(body))
    # The logs endpoint.
    r = both(lambda base: jcall(base, "GET", f"/api/v1/agents/{canvas}/logs/abc", token=api))
    same("logs for a message with no log", r["go"][1], r["rust"][1])
    r = both(lambda base: jcall(base, "GET", f"/api/v1/agentbots/{canvas}/inputs", token="Bearer ragflow-e2e-beta-token-own"))
    same("agentbot inputs with an API token", r["go"][1], r["rust"][1])
    r = both(lambda base: jcall(base, "GET", f"/api/v1/agentbots/{canvas}/inputs", token="Bearer e2e-beta-own"))
    same("agentbot inputs with a beta token", r["go"][1], r["rust"][1])


def strip_resp(v):
    return strip(v)


def norm_resp(v, ids):
    text = json.dumps(v, ensure_ascii=False)
    for real, name in ids.items():
        text = text.replace(real, name)
    out = json.loads(text)
    return strip(out)


def auth_matrix(stack, secret, canvas, api):
    users = stack.rows("SELECT JSON_OBJECT('id',id,'t',access_token,'su',is_superuser,'email',email) FROM `user`")
    owner = next(u for u in users if not u["su"])
    admin = next(u for u in users if u["su"])
    now = int(time.time())
    cases = {
        "no header": None,
        "garbage": "Bearer garbage",
        "valid session token": "Bearer " + mint(owner["t"], secret, now),
        "valid session token, no Bearer": mint(owner["t"], secret, now),
        "expired session token (25 h)": "Bearer " + mint(owner["t"], secret, now - 25 * 3600),
        "just inside 24 h": "Bearer " + mint(owner["t"], secret, now - 24 * 3600 + 30),
        "timestamp in the future": "Bearer " + mint(owner["t"], secret, now + 3600),
        "signed with another secret": "Bearer " + mint(owner["t"], "not-the-secret", now),
        "unknown access token": "Bearer " + mint("0" * 32, secret, now),
        "short access token": "Bearer " + mint("short", secret, now),
        "superuser session token": "Bearer " + mint(admin["t"], secret, now),
        "api token": "Bearer " + api,
        "api token without scheme": api,
        "api token, any scheme": "Foo " + api,
        "api token, two spaces": "Bearer  " + api,
        "unknown api token": "Bearer ragflow-nope",
        "beta token on a standard endpoint": "Bearer e2e-beta-bound",
    }
    for name, token in cases.items():
        gs, gb = jcall(GO, "GET", f"/api/v1/agents/{canvas}/sessions?page_size=1&dsl=false", token=token)
        rs, rb = jcall(RUST, "GET", f"/api/v1/agents/{canvas}/sessions?page_size=1&dsl=false", token=token)
        ok = gs == rs and (gb.get("code") if isinstance(gb, dict) else gb) == (rb.get("code") if isinstance(rb, dict) else rb) \
            and (gb.get("message") if isinstance(gb, dict) else 0) == (rb.get("message") if isinstance(rb, dict) else 0)
        record(f"auth: {name} -> http {gs} code {gb.get('code') if isinstance(gb, dict) else '?'}", "PASS" if ok else "FAIL",
               f"go={gs} {str(gb)[:200]} rust={rs} {str(rb)[:200]}")
    # Beta endpoints answer failures with a business error, not HTTP 401.
    for name, token in (("none", None), ("garbage", "Bearer garbage"), ("beta", "Bearer e2e-beta-bound"), ("beta bare", "e2e-beta-bound"), ("Bearer alone", "Bearer"),
                        ("session token", cases["valid session token"])):
        gs, gb = jcall(GO, "GET", f"/api/v1/agentbots/{canvas}/inputs", token=token)
        rs, rb = jcall(RUST, "GET", f"/api/v1/agentbots/{canvas}/inputs", token=token)
        d = diff_json(gb, rb) if gs == rs else [f"status {gs}/{rs}"]
        record(f"auth (beta endpoint): {name} -> {gb.get('code') if isinstance(gb, dict) else gb}", "FAIL" if d else "PASS", "\n         ".join(d[:4]))
    gs, gb = jcall(GO, "GET", f"/api/v1/agentbots/{canvas}/inputs", headers={"Cookie": "ragflow_auth=" + cases["valid session token"].replace(" ", "%20")})
    rs, rb = jcall(RUST, "GET", f"/api/v1/agentbots/{canvas}/inputs", headers={"Cookie": "ragflow_auth=" + cases["valid session token"].replace(" ", "%20")})
    record("auth (beta endpoint): cookie fallback", "PASS" if gs == rs and gb == rb else "FAIL", f"{gs}/{rs} {str(gb)[:120]} {str(rb)[:120]}")


def beta(stack, canvas, out):
    # The binding: a beta token bound to another agent.
    gs, gf, _ = sse(GO, f"/api/v1/agentbots/{canvas}/completions", "Bearer e2e-beta-bound", {"query": "hello beta"})
    rs, rf, _ = sse(RUST, f"/api/v1/agentbots/{canvas}/completions", "Bearer e2e-beta-bound", {"query": "hello beta"})
    go_ran = any(f != "[DONE]" and f.get("event") == "workflow_finished" for f in gf)
    rust_text = json.dumps(rf)
    status_go = "go ran another agent with a token bound elsewhere" if go_ran else f"go refused too: {str(gf)[:160]}"
    rust_refused, rust_body = jcall(RUST, "POST", f"/api/v1/agentbots/{canvas}/completions", token="Bearer e2e-beta-bound", body={"query": "hello beta"})
    ok = go_ran and isinstance(rust_body, dict) and rust_body.get("code") == 401 and "not authorized" in rust_body.get("message", "")
    record("beta token bound to another agent: go runs it, rust refuses (ruled difference)", "KNOWN" if ok else "FAIL",
           f"{status_go}; rust: {str(rust_body)[:200]}")
    # A token bound to this agent runs on both.
    for side, base in (("go", GO), ("rust", RUST)):
        st, fr, _ = sse(base, f"/api/v1/agentbots/{canvas}/completions", "Bearer e2e-beta-own", {"query": "hello beta"})
        ok = st == 200 and any(f != "[DONE]" and f.get("event") == "workflow_finished" for f in fr)
        record(f"beta token bound to the agent runs on {side}", "PASS" if ok else "FAIL", str(fr)[:200])
    # Logs endpoint enforces the binding on both.
    gs, gb = jcall(GO, "GET", f"/api/v1/agentbots/{canvas}/logs/abc", token="Bearer e2e-beta-bound")
    rs, rb = jcall(RUST, "GET", f"/api/v1/agentbots/{canvas}/logs/abc", token="Bearer e2e-beta-bound")
    record("beta logs endpoint refuses a token bound elsewhere", "PASS" if gb == rb and gb.get("code") == 401 else "FAIL", f"go={gb} rust={rb}")
    gs, gb = jcall(GO, "GET", f"/api/v1/agentbots/{canvas}/logs/abc", token="Bearer e2e-beta-own")
    rs, rb = jcall(RUST, "GET", f"/api/v1/agentbots/{canvas}/logs/abc", token="Bearer e2e-beta-own")
    record("beta logs endpoint, own agent, no log", "PASS" if gb == rb else "FAIL", f"go={gb} rust={rb}")
    # release=true: nothing released yet -> both fail alike; then release the newest version.
    gs, gf, gt = sse(GO, f"/api/v1/agentbots/{canvas}/completions?release=true", "Bearer e2e-beta-own", {"query": "hi"})
    rs, rf, rt = sse(RUST, f"/api/v1/agentbots/{canvas}/completions?release=true", "Bearer e2e-beta-own", {"query": "hi"})
    gb = json.loads(gt) if gt.startswith("{") else gf
    rb = json.loads(rt) if rt.startswith("{") else rf
    record("release=true with nothing released", "PASS" if gb == rb else "FAIL", f"go={str(gb)[:200]} rust={str(rb)[:200]}")
    stack.sql(f"UPDATE user_canvas_version SET `release`=1 WHERE user_canvas_id='{canvas}' ORDER BY create_time DESC LIMIT 1")
    ok = True
    for side, base in (("go", GO), ("rust", RUST)):
        st, fr, _ = sse(base, f"/api/v1/agentbots/{canvas}/completions?release=true", "Bearer e2e-beta-own", {"query": "hi"})
        ok &= any(f != "[DONE]" and f.get("event") == "workflow_finished" for f in fr)
    record("release=true runs the released version on both", "PASS" if ok else "FAIL")
    stack.sql(f"UPDATE user_canvas_version SET `release`=0 WHERE user_canvas_id='{canvas}'")


def team(stack, canvas, api):
    found = stack.rows("SELECT JSON_OBJECT('t', tenant_id) FROM api_token WHERE token='ragflow-e2e-invitee'")
    if not found:
        record("team member checks", "SKIP", "the copy has no invited member")
        return
    member = found[0]["t"]
    owner_tenant = stack.rows(f"SELECT JSON_OBJECT('t', user_id) FROM user_canvas WHERE id='{canvas}'")[0]["t"]
    token = "Bearer ragflow-e2e-invitee"
    for role in ("invite", "normal"):
        stack.sql(f"UPDATE user_tenant SET role='{role}' WHERE user_id='{member}' AND tenant_id='{owner_tenant}'")
        outcome = {}
        for side, base in (("go", GO), ("rust", RUST)):
            st, fr, raw = sse(base, "/api/v1/agents/chat/completions", token, {"agent_id": canvas, "query": f"team {role}", "stream": True})
            if raw.lstrip().startswith("{"):
                body = json.loads(raw)
                outcome[side] = ("refused", body.get("code"), body.get("message"))
            else:
                outcome[side] = ("ran", answer_of(fr)[:30] and "answer")
        ls = {s: jcall(b, "GET", f"/api/v1/agents/{canvas}/sessions?page_size=1&dsl=false", token=token) for s, b in (("go", GO), ("rust", RUST))}
        listed = {s: (v[1].get("code"), v[1].get("total")) for s, v in ls.items()}
        record(f"team member with role '{role}' runs the canvas: go={outcome['go'][0]} rust={outcome['rust'][0]}; lists {listed['go']} / {listed['rust']}",
               "PASS" if outcome["go"] == outcome["rust"] else "FAIL", f"go={outcome['go']} rust={outcome['rust']}")
        if role == "normal":
            # The member sees the owner's sessions but cannot continue them.
            sess = stack.rows(f"SELECT JSON_OBJECT('id',id) FROM api_4_conversation WHERE dialog_id='{canvas}' AND user_id<>'{member}' LIMIT 1")[0]["id"]
            res = {}
            for side, base in (("go", GO), ("rust", RUST)):
                st, body = jcall(base, "POST", "/api/v1/agents/chat/completions", token=token, body={"agent_id": canvas, "query": "x", "session_id": sess, "stream": True})
                res[side] = (body.get("code"), body.get("message")) if isinstance(body, dict) else body[:80]
                gst, gbody = jcall(base, "GET", f"/api/v1/agents/{canvas}/sessions/{sess}", token=token)
                res[side + "-get"] = gbody.get("code") if isinstance(gbody, dict) else gst
            record("team member cannot run another member's session; can read it", "PASS" if res["go"] == res["rust"] and res["go-get"] == res["rust-get"] else "FAIL", str(res))
    stack.sql(f"UPDATE user_tenant SET role='invite' WHERE user_id='{member}' AND tenant_id='{owner_tenant}'")


def cancel(stack, canvas, api):
    def start(base, session, query):
        holder = {}

        def run():
            holder["result"] = sse(base, "/api/v1/agents/chat/completions", api,
                                   {"agent_id": canvas, "query": query, "stream": True, "session_id": session})
        t = threading.Thread(target=run)
        t.start()
        return t, holder

    # A first turn on each side creates a session to run SLOW turns in.
    sids = {}
    for side, base in (("go", GO), ("rust", RUST)):
        st, fr, _ = sse(base, "/api/v1/agents/chat/completions", api, {"agent_id": canvas, "query": f"cancel warm-up {side}", "stream": True})
        sids[side] = next(f["session_id"] for f in fr if f != "[DONE]" and "session_id" in f)
    for runner, canceller in (("go", "rust"), ("rust", "go"), ("go", "go"), ("rust", "rust")):
        base = GO if runner == "go" else RUST
        other = GO if canceller == "go" else RUST
        sid = sids[runner]
        rounds_before = session_rows(stack, sid)["round"]
        t, holder = start(base, sid, "SLOW parking fee please")
        time.sleep(3)
        lease = stack.redis("EXISTS", f"agent:active-session:{sid}")
        started = time.time()
        st, body = jcall(other, "POST", f"/api/v1/tasks/{sid}/cancel", token=api)
        t.join(60)
        elapsed = time.time() - started
        fr = holder.get("result", (0, [], ""))[1]
        kinds = [f.get("event") if f != "[DONE]" else "[DONE]" for f in fr]
        after = session_rows(stack, sid)
        cancelled = "cancelled" in kinds
        ok = lease == "1" and body.get("code") == 0 and cancelled and elapsed < 15 and kinds[-1] == "[DONE]" \
            and after["round"] == rounds_before and "workflow_finished" not in kinds
        left = stack.redis("EXISTS", f"agent:active-session:{sid}"), stack.redis("EXISTS", f"{sid}-cancel")
        record(f"cancel: {runner} runs, {canceller} cancels ({elapsed:.1f}s, lease then {lease}, keys after {left}, events end {kinds[-3:]})",
               "PASS" if ok and left == ("0", "0") else "FAIL", f"cancel reply {body}; kinds {kinds}")
    # Someone else's cancel is refused alike.
    st_g, b_g = jcall(GO, "POST", f"/api/v1/tasks/{sids['go']}/cancel", token="Bearer ragflow-e2e-outsider")
    st_r, b_r = jcall(RUST, "POST", f"/api/v1/tasks/{sids['go']}/cancel", token="Bearer ragflow-e2e-outsider")
    record("cancel by someone with no lease to cancel is a quiet success on both", "PASS" if b_g == b_r else "FAIL", f"go={b_g} rust={b_r}")
    # Ownership: cancel a running SLOW turn as another user.
    t, holder = start(GO, sids["go"], "SLOW parking owner check")
    time.sleep(3)
    res = {}
    for side, base in (("go", GO), ("rust", RUST)):
        st, body = jcall(base, "POST", f"/api/v1/tasks/{sids['go']}/cancel", token="Bearer ragflow-e2e-outsider")
        res[side] = body
    record("cancel by another user while the run is live is refused alike", "PASS" if res["go"] == res["rust"] and res["go"].get("code") == 103 else "FAIL", str(res))
    jcall(GO, "POST", f"/api/v1/tasks/{sids['go']}/cancel", token=api)
    t.join(60)
    # Contention: two requests on one session; the second waits for the first.
    for side, base in (("go", GO), ("rust", RUST)):
        sid = sids[side]
        t1, h1 = start(base, sid, "contend one")
        time.sleep(0.05)
        t2, h2 = start(base, sid, "contend two")
        t1.join(60)
        t2.join(60)
        ok = all(any(f != "[DONE]" and f.get("event") == "workflow_finished" for f in h["result"][1]) for h in (h1, h2))
        record(f"two requests on one {side} session both complete", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepares the throwaway copy for the comparison: the stub model replaces the
hosted one in every OpenAI instance, API and beta tokens are added, the agent
is copied once so a beta token can be bound to a different one, and the stub's
routing rules are written from the probe plan.

    prepare.py --state STATE_DIR --canvas CANVAS_ID --probes probes.json
"""
import argparse
import json
import os
import subprocess
import sys
import uuid

sys.path.insert(0, os.path.dirname(__file__))
from e2e import Stack  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--canvas", required=True)
    ap.add_argument("--probes", required=True)
    args = ap.parse_args()
    stack = Stack(args.state)
    canvas = args.canvas
    stack.sql("""UPDATE tenant_model_instance i JOIN tenant_model_provider p ON p.id = i.provider_id
        SET i.api_key = 'stub-key', i.extra = JSON_SET(i.extra, '$.base_url', 'http://11.250.0.13:8000/v1')
        WHERE p.provider_name = 'OpenAI'""")
    owner_tenant = stack.rows(f"SELECT JSON_OBJECT('u', user_id) FROM user_canvas WHERE id='{canvas}'")[0]["u"]
    invited = {r["u"] for r in stack.rows(
        f"SELECT JSON_OBJECT('u', user_id) FROM user_tenant WHERE tenant_id='{owner_tenant}' AND role='invite' AND status='1'")}
    others = stack.rows(f"SELECT JSON_OBJECT('id', id) FROM `user` WHERE id<>'{owner_tenant}' AND is_superuser=0 ORDER BY id")
    canvas2 = uuid.uuid4().hex
    stack.sql(f"""DELETE FROM user_canvas WHERE id='{canvas2}';
        CREATE TEMPORARY TABLE c2 SELECT * FROM user_canvas WHERE id='{canvas}';
        UPDATE c2 SET id='{canvas2}', title='second agent', permission='team';
        INSERT INTO user_canvas SELECT * FROM c2;""")
    rows = [(owner_tenant, "ragflow-e2e-owner", None, None)]
    # An invited-but-not-accepted member and a stranger, when the copy has them.
    for label, wanted in (("invitee", True), ("outsider", False)):
        pick = next((u for u in others if (u["id"] in invited) == wanted), None)
        if pick:
            rows.append((pick["id"], f"ragflow-e2e-{label}", None, None))
    rows.append((owner_tenant, "ragflow-e2e-beta-token-bound", canvas2, "e2e-beta-bound"))
    rows.append((owner_tenant, "ragflow-e2e-beta-token-own", canvas, "e2e-beta-own"))
    stack.sql("DELETE FROM api_token WHERE token LIKE 'ragflow-e2e-%'")
    for tenant, token, dialog, beta in rows:
        d = f"'{dialog}'" if dialog else "NULL"
        b = f"'{beta}'" if beta else "NULL"
        stack.sql(f"""INSERT INTO api_token (tenant_id, token, dialog_id, source, beta, create_time, create_date, update_time, update_date)
            VALUES ('{tenant}', '{token}', {d}, 'none', {b}, 1, '2026-01-01', 1, '2026-01-01')""")
    plan = json.load(open(args.probes, encoding="utf-8"))
    stub_dir = os.path.join(args.state, "stub")
    json.dump(plan.get("rules", []), open(os.path.join(stub_dir, "rules.json"), "w"), ensure_ascii=False)
    subprocess.run(["docker", "restart", "rfe2e-stub"], check=True, capture_output=True)
    print(json.dumps({"canvas": canvas, "second_canvas": canvas2, "owner_tenant": owner_tenant}))


if __name__ == "__main__":
    main()

"""Push changed dashboard files (site/data/*.json) into the private Supabase table behind the login.
Needs GitHub secret ORACLE_SYNC_TOKEN. Supabase URL + publishable key are public and live in config.yaml."""
from __future__ import annotations

import hashlib
import json
import os

import requests

from .core import SITE_DATA, STATE_DIR, load_config, log, read_json, write_json


def publish() -> int:
    token = os.environ.get("ORACLE_SYNC_TOKEN")
    sb = load_config().get("supabase", {})
    if not token or not sb.get("url"):
        log("publish skipped (no ORACLE_SYNC_TOKEN secret)")
        return 0
    seen = read_json(STATE_DIR / "published.json", {})
    batch, size, sent = [], 0, 0
    url = sb["url"].rstrip("/") + "/rest/v1/rpc/oracle_ingest"
    hdr = {"apikey": sb["anon_key"], "Authorization": f"Bearer {sb['anon_key']}", "Content-Type": "application/json"}

    def flush():
        nonlocal batch, size, sent
        if not batch:
            return
        r = requests.post(url, headers=hdr, json={"p_token": token, "p_docs": batch}, timeout=120)
        if not r.ok:
            raise RuntimeError(f"publish failed: {r.status_code} {r.text[:200]}")
        for d in batch:
            seen[d["path"]] = d["sha"]
        sent += len(batch)
        batch, size = [], 0

    for f in sorted(SITE_DATA.rglob("*.json")):
        rel = f.relative_to(SITE_DATA).as_posix()
        raw = f.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if seen.get(rel) == sha:
            continue
        try:
            body = json.loads(raw)
        except ValueError:
            continue
        batch.append({"path": rel, "sha": sha, "body": body})
        size += len(raw)
        if size > 3_000_000:
            flush()
    flush()
    write_json(STATE_DIR / "published.json", seen)
    log(f"published {sent} files to Supabase")
    return sent

#!/usr/bin/env python3
"""Pull the GitHub Actions (cloud) cycle state into the dashboard.

The always-on pipeline runs on GitHub Actions; its database travels via the
`ytauto-state` artifact. This module fetches that artifact, swaps the SQLite
file into data/ci/ytauto.db (the dashboard's preferred DB), and writes
data/ci/runs.json with the last N workflow runs so the dashboard can show
live cloud cycles.

Stdlib only; talks to GitHub through the local `gh` CLI (already authed).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

REPO = "techtricks/rumble-studio"
CI_DIR = Path(__file__).resolve().parent.parent / "data" / "ci"
DB_OUT = CI_DIR / "ytauto.db"
STAMP = CI_DIR / "state.stamp"
RUNS_OUT = CI_DIR / "runs.json"


def _gh() -> str:
    for cand in (Path.home() / "tools" / "gh", shutil.which("gh")):
        if cand and Path(cand).exists():
            return str(cand)
    raise FileNotFoundError("gh CLI not found (~/tools/gh)")


def _api(path: str, out: Path | None = None, *, stream: bool = False) -> tuple:
    """GET a GitHub API path via gh. Returns (ok, json_or_exit_error)."""
    cmd = [_gh(), "api", path]
    # NOTE: no custom Accept header — the workflow's restore step works the
    # same way (plain gh api follows the redirect and writes the zip).
    tmp = out.with_suffix(".part") if out else None
    try:
        if out is None:
            r = subprocess.run(cmd, capture_output=True, timeout=60)
        else:
            with open(tmp, "wb") as fh:
                r = subprocess.run(cmd, stdout=fh, stderr=subprocess.PIPE,
                                   timeout=180)
        if r.returncode == 0:
            if out is None:
                return True, json.loads(r.stdout or b"{}")
            tmp.replace(out)
            tmp = None
            return True, None
        msg = r.stderr.decode(errors="replace").strip()[:200]
        return False, msg
    except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
        return False, str(exc)[:200]
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink(missing_ok=True)


def _latest_artifact() -> dict:
    ok, data = _api(f"repos/{REPO}/actions/artifacts?name=ytauto-state&per_page=1")
    if not ok:
        return {}
    arts = (data.get("artifacts") or [])
    art = arts[0] if arts else {}
    return {} if art.get("expired") else art


def _sync_db(art: dict) -> bool:
    """Download a newer state artifact and swap ytauto.db atomically."""
    art_id = str(art.get("id", ""))
    if not art_id or (STAMP.exists() and STAMP.read_text().strip() == art_id):
        return False
    with tempfile.TemporaryDirectory() as td:
        zpath = Path(td) / "state.zip"
        ok, err = _api(f"repos/{REPO}/actions/artifacts/{art_id}/zip",
                       out=zpath, stream=True)
        if not ok:
            print(f"[ci-sync] zip download failed: {err}", flush=True)
            return False
        try:
            with zipfile.ZipFile(zpath) as zf:
                db_tmp = Path(td) / "incoming.db"
                targets = {n for n in zf.namelist()
                           if Path(n).name in ("ytauto.db", "rumble_stats.json")}
                db_target = next((n for n in targets
                                  if Path(n).name == "ytauto.db"), None)
                if not db_target:
                    print("[ci-sync] artifact had no ytauto.db", flush=True)
                    return False
                db_tmp.write_bytes(zf.read(db_target))
                for n in targets:
                    if Path(n).name == "rumble_stats.json":
                        stats_tmp = Path(td) / "rumble_stats.json"
                        stats_tmp.write_bytes(zf.read(n))
                        stats_tmp.replace(CI_DIR / "rumble_stats.json")
            os.replace(db_tmp, DB_OUT)  # atomic: dashboard never sees a half DB
        except (zipfile.BadZipFile, KeyError, OSError) as exc:
            print(f"[ci-sync] bad artifact: {exc}", flush=True)
            return False
    STAMP.parent.mkdir(parents=True, exist_ok=True)
    STAMP.write_text(art_id)
    return True


def _sync_runs() -> None:
    ok, data = _api(
        f"repos/{REPO}/actions/runs?per_page=10&_={int(time.time())}")
    if not ok:
        return
    runs = [{
        "id": r.get("id"),
        "number": r.get("run_number"),
        "title": r.get("display_title"),
        "event": r.get("event"),
        "status": r.get("status"),
        "conclusion": r.get("conclusion"),
        "created_at": r.get("created_at"),
        "updated_at": r.get("updated_at"),
        "url": r.get("html_url"),
    } for r in (data.get("workflow_runs") or [])]
    CI_DIR.mkdir(parents=True, exist_ok=True)
    out = {"synced_at": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": runs}
    tmp = RUNS_OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1))
    os.replace(tmp, RUNS_OUT)


def sync_once() -> None:
    """One full sync pass (never raises)."""
    CI_DIR.mkdir(parents=True, exist_ok=True)
    art = _latest_artifact()
    if art and _sync_db(art):
        print(f"[ci-sync] state DB updated (artifact {art['id']})", flush=True)
    _sync_runs()


def loop(every: int = 180) -> None:
    """Daemon-thread entry: sync every `every` seconds until killed."""
    _sync_runs()  # run-status first — cheap and shows cloud liveness early
    while True:
        try:
            sync_once()
        except Exception as exc:  # noqa: BLE001 - the log is our only voice
            print(f"[ci-sync] pass failed: {exc}", flush=True)
        time.sleep(every)


def run_status() -> dict:
    """What runs.json last said (for the dashboard's live badge)."""
    if not RUNS_OUT.exists():
        return {"synced_at": None, "runs": []}
    try:
        return json.loads(RUNS_OUT.read_text())
    except (json.JSONDecodeError, OSError):
        return {"synced_at": None, "runs": []}

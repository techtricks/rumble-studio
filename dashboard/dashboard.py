#!/usr/bin/env python3
"""ytauto live dashboard — sidebar-style monitor with clickable cards.

No extra dependencies: stdlib http.server + sqlite3 + the project's existing
logs/database. Serves:

  /            dashboard page (dark neon theme, auto-refresh every 5 s)
  /api/*       JSON snapshots (status, videos, topics, runs, logs, settings)
  /media/*     rendered thumbnails (jpg/png) straight from data/media/

Cards are clickable: clicking e.g. "failed" opens the Videos view filtered to
failed videos. Views are hash-routed (#videos:failed) so refresh keeps context.

Start:  .venv/bin/python dashboard/dashboard.py
        (a LaunchAgent starts this automatically at login — see
        ~/Library/LaunchAgents/com.ytauto.dashboard.plist)
"""
from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "ytauto.db"
CI_DB = ROOT / "data" / "ci" / "ytauto.db"   # synced from GitHub Actions
LOG = ROOT / "data" / "logs" / "launchd.log"
ENV = ROOT / ".env"
MEDIA = ROOT / "data" / "media"
PORT = 8600


def active_db() -> Path:
    """The cloud (GitHub Actions) DB wins whenever it has synced."""
    return CI_DB if CI_DB.exists() else DB


def db_rows(sql: str, args: tuple = ()) -> list[dict]:
    db = active_db()
    if not db.exists():
        return []
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, args)]
    finally:
        conn.close()


def _live_cycle() -> dict:
    """Is a pipeline cycle running right now?"""
    try:
        out = subprocess.run(
            ["/usr/bin/pgrep", "-fl", "ytauto (run|rumble)|launcher/run.sh"],
            capture_output=True, text=True, timeout=5,
        )
        procs = [l for l in out.stdout.splitlines() if "dashboard" not in l]
        return {"running": bool(procs), "procs": procs}
    except Exception as exc:  # noqa: BLE001
        return {"running": False, "procs": [], "error": str(exc)}


def api_settings() -> dict:
    """Selected .env values so the Settings view mirrors real config."""
    keys = ("NICHE", "MIN_NICHE_SCORE", "MIN_TOTAL_SCORE", "SUBREDDITS",
            "RSS_FEEDS", "AUTO_APPROVE", "DRY_RUN", "RUMBLE_CHANNEL",
            "LONGFORM_TARGET_S", "IMAGES_PER_VIDEO")
    vals = {}
    if ENV.exists():
        for line in ENV.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() in keys:
                vals[k.strip()] = v.strip()
    return vals


def api_status() -> dict:
    counts = {r["status"]: r["c"] for r in db_rows(
        "select status, count(*) c from videos group by status")}
    topic_counts = {r["status"]: r["c"] for r in db_rows(
        "select status, count(*) c from topics group by status")}
    runs = db_rows("select * from runs order by started_at desc limit 20")
    last_run = runs[0] if runs else None
    running = _live_cycle()

    # Online mode: cloud (GitHub Actions) activity drives the live badge and
    # the heartbeat — the Mac's own cycle is of historical interest only.
    ci = ci_sync.run_status()
    online = CI_DB.exists()
    if online:
        active = [r for r in ci.get("runs") or []
                  if r.get("status") in ("in_progress", "queued")]
        if active:
            running = {"running": True,
                       "procs": [f"github-actions run #{active[0].get('number', '')}"]}
        heartbeat = None
        for r in ci.get("runs") or []:
            if r.get("status") == "completed":
                heartbeat = {"last_start": r.get("created_at"),
                             "last_end": r.get("updated_at")}
                break

    # scheduler heartbeat from the launchd log
    heartbeat = None
    if LOG.exists():
        text = LOG.read_text(errors="replace")
        starts = re.findall(r"=== (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) cycle start", text)
        ends = re.findall(r"=== (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) cycle end", text)
        if starts:
            heartbeat = {"last_start": starts[-1],
                         "last_end": ends[-1] if len(ends) == len(starts) else None}

    cookies = ROOT / "secrets" / "rumble_cookies.json"
    alive = sum(v for k, v in counts.items() if k not in ("archived", "failed"))
    published = counts.get("published", 0)
    attempts = db_rows("select count(*) c from videos") or [{"c": 0}]
    total_ever = attempts[0]["c"] if attempts else 0
    success = round(100 * published / total_ever) if total_ever else 0

    # GitHub Actions connectivity: runs.json is rewritten every sync pass
    # (~3 min); older than 15 min means the pull is broken → offline.
    gh_runs = ci.get("runs") or []
    live = [r for r in gh_runs
            if r.get("status") in ("in_progress", "queued")]
    show_run = (live[0] if live else (gh_runs[0] if gh_runs else None))
    online = False
    if ci.get("synced_at"):
        try:
            age = time.time() - time.mktime(
                time.strptime(ci["synced_at"], "%Y-%m-%d %H:%M:%S"))
            online = 0 <= age < 900
        except ValueError:
            pass
    github = {
        "online": online,
        "repo": "https://github.com/techtricks/rumble-studio/actions",
        "run_url": (show_run or {}).get("url"),
        "run_number": (show_run or {}).get("number"),
        "run_status": (show_run or {}).get("status"),
        "run_conclusion": (show_run or {}).get("conclusion"),
        "run_started_at": (show_run or {}).get("started_at"),
        "run_updated_at": (show_run or {}).get("updated_at"),
        "run_event": (show_run or {}).get("event"),
    }

    # Channel money+reach snapshot written by tools/rumble_stats.py (CI adds
    # the file to the state artifact; locally the checker writes data/).
    rumble_stats = None
    for cand in (CI_DB.parent / "rumble_stats.json",
                 ROOT / "data" / "rumble_stats.json"):
        if cand.exists():
            try:
                rumble_stats = json.loads(cand.read_text())
            except (json.JSONDecodeError, OSError):
                rumble_stats = None
            if rumble_stats:
                vids = rumble_stats.get("videos") or []
                rumble_stats["monetized"] = sum(
                    1 for v in vids if v.get("monetization") == "monetized")
                rumble_stats["pending"] = sum(
                    1 for v in vids if v.get("monetization") == "pending")
            break
    return {
        "now": time.strftime("%Y-%m-%d %H:%M:%S"),
        "live": running,
        "heartbeat": heartbeat,
        "cycle_state": "RUNNING NOW" if running["running"] else ("idle" if heartbeat else "never run"),
        "source": "cloud" if online else "mac",
        "ci_synced_at": ci.get("synced_at"),
        "github": github,
        "rumble_stats": rumble_stats,
        "video_counts": counts,
        "topic_counts": topic_counts,
        "videos_alive": alive,
        "success_rate": success,
        "last_run": last_run,
        "runs": runs,
        "rumble_cookies_since": cookies.exists()
        and time.strftime("%Y-%m-%d", time.gmtime(cookies.stat().st_mtime)),
        "links": {
            "channel": "https://rumble.com/c/IndianHistoryOfficial",
            "content": "https://rumble.com/account/content",
        },
    }


def api_videos() -> list[dict]:
    rows = db_rows(
        """select id, status, kind, title, duration_s, attempts, youtube_id,
                  substr(last_error,1,160) as error,
                  created_at, updated_at, video_path, thumb_path
           from videos order by updated_at desc limit 60""")
    for r in rows:
        vp = r.get("video_path") or ""
        tp = r.get("thumb_path") or ""
        r["mp4"] = Path(vp).name if vp else None
        r["thumb"] = Path(tp).name if tp else None
    return rows


def api_topics() -> list[dict]:
    return db_rows(
        """select id, status, source, round(niche_score,2) niche,
                  round(total_score,2) total, substr(reject_reason,1,120) reason,
                  substr(title,1,90) title, created_at
           from topics order by created_at desc limit 60""")


def api_logs() -> dict:
    if not LOG.exists():
        return {"text": "(launchd log not created yet)", "lines": 0}
    lines = LOG.read_text(errors="replace").splitlines()
    return {"lines": len(lines), "text": "\n".join(lines[-160:])}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data) -> None:
        self._send(200, "application/json", json.dumps(data).encode())

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        if self.path == "/":
            self._send(200, "text/html", PAGE.encode())
        elif self.path == "/api/status":
            self._json(api_status())
        elif self.path == "/api/videos":
            self._json(api_videos())
        elif self.path == "/api/topics":
            self._json(api_topics())
        elif self.path == "/api/logs":
            self._json(api_logs())
        elif self.path == "/api/settings":
            self._json(api_settings())
        elif self.path.startswith("/api/all"):
            self._json({
                "status": api_status(),
                "videos": api_videos(),
                "topics": api_topics(),
                "logs": api_logs(),
                "settings": api_settings(),
            })
        elif self.path.startswith("/media/"):
            f = (MEDIA / self.path[len("/media/"):]).resolve()
            if f.suffix.lower() in (".jpg", ".png", ".webp") and f.exists() and MEDIA in f.parents:
                self._send(200, "image/jpeg" if f.suffix == ".jpg" else "image/png", f.read_bytes())
            else:
                self._send(404, "text/plain", b"no file")
        elif self.path == "/favicon.ico":
            self._send(204, "image/x-icon", b"")
        else:
            self._send(404, "text/plain", b"not found")

    def log_message(self, *args) -> None:  # silence request spam
        return


PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rumble Studio — Indian History pipeline</title>
<style>
  :root {
    --bg:#071312; --side:#0a1a19; --card:#0d211f; --card2:#0a1a19;
    --line:#16413c; --line2:#1d544d; --txt:#d7efe9; --dim:#6d948c;
    --teal:#2dd4bf; --teal2:#5eead4; --green:#4ade80; --red:#f87171;
    --amber:#fbbf24; --blue:#60a5fa; --purple:#a78bfa; --pink:#f472b6;
  }
  * { box-sizing:border-box; margin:0 }
  body { background:var(--bg); color:var(--txt);
         font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }
  body::before { content:''; position:fixed; inset:0; pointer-events:none;
    background:
      radial-gradient(900px 500px at 70% -10%, rgba(45,212,191,.07), transparent 60%),
      radial-gradient(700px 400px at 100% 100%, rgba(45,212,191,.05), transparent 60%); }

  /* ---------- layout ---------- */
  .app { display:flex; min-height:100vh }
  aside { width:228px; flex:0 0 228px; background:var(--side);
          border-right:1px solid var(--line); padding:16px 12px;
          position:sticky; top:0; height:100vh; overflow-y:auto }
  main { flex:1; padding:26px 30px 60px; min-width:0 }

  /* ---------- sidebar ---------- */
  .nav { display:flex; flex-direction:column; gap:4px }
  .nav a { display:flex; align-items:center; gap:11px; padding:10px 12px;
           border-radius:10px; color:var(--dim); text-decoration:none;
           font-weight:600; font-size:14px; border:1px solid transparent;
           cursor:pointer; transition:.15s }
  .nav a .ic { font-size:16px; width:22px; text-align:center }
  .nav a:hover { color:var(--txt); background:rgba(45,212,191,.06) }
  .nav a.lo { color:var(--red); opacity:.75; font-weight:600 }
  .nav a.lo:hover { color:var(--red); opacity:1; background:rgba(248,113,113,.08) }
  .nav a.active { color:var(--teal2); background:linear-gradient(90deg,rgba(45,212,191,.14),rgba(45,212,191,.03));
                  border-color:var(--line2); box-shadow:inset 0 0 18px rgba(45,212,191,.08) }
  .nav .ext { margin-top:auto }
  .nav .lo-gap { margin-top:8px }
  .brand { display:flex; align-items:center; gap:10px; padding:6px 10px 18px }
  .brand .logo { width:34px; height:34px; border-radius:9px; display:grid; place-items:center;
    background:linear-gradient(135deg,#134e4a,#0f766e); box-shadow:0 0 14px rgba(45,212,191,.35);
    font-size:16px }
  .brand b { font-size:15px } .brand small { display:block; color:var(--dim); font-weight:400; font-size:11px }

  /* ---------- header ---------- */
  .hd { display:flex; align-items:baseline; justify-content:space-between; gap:12px; margin-bottom:20px }
  .hd h1 { font-size:28px; letter-spacing:-.5px }
  .hd .sub2 { color:var(--dim) }
  .livebadge { display:inline-flex; align-items:center; gap:7px; padding:5px 12px; border-radius:16px;
    font-size:12px; font-weight:700; border:1px solid var(--line2) }
  .livebadge.on  { background:rgba(74,222,128,.1); color:var(--green); box-shadow:0 0 14px rgba(74,222,128,.15) }
  .livebadge.off { background:rgba(109,148,140,.08); color:var(--dim) }
  .dot { width:8px; height:8px; border-radius:50%; background:currentColor }
  .livebadge.on .dot { animation:pulse 1.6s infinite }
  @keyframes pulse { 50% { opacity:.35 } }

  /* ---------- cards ---------- */
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:14px; margin-bottom:22px }
  .card { background:linear-gradient(160deg,var(--card),var(--card2));
          border:1px solid var(--line); border-radius:14px; padding:14px 16px;
          cursor:pointer; transition:.15s; position:relative; overflow:hidden }
  .card:hover { border-color:var(--teal); transform:translateY(-2px);
                box-shadow:0 6px 24px rgba(45,212,191,.12), inset 0 0 24px rgba(45,212,191,.05) }
  .card .ic { width:38px; height:38px; border-radius:10px; display:grid; place-items:center;
              font-size:17px; margin-bottom:12px; background:rgba(45,212,191,.1);
              border:1px solid var(--line2) }
  .card .n { font-size:30px; font-weight:800; letter-spacing:-1px; line-height:1.1 }
  .card .l { color:var(--dim); font-size:12.5px; margin-top:3px }
  .card.c-green .n{color:var(--green)} .card.c-red .n{color:var(--red)}
  .card.c-amber .n{color:var(--amber)} .card.c-blue .n{color:var(--blue)}
  .card.c-purple .n{color:var(--purple)} .card.c-teal .n{color:var(--teal2)}

  /* ---------- panels ---------- */
  .panels { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:14px; margin-bottom:24px }
  .panel { background:linear-gradient(165deg,var(--card),#091615); border:1px solid var(--line);
           border-radius:14px; padding:18px; min-width:0; overflow:hidden }
  .panel h3 { font-size:12.5px; letter-spacing:.09em; text-transform:uppercase; color:var(--teal2);
              display:flex; align-items:center; gap:8px; margin-bottom:14px }
  .chip { padding:3px 10px; border-radius:12px; font-size:11px; font-weight:700; margin-left:auto }
  .chip.ip { background:rgba(251,191,36,.12); color:var(--amber); border:1px solid rgba(251,191,36,.3) }
  .chip.ok { background:rgba(74,222,128,.12); color:var(--green); border:1px solid rgba(74,222,128,.3) }
  .chip.bad { background:rgba(248,113,113,.12); color:var(--red); border:1px solid rgba(248,113,113,.3) }
  .bar-row { margin:10px 0 4px; font-size:13px; display:flex; justify-content:space-between }
  .bar-row small { color:var(--dim) }
  .track { height:9px; border-radius:6px; background:rgba(45,212,191,.08);
           border:1px solid rgba(45,212,191,.18); overflow:hidden; margin-bottom:10px }
  .fill { height:100%; border-radius:6px; background:linear-gradient(90deg,#0f766e,var(--teal));
          box-shadow:0 0 10px rgba(45,212,191,.5); transition:width .6s }
  .panel p.note { color:var(--dim); font-size:12px; margin-top:8px; line-height:1.55 }

  /* ---------- auto-scroll data feeds (cards of identical size) ---------- */
  .panel.feedcard { height:330px; display:flex; flex-direction:column; overflow:hidden;
    grid-auto-rows:1fr }
  .feedcard .feed { flex:1; min-height:0; overflow:hidden; position:relative;
    -webkit-mask-image:linear-gradient(180deg,transparent 0,#000 22px,#000 calc(100% - 22px),transparent 100%);
            mask-image:linear-gradient(180deg,transparent 0,#000 22px,#000 calc(100% - 22px),transparent 100%) }
  .feedcard:hover .feed-inner { animation-play-state:paused }
  .feed-inner { animation:scrollup 40s linear infinite; will-change:transform }
  @keyframes scrollup { from{transform:translateY(0)} to{transform:translateY(-50%)} }
  .feed dl.kv { height:auto }
  .feed .kv dd { overflow-wrap:anywhere }

  /* ---------- tables ---------- */
  h2.sec { font-size:13px; letter-spacing:.09em; text-transform:uppercase; color:var(--teal2); margin:26px 0 10px }
  table { width:100%; border-collapse:collapse; background:var(--card2); border:1px solid var(--line);
          border-radius:12px; overflow:hidden; margin-bottom:20px }
  th,td { padding:9px 12px; text-align:left; border-bottom:1px solid rgba(22,65,60,.5);
          font-size:13px; vertical-align:top }
  th { color:var(--dim); font-size:11px; text-transform:uppercase; letter-spacing:.06em;
       background:rgba(45,212,191,.04) }
  tr:last-child td { border-bottom:0 }
  tr:hover td { background:rgba(45,212,191,.03) }
  .tag { padding:3px 9px; border-radius:11px; font-size:11px; font-weight:700; white-space:nowrap }
  .t-published{ background:rgba(74,222,128,.12); color:var(--green) }
  .t-rendered{ background:rgba(96,165,250,.12); color:var(--blue) }
  .t-approved,.t-scripted{ background:rgba(251,191,36,.12); color:var(--amber) }
  .t-scheduled{ background:rgba(167,139,250,.12); color:var(--purple) }
  .t-failed{ background:rgba(248,113,113,.12); color:var(--red) }
  .t-archived,.t-used{ background:rgba(109,148,140,.1); color:var(--dim) }
  .t-new{ background:rgba(96,165,250,.12); color:var(--blue) }
  .t-rejected{ background:rgba(248,113,113,.12); color:var(--red) }
  .err { color:var(--red) } .dim { color:var(--dim) }
  img.th { width:88px; border-radius:8px; border:1px solid var(--line2) }
  a { color:var(--teal2) }
  .filters { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px }
  .filters button { background:var(--card2); color:var(--dim); border:1px solid var(--line);
    border-radius:14px; padding:5px 14px; font-weight:600; font-size:12.5px; cursor:pointer }
  .filters button:hover { color:var(--txt); border-color:var(--line2) }
  .filters button.on { color:var(--teal2); border-color:var(--teal); background:rgba(45,212,191,.08) }
  details pre, pre#logs { font:12px/1.55 ui-monospace,Menlo,monospace; padding:14px; max-height:60vh;
    overflow:auto; background:#050d0c; border:1px solid var(--line); border-radius:12px;
    white-space:pre-wrap; color:#9fd8cc }
  .kv { display:grid; grid-template-columns:minmax(110px,170px) 1fr; gap:8px 16px; font-size:13.5px }
  .kv dt { color:var(--dim) }
  .kv dd { font-weight:600; min-width:0; overflow-wrap:anywhere; word-break:break-word }
  .view { display:none } .view.on { display:block }
  @media (max-width:860px){ .app{flex-direction:column} aside{width:100%;height:auto;position:static;
    display:flex;overflow-x:auto} .nav{flex-direction:row} .brand{display:none} .nav .ext{margin-top:0} main{padding:18px} }
</style></head>
<body>
<div class="app">
  <aside>
    <div class="brand"><div class="logo">📜</div><div><b>Rumble Studio</b><small>Indian History · Auto videos</small></div></div>
    <nav class="nav" id="nav">
      <a data-go="dash"><span class="ic">▦</span>Dashboard</a>
      <a data-go="videos"><span class="ic">🎬</span>Videos</a>
      <a data-go="topics"><span class="ic">📚</span>Topics</a>
      <a data-go="jobs"><span class="ic">⚙</span>Cycles</a>
      <a data-go="logs"><span class="ic">▤</span>Logs</a>
      <a data-go="policies"><span class="ic">✓</span>Policies</a>
      <a data-go="settings"><span class="ic">⚙</span>Settings</a>
      <a class="ext" href="{{CHANNEL}}" target="_blank"><span class="ic">▸</span>Rumble Channel ↗</a><span class="lo-slot">{{LOGOUT}}</span>
    </nav>
  </aside>
  <main>
    <div class="hd">
      <h1 id="pagetitle">Dashboard</h1>
      <div style="display:flex;gap:14px;align-items:center">
        <span class="sub2" id="meta">loading…</span>
        <span class="livebadge off" id="live"><span class="dot"></span><span id="livetext">idle</span></span>
      </div>
    </div>

    <!-- ============ DASHBOARD ============ -->
    <section class="view" id="v-dash">
      <div class="cards" id="cards"></div>
      <div class="panels" id="panels"></div>
      <h2 class="sec">Recent cycles</h2>
      <table><thead><tr><th>run id</th><th>started</th><th>ended</th><th>status</th><th>detail</th></tr></thead>
      <tbody id="runs-mini"></tbody></table>
    </section>

    <!-- ============ VIDEOS ============ -->
    <section class="view" id="v-videos">
      <div class="filters" id="vf"></div>
      <table><thead><tr><th>thumb</th><th>title</th><th>stage</th><th>len</th><th>last error</th><th>updated</th></tr></thead>
      <tbody id="videos"></tbody></table>
    </section>

    <!-- ============ TOPICS ============ -->
    <section class="view" id="v-topics">
      <div class="filters" id="tf"></div>
      <table><thead><tr><th>title</th><th>src</th><th>niche</th><th>total</th><th>stage</th><th>note</th></tr></thead>
      <tbody id="topics"></tbody></table>
    </section>

    <!-- ============ JOBS ============ -->
    <section class="view" id="v-jobs">
      <div class="panels" id="jobpanels"></div>
      <table><thead><tr><th>run id</th><th>started</th><th>ended</th><th>status</th><th>detail</th></tr></thead>
      <tbody id="runs"></tbody></table>
      <h2 class="sec">Raw pipeline log (live)</h2>
      <pre id="logs"></pre>
    </section>
    <!-- ============ LOGS ============ -->
    <section class="view" id="v-logs">
      <p class="dim" style="margin-bottom:10px">data/logs/launchd.log — scheduler ka poora output, har 5s refresh</p>
      <pre id="logs2"></pre>
    </section>

    <!-- ============ POLICIES ============ -->
    <section class="view" id="v-policies">
      <div class="panels">
        <div class="panel"><h3>✓ Content safety</h3>
          <p class="note">• Sirf Indian-history topics (curated sources: LiveHistoryIndia, Ancient Origins, r/IndianHistory)<br>
          • Research confidence ≥ 0.4 zaroori — thin topics kabhi script nahi hote<br>
          • Duplicate-topic gate — ek baar likha topic dobara nahi banta<br>
          • Purane tech videos = <b style="color:var(--red)">archived</b>, kabhi publish nahi honge</p></div>
        <div class="panel"><h3>✓ Media / copyright</h3>
          <p class="note">• Sirf CC-licensed / public-domain images<br>
          • Har video ke saath <b>credits file</b> (image attributions)<br>
          • Rotating narrator voices, self-rendered captions<br>
          • Thumbnail self-generated — koi stolen-media risk nahi</p></div>
        <div class="panel"><h3>✓ Automation rails</h3>
          <p class="note">• AUTO_APPROVE ke baad bhi validation gates chalte hain (length, opener variety, facts)<br>
          • Groq 429 pe auto-retry — kabhi crash nahi<br>
          • Single-instance lock — do cycle overlap nahi kar sakti<br>
          • Failed videos render-queue se bahar (no wasted cycles)</p></div>
      </div>
    </section>

    <!-- ============ SETTINGS ============ -->
    <section class="view" id="v-settings"></section>
  </main>
</div>

<script>
const esc = s => (s==null?'':String(s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
let D = null;                 // cached /api/all payload
let view = 'dash', vfilter = 'all', tfilter = 'all';

/* ---------- routing ---------- */
function parseHash(){
  const h = location.hash.replace('#','');
  const [v,f] = h.split(':');
  if (['dash','videos','topics','jobs','logs','policies','settings'].includes(v)) { view=v; vfilter=f||'all'; tfilter=f||'all'; }
  else view = 'dash';
  document.querySelectorAll('#nav a[data-go]').forEach(a=>a.classList.toggle('active', a.dataset.go===view));
  document.querySelectorAll('.view').forEach(s=>s.classList.toggle('on', s.id==='v-'+view));
  const titles = {dash:'Dashboard',videos:'Videos',topics:'Topics',jobs:'Cycles',logs:'Logs',policies:'Policies',settings:'Settings'};
  document.getElementById('pagetitle').textContent = titles[view]||'Dashboard';
}
document.querySelectorAll('#nav a[data-go]').forEach(a=>a.onclick=()=>location.hash=a.dataset.go);
window.onhashchange = ()=>{ parseHash(); render(); };
addEventListener('DOMContentLoaded', parseHash);

/* ---------- cards: click -> filtered view ---------- */
const CARDS = [
  {n:'published', l:'published', c:'c-green', ic:'📺', go:'#videos:published'},
  {n:'active',    l:'approved / rendered', c:'c-amber', ic:'🎬', go:'#videos:active'},
  {n:'scripted',  l:'awaiting approval', c:'c-amber', ic:'📝', go:'#videos:scripted'},
  {n:'failed',    l:'failed', c:'c-red', ic:'⛔', go:'#videos:failed'},
  {n:'alive',     l:'total alive videos', c:'c-blue', ic:'🗂', go:'#videos:all'},
  {n:'archived',  l:'archived', c:'c-teal', ic:'🗄', go:'#videos:archived'},
  {n:'tnew',      l:'topics queued', c:'c-purple', ic:'📚', go:'#topics:new'},
  {n:'trej',      l:'topics rejected', c:'c-red', ic:'🚫', go:'#topics:rejected'},
];
function renderCards(s){
  const c = s.video_counts, tc = s.topic_counts;
  const vals = {published:c.published||0, active:(c.approved||0)+(c.rendered||0),
    scripted:c.scripted||0, failed:c.failed||0, alive:s.videos_alive, archived:c.archived||0,
    tnew:tc.new||0, trej:tc.rejected||0};
  const gh = s.github || {online:false};
  const ghStatusWord = {queued:'⏳ queued', in_progress:'🏃 running',
    completed:(gh.run_conclusion==='success'?'✅ success':
               gh.run_conclusion==='failure'?'❌ failed':
               gh.run_conclusion==='cancelled'?'🛑 cancelled':
               (gh.run_conclusion||'done'))}[gh.run_status] || gh.run_status || 'no runs yet';
  const ghSub = gh.run_number ? ('#' + gh.run_number + ' ' + ghStatusWord) : 'no runs yet';
  const ghCard = `<div class="card ${gh.online?'c-green':'c-red'}" onclick="location.href='${gh.run_url||gh.repo}'" style="cursor:pointer" title="GitHub Actions — click to open">
      <div class="ic">☁</div><div class="n">${gh.online?'ONLINE':'OFFLINE'}</div>
      <div class="l">GitHub Actions · ${ghSub}</div></div>`;
  document.getElementById('cards').innerHTML = ghCard + CARDS.map(k=>
    `<div class="card ${k.c}" onclick="location.hash='${k.go}'">
       <div class="ic">${k.ic}</div><div class="n">${vals[k.n]}</div><div class="l">${k.l}</div>
     </div>`).join('');
}
function renderPanels(s){
  const c=s.video_counts, tc=s.topic_counts;
  const started=s.heartbeat?.last_start||'—', ended=s.heartbeat?.last_end;
  const st=s.rumble_stats;
  const stPanel = st ? `
  <div class="panel"><h3>💰 Rumble earnings <span class="chip ${st.monetized&&!st.pending?'ok':'ip'}">${st.monetized}✅/${st.pending}⏳</span></h3>
    <div class="bar-row"><span>All-time</span><small>${st.all_time.earnings} · ${st.all_time.views} views</small></div>
    <div class="bar-row"><span>Last 30 days</span><small>${st.last30.earnings} · ${st.last30.views} views · ${st.last30.hours}h watched</small></div>
    <div class="bar-row"><span>Balance (payout $50 pe)</span><small>${st.balance}</small></div>
    <div class="bar-row"><span>Followers</span><small>${st.all_time.followers}</small></div>
    <p class="note">Har video pe earnings: ${(st.earnings_rows||[]).slice(0,3).map(r=>esc(r.title.slice(0,38))+' — '+r.views+' views').join(' · ')} …<br>Stats checked: ${st.checked_at} (CI har cycle ke baad refresh karti hai). Pending wale videos kuch ghanton mein review hokar Monetized ban jate hain — earning 0 isliye hai kyunki views abhi kam hain, ye normal hai 📈</p></div>` : '';
  const success=s.success_rate||0;
  const gh2 = s.github || {};
  const ghRun = gh2.run_status ? ('#' + gh2.run_number + ' ' + ({
      queued:'⏳ queued (cloud soon)', in_progress:'🏃 running in cloud',
      completed: gh2.run_conclusion==='success' ? '✅ finished — videos published'
               : gh2.run_conclusion==='failure' ? '❌ failed — see logs'
               : gh2.run_conclusion==='cancelled' ? '🛑 cancelled' :
                 (gh2.run_conclusion||'done')
    })[gh2.run_status] || gh2.run_status) : 'no cycle yet';
  const ghLive = (gh2.run_status==='in_progress' || gh2.run_status==='queued');
  document.getElementById('panels').innerHTML = `
  <div class="panel"><h3>☁ GitHub cycle <span class="chip ${ghLive?'ip':'ok'}">${ghLive?'live':'idle'}</span></h3>
    <div class="bar-row"><span>Current run</span><small>${gh2.run_url ? `<a href="${gh2.run_url}" target="_blank">${ghRun} ↗</a>` : ghRun}</small></div>
    <div class="bar-row"><span>Started</span><small>${gh2.run_started_at ? (gh2.run_started_at+' '+(gh2.run_event||'')) : (s.heartbeat?.last_start||'—')}</small></div>
    <div class="track"><div class="fill" style="width:${ghLive?55:100}%"></div></div>
    <div class="bar-row"><span>Cloud updated</span><small>${gh2.run_updated_at||s.heartbeat?.last_end||'—'}</small></div>
    <p class="note">Scripts → approve → Edge-TTS render → <b>Rumble upload</b> (+ short). Scheduled: roz 2:30 AM IST (+ 9:00 AM IST backup-sirf-fail-par). Status: <b>${esc(s.cycle_state)}</b></p></div>
  <div class="panel"><h3>📈 Pipeline success <span class="chip ${success>60?'ok':success>25?'ip':'bad'}">${success}%</span></h3>
    <div class="bar-row"><span>Published / ever created</span><small>${c.published||0} / ${Object.values(c).reduce((a,b)=>a+b,0)}</small></div>
    <div class="track"><div class="fill" style="width:${success}%"></div></div>
    <div class="bar-row"><span>Queued topics</span><small>${tc.new||0} ready</small></div>
    <div class="track"><div class="fill" style="width:${Math.min(100,(tc.new||0)*2)}%"></div></div>
    <p class="note">Topic-bank se har cycle guaranteed content nikalta hai — rejected topics automatically skip.</p></div>
  <div class="panel"><h3>🛰 Rumble upload <span class="chip ok">live</span></h3>
    <div class="bar-row"><span>Channel</span><small><a href="${s.links.channel}" target="_blank">IndianHistoryOfficial ↗</a></small></div>
    <div class="track"><div class="fill" style="width:100%"></div></div>
    <div class="bar-row"><span>Cookies session</span><small>${s.rumble_cookies_since||'none'}</small></div>
    <div class="bar-row"><span>Content dashboard</span><small><a href="${s.links.content}" target="_blank">rumble.com/account ↗</a></small></div>
    <p class="note">Published videos turant channel pe dikhte hain — upload fully automated.</p></div>` + stPanel;
  document.getElementById('runs-mini').innerHTML = runRows(s.runs.slice(0,6));
}
const runRows = runs => runs.map(r=>`
  <tr><td class="dim">${esc(r.id.slice(0,8))}</td><td>${esc((r.started_at||'').slice(5,19))}</td>
  <td>${esc((r.ended_at||'').slice(5,19))}</td>
  <td><span class="tag ${r.status==='ok'?'t-published':r.status==='running'?'t-approved':'t-failed'}">${esc(r.status)}</span></td>
  <td class="dim">${esc(r.detail||'').slice(0,200)}</td></tr>`).join('');

/* ---------- videos view ---------- */
function renderVideos(){
  const fs=[['all','all'],['published','published'],['active','active'],['scripted','scripted'],
            ['rendered','rendered'],['failed','failed'],['archived','archived']];
  document.getElementById('vf').innerHTML = fs.map(([k,l])=>
    `<button class="${vfilter===k?'on':''}" onclick="location.hash='videos:${k}'">${l}</button>`).join('');
  let rows = D.videos;
  if (vfilter==='published') rows=rows.filter(v=>v.status==='published');
  else if (vfilter==='active') rows=rows.filter(v=>['approved','rendered'].includes(v.status));
  else if (vfilter!=='all') rows=rows.filter(v=>v.status===vfilter);
  document.getElementById('videos').innerHTML = rows.length ? rows.map(v=>`
    <tr><td>${v.thumb?`<img class="th" src="/media/${v.thumb}">`:'<span class="dim">—</span>'}</td>
    <td><b>${esc(v.title||v.id)}</b><div class="dim">${esc(v.kind||'')} · ${esc(v.id)}</div></td>
    <td><span class="tag t-${esc(v.status)}">${esc(v.status)}</span></td>
    <td class="dim">${v.duration_s?Math.round(v.duration_s/60*10)/10+'m':'—'}</td>
    <td class="err">${esc(v.error||'')}</td><td class="dim">${esc((v.updated_at||'').slice(0,16))}</td></tr>`).join('')
    : '<tr><td colspan="6" class="dim">koi video nahi (is filter mein)</td></tr>';
}

/* ---------- topics view ---------- */
function renderTopics(){
  const fs=[['all','all'],['new','queued'],['used','used'],['rejected','rejected']];
  document.getElementById('tf').innerHTML = fs.map(([k,l])=>
    `<button class="${tfilter===k?'on':''}" onclick="location.hash='topics:${k}'">${l}</button>`).join('');
  let rows = D.topics;
  if (tfilter!=='all') rows=rows.filter(t=>t.status===tfilter);
  document.getElementById('topics').innerHTML = rows.length ? rows.map(t=>`
    <tr><td>${esc(t.title)}</td><td class="dim">${esc(t.source)}</td><td>${esc(t.niche)}</td>
    <td>${esc(t.total)}</td><td><span class="tag t-${esc(t.status)}">${esc(t.status)}</span></td>
    <td class="err">${esc(t.reason||'')}</td></tr>`).join('')
    : '<tr><td colspan="6" class="dim">empty</td></tr>';
}

/* ---------- master render ---------- */
function feed(html){           // duplicated content = seamless bottom->top loop
  return `<div class="feed"><div class="feed-inner">${html}${html}</div></div>`;
}
function shorten(p){           // long command lines -> readable one-liners
  return esc(p)
    .replace(/\/Users\/mac\/Documents\/rumble\//g, '~/')
    .replace(/\/Library\/Frameworks\/Python[^ ]*MacOS\/Python/g, 'python')
    .replace(/\/Applications\/Google Chrome[\s\S]*/g, 'chrome');
}
function renderJobs(){
  const s=D.status;
  const kv = rows => `<dl class="kv">${rows.join('')}</dl>`;
  document.getElementById('jobpanels').innerHTML = `
  <div class="panel feedcard"><h3>⏱ Scheduler <span class="chip ${s.live.running?'ip':'ok'}">${s.live.running?'running':'idle'}</span></h3>
    ${feed(kv([
      `<dt>state</dt><dd>${esc(s.cycle_state)}</dd>`,
      `<dt>last start</dt><dd>${esc(s.heartbeat?.last_start||'—')}</dd>`,
      `<dt>last end</dt><dd>${esc(s.heartbeat?.last_end||'—')}</dd>`,
      `<dt>db</dt><dd>data/ytauto.db</dd>`,
      `<dt>media</dt><dd>data/media/</dd>`,
      `<dt>log</dt><dd>data/logs/launchd.log</dd>`,
      `<dt>cookies</dt><dd>${esc(s.rumble_cookies_since||'none')}</dd>`,
      `<dt>channel</dt><dd><a href="${s.links.channel}" target="_blank">IndianHistoryOfficial ↗</a></dd>`,
  ]))}</div>
  <div class="panel feedcard"><h3>🧠 Live processes</h3>
    ${feed(kv(s.live.procs.length
      ? s.live.procs.map(p=>`<dt>pid</dt><dd>${shorten(p)}</dd>`)
      : ['<dt>—</dt><dd class="dim">kuch nahi chal raha</dd>']))}</div>
  <div class="panel feedcard"><h3>🗓 Recent runs</h3>
    ${feed(s.runs.map(r=>`
      <div style="padding:7px 0;border-bottom:1px solid rgba(22,65,60,.5)">
        <b class="dim">${esc(r.id.slice(0,8))}</b>
        <span class="tag ${r.status==='ok'?'t-published':r.status==='running'?'t-approved':'t-failed'}">${esc(r.status)}</span>
        <div style="font-size:12px" class="dim">${esc((r.started_at||'').slice(5,19))} → ${esc((r.ended_at||'').slice(5,19))}</div>
        <div class="err" style="font-size:12px">${esc((r.detail||'')||'').slice(0,180)}</div>
      </div>`).join(''))}</div>`;
  document.getElementById('runs').innerHTML = runRows(s.runs);
  document.getElementById('logs').textContent = D.logs.text;
}

/* ---------- settings view ---------- */
function renderSettings(){
  const st=D.settings||{};
  const pretty = k => k.toLowerCase().replace(/_/g,' ');
  const rows = Object.keys(st).length
    ? Object.entries(st).map(([k,v])=>`<dt>${esc(pretty(k))}</dt><dd>${esc(v)}</dd>`)
    : ['<dt>—</dt><dd class="dim">.env values not found</dd>'];
  document.getElementById('v-settings').innerHTML = `
  <div class="panels">
    <div class="panel feedcard"><h3>⚙ Config (.env mirror)</h3>${feed(`<dl class="kv">${rows.join('')}</dl>`)}</div>
    <div class="panel feedcard"><h3>⚙ Runtime</h3>${feed(`<dl class="kv">
      <dt>Rumble channel</dt><dd><a href="https://rumble.com/c/IndianHistoryOfficial" target="_blank">IndianHistoryOfficial ↗</a></dd>
      <dt>Cookies saved</dt><dd>${esc(D.status.rumble_cookies_since||'none')}</dd>
      <dt>Last cycle start</dt><dd>${esc(D.status.heartbeat?.last_start||'—')}</dd>
      <dt>Last cycle end</dt><dd>${esc(D.status.heartbeat?.last_end||'—')}</dd>
      <dt>Dashboard port</dt><dd>8600 (local only)</dd>
      <dt>DB file</dt><dd>data/ytauto.db</dd>
      <dt>Media dir</dt><dd>data/media/</dd>
      <dt>Scheduler log</dt><dd>data/logs/launchd.log</dd>
    </dl>`)}</div>
  </div>`;
}

/* ---------- master render ---------- */
function render(){
  if (!D) return;
  const s = D.status;
  document.getElementById('live').className = 'livebadge ' + (s.cycle_state==='RUNNING NOW'?'on':'off');
  document.getElementById('livetext').textContent = s.cycle_state;
  document.getElementById('meta').textContent =
    'updated ' + s.now + (s.source==='cloud'
      ? ' · ☁ live (GitHub Actions' + (s.ci_synced_at ? ' · sync ' + s.ci_synced_at : '') + ')'
      : ' · ⛁ local');
  if (view==='dash'){ renderCards(s); renderPanels(s); }
  else if (view==='videos') renderVideos();
  else if (view==='topics') renderTopics();
  else if (view==='jobs') renderJobs();
  else if (view==='logs'){ document.getElementById('logs2').textContent = D.logs.text; }
  else if (view==='settings') renderSettings();
}

async function refresh(){
  try {
    D = await (await fetch('/api/all')).json();
    render();
  } catch(err) {
    document.getElementById('meta').textContent = 'server down? dashboard.py start karo';
    console.error(err);
  }
}
refresh(); setInterval(refresh, 5000);
</script>
</body></html>"""

if __name__ == "__main__":
    # No login gate exists locally, so the logout entry stays off here.
    PAGE = PAGE.replace('{{LOGOUT}}', '')
    PAGE = PAGE.replace("{{CHANNEL}}", "https://rumble.com/c/IndianHistoryOfficial")
    import threading
    try:
        import ci_sync  # noqa: E402 - same directory as this script

        threading.Thread(target=ci_sync.loop, daemon=True,
                         name="ci-sync").start()
    except Exception as exc:  # noqa: BLE001 - dashboard still works local-only
        print(f"ci-sync disabled: {exc}", flush=True)
    print(f"ytauto dashboard → http://localhost:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

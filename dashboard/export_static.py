#!/usr/bin/env python3
"""Build the static online dashboard for Netlify (or any static host).

The local dashboard at 127.0.0.1:8600 needs this Mac. The online one does
not: this script bakes the exact same UI into a folder of plain files —
  index.html   the dashboard page (frontend reads ./all.json instead of /api)
  all.json     the same payload as the local server's /api/all
  media/       every thumbnail that exists locally

GitHub Actions runs this after every cycle (and on a 30-minute timer) and
pushes the folder to Netlify, so the user can open the dashboard from any
device. Stdlib only; reuses the live dashboard's page template and snapshot
builders so the two always look identical.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dashboard"))


def _status() -> dict:
    """Cloud-flavoured /api/status: the DB is the synced CI one."""
    import dashboard as dash
    import ci_sync
    from ci_sync import run_status

    # api_status() reads the module-global `ci_sync`, which the server binds
    # at startup (its __main__ block). Bind it here the same way.
    dash.ci_sync = ci_sync

    s = dash.api_status()
    ci = run_status()
    gh_runs = ci.get("runs") or []
    active = [r for r in gh_runs if r.get("status") in ("in_progress", "queued")]
    # On any static host there is no local process to probe: the cycle badge
    # is driven purely by GitHub Actions.
    s["live"] = {"running": bool(active),
                 "procs": [f"github-actions run #{active[0].get('number', '')}"]
                 if active else []}
    heartbeat = None
    for r in gh_runs:
        if r.get("status") == "completed":
            heartbeat = {"last_start": (r.get("created_at") or "")[:19],
                         "last_end": (r.get("updated_at") or "")[:19]}
            break
    s["heartbeat"] = heartbeat
    s["cycle_state"] = ("RUNNING NOW" if active
                        else ("idle" if heartbeat else "never run"))
    s["source"] = "cloud"
    s["now"] = time.strftime("%Y-%m-%d %H:%M:%S")
    show_run = active[0] if active else (
        gh_runs[0] if gh_runs else {})
    s["github"] = {
        "online": bool(gh_runs),
        "repo": "https://github.com/techtricks/rumble-studio/actions",
        "run_url": show_run.get("url"),
        "run_number": show_run.get("number"),
        "run_status": show_run.get("status"),
        "run_conclusion": show_run.get("conclusion"),
        "run_started_at": (show_run.get("started_at") or "")[:19],
        "run_updated_at": (show_run.get("updated_at") or "")[:19],
        "run_event": show_run.get("event"),
    }
    s["runs"] = [
        {"id": str(r.get("id") or "")[-8:], "status": r.get("status"),
         "started_at": (r.get("started_at") or "")[:19],
         "ended_at": (r.get("updated_at") or "")[:19],
         "detail": f"#{r.get('number')} {r.get('conclusion') or ''}".strip()}
        for r in gh_runs[:20]
    ]
    return s


def _settings_map() -> dict:
    """Mirror the local /api/settings: .env first, config defaults second."""
    import dashboard as dash

    vals = dash.api_settings()
    if vals:
        return {k: str(v) for k, v in vals.items() if v not in (None, "", [])}
    try:
        from ytauto.config import get_settings
    except Exception:  # noqa: BLE001 — settings are cosmetic; never block the build
        return {}

    vals = dash.api_settings()
    s = get_settings()
    fallback = {
        "NICHE": getattr(s, "niche", ""),
        "MIN_NICHE_SCORE": getattr(s, "min_niche_score", ""),
        "MIN_TOTAL_SCORE": getattr(s, "min_total_score", ""),
        "SUBREDDITS": getattr(s, "subreddits", None),
        "RSS_FEEDS": getattr(s, "rss_feeds", None),
        "AUTO_APPROVE": getattr(s, "auto_approve", ""),
        "DRY_RUN": getattr(s, "dry_run", ""),
        "RUMBLE_CHANNEL": getattr(s, "rumble_channel", ""),
        "LONGFORM_TARGET_S": getattr(s, "longform_target_s", ""),
        "IMAGES_PER_VIDEO": getattr(s, "images_per_video", ""),
    }
    out = {}
    for k, v in fallback.items():
        shown = vals.get(k)
        if shown is None:
            if isinstance(v, (list, tuple)):
                shown = ", ".join(str(x) for x in v[:4]) + (f" (+{len(v)-4})" if len(v) > 4 else "")
            else:
                shown = v
        if isinstance(shown, bool):
            shown = "1" if shown else "0"
        if shown not in (None, "", []):
            out[k] = shown
    return out


def _media(db_thumbs: list[str], site: Path) -> set[str]:
    """Copy thumbnails that exist locally; blank out the rest so the table
    degrades to a dash instead of a broken-image box. CI renders carry no
    media; the local Mac (and previously published videos) may have them."""
    media = site / "media"
    copied: set[str] = set()
    MEDIA = ROOT / "data" / "media"
    if MEDIA.exists():
        for name in dict.fromkeys(t for t in db_thumbs if t):
            src = MEDIA / name
            if src.suffix.lower() in (".jpg", ".png", ".webp") and src.exists():
                media.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, media / src.name)
                copied.add(src.name)
    return copied


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default=str(ROOT / "netlify_site"),
                    help="output folder (default netlify_site/)")
    args = ap.parse_args()
    site = Path(args.dest).resolve()
    if site.exists():
        shutil.rmtree(site)
    site.mkdir(parents=True)

    import dashboard as dash

    status = _status()
    videos = dash.api_videos()
    topics = dash.api_topics()
    if not videos and not topics:
        # An empty snapshot is never legitimate once the pipeline has run:
        # it means the CI DB failed to sync (that's how the online dashboard
        # once showed "no data" while everything was fine). Fail loudly so
        # the workflow stops here and the live site keeps its last good copy.
        print("FATAL: 0 videos and 0 topics — data/ci/ytauto.db did not sync "
              "(see ci_sync output above); refusing to publish an empty site",
              file=sys.stderr)
        sys.exit(1)
    copied = _media([v.get("thumb") or "" for v in videos], site)
    for v in videos:
        if v.get("thumb") and v["thumb"] not in copied:
            v["thumb"] = None
    logs = {
        "lines": 0,
        "text": "Cloud snapshot — live logs GitHub Actions run pe hi hain.\n"
                "Dashboard > GitHub card <ha> run page khol deta hai "
                "(xiety logs ke saath).\n"
                "https://github.com/techtricks/rumble-studio/actions",
    }
    payload = {
        "status": status,
        "videos": videos,
        "topics": topics,
        "logs": logs,
        "settings": _settings_map(),
    }
    (site / "all.json").write_text(json.dumps(payload, default=str))

    phtml = dash.PAGE.replace(
        "{{CHANNEL}}", "https://rumble.com/c/IndianHistoryOfficial")
    # static reads: all.json instead of /api; the rebuild timer (Actions)
    # refreshes the fact, the page just keeps itself modern on the screen.
    phtml = phtml.replace("fetch('/api/all')", "fetch('./all.json')")
    # logout: server-side cookie kill via the edge gate, with a confirm so a
    # stray click doesn't lock the user out mid-review.
    phtml = phtml.replace(
        '<span class="lo-slot">{{LOGOUT}}</span>',
        '<a class="ext lo-gap lo" href="/__dash_logout" '
        'onclick="return confirm(&quot;Log out? Url dobara kholne pe password '
        'maangega.&quot;)"><span class="ic">🚪</span>Logout</a>')
    phtml = phtml.replace(
        "refresh(); setInterval(refresh, 5000);",
        "refresh(); setInterval(() => location.reload(), 240000);")
    phtml = phtml.replace(
        "server down? dashboard.py start karo",
        "snapshot unavailable (all.json missing?) — Actions deploy dekho")
    # Auto-logout: a public-ish private dashboard shouldn't stay signed in on
    # an unattended screen. Any human signal (mouse, keys, scroll, touch)
    # resets the 2-minute idle clock; 30 s before the cut a toast warns, at
    # 2:00 the browser is sent to /__dash_logout and the edge gate kills the
    # session cookie. Browser-side only — good enough for a status dashboard.
    phtml = phtml.replace("</body>", """<script id="idle-logout">
(function () {
  var IDLE_MS = 240000/2;          // 120000 ms — 2 minutes
  var WARN_AT = 30000;             // toast when 30 s remain
  var idle = 0, toast = null;
  function reset() {
    idle = 0;
    if (toast) { toast.remove(); toast = null; }
  }
  ['pointermove','pointerdown','keydown','wheel','scroll','touchstart']
    .forEach(function (ev) {
      window.addEventListener(ev, reset, {passive: true});
    });
  setInterval(function () {
    idle += 1000;
    if (IDLE_MS - idle === WARN_AT && !toast) {
      toast = document.createElement('div');
      toast.textContent = '⏳ 2 min of no activity → auto-logout in 30s';
      toast.setAttribute('style',
        'position:fixed;bottom:18px;right:18px;z-index:999;' +
        'background:#2a1215;color:#f87171;border:1px solid #7f1d1d;' +
        'border-radius:12px;padding:12px 18px;font-size:14px;' +
        'box-shadow:0 10px 30px rgba(0,0,0,.5)');
      document.body.appendChild(toast);
    }
    if (idle >= IDLE_MS) location.href = '/__dash_logout';
  }, 1000);
})();
</script></body>""", 1)
    (site / "index.html").write_text(phtml)

    print(f"static dashboard ready: {site}  "
          f"({len(videos)} videos, {len(topics)} topics, {len(copied)} thumbs)")


if __name__ == "__main__":
    main()

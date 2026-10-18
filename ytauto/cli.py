"""Command line interface.

`python -m ytauto <command>`. Every command is safe to run twice and safe to
run with no credentials: `status`, `doctor`, `discover` and `render` work
offline, and anything that would touch the channel says so first because DRY_RUN
defaults to on.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__
from .config import get_settings
from .db import get_db
from .logging_setup import setup_logging
from .util import clip, humanize_duration

# ---------------------------------------------------------------- output
_OK = "\033[0;32m"
_WARN = "\033[0;33m"
_ERR = "\033[1;31m"
_DIM = "\033[2;37m"
_OFF = "\033[0m"


def _colour() -> bool:
    return sys.stdout.isatty()


def out(msg: str = "") -> None:
    print(msg)


def head(msg: str) -> None:
    c = _colour()
    print(f"{_DIM}──{_OFF} {msg}" if c else f"-- {msg}")


def ok(msg: str) -> None:
    print(f"{_OK}OK{_OFF}  {msg}" if _colour() else f"OK   {msg}")


def warn(msg: str) -> None:
    print(f"{_WARN}!!{_OFF}  {msg}" if _colour() else f"!!   {msg}")


def bad(msg: str) -> None:
    print(f"{_ERR}XX{_OFF}  {msg}" if _colour() else f"XX   {msg}")


def table(rows: List[Dict[str, Any]], columns: List[str]) -> None:
    if not rows:
        out("  (none)")
        return
    widths = {c: max(len(c), *(len(_cell(r.get(c))) for r in rows)) for c in columns}
    line = "  ".join(c.ljust(widths[c]) for c in columns)
    out("  " + (_DIM + line + _OFF if _colour() else line))
    for r in rows:
        cells = "  ".join(_cell(r.get(c)).ljust(widths[c]) for c in columns)
        out("  " + cells)


def _cell(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)


# --------------------------------------------------------------- commands
def cmd_status(args: argparse.Namespace) -> int:
    s = get_settings()
    db = get_db()
    st = db.stats()

    head("channel")
    if s.dry_run:
        warn("DRY_RUN=1 — nothing will be uploaded (set DRY_RUN=0 to go live)")
    else:
        ok("DRY_RUN=0 — uploads are live")
    from .youtube.auth import auth_status

    a = auth_status()
    out(f"  channel_id   {a['channel_id'] or '(not detected — run `auth`)'}")
    out(f"  token        {'valid' if a['token_valid'] else 'missing/expired'}")
    out(f"  quota today  {st['quota_used_today']} / {s.daily_quota} used, "
        f"{st['quota_remaining']} safe to spend")

    head("topics")
    out(f"  new          {st['topics_new']}")

    head("videos")
    by_status = st["videos"]
    if not by_status:
        out("  (none yet — run `discover`)")
    for status in ("discovered", "scripted", "approved", "rendered", "scheduled",
                   "published", "failed", "archived"):
        if status in by_status:
            out(f"  {status:<12} {by_status[status]}")

    head("upcoming")
    from .schedule import upcoming

    cal = upcoming(db, limit=10)
    table(
        [
            {
                "when": u["publish_at"].strftime("%Y-%m-%d %H:%M"),
                "in": u["in"],
                "kind": u["kind"],
                "status": u["status"],
                "title": clip(u["title"] or "", 52),
            }
            for u in cal
        ],
        ["when", "in", "kind", "status", "title"],
    )
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Everything that must be true before the pipeline can do real work."""
    s = get_settings()
    problems: List[str] = []
    optional: List[str] = []

    head("configuration")
    out(f"  python        {sys.version.split()[0]}")
    out(f"  package       {__version__}")
    out(f"  timezone      {s.timezone}")
    out(f"  niche         {s.niche}")
    out(f"  longform      {', '.join(s.longform_times)}")
    out(f"  shorts        {', '.join(s.shorts_times)}")
    out(f"  dry_run       {s.dry_run}")
    out(f"  auto_approve  {s.auto_approve}")

    head("secrets")
    from .youtube.auth import auth_status

    a = auth_status()
    if not a["client_secret"]:
        problems.append("no OAuth client: put client_secret.json in secrets/ (see README)")
    else:
        ok(f"client secret {Path(a['client_secret']).name}")
    if not a["token"]:
        problems.append("not authorised yet: run `python -m ytauto auth`")
    elif not a["token_valid"]:
        problems.append("token expired/corrupt: run `python -m ytauto auth` to re-authorise")
    else:
        ok(f"token valid, scopes={len(a['scopes'])}")
        if a.get("reauth_due"):
            # A Testing-mode consent screen invalidates the refresh token after
            # 7 days and nothing in the token file records that deadline, so the
            # cron would just stop one week after setup with no other signal.
            warn(
                "token is %.0f days old: a consent screen left in \"Testing\" "
                "expires refresh tokens after %d days. Re-run `python -m ytauto "
                "auth`, or click \"Publish app\" on the consent screen to make "
                "the token non-expiring",
                a["token_age_days"], a["testing_token_days"],
            )
    provider = s.resolve_llm_provider()
    if not provider:
        problems.append(
            "no LLM key set — set GEMINI_API_KEY (free tier, no billing) or "
            "ANTHROPIC_API_KEY; scripts will otherwise use the degraded template"
        )
    else:
        ok(f"llm {provider} model {s.provider_model(provider)}")

    head("tools")
    from .render import ffmpeg as ff
    from .render import voice as vo

    if ff.find_ffmpeg():
        ok(f"ffmpeg {ff.version_of()}")
    else:
        problems.append("ffmpeg missing: run `python -m ytauto setup-ffmpeg`")
    if vo.say_available():
        pool = s.tts_voice or vo.DEFAULT_VOICE_POOL
        ok(f"tts `say` available, {len([v for v in pool.split(',') if v])} voice(s) in rotation")
    else:
        problems.append("no TTS: macOS `say` not found (rendering cannot narrate)")
    if ff.find_font():
        ok("truetype font available for captions")
    else:
        problems.append("no TTF font for burned-in captions")

    head("monetisation policy")
    # These are the settings that decide whether a published video can earn, so
    # they belong next to the other pre-flight checks rather than in the README
    # only. A wrong value here is not a quality problem, it is a channel problem.
    if s.allow_article_images:
        warn(
            "ALLOW_ARTICLE_IMAGES=1: the source article's own photo is on screen. "
            "That is the outlet's copyrighted image, not a licensed one"
        )
    else:
        ok("article photos off; only CC0/PDM/BY/BY-SA pictures are used")
    ok(f"picture budget {s.images_per_video} per video, voices rotate per video")
    from .policy import manual_steps

    for step in manual_steps():
        out(f"  manual: {step}")

    head("sources")
    out(f"  rss feeds     {len(s.rss_feeds)}")
    out(f"  subreddits    {len(s.subreddits)}")
    if not s.rss_feeds and not s.subreddits:
        problems.append("no topic sources configured")

    head("verdict")
    if problems:
        for p in problems:
            bad(p)
        out("")
        out(f"  {len(problems)} problem(s). The pipeline will not publish until these are fixed.")
        return 1
    for o in optional:
        warn(o)
    ok("ready to run")
    if s.dry_run:
        warn("DRY_RUN is on: uploads will be simulated, not performed")
    return 0


def cmd_auth(args: argparse.Namespace) -> int:
    from .youtube.auth import AuthError, authorize, auth_status, load_credentials, revoke

    if args.revoke:
        revoke()
        ok("token removed")
        return 0
    if args.status:
        print(json.dumps(auth_status(), indent=2))
        return 0
    try:
        authorize()
    except AuthError as exc:
        bad(str(exc))
        return 1
    ok("authorised")
    # Detect and cache the channel id so later runs need no extra call.
    try:
        from .youtube.client import YouTubeClient

        YouTubeClient().channel_id()
    except Exception as exc:  # noqa: BLE001
        warn(f"authorised, but channel detection failed: {exc}")
    return 0


def cmd_discover(args: argparse.Namespace) -> int:
    from .pipeline import discover as stage

    db = get_db()
    if args.show:
        head("top topics (no network)")
        table(stage.preview(db, limit=args.limit), ["id", "source", "niche", "fresh", "total", "title"])
        return 0
    if args.add:
        try:
            tid = stage.add_manual(db, args.add, url=args.url, summary=args.summary or "")
        except ValueError as exc:
            bad(str(exc))
            return 1
        ok(f"topic {tid} added")
        return 0
    if args.rescore:
        # Scores are materialised when a topic is discovered, so editing NICHE
        # in .env would otherwise leave every stored topic scored against the
        # old niche forever. This is the only way to re-apply it.
        s = get_settings()
        n = db.score_topics(s.niche_terms, half_life_h=s.topic_half_life_h)
        ok(f"rescored {n} topic(s) against niche {s.niche}")
        return 0

    result = stage.run(db=db, limit_each=args.limit_each, only=args.only)
    out("")
    head(f"fetched {result.fetched}, added {result.added}, duplicates {result.duplicates}")
    table(
        [
            {
                "id": k["id"],
                "source": k["source"],
                "niche": k["niche"],
                "fresh": k["fresh"],
                "score": k["total"],
                "title": clip(k["title"], 60),
            }
            for k in result.kept
        ],
        ["id", "source", "niche", "fresh", "score", "title"],
    )
    if not result.kept:
        warn("nothing cleared the threshold — lower MIN_TOTAL_SCORE or add sources")
    return 0


def cmd_script(args: argparse.Namespace) -> int:
    from .pipeline import script as stage

    db = get_db()
    if args.review:
        head("awaiting review")
        table(
            stage.pending_review(db),
            ["id", "kind", "status", "duration", "title"],
        )
        return 0
    if args.show:
        record = stage.show(args.show, db=db)
        out(json.dumps(record, indent=2, ensure_ascii=False, default=str))
        return 0

    if args.topic:
        result = stage.script_topic(args.topic, kind=args.kind, db=db, force=args.force)
        _report_script(result)
        return 0 if result.ok else 1

    s = get_settings()
    rows = db.top_topics(limit=args.count, min_score=s.min_total_score, min_niche=s.min_niche_score)
    if not rows:
        warn("no topics cleared the threshold; run `discover` first")
        return 1
    failed = 0
    for row in rows:
        result = stage.script_topic(row["id"], kind=args.kind, db=db, force=args.force)
        _report_script(result)
        failed += 0 if result.ok else 1
    return 1 if failed else 0


def _report_script(result: Any) -> None:
    if result.status == "failed" and result.errors:
        bad(f"{result.video_id} {clip(result.title, 50)}")
        for e in result.errors[:6]:
            out(f"      - {e}")
    else:
        ok(f"{result.video_id} [{result.status}] {clip(result.title, 60)}")
    for w in getattr(result, "warnings", [])[:3]:
        warn(f"  {w}")


def cmd_approve(args: argparse.Namespace) -> int:
    from .pipeline import script as stage

    db = get_db()
    targets = args.ids or [r["id"] for r in stage.pending_review(db)]
    if not targets:
        warn("nothing awaiting approval")
        return 0
    n = 0
    for vid in targets:
        if stage.approve(vid, db=db, force=args.force):
            n += 1
    ok(f"{n}/{len(targets)} approved")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    from .pipeline import render as stage

    db = get_db()
    if args.cleanup:
        stage.cleanup(args.cleanup)
        return 0
    if args.id:
        result = stage.render_video(args.id, db=db, force=args.force)
        if result.ok:
            ok(f"{result.video_id} -> {result.video_path} ({humanize_duration(int(result.duration_s or 0))})")
            return 0
        bad(f"{result.video_id}: {result.error}")
        return 1

    results = stage.render_queue(db=db, limit=args.limit, kind=args.kind)
    if not results:
        warn("no approved videos to render")
        return 0
    failed = 0
    for r in results:
        if r.ok:
            ok(f"{r.video_id} -> {Path(r.video_path).name} ({humanize_duration(int(r.duration_s or 0))})")
        else:
            bad(f"{r.video_id}: {clip(r.error, 90)}")
            failed += 1
    return 1 if failed else 0


def cmd_upload(args: argparse.Namespace) -> int:
    from .pipeline import upload as stage

    db = get_db()
    dry = None if args.dry_run is None else args.dry_run
    if args.now:
        outcome = stage.go_live_now(args.now, db=db, dry_run=dry)
        (ok if outcome.ok else bad)(outcome.summary())
        return 0 if outcome.ok else 1
    if args.sync:
        n = stage.sync_published(db=db, dry_run=dry)
        ok(f"{n} video(s) newly live")
        return 0

    if args.id:
        outcomes = [stage.upload_video(args.id, db=db, dry_run=dry, schedule=not args.no_schedule)]
    else:
        outcomes = stage.upload_queue(db=db, limit=args.limit, kind=args.kind, dry_run=dry)
    if not outcomes:
        warn("no rendered videos to upload")
        return 0
    failed = 0
    for o in outcomes:
        if o.ok:
            ok(o.summary())
            for n in o.notes:
                out(f"      {n}")
        else:
            bad(o.summary())
            failed += 1
    return 1 if failed else 0


def cmd_publish(args: argparse.Namespace) -> int:
    """Alias for `upload --sync`; kept because "publish" is what people type."""
    from .pipeline import upload as stage

    n = stage.sync_published(db=get_db())
    ok(f"{n} video(s) newly live")
    return 0


def cmd_rumble(args: argparse.Namespace) -> int:
    """Upload rendered videos to Rumble via Playwright browser automation.

    Needs: playwright installed, RUMBLE_EMAIL/RUMBLE_PASSWORD (or saved
    cookies) in .env. Rumble has no public upload API, so this drives a real
    browser and can break if the upload page changes.
    """
    from .pipeline.upload_rumble import RumbleUploader

    db = get_db()
    uploader = RumbleUploader(db=db, headless=not args.headed)
    if args.video_id:
        ids = [args.video_id]
    else:
        ids = [v["id"] for v in db.videos_in_status("rendered")]
        if args.limit:
            ids = ids[: args.limit]
    if not ids:
        warn("no rendered videos to upload — run `ytauto render` first")
        return 0
    failed = 0
    for vid in ids:
        try:
            uploader.upload(vid)
            ok(f"uploaded to Rumble: {vid}")
        except Exception as exc:  # noqa: BLE001
            bad(f"{vid}: {exc}")
            failed += 1
    return 1 if failed else 0


def cmd_analytics(args: argparse.Namespace) -> int:
    from .youtube import analytics as stage

    db = get_db()
    if args.fetch:
        n = stage.fetch_daily(db, days=args.days)
        ok(f"{n} analytics row(s)")
        return 0
    data = stage.summary(db, days=args.days)
    head(f"last {data['days']} days")
    t = data["totals"]
    out(f"  views        {t['views']:,}")
    out(f"  impressions  {t['impressions']:,}")
    out(f"  ctr          {t['ctr'] * 100:.2f}%")
    out(f"  retention    {t['retention'] * 100:.1f}%")
    out(f"  subs gained  {t['subs_gained']:,}")
    out(f"  revenue est  ${t['estimated_revenue']:,.2f}")
    head("top videos")
    table(
        [
            {
                "id": v["id"],
                "kind": v["kind"],
                "views": v["views"],
                "ctr": v["ctr"] or 0.0,
                "retention": v["retention"] or 0.0,
                "title": clip(v["title"] or "", 46),
            }
            for v in data["top_videos"]
        ],
        ["id", "kind", "views", "ctr", "retention", "title"],
    )
    head("top topics")
    table(
        [
            {
                "key": t_["topic_key"],
                "videos": t_["videos"],
                "views": t_["total_views"],
                "ctr": t_["ctr"] or 0.0,
            }
            for t_ in data["top_topics"]
        ],
        ["key", "videos", "views", "ctr"],
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import run as stage

    reports = stage.auto_produce(cycles=args.cycles)
    for report in reports:
        head(f"{report.job} (run {report.run_id})")
        for line in report.lines():
            out(f"  {line}")
        for e in report.errors:
            bad(e)
    return 0 if all(r.ok for r in reports) else 1


def cmd_llm(args: argparse.Namespace) -> int:
    """Prove the LLM path works end to end with one real, tiny call.

    `doctor` can only see whether a key string exists. This actually dials the
    API, so a typo, an expired key, a wrong model name or a rate limit shows up
    here as a clear error instead of as a failed video three stages later.
    """
    import time

    from .llm import LLM, LLMError, LLMUnavailable

    s = get_settings()
    head("llm")
    provider = s.resolve_llm_provider()
    if not provider:
        bad(LLM()._missing_key_message())
        return 1
    ok(f"provider   {provider}  (LLM_PROVIDER={s.llm_provider})")

    if args.list_models:
        if provider in ("groq", "nvidia"):
            return _list_openai_compat_models(provider, s)
        if provider != "gemini":
            warn("--list-models is not implemented for %s", provider)
            return 1
        return _list_gemini_models(s)

    llm = LLM(s)
    model = llm.model
    ok(f"model      {model}")

    # The real research schema, not a toy one: if the model cannot satisfy the
    # schema the pipeline actually uses, there is no point finding out during
    # scripting.
    schema = {
        "type": "object",
        "properties": {
            "angle": {"type": "string"},
            "key_facts": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
        },
        "required": ["angle", "key_facts", "confidence"],
    }
    prompt = (
        f"{args.prompt}\n\n"
        "Return JSON with: angle (one sentence), key_facts (1-3 short strings), "
        "confidence (0-1). If you are unsure, say so in the facts and set a low "
        "confidence rather than guessing."
    )

    out("")
    t0 = time.monotonic()
    try:
        data = llm.complete(prompt, schema, system="You are a research assistant.", max_tokens=600)
    except LLMUnavailable as exc:
        bad(str(exc))
        return 1
    except LLMError as exc:
        bad(f"call failed: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        bad(f"unexpected {type(exc).__name__}: {exc}")
        return 1
    took = time.monotonic() - t0

    ok(f"round trip {took:.1f}s")
    missing = [k for k in schema["required"] if k not in data]
    if missing:
        bad(f"schema not satisfied: missing {missing}")
        return 1
    if not isinstance(data.get("key_facts"), list):
        bad("key_facts is not a list — the schema was ignored")
        return 1

    head("response")
    out(f"  angle      {str(data.get('angle'))[:90]}")
    for f in (data.get("key_facts") or [])[:3]:
        out(f"  fact       {str(f)[:90]}")
    out(f"  confidence {data.get('confidence')}")
    out("")
    ok("LLM path is working")
    return 0


def _list_openai_compat_models(provider: str, s: Any) -> int:
    """List what a Groq or NVIDIA key can actually reach."""
    from .llm_openai_compat import OpenAICompatBackend

    try:
        names = OpenAICompatBackend(provider, s).list_models()
    except Exception as exc:  # noqa: BLE001
        bad(f"could not list {provider} models: {exc}")
        return 1
    if not names:
        bad(f"{provider} returned no models for this key")
        return 1

    ok(f"provider   {provider}")
    ok(f"{len(names)} models reachable")
    print()
    for n in names:
        mark = " *" if n == OpenAICompatBackend(provider, s).default_model() else "  "
        print(f"{mark} {n}")
    print()
    print("(* = default; a 404 here means the id is listed but not served to your key)")
    return 0


def _list_gemini_models(s: Any) -> int:
    """Show which models this key can reach.

    Necessary because the models endpoint lists retired point versions too:
    `gemini-2.5-flash` is listed but 404s for a newly created key, so the list
    alone is not proof. This only filters to what supports generateContent, so
    treat it as candidates and let `ytauto llm` prove the one you picked.
    """
    import requests

    from .llm_gemini import API_ROOT

    root = API_ROOT.rsplit("/models", 1)[0]
    try:
        resp = requests.get(
            f"{root}/models", params={"key": s.gemini_api_key}, timeout=60
        )
    except Exception as exc:  # noqa: BLE001
        bad(f"could not reach the models endpoint: {exc}")
        return 1
    if resp.status_code >= 400:
        bad(f"HTTP {resp.status_code}: {resp.text[:200]}")
        return 1

    names = sorted(
        m["name"].split("/")[-1]
        for m in resp.json().get("models", [])
        if "generateContent" in (m.get("supportedGenerationMethods") or [])
    )
    if not names:
        warn("no models returned — the key may be invalid")
        return 1
    out("")
    out(f"  {len(names)} model(s) support generateContent:")
    for n in names:
        mark = " <- in use" if n in (s.gemini_model, s.gemini_model_fast) else ""
        out(f"    {n}{mark}")
    out("")
    out("  Set GEMINI_MODEL / GEMINI_MODEL_FAST in .env to choose one.")
    return 0


def cmd_quota(args: argparse.Namespace) -> int:
    from .youtube.client import YouTubeClient

    snap = YouTubeClient().quota_snapshot()
    head(f"quota {snap['day']}")
    out(f"  used       {snap['used']:,}")
    out(f"  remaining  {snap['remaining']:,}  (safe to spend)")
    out(f"  cap        {snap['cap']:,}")
    head("by endpoint")
    table(snap["by_endpoint"], ["endpoint", "calls", "units"])
    if not snap["by_endpoint"]:
        out("  (no API calls recorded yet)")
    return 0


def cmd_setup_ffmpeg(args: argparse.Namespace) -> int:
    from .render import ffmpeg as ff

    existing = ff.find_ffmpeg()
    if existing and not args.force:
        ok(f"ffmpeg already at {existing}")
        out(f"     {ff.version_of(existing)}")
        return 0
    try:
        path = ff.install_macos(force=args.force)
    except Exception as exc:  # noqa: BLE001
        bad(str(exc))
        return 1
    ok(f"ffmpeg installed to {path}")
    out(f"     {ff.version_of(path)}")
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    from .config import log_dir

    path = log_dir() / "ytauto.log"
    if not path.exists():
        warn(f"no log file at {path}")
        return 1
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[-args.lines :]:
        out(line)
    return 0


# ----------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ytauto",
        description="Autonomous YouTube research → script → render → publish pipeline.",
    )
    p.add_argument("--version", action="version", version=f"ytauto {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    p.add_argument("-q", "--quiet", action="store_true", help="warnings and errors only")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="channel, quota, pipeline and calendar at a glance").set_defaults(
        func=cmd_status
    )

    d = sub.add_parser("doctor", help="check everything needed to publish; non-zero if not ready")
    d.set_defaults(func=cmd_doctor)

    a = sub.add_parser("auth", help="grant/revoke YouTube access")
    a.add_argument("--revoke", action="store_true", help="delete the stored token")
    a.add_argument("--status", action="store_true", help="print auth state as JSON")
    a.set_defaults(func=cmd_auth)

    dsc = sub.add_parser("discover", help="harvest and score new topics")
    dsc.add_argument("--show", action="store_true", help="show stored topics, no network")
    dsc.add_argument("--limit", type=int, default=15, help="rows to show with --show")
    dsc.add_argument("--add", metavar="TITLE", help="add a topic by hand")
    dsc.add_argument(
        "--rescore",
        action="store_true",
        help="re-score stored topics against the current NICHE, no network",
    )
    dsc.add_argument("--url", help="source url for --add")
    dsc.add_argument("--summary", help="source text for --add")
    dsc.add_argument("--limit-each", type=int, default=20, help="max items per source")
    dsc.add_argument("--only", nargs="*", choices=["rss", "reddit"], help="restrict to these sources")
    dsc.set_defaults(func=cmd_discover)

    sc = sub.add_parser("script", help="turn topics into researched, validated scripts")
    sc.add_argument("--topic", help="specific topic id (default: best unused)")
    sc.add_argument("--kind", choices=["longform", "shorts"], default="longform")
    sc.add_argument("--count", type=int, default=1, help="how many topics to script")
    sc.add_argument("--force", action="store_true", help="re-script even if one exists")
    sc.add_argument("--review", action="store_true", help="list videos awaiting approval")
    sc.add_argument("--show", metavar="VIDEO_ID", help="dump one video record as JSON")
    sc.set_defaults(func=cmd_script)

    ap = sub.add_parser("approve", help="sign off scripted videos for rendering")
    ap.add_argument("ids", nargs="*", help="video ids (default: all awaiting review)")
    ap.add_argument("--force", action="store_true", help="approve even failed validation")
    ap.set_defaults(func=cmd_approve)

    rd = sub.add_parser("render", help="narrate and composite approved videos")
    rd.add_argument("--id", help="one video id (default: the queue)")
    rd.add_argument("--kind", choices=["longform", "shorts"], help="filter the queue")
    rd.add_argument("--limit", type=int, default=3, help="max videos from the queue")
    rd.add_argument("--force", action="store_true", help="re-render even if media exists")
    rd.add_argument("--cleanup", metavar="VIDEO_ID", help="delete rendered files for a video")
    rd.set_defaults(func=cmd_render)

    up = sub.add_parser("upload", help="upload and schedule rendered videos")
    up.add_argument("--id", help="one video id (default: the queue)")
    up.add_argument("--kind", choices=["longform", "shorts"], help="filter the queue")
    up.add_argument("--limit", type=int, default=2, help="max uploads this run")
    up.add_argument("--no-schedule", action="store_true", help="upload without a publish time")
    up.add_argument("--sync", action="store_true", help="verify scheduled videos went live")
    up.add_argument("--now", metavar="VIDEO_ID", help="make an uploaded video public immediately")
    group = up.add_mutually_exclusive_group()
    group.add_argument("--dry-run", dest="dry_run", action="store_true", default=None,
                       help="force simulation")
    group.add_argument("--no-dry-run", dest="dry_run", action="store_false",
                       help="force real upload (overrides DRY_RUN)")
    up.set_defaults(func=cmd_upload)

    pb = sub.add_parser("publish", help="confirm which scheduled videos are live")
    pb.set_defaults(func=cmd_publish)

    an = sub.add_parser("analytics", help="performance report, and the API fetch")
    an.add_argument("--fetch", action="store_true", help="pull fresh data (costs 1000 units)")
    an.add_argument("--days", type=int, default=28, help="window in days")
    an.set_defaults(func=cmd_analytics)

    rn = sub.add_parser("run", help="the full unattended loop, once or repeated")
    rn.add_argument("--cycles", type=int, default=1, help="how many times to run the loop")
    rn.set_defaults(func=cmd_run)

    q = sub.add_parser("quota", help="today's YouTube API spend")
    q.set_defaults(func=cmd_quota)

    lc = sub.add_parser(
        "llm", help="verify the LLM key and model with one real call"
    )
    lc.add_argument(
        "--prompt",
        default="Name one AI chip released in the last year and give its specs.",
        help="what to ask, to eyeball the model's actual quality",
    )
    lc.add_argument(
        "--list-models",
        action="store_true",
        help="list the models this key can actually use, and exit",
    )
    lc.set_defaults(func=cmd_llm)

    sf = sub.add_parser("setup-ffmpeg", help="download a project-local static ffmpeg")
    sf.add_argument("--force", action="store_true", help="reinstall even if present")
    sf.set_defaults(func=cmd_setup_ffmpeg)

    rm = sub.add_parser(
        "rumble",
        help="upload rendered videos to Rumble (Playwright; needs RUMBLE_EMAIL/RUMBLE_PASSWORD)",
    )
    rm.add_argument("--video", dest="video_id", help="upload one specific video id")
    rm.add_argument("--limit", type=int, help="at most N videos")
    rm.add_argument(
        "--headed",
        action="store_true",
        help="show the browser window while uploading (default: headless)",
    )
    rm.set_defaults(func=cmd_rumble)

    lg = sub.add_parser("logs", help="tail the log file")
    lg.add_argument("--lines", type=int, default=40, help="how many lines")
    lg.set_defaults(func=cmd_logs)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _quiet_eol_warnings()
    setup_logging(level="DEBUG" if args.verbose else "INFO", quiet=args.quiet)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        warn("interrupted")
        return 130
    except Exception as exc:  # noqa: BLE001 - a CLI should not dump a traceback
        if args.verbose:
            raise
        bad(str(exc))
        return 1


def _quiet_eol_warnings() -> None:
    """Drop the per-command "Python 3.9 is past its end of life" banner.

    google-auth emits this on import, so it otherwise appears in front of every
    single command's output. It is accurate but not actionable here — the only
    interpreter on this machine is the system 3.9 — and the requirement is
    recorded in the README instead. Other FutureWarnings still show.
    """
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=r".*past its end of life.*",
        category=FutureWarning,
    )


if __name__ == "__main__":
    sys.exit(main())

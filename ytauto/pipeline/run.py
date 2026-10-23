"""The daily run — one command that does the whole loop.

Ordering is not arbitrary. Quota is the scarcest resource (10,000 units/day,
1600 per upload), so the run:

  1. confirms what actually went live            (1 unit/video, only if due)
  2. refreshes analytics                         (1000 units, once)
  3. discovers new topics                        (0 quota)
  4. scripts the best unused topics              (0 quota, LLM tokens)
  5. auto-approves only if AUTO_APPROVE=1        (no quota)
  6. renders approved videos                     (0 quota, CPU)
  7. uploads + schedules, cheapest-risk first    (1700 units each)

Doing upload last means a quota-exhausted day still gets discovery, scripting
and rendering done, so nothing is wasted and the next day picks up where this
one stopped.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from ..config import Settings, get_settings
from ..db import Database, get_db, iso, utcnow
from ..logging_setup import get_logger
from ..util import clip
from . import discover as discover_stage
from . import render as render_stage
from . import script as script_stage
from . import shorts as shorts_stage
from . import upload as upload_stage
from ..youtube import analytics as analytics_stage

log = get_logger("pipeline.run")


@dataclass
class RunReport:
    job: str
    run_id: str
    published: int = 0
    analytics_rows: int = 0
    topics_found: int = 0
    topics_added: int = 0
    scripted: List[str] = field(default_factory=list)
    approved: List[str] = field(default_factory=list)
    rendered: List[str] = field(default_factory=list)
    uploaded: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def lines(self) -> List[str]:
        out = [
            f"published     {self.published}",
            f"analytics     {self.analytics_rows} rows",
            f"topics        {self.topics_added} new of {self.topics_found} fetched",
            f"scripted      {len(self.scripted)}",
            f"approved      {len(self.approved)}",
            f"rendered      {len(self.rendered)}",
            f"uploaded      {len(self.uploaded)}",
        ]
        if self.skipped:
            out.append(f"skipped       {len(self.skipped)}")
        if self.errors:
            out.append(f"errors        {len(self.errors)}")
        return out

    def detail(self) -> str:
        return "; ".join(
            f"{k}={v}" for k, v in (
                ("published", self.published),
                ("analytics", self.analytics_rows),
                ("added", self.topics_added),
                ("scripted", len(self.scripted)),
                ("approved", len(self.approved)),
                ("rendered", len(self.rendered)),
                ("uploaded", len(self.uploaded)),
                ("errors", len(self.errors)),
            )
        )


def daily(
    *,
    db: Optional[Database] = None,
    settings: Optional[Settings] = None,
    job: str = "daily",
    discover: bool = True,
    script_count: int = 2,
    render_count: int = 3,
    upload_count: int = 2,
    fetch_analytics: bool = True,
    sync: bool = True,
) -> RunReport:
    """Run the full loop once. Never raises; collects problems into the report."""
    script_count = int(os.getenv("SCRIPTS_PER_CYCLE", str(script_count)))
    render_count = int(os.getenv("RENDERS_PER_CYCLE", str(render_count)))
    upload_count = int(os.getenv("UPLOADS_PER_CYCLE", str(upload_count)))
    s = settings or get_settings()
    db = db or get_db()

    # Catch-up guard: a backup schedule may fire on the same day as the main
    # one. If a cycle already finished cleanly within CATCHUP_HOURS, this
    # invocation is a no-op instead of producing a duplicate video.
    catchup_h = float(os.getenv("CATCHUP_HOURS", "0"))
    if catchup_h > 0:
        since = iso(utcnow() - timedelta(hours=catchup_h))
        row = db.conn.execute(
            "select id from runs where ended_at>=? and status='ok' "
            "order by ended_at desc limit 1", (since,)).fetchone()
        if row:
            log.info("catch-up guard: last clean cycle %s is recent "
                     "(CATCHUP_HOURS=%g) — skipping this invocation",
                     row[0], catchup_h)
            rep = RunReport(job=job, run_id=row[0])
            rep.skipped.append(f"guard: clean cycle {row[0]} within "
                               f"{catchup_h:g}h — nothing to do")
            return rep
    run_id = db.start_run(job)
    report = RunReport(job=job, run_id=run_id)
    log.info("=== %s run %s starting ===", job, run_id)

    # 0. stale-media sweep: a previous cycle can have rows whose media never
    # made it out (mid-run kill, artifact that carries only the database).
    # A rendered-but-missing longform re-enters the render queue; a ghost
    # short row would block build_missing from ever creating the real short,
    # so it is deleted outright.
    try:
        stale = db.conn.execute(
            "select id, kind, video_path from videos "
            "where status in ('rendered','scheduled') and video_path is not null"
        ).fetchall()
        for r in stale:
            if r["video_path"] and not __import__("pathlib").Path(r["video_path"]).exists():
                if r["kind"] == "shorts":
                    db.conn.execute("delete from videos where id=?", (r["id"],))
                    log.info("swept ghost short %s (media missing)", r["id"])
                else:
                    db.set_status(r["id"], "approved")
                    db.release(r["id"], owner="stale-media-sweep")
                    log.info("requeued %s: media missing", r["id"])
        db.conn.commit()
    except Exception as exc:  # noqa: BLE001
        log.warning("stale-media sweep skipped: %s", exc)

    try:
        # 1. what went live
        if sync:
            try:
                report.published = upload_stage.sync_published(db=db, settings=s)
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"publish sync: {exc}")
                log.error("publish sync failed: %s", exc)

        # 2. analytics (skipped automatically when quota is short)
        if fetch_analytics:
            try:
                report.analytics_rows = analytics_stage.fetch_daily(db, settings=s)
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"analytics: {exc}")
                log.warning("analytics skipped: %s", exc)

        # 3. discover
        if discover:
            try:
                found = discover_stage.run(db=db, settings=s)
                report.topics_found = found.fetched
                report.topics_added = found.added
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"discover: {exc}")
                log.error("discover failed: %s", exc)

        # 4. script
        for topic_id in _pick_topics(db, s, script_count):
            try:
                result = script_stage.script_topic(topic_id, db=db, settings=s)
                report.scripted.append(result.video_id)
                if result.status == "failed":
                    report.skipped.append(f"{result.video_id}: {clip('; '.join(result.errors), 80)}")
                    continue
                # 5. approve
                if s.auto_approve:
                    if result.degraded:
                        # ALLOW_DEGRADED exists for dry runs, not for publishing.
                        # Template content must never reach a public channel
                        # without a human pressing `approve`.
                        report.skipped.append(
                            f"{result.video_id}: degraded (no LLM) — needs manual `approve`"
                        )
                        log.warning(
                            "video %s is degraded; refusing to auto-approve", result.video_id
                        )
                    elif script_stage.approve(result.video_id, db=db, settings=s, force=True):
                        report.approved.append(result.video_id)
                else:
                    report.skipped.append(f"{result.video_id}: awaiting `approve`")
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"script {topic_id}: {exc}")
                log.error("scripting %s failed: %s", topic_id, exc)

        # 6. render
        try:
            for r in render_stage.render_queue(db=db, settings=s, limit=render_count):
                if r.ok:
                    report.rendered.append(r.video_id)
                else:
                    report.errors.append(f"render {r.video_id}: {clip(r.error, 100)}")
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"render queue: {exc}")

        # 6.5 shorts — every rendered/published longform gets a vertical cut
        try:
            for sh in shorts_stage.build_missing(db=db, limit=render_count + 2):
                if sh.ok:
                    report.rendered.append(sh.video_id)
                else:
                    log.info("short skip %s: %s", sh.video_id, sh.error)
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"shorts: {exc}")

        # 7. upload + schedule
        try:
            for outcome in upload_stage.upload_queue(db=db, settings=s, limit=upload_count):
                if outcome.ok:
                    report.uploaded.append(outcome.video_id)
                else:
                    report.errors.append(f"upload {outcome.video_id}: {clip(outcome.error, 100)}")
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"upload queue: {exc}")

    except Exception as exc:  # noqa: BLE001 - a run must always close its ledger row
        report.errors.append(f"unhandled: {exc}")
        log.exception("daily run %s crashed", run_id)

    status = "ok" if report.ok else "error"
    db.finish_run(run_id, status, report.detail() + (" | " + "; ".join(report.errors[:3]) if report.errors else ""))
    log.info("=== %s run %s finished: %s ===", job, run_id, report.detail())
    return report


def _pick_topics(db: Database, s: Settings, count: int) -> List[str]:
    """Best unused topics that clear every threshold."""
    rows = db.top_topics(
        limit=count * 3,
        min_score=s.min_total_score,
        min_niche=s.min_niche_score,
    )
    out: List[str] = []
    for r in rows:
        if db.find_video_by_topic(r["id"]) is not None:
            continue
        out.append(r["id"])
        if len(out) >= count:
            break
    if not out:
        log.info("no unused topic clears the threshold (min_total=%.2f min_niche=%.2f)",
                 s.min_total_score, s.min_niche_score)
    return out


def auto_produce(
    *,
    db: Optional[Database] = None,
    settings: Optional[Settings] = None,
    cycles: int = 1,
) -> List[RunReport]:
    """`daily` repeated. Used by the `loop` command to run unattended."""
    s = settings or get_settings()
    db = db or get_db()
    reports: List[RunReport] = []
    for i in range(max(1, cycles)):
        reports.append(daily(db=db, settings=s, job=f"loop-{i + 1}", fetch_analytics=(i == 0)))
    return reports

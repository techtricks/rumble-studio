"""Stage 2 — script: topic -> research -> script -> metadata -> validated video row.

This stage spends LLM tokens but no quota. A video lands in `scripted` with
everything needed to render, and is only allowed to advance to `approved` if
every gate in `validate` passes.

A short is derived from the same topic as the long-form video, not invented
separately: the pipeline reuses the research and writes a tighter script, so
the two uploads are consistent and only cost one extra research pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..config import Settings, get_settings
from ..content import research_topic, write_metadata, write_script
from ..db import Database, get_db
from ..llm import LLM, get_llm
from ..logging_setup import get_logger
from ..sources.article import enrich, needs_enrichment
from ..util import clip, humanize_duration
from ..validate import validate_package

log = get_logger("pipeline.script")


@dataclass
class ScriptedVideo:
    video_id: str
    kind: str
    title: str
    status: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    degraded: bool = False

    @property
    def ok(self) -> bool:
        return not self.errors


def script_topic(
    topic_id: str,
    *,
    kind: str = "longform",
    db: Optional[Database] = None,
    settings: Optional[Settings] = None,
    llm: Optional[LLM] = None,
    force: bool = False,
) -> ScriptedVideo:
    """Build a scripted video row from a topic."""
    s = settings or get_settings()
    db = db or get_db()
    llm = llm or get_llm()

    topic = db.get_topic(topic_id)
    if topic is None:
        raise ValueError(f"unknown topic {topic_id}")

    existing = db.find_video_by_topic(topic_id)
    if existing is not None and existing["kind"] == kind and not force:
        log.info("topic %s already has a %s video (%s)", topic_id, kind, existing["id"])
        return ScriptedVideo(
            video_id=existing["id"], kind=kind, title=existing["title"] or "",
            status=existing["status"],
        )

    if existing is not None:
        db.set_status(existing["id"], "archived", error="superseded by a re-script")
        log.info("archived superseded video %s", existing["id"])

    tdict = {k: topic[k] for k in topic.keys()}
    log.info("scripting %s from topic %s: %s", kind, topic_id, clip(tdict.get("title", ""), 60))

    # Feeds carry teasers, not articles. Widen the source before researching, but
    # only for the topic actually selected -- enriching all 20 candidates at
    # discovery time would mean 20 article fetches a day, most of them unused.
    if needs_enrichment(tdict.get("summary") or ""):
        tdict["summary"] = enrich(tdict.get("summary") or "", tdict.get("url") or "")

    research = research_topic(tdict, settings=s, llm=llm)
    # One request returns the script and its metadata together; the free tier
    # allows 20 generateContent calls a day, so metadata must not cost its own.
    meta_fields: dict = {}
    script = write_script(
        kind, tdict, research, settings=s, llm=llm, meta_out=meta_fields
    )
    meta = write_metadata(
        kind, tdict, research, script, settings=s, llm=llm, prefetched=meta_fields
    )

    target_s = s.shorts_target_s if kind == "shorts" else s.longform_target_s
    errors, warnings = validate_package(
        kind=kind,
        title=meta.title,
        title_b=meta.title_b,
        description=meta.description,
        tags=meta.tags,
        script=script.text,
        thumbnail_plan=meta.thumbnail_plan,
        research={
            "angle": research.angle,
            "key_facts": research.key_facts,
            "confidence": research.confidence,
        },
        chapters=meta.chapters,
        max_title_len=55 if kind == "shorts" else s.title_max_len,
        niche_terms=s.niche_terms,
        target_s=target_s,
        wpm=s.words_per_minute,
        source_url=tdict.get("url"),
    )

    est_s = int(script.word_count / max(1, s.words_per_minute) * 60)
    duration_s = _duration_s(est_s, kind)
    source_note = _source_note(research, tdict)

    degraded = getattr(research, "degraded", False)
    if degraded:
        # Recorded on the row so an operator can tell, months later, that this
        # video was never fact-checked.
        source_note = ("[DEGRADED: written without an LLM, not fact-checked] "
                       + source_note).strip()
        if s.allow_degraded:
            # Forgive *only* the confidence complaint, so the pipeline can be
            # exercised end-to-end offline. A template that could not parse the
            # source at all is still a real failure.
            forgiven = [e for e in errors if e.startswith("research confidence")]
            if forgiven:
                errors = [e for e in errors if e not in forgiven]
                warnings = list(warnings) + [
                    "degraded content accepted (ALLOW_DEGRADED=1): " + forgiven[0]
                ]
        else:
            errors = list(errors) + [
                "written without an LLM — set ANTHROPIC_API_KEY, or "
                "ALLOW_DEGRADED=1 to allow template content for dry runs"
            ]

    video_id = db.create_video(
        kind,
        meta.title,
        script=script.text,
        description=meta.description,
        tags=meta.tags,
        hook=script.hook,
        thumbnail_plan=meta.thumbnail_plan,
        source_note=source_note,
        topic_id=topic_id,
        duration_s=duration_s,
    )
    db.set_status(
        video_id,
        "scripted",
        title_variant_b=meta.title_b,
        ab_test_active=1 if meta.title_b else 0,
    )
    for w in warnings:
        log.warning("video %s: %s", video_id, w)

    if errors:
        reason = "; ".join(errors[:5])
        db.mark_failed(video_id, "validation: " + reason)
        # Not 'used': no video came of this topic. Mark it rejected with the
        # reason instead, so the same unusable source is not silently retried
        # forever by `run`, and so an operator can see why it was dropped.
        db.mark_topic_rejected(topic_id, reason, video_id=video_id)
        log.error("video %s REJECTED by validation: %s", video_id, reason)
        return ScriptedVideo(
            video_id=video_id, kind=kind, title=meta.title, status="failed",
            errors=errors, warnings=warnings,
        )

    db.mark_topic_used(topic_id, video_id)
    log.info(
        "video %s scripted: %r (%s, %d words, ~%s, %d tags)",
        video_id, clip(meta.title, 50), kind, script.word_count,
        humanize_duration(duration_s), len(meta.tags),
    )
    return ScriptedVideo(
        video_id=video_id, kind=kind, title=meta.title, status="scripted",
        warnings=warnings, degraded=degraded,
    )


def _duration_s(est_s: int, kind: str) -> int:
    """Round to a whole second and keep inside YouTube's format limits."""
    if kind == "shorts":
        return max(5, min(180, est_s))       # Shorts cap is 3 minutes
    return max(30, min(43_200, est_s))       # long-form cap is 12 hours


def _source_note(research: Any, tdict: Dict[str, Any]) -> str:
    """Attribution kept with the video: what we based it on, and what is shaky."""
    lines: List[str] = []
    if research.sources:
        lines.append("Sources: " + "; ".join(research.sources[:6]))
    if tdict.get("url"):
        lines.append(f"Origin: {tdict['url']}")
    if research.risks:
        lines.append("Caveats: " + "; ".join(research.risks[:4]))
    lines.append(f"Research confidence: {research.confidence:.2f}")
    return "\n".join(lines)


def script_for_short(
    topic_id: str,
    *,
    db: Optional[Database] = None,
    settings: Optional[Settings] = None,
    llm: Optional[LLM] = None,
) -> Optional[ScriptedVideo]:
    """Derive a Short from a topic that already has a long-form video."""
    db = db or get_db()
    existing = db.conn.execute(
        "SELECT id FROM videos WHERE topic_id=? AND kind='shorts'", (topic_id,)
    ).fetchone()
    if existing:
        log.info("topic %s already has a short (%s)", topic_id, existing["id"])
        return None
    return script_topic(topic_id, kind="shorts", db=db, settings=settings, llm=llm)


def backfill_short(
    topic_id: str,
    *,
    db: Optional[Database] = None,
    settings: Optional[Settings] = None,
) -> Optional[ScriptedVideo]:
    """Alias kept for readability at the call site."""
    return script_for_short(topic_id, db=db, settings=settings)


def approve(
    video_id: str,
    *,
    db: Optional[Database] = None,
    settings: Optional[Settings] = None,
    force: bool = False,
) -> bool:
    """Human sign-off. The only legitimate path from `scripted` to `approved`.

    `force` exists for the offline/fallback path, where validation failures are
    expected and a human has explicitly looked at it.
    """
    db = db or get_db()
    s = settings or get_settings()
    row = db.get_video(video_id)
    if row is None:
        raise ValueError(f"unknown video {video_id}")
    if row["status"] not in ("scripted", "failed"):
        log.warning("video %s is %s; nothing to approve", video_id, row["status"])
        return False
    if row["status"] == "failed" and not force:
        log.error(
            "video %s failed validation: %s. Fix it, or re-run with --force to override.",
            video_id, row["last_error"],
        )
        return False
    if not row["script"] or not row["title"]:
        log.error("video %s has no script/title; cannot approve", video_id)
        return False
    db.set_status(video_id, "approved", error=None)
    log.info("video %s approved: %r", video_id, clip(row["title"], 60))
    return True


def pending_review(db: Optional[Database] = None) -> List[Dict[str, Any]]:
    db = db or get_db()
    rows = db.videos_in_status("scripted", "failed")
    return [
        {
            "id": r["id"],
            "kind": r["kind"],
            "status": r["status"],
            "title": r["title"],
            "duration": humanize_duration(r["duration_s"]),
            "error": r["last_error"],
        }
        for r in rows
    ]


def show(video_id: str, *, db: Optional[Database] = None) -> Dict[str, Any]:
    """Full record for `ytauto show <id>`."""
    db = db or get_db()
    row = db.get_video(video_id)
    if row is None:
        raise ValueError(f"unknown video {video_id}")
    d = {k: row[k] for k in row.keys()}
    for key in ("tags", "thumbnail_plan"):
        try:
            d[key] = json.loads(d[key] or ("[]" if key == "tags" else "{}"))
        except (json.JSONDecodeError, TypeError):
            pass
    return d

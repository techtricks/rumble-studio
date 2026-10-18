from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .config import COST, DB_PATH, SAFETY_MARGIN, get_settings
from .logging_setup import get_logger
from .util import has_token, parse_dt

log = get_logger("db")

SCHEMA_VERSION = 7

KINDS = ("longform", "shorts")

# The production state machine. Kept as a CHECK constraint so an illegal
# transition can never be written to disk, not even by a buggy script.
VIDEO_STATUSES = (
    "discovered",  # topic accepted, not scripted yet
    "scripted",    # script + metadata drafted, awaiting review
    "approved",    # human signed off, ready to render
    "rendered",    # media files exist
    "scheduled",   # uploaded as private with publishAt in the future
    "published",   # live
    "failed",      # see videos.last_error
    "archived",    # retired from the pipeline
)

TERMINAL = ("published", "archived")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: Optional[datetime] = None) -> str:
    dt = dt or utcnow()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex[:16]


DDL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Candidate topics mined from RSS/Reddit, scored for niche fit and freshness.
CREATE TABLE IF NOT EXISTS topics (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    url           TEXT,
    source        TEXT NOT NULL,              -- rss | reddit | manual
    source_score  REAL DEFAULT 0,             -- raw popularity from the source
    niche_score   REAL DEFAULT 0,             -- keyword match against NICHE
    freshness     REAL DEFAULT 0,             -- 1.0 = just published
    total_score   REAL DEFAULT 0,
    summary       TEXT,
    raw           TEXT,                        -- JSON blob from the source
    status        TEXT NOT NULL DEFAULT 'new', -- new | used | rejected
    reject_reason TEXT,                        -- why, when status='rejected'
    topic_video_id TEXT,                       -- video built from this topic
    published_at  TEXT,                        -- when the SOURCE published it
    created_at    TEXT NOT NULL,
    used_at       TEXT,
    FOREIGN KEY (topic_video_id) REFERENCES videos(id) ON DELETE SET NULL
);

-- One row per video the pipeline owns, across its whole lifecycle.
CREATE TABLE IF NOT EXISTS videos (
    id              TEXT PRIMARY KEY,
    kind            TEXT NOT NULL CHECK (kind IN ('longform', 'shorts')),
    status          TEXT NOT NULL CHECK (status IN (
                        'discovered','scripted','approved','rendered',
                        'scheduled','published','failed','archived')),
    title           TEXT,
    description     TEXT,
    tags            TEXT,                       -- JSON list
    script          TEXT,                       -- full script / narration
    hook            TEXT,                       -- first 3s hook for retention
    thumbnail_plan  TEXT,                       -- JSON brief for the thumbnail
    source_note     TEXT,                       -- attribution / citations
    video_path      TEXT,
    thumb_path      TEXT,
    topic_id        TEXT,
    youtube_id      TEXT UNIQUE,                -- assigned by YouTube on upload
    publish_at      TEXT,                       -- when it goes live (UTC ISO)
    published_at    TEXT,                       -- confirmed live (UTC ISO)
    scheduled_at    TEXT,                       -- when we uploaded w/ publishAt
    duration_s      INTEGER,
    title_variant_b TEXT,                       -- A/B title
    ab_test_active  INTEGER NOT NULL DEFAULT 0,
    image_credits   TEXT,                       -- JSON: licence+creator per picture
    narrator_voice  TEXT,                       -- which voice was actually used
    policy_error    TEXT,                       -- why the publish gate refused
    last_error      TEXT,
    attempts        INTEGER NOT NULL DEFAULT 0,
    lease_until     TEXT,                       -- crash-safe claim for workers
    lease_owner     TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_videos_status      ON videos(status);
CREATE INDEX IF NOT EXISTS idx_videos_publish_at  ON videos(publish_at);
CREATE INDEX IF NOT EXISTS idx_videos_kind_status ON videos(kind, status);
CREATE INDEX IF NOT EXISTS idx_topics_status     ON topics(status, total_score DESC);

-- Daily API spend. YouTube's cap is 10,000 units and a failed overage blocks
-- the key for 24h, so every single API call is recorded here.
--
-- The primary key is a surrogate id, not (day, at, endpoint): `at` is written at
-- second resolution, so two calls to the same endpoint inside one second used
-- to violate the key, crash the run, and lose the very record that exists to
-- stop us over-spending. Rows are append-only and the id keeps them unique.
CREATE TABLE IF NOT EXISTS quota_ledger (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    day       TEXT NOT NULL,
    cost      INTEGER NOT NULL,
    endpoint  TEXT NOT NULL,
    note      TEXT,
    at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_quota_day ON quota_ledger(day);
CREATE INDEX IF NOT EXISTS idx_quota_day_endpoint ON quota_ledger(day, endpoint);

-- Daily performance snapshots pulled from YouTube Analytics.
CREATE TABLE IF NOT EXISTS analytics (
    video_id                  TEXT NOT NULL,
    day                       TEXT NOT NULL,
    views                     INTEGER DEFAULT 0,
    impressions               INTEGER DEFAULT 0,
    impressions_ctr           REAL,
    avg_view_duration_s       REAL,
    avg_view_pct             REAL,
    likes                     INTEGER DEFAULT 0,
    comments                  INTEGER DEFAULT 0,
    subs_gained               INTEGER DEFAULT 0,
    estimated_revenue         REAL,
    fetched_at                TEXT NOT NULL,
    PRIMARY KEY (video_id, day),
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_analytics_day ON analytics(day);

-- Topic-level feedback: what actually worked, fed back into topic scoring.
CREATE TABLE IF NOT EXISTS topic_performance (
    topic_key   TEXT PRIMARY KEY,
    videos      INTEGER NOT NULL DEFAULT 0,
    total_views INTEGER NOT NULL DEFAULT 0,
    total_impressions INTEGER NOT NULL DEFAULT 0,
    ctr_sum     REAL NOT NULL DEFAULT 0,
    retention_sum REAL NOT NULL DEFAULT 0,
    updated_at  TEXT NOT NULL
);

-- Job execution log, so cron runs are auditable.
CREATE TABLE IF NOT EXISTS runs (
    id         TEXT PRIMARY KEY,
    job        TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at   TEXT,
    status     TEXT NOT NULL,             -- ok | error
    detail     TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at DESC);
"""

# Forward-only migrations, applied in order for databases created by an older
# build. `CREATE TABLE IF NOT EXISTS` in DDL cannot add a column to a table that
# already exists, so column additions live here instead.
#
# Key = the schema_version the migration produces. Each entry is a list of
# idempotent statements.
# Columns a caller may write. Kept explicit so a typo cannot become an UPDATE
# against something that is not a column.
_UPDATABLE_VIDEO_COLUMNS = frozenset({
    "title", "description", "tags", "script", "hook", "thumbnail_plan",
    "source_note", "video_path", "thumb_path", "youtube_id", "publish_at",
    "published_at", "scheduled_at", "duration_s", "title_variant_b",
    "ab_test_active", "last_error", "image_credits", "narrator_voice",
    "policy_error",
})

MIGRATIONS: Dict[int, List[str]] = {
    4: [
        "ALTER TABLE topics ADD COLUMN published_at TEXT",
    ],
    6: [
        # The reason a topic was rejected used to be appended to `summary`,
        # which is the source text the research step reads. A re-run
        # therefore researched a status message instead of the article, which
        # is how "OpenAI ships GPT-6" came back with zero facts.
        "ALTER TABLE topics ADD COLUMN reject_reason TEXT",
    ],
    5: [
        # Rebuild quota_ledger with a surrogate primary key. The old key was
        # (day, at, endpoint) with second-resolution timestamps, so two calls to
        # one endpoint in the same second collided. Copy the rows across; the
        # old table name is swapped back and forth to stay transactional.
        "ALTER TABLE quota_ledger RENAME TO quota_ledger_old",
        """CREATE TABLE quota_ledger (
               id        INTEGER PRIMARY KEY AUTOINCREMENT,
               day       TEXT NOT NULL,
               cost      INTEGER NOT NULL,
               endpoint  TEXT NOT NULL,
               note      TEXT,
               at        TEXT NOT NULL
           )""",
        """INSERT INTO quota_ledger(day, cost, endpoint, note, at)
               SELECT day, cost, endpoint, note, at FROM quota_ledger_old""",
        "DROP TABLE quota_ledger_old",
        "CREATE INDEX IF NOT EXISTS idx_quota_day ON quota_ledger(day)",
        "CREATE INDEX IF NOT EXISTS idx_quota_day_endpoint ON quota_ledger(day, endpoint)",
    ],
    7: [
        # Monetisation compliance needs three facts the schema could not hold.
        # image_credits: the licence and creator of every picture that reached
        # the screen, so the upload step can build an attributed description
        # instead of leaving credits in a file nobody reads. narrator_voice:
        # which voice was actually used, for the same disclosure. policy_error:
        # why the publish gate refused, so a human can see the reason on the row
        # rather than only in a log line.
        "ALTER TABLE videos ADD COLUMN image_credits TEXT",
        "ALTER TABLE videos ADD COLUMN narrator_voice TEXT",
        "ALTER TABLE videos ADD COLUMN policy_error TEXT",
    ],
}


class Database:
    def __init__(self, path: Optional[Path] = None) -> None:
        # Read DB_PATH from the environment at construction time rather than
        # trusting the import-time constant: config resolves it as a module
        # global, so anything that wants a different database (a test, a
        # second channel, a one-off inspection) would otherwise be locked to
        # whatever was set when ytauto.config was first imported.
        self.path = Path(path or os.getenv("DB_PATH") or DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    # ---------------------------------------------------------------- conn
    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA foreign_keys=ON")
            c.execute("PRAGMA busy_timeout=30000")
            c.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = c
        return c

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """Immediate transaction. Serialises writers so two cron runs
        can never claim the same video."""
        c = self.conn
        c.execute("BEGIN IMMEDIATE")
        try:
            yield c
        except Exception:
            c.execute("ROLLBACK")
            raise
        else:
            c.execute("COMMIT")

    def close(self) -> None:
        c = getattr(self._local, "conn", None)
        if c is not None:
            c.close()
            self._local.conn = None

    def _init_schema(self) -> None:
        c = self.conn
        c.executescript(DDL)
        current = self.get_meta("schema_version")
        if current is None:
            self.set_meta("schema_version", str(SCHEMA_VERSION))
            log.info("db initialised at %s (schema v%s)", self.path, SCHEMA_VERSION)
            return
        from_version = int(current)
        if from_version == SCHEMA_VERSION:
            return
        if from_version > SCHEMA_VERSION:
            raise RuntimeError(
                f"database schema is v{from_version} but this build only "
                f"understands v{SCHEMA_VERSION}. Downgrade the code, not the db."
            )
        self._migrate(from_version)
        self.set_meta("schema_version", str(SCHEMA_VERSION))
        log.info("db migrated v%s -> v%s", from_version, SCHEMA_VERSION)

    def _migrate(self, from_version: int) -> None:
        """Apply every pending version inside one transaction.

        The connection is in autocommit mode, so without the explicit BEGIN a
        crash part-way through a multi-statement migration (like the v5
        quota_ledger rebuild) would leave the table renamed and the rows
        stranded. All-or-nothing is the only safe option here.
        """
        self.conn.execute("BEGIN")
        try:
            for version in range(from_version + 1, SCHEMA_VERSION + 1):
                for stmt in MIGRATIONS.get(version, []):
                    try:
                        self.conn.execute(stmt)
                    except sqlite3.OperationalError as exc:
                        # "duplicate column name" => already applied by a partial run.
                        if "duplicate column" in str(exc).lower():
                            continue
                        raise
                    log.info("applied v%s: %s", version, stmt.strip()[:80])
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        self.conn.execute("COMMIT")

    # ---------------------------------------------------------------- meta
    def get_meta(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    # -------------------------------------------------------------- quota
    def quota_used_today(self, day: Optional[str] = None) -> int:
        day = day or utcnow().strftime("%Y-%m-%d")
        row = self.conn.execute(
            "SELECT COALESCE(SUM(cost),0) AS c FROM quota_ledger WHERE day=?", (day,)
        ).fetchone()
        return int(row["c"])

    def quota_remaining(self, day: Optional[str] = None) -> int:
        s = get_settings()
        used = self.quota_used_today(day)
        return max(0, int(s.daily_quota * SAFETY_MARGIN) - used)

    def can_afford(self, endpoint: str) -> bool:
        return self.quota_remaining() >= COST.get(endpoint, 100)

    def record_quota(self, endpoint: str, note: str = "", cost: Optional[int] = None) -> int:
        c = COST.get(endpoint, 100) if cost is None else cost
        self.conn.execute(
            "INSERT INTO quota_ledger(day,cost,endpoint,note,at) VALUES(?,?,?,?,?)",
            (utcnow().strftime("%Y-%m-%d"), c, endpoint, note, iso()),
        )
        return c

    # -------------------------------------------------------------- topics
    def add_topic(
        self,
        title: str,
        source: str,
        url: Optional[str] = None,
        source_score: float = 0.0,
        summary: str = "",
        raw: Optional[Dict[str, Any]] = None,
        published_at: Optional[str] = None,
    ) -> Optional[str]:
        """Insert a topic, or return None if we already track this URL/title.

        Checks the URL *and* the normalised title. A URL-only check let the same
        story through twice whenever two feeds carried it under different links
        — one RSS feed and the subreddit that posted the same headline both
        landed as separate topics, and `discover` then offered the channel two
        videos on identical news. Headlines are the thing that repeats; URLs
        usually do not.
        """
        norm_title = " ".join(title.split()).lower()
        if url:
            existing = self.conn.execute(
                "SELECT id FROM topics WHERE url = ?", (url,)
            ).fetchone()
        else:
            existing = None
        if not existing:
            existing = self.conn.execute(
                "SELECT id FROM topics WHERE lower(title) = ?", (norm_title,)
            ).fetchone()
        if existing:
            return None
        tid = new_id()
        self.conn.execute(
            "INSERT INTO topics(id,title,url,source,source_score,summary,raw,"
            "published_at,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                tid,
                title.strip(),
                url,
                source,
                source_score,
                summary,
                json.dumps(raw or {}, ensure_ascii=False),
                published_at,
                iso(),
            ),
        )
        return tid

    def ingest_topics(self, topics: Iterable[Any]) -> Dict[str, int]:
        """Bulk-insert harvested `sources.Topic` objects. Returns counts."""
        added = dupes = 0
        for t in topics:
            row = t.to_row() if hasattr(t, "to_row") else dict(t)
            tid = self.add_topic(
                title=row["title"],
                source=row.get("source", "manual"),
                url=row.get("url"),
                source_score=row.get("source_score", 0.0),
                summary=row.get("summary", ""),
                raw=row.get("raw"),
                published_at=row.get("published_at"),
            )
            if tid:
                added += 1
            else:
                dupes += 1
        return {"added": added, "duplicates": dupes}

    def score_topics(self, niche_terms: Iterable[str], half_life_h: float = 24.0) -> int:
        """Recompute niche + freshness + total score for all unscored topics."""
        terms = [t.lower() for t in niche_terms]
        ids = [
            r["id"]
            for r in self.conn.execute("SELECT id FROM topics WHERE status='new'").fetchall()
        ]
        for tid in ids:
            self.score_topic(tid, terms, half_life_h)
        return len(ids)

    def score_topic(
        self, topic_id: str, niche_terms: Iterable[str], half_life_h: float = 24.0
    ) -> bool:
        """Score one topic.

        Age is measured from `published_at` (when the story broke), not
        `created_at` (when we happened to crawl it) — otherwise a three-day-old
        article found today would look brand new and outrank everything.
        """
        terms = [t.lower() for t in niche_terms]
        row = self.conn.execute(
            "SELECT id, title, summary, created_at, published_at FROM topics WHERE id=?",
            (topic_id,),
        ).fetchone()
        if row is None:
            return False

        # Whole-word matching only. A substring test on keywords this short is
        # actively harmful here: "ai" hits expl-ai-ned and "tech" hits
        # "techniques", so off-topic stories scored a full 1.0 and outranked
        # real ones. This is the top of the ranking, so it has to be right.
        hay = f"{row['title']} {row['summary'] or ''}"
        hits = sum(1 for t in terms if has_token(hay, t))
        # A wildcard term means "no niche filter" (manual seeds), not "match nothing".
        # The old formula (hits / len(terms)) penalised a broad vocabulary: a real
        # tech story naturally hits 3-5 terms regardless of vocabulary size. Score
        # on absolute hits instead: 1 hit → 0.4, 2 → 0.8, 3+ → 1.0.
        niche_score = 1.0 if "*" in terms else min(1.0, hits * 0.4)
        # Decay rather than a hard cliff: a story 30h old is still usable.
        ref = parse_dt(row["published_at"]) or parse_dt(row["created_at"])
        if ref is None:
            freshness = 0.0
        else:
            age_h = max(0.0, (utcnow() - ref).total_seconds() / 3600.0)
            freshness = 0.5 ** (age_h / max(1e-6, half_life_h))
        total = round(niche_score * 0.6 + freshness * 0.4, 4)
        self.conn.execute(
            "UPDATE topics SET niche_score=?, freshness=?, total_score=? WHERE id=?",
            (round(niche_score, 4), round(freshness, 4), total, row["id"]),
        )
        return True

    def top_topics(
        self,
        limit: int = 10,
        *,
        min_score: float = 0.0,
        min_niche: float = 0.0,
        exclude_used: bool = True,
    ) -> List[sqlite3.Row]:
        """Best unbuilt topics, best first.

        `min_niche` is the real quality gate: a 0.9-freshness story about
        something outside the niche must never outrank a solid in-niche one.
        """
        sql = "SELECT * FROM topics WHERE 1=1"
        args: List[Any] = []
        if exclude_used:
            sql += " AND status='new'"
        if min_score > 0:
            sql += " AND total_score >= ?"
            args.append(min_score)
        if min_niche > 0:
            sql += " AND niche_score >= ?"
            args.append(min_niche)
        sql += " ORDER BY total_score DESC, created_at DESC LIMIT ?"
        args.append(limit)
        return self.conn.execute(sql, args).fetchall()

    def get_topic(self, topic_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM topics WHERE id=?", (topic_id,)).fetchone()

    def mark_topic_used(self, topic_id: str, video_id: str) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE topics SET status='used', used_at=?, topic_video_id=? WHERE id=?",
                (iso(), video_id, topic_id),
            )
            c.execute("UPDATE videos SET topic_id=? WHERE id=?", (topic_id, video_id))

    def mark_topic_rejected(
        self, topic_id: str, why: str, video_id: Optional[str] = None
    ) -> None:
        """Record that a topic produced a script which failed validation.

        Distinct from `mark_topic_used`: nothing was published from it. Keeps
        the topic out of the selection queue (so `run` does not loop on the same
        unusable source) while preserving the reason for a human to read.
        """
        # Never write the reason into `summary`: that column is the source
        # text the research step is given, so appending to it made a re-run
        # research a rejection notice rather than the article, and grew the
        # field without bound on repeated rejections.
        self.conn.execute(
            "UPDATE topics SET status='rejected', reject_reason=?, "
            "topic_video_id=COALESCE(?, topic_video_id) WHERE id=?",
            (why, video_id, topic_id),
        )
        if video_id:
            self.conn.execute(
                "UPDATE videos SET topic_id=? WHERE id=?", (topic_id, video_id)
            )

    # ------------------------------------------------------------- videos
    def create_video(
        self,
        kind: str,
        title: str,
        *,
        script: str = "",
        description: str = "",
        tags: Optional[List[str]] = None,
        hook: str = "",
        thumbnail_plan: Optional[Dict[str, Any]] = None,
        source_note: str = "",
        topic_id: Optional[str] = None,
        duration_s: Optional[int] = None,
    ) -> str:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        vid = new_id()
        now = iso()
        self.conn.execute(
            "INSERT INTO videos(id,kind,status,title,description,tags,script,hook,"
            "thumbnail_plan,source_note,topic_id,duration_s,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                vid,
                kind,
                "discovered" if not script else "scripted",
                title,
                description,
                json.dumps(tags or [], ensure_ascii=False),
                script,
                hook,
                json.dumps(thumbnail_plan or {}, ensure_ascii=False),
                source_note,
                topic_id,
                duration_s,
                now,
                now,
            ),
        )
        return vid

    def get_video(self, video_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()

    def find_video_by_topic(self, topic_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM videos WHERE topic_id=?", (topic_id,)
        ).fetchone()

    def videos_in_status(self, *statuses: str) -> List[sqlite3.Row]:
        if not statuses:
            return []
        qs = ",".join("?" * len(statuses))
        return self.conn.execute(
            f"SELECT * FROM videos WHERE status IN ({qs}) ORDER BY created_at",
            list(statuses),
        ).fetchall()

    def set_status(
        self,
        video_id: str,
        status: str,
        *,
        error: Optional[str] = None,
        bump_attempts: bool = False,
        **fields: Any,
    ) -> None:
        if status not in VIDEO_STATUSES:
            raise ValueError(f"invalid status {status!r}")
        sets = ["status=?", "updated_at=?"]
        args: List[Any] = [status, iso()]
        for k, v in fields.items():
            if k not in _UPDATABLE_VIDEO_COLUMNS:
                raise ValueError(f"refusing to write unknown column {k!r}")
            sets.append(f"{k}=?")
            args.append(json.dumps(v, ensure_ascii=False) if k in {"tags", "thumbnail_plan"} else v)
        if error is not None:
            sets.append("last_error=?")
            args.append(error[:2000])
        if bump_attempts:
            sets.append("attempts=attempts+1")
        args.append(video_id)
        self.conn.execute(
            f"UPDATE videos SET {', '.join(sets)} WHERE id=?", args
        )

    def update_video(self, video_id: str, **fields: Any) -> None:
        """Update a video row without touching its status.

        Used for facts the pipeline learns while rendering — which voice was
        used, which pictures reached the screen — which must not by themselves
        move a video between lifecycle states.
        """
        if not fields:
            return
        sets: List[str] = []
        args: List[Any] = []
        for k, v in fields.items():
            if k not in _UPDATABLE_VIDEO_COLUMNS:
                raise ValueError(f"refusing to write unknown column {k!r}")
            sets.append(f"{k}=?")
            args.append(v)
        sets.append("updated_at=?")
        args.append(iso())
        args.append(video_id)
        self.conn.execute(f"UPDATE videos SET {', '.join(sets)} WHERE id=?", args)

    def mark_failed(self, video_id: str, error: str) -> None:
        self.set_status(video_id, "failed", error=error, bump_attempts=True)
        log.error("video %s failed: %s", video_id, error)

    # ------------------------------------------------------- job leasing
    def claim(self, video_id: str, owner: str, minutes: int = 30) -> bool:
        """Atomically take ownership of a video.

        Returns False if another worker holds a live lease, so overlapping
        cron invocations can never double-publish the same video.
        """
        cutoff = iso(utcnow() + timedelta(minutes=minutes))
        now = iso()
        with self.tx() as c:
            cur = c.execute(
                "UPDATE videos SET lease_owner=?, lease_until=?, updated_at=? "
                "WHERE id=? AND (lease_until IS NULL OR lease_until < ?)",
                (owner, cutoff, now, video_id, now),
            )
            return cur.rowcount == 1

    def release(self, video_id: str, owner: str) -> None:
        self.conn.execute(
            "UPDATE videos SET lease_owner=NULL, lease_until=NULL, updated_at=? "
            "WHERE id=? AND lease_owner=?",
            (iso(), video_id, owner),
        )

    def owner_token(self) -> str:
        return f"host-{__import__('socket').gethostname()}-{new_id()[:6]}"

    # -------------------------------------------------------------- runs
    def start_run(self, job: str) -> str:
        rid = new_id()
        self.conn.execute(
            "INSERT INTO runs(id,job,started_at,status) VALUES(?,?,?,'running')",
            (rid, job, iso()),
        )
        return rid

    def finish_run(self, run_id: str, status: str, detail: str = "") -> None:
        self.conn.execute(
            "UPDATE runs SET ended_at=?, status=?, detail=? WHERE id=?",
            (iso(), status, detail[:4000], run_id),
        )

    # ---------------------------------------------------------- analytics
    def upsert_analytics(self, video_id: str, row: Dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO analytics(video_id,day,views,impressions,impressions_ctr,"
            "avg_view_duration_s,avg_view_pct,likes,comments,subs_gained,"
            "estimated_revenue,fetched_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(video_id,day) DO UPDATE SET "
            "views=excluded.views, impressions=excluded.impressions, "
            "impressions_ctr=excluded.impressions_ctr, "
            "avg_view_duration_s=excluded.avg_view_duration_s, "
            "avg_view_pct=excluded.avg_view_pct, likes=excluded.likes, "
            "comments=excluded.comments, subs_gained=excluded.subs_gained, "
            "estimated_revenue=excluded.estimated_revenue, fetched_at=excluded.fetched_at",
            (
                video_id,
                row["day"],
                int(row.get("views") or 0),
                int(row.get("impressions") or 0),
                _f(row.get("impressions_ctr")),
                _f(row.get("avg_view_duration_s")),
                _f(row.get("avg_view_pct")),
                int(row.get("likes") or 0),
                int(row.get("comments") or 0),
                int(row.get("subs_gained") or 0),
                _f(row.get("estimated_revenue")),
                iso(),
            ),
        )

    def recent_analytics(self, days: int = 7) -> List[sqlite3.Row]:
        since = (utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        return self.conn.execute(
            "SELECT v.id, v.kind, v.title, v.publish_at, a.* FROM analytics a "
            "JOIN videos v ON v.id=a.video_id WHERE a.day>=? ORDER BY a.day DESC",
            (since,),
        ).fetchall()

    def upsert_topic_performance(self, topic_key: str, **agg: Any) -> None:
        self.conn.execute(
            "INSERT INTO topic_performance(topic_key,videos,total_views,total_impressions,"
            "ctr_sum,retention_sum,updated_at) VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(topic_key) DO UPDATE SET "
            "videos=excluded.videos, total_views=excluded.total_views, "
            "total_impressions=excluded.total_impressions, ctr_sum=excluded.ctr_sum, "
            "retention_sum=excluded.retention_sum, updated_at=excluded.updated_at",
            (
                topic_key,
                int(agg.get("videos") or 0),
                int(agg.get("total_views") or 0),
                int(agg.get("total_impressions") or 0),
                float(agg.get("ctr_sum") or 0.0),
                float(agg.get("retention_sum") or 0.0),
                iso(),
            ),
        )

    # ------------------------------------------------------------- stats
    def stats(self) -> Dict[str, Any]:
        c = self.conn
        by_status = {
            r["status"]: r["n"]
            for r in c.execute(
                "SELECT status, COUNT(*) n FROM videos GROUP BY status"
            ).fetchall()
        }
        return {
            "topics_new": c.execute(
                "SELECT COUNT(*) c FROM topics WHERE status='new'"
            ).fetchone()["c"],
            "videos": by_status,
            "scheduled": c.execute(
                "SELECT COUNT(*) c FROM videos WHERE status='scheduled'"
            ).fetchone()["c"],
            "published": c.execute(
                "SELECT COUNT(*) c FROM videos WHERE status='published'"
            ).fetchone()["c"],
            "quota_used_today": self.quota_used_today(),
            "quota_remaining": self.quota_remaining(),
        }


def _f(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


_db: Optional[Database] = None
_db_lock = threading.Lock()


def get_db() -> Database:
    """The process-wide database handle, rebuilt if the configured path changes.

    Keyed on the resolved path rather than cached blindly. A plain "build once"
    singleton silently ignores a later DB_PATH — which is both confusing in
    production (you set the variable, nothing happens) and the reason a test
    suite sharing one process would leak state from one test into the next.
    """
    global _db
    want = Path(os.getenv("DB_PATH") or DB_PATH)
    if _db is not None and _db.path == want:
        return _db
    with _db_lock:
        if _db is None or _db.path != want:
            _db = Database(want)
    return _db

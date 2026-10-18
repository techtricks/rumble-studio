"""Publishing calendar: no collisions, timezone handling, idempotence."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ytauto.schedule import ScheduleError, assign, next_slot, tz, upcoming
from ytauto.util import parse_dt


def _video(db, kind="longform"):
    return db.create_video(kind, f"Video {kind}")


def test_next_slot_is_in_the_configured_local_time(db, settings):
    v = _video(db)
    slot = next_slot("longform", db, settings=settings)
    assert slot is not None
    local = slot.when.astimezone(tz(settings))
    assert local.strftime("%H:%M") in settings.longform_times


def test_slots_respect_the_minimum_gap(db, settings):
    first = _video(db, "longform")
    a = assign(db, first, "longform", settings=settings)
    second = _video(db, "longform")
    b = assign(db, second, "longform", settings=settings)
    assert a and b
    assert a.when != b.when
    gap = abs((b.when - a.when).total_seconds()) / 60.0
    assert gap >= settings.min_gap_minutes - 1e-6


def test_no_two_videos_share_a_slot(db, settings):
    slots = []
    for i in range(6):
        v = _video(db, "longform")
        s = assign(db, v, "longform", settings=settings)
        assert s is not None
        slots.append(s.when)
    assert len(set(slots)) == len(slots)


def test_longform_and_shorts_do_not_collide(db, settings):
    seen = {}
    for i in range(3):
        for kind, vid in (("longform", _video(db, "longform")),
                          ("shorts", _video(db, "shorts"))):
            s = assign(db, vid, kind, settings=settings)
            assert s is not None
            seen.setdefault(s.when, []).append(kind)
    for when, kinds in seen.items():
        assert len(kinds) == 1, f"{when} double-booked: {kinds}"


def test_assign_is_idempotent(db, settings):
    v = _video(db)
    first = assign(db, v, "longform", settings=settings)
    second = assign(db, v, "longform", settings=settings)
    assert first.when == second.when
    assert second.reason == "already scheduled"


def test_a_past_slot_is_replanned(db, settings):
    v = _video(db)
    db.set_status(v, "rendered", publish_at="2020-01-01T00:00:00+00:00")
    slot = assign(db, v, "longform", settings=settings)
    assert slot is not None
    assert slot.when > datetime.now(timezone.utc)


def test_assign_can_plan_without_saving(db, settings):
    v = _video(db)
    slot = assign(db, v, "longform", settings=settings, persist=False)
    assert slot is not None
    assert db.get_video(v)["publish_at"] is None


def test_assign_rejects_an_unknown_video(db, settings):
    with pytest.raises(ScheduleError):
        assign(db, "nope", "longform", settings=settings)


def test_next_slot_returns_none_when_the_horizon_is_full(db, settings, monkeypatch):
    v = _video(db)
    db.set_status(v, "rendered", publish_at="2020-01-01T00:00:00+00:00")
    monkeypatch.setattr("ytauto.schedule.next_slot", lambda *a, **k: None)
    assert assign(db, v, "longform", settings=settings) is None


def test_upcoming_is_ordered_and_future_only(db, settings):
    for i in range(3):
        v = _video(db)
        assign(db, v, "longform", settings=settings)
    rows = upcoming(db, limit=10)
    assert rows
    whens = [r["publish_at"] for r in rows]
    assert whens == sorted(whens)
    assert all(w > datetime.now(timezone.utc) for w in whens)


def test_slot_times_are_timezone_aware(db, settings):
    v = _video(db)
    s = assign(db, v, "longform", settings=settings)
    assert s.when.tzinfo is not None
    # The stored value round-trips through the text format unchanged.
    assert parse_dt(db.get_video(v)["publish_at"]) == s.when


def test_tz_helper_rejects_a_nonsense_timezone(monkeypatch):
    from ytauto.config import get_settings

    monkeypatch.setenv("TIMEZONE", "Mars/Olympus_Mons")
    with pytest.raises(ScheduleError, match="invalid TIMEZONE"):
        tz(get_settings())


def test_minimum_gap_is_configurable(db, monkeypatch):
    from ytauto.config import get_settings

    monkeypatch.setenv("MIN_GAP_MINUTES", "1440")
    s = get_settings()
    assert s.min_gap_minutes == 1440
    v = _video(db)
    assert assign(db, v, "longform", settings=s) is not None

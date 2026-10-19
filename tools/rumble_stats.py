#!/usr/bin/env python3
"""Mini-checker: scrape the live channel's views + monetization + earnings
from the logged-in Rumble account pages and write data/rumble_stats.json.

Reported by the dashboard's "Rumble earnings" panel and refreshed inside the
GitHub Actions cycle after every publish (cookies come from
RUMBLE_COOKIES_B64 there, from secrets/rumble_cookies.json locally).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COOKIES = ROOT / "secrets" / "rumble_cookies.json"

CONTENT_URL = "https://rumble.com/account/content"
DASH_URL = "https://rumble.com/account/dashboard?type=earnings&interval=30"

DOLLAR = r"\$([0-9,]+\.\d+)"


def _launch():
    from playwright.sync_api import sync_playwright
    pw = sync_playwright().start()
    try:
        browser = pw.chromium.launch(
            headless=True, channel="chrome",
            args=["--disable-blink-features=AutomationControlled"])
    except Exception:
        browser = pw.chromium.launch(headless=True)
    ctx = browser.new_context(viewport={"width": 1300, "height": 1000})
    if COOKIES.exists():
        ctx.add_cookies(json.loads(COOKIES.read_text()))
    return pw, browser, ctx


def scrape() -> dict:
    import json as _json
    if not COOKIES.exists():
        raise SystemExit(f"no cookies at {COOKIES} — run a rumble login first")
    pw, browser, ctx = _launch()
    stats: dict = {"checked_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        page = ctx.new_page()
        page.goto(CONTENT_URL, timeout=60000)
        page.wait_for_timeout(4000)
        lines = [l.strip() for l in page.inner_text("body").splitlines()]
        stats["videos"] = _parse_content(lines)

        page.goto(DASH_URL, timeout=60000)
        page.wait_for_timeout(4000)
        lines = [l.strip() for l in page.inner_text("body").splitlines()]
        stats["all_time"], stats["last30"], stats["balance"] = _parse_dash(lines)
        stats["earnings_rows"] = _parse_earnings_rows(lines)
    finally:
        browser.close()
        pw.stop()
    return stats


def _parse_content(lines: list[str]) -> list[dict]:
    """Content page rows: [badge] title … date: X … raw views: N…"""
    videos, i, n = [], 0, len(lines)
    while i < n:
        badge = None
        if lines[i] == "Monetized":
            badge = "monetized"
        elif lines[i].startswith("Ready to share"):
            badge = "pending"
        if badge is None:
            i += 1
            continue
        title = lines[i + 1] if i + 1 < n else ""
        raw = unique = None
        j = i + 1
        while j < n and j < i + 14:
            m = re.search(r"raw views:\s*(\d+)", lines[j])
            if m:
                raw = int(m.group(1))
                u = re.search(r"unique views:\s*(\d+)", lines[j + 1] or "")
                unique = int(u.group(1)) if u else None
                break
            j += 1
        if raw is not None and title:
            videos.append({"title": title[:100], "monetization": badge,
                           "raw_views": raw, "unique_views": unique})
            i = j
        i += 1
    return videos


def _parse_dash(lines: list[str]) -> tuple[dict, dict, str]:
    def after(label: str, lines: list[str]) -> str | None:
        for i, l in enumerate(lines):
            if l == label:
                for nxt in (x for x in lines[i + 1:i + 4] if x):
                    if re.match(DOLLAR + r"$", nxt) or re.match(r"^\d+$", nxt):
                        return nxt
        return None

    all_time = {"earnings": after("ALL TIME EARNINGS", lines) or "$0.00",
                "views": after("ALL TIME VIEWS", lines) or "0",
                "followers": after("TOTAL FOLLOWERS", lines) or "0"}
    last30 = {"earnings": after("EARNINGS", lines) or "$0.00",
              "views": after("VIEWS", lines) or "0",
              "hours": after("HOURS VIEWED", lines) or "0"}
    balance = "unknown"
    for i, l in enumerate(lines):
        if l == "BALANCE" and i + 3 < len(lines):
            for cand in lines[i + 1:i + 6]:
                if re.match(DOLLAR + r"", cand):
                    balance = cand
                    break
            break
    return all_time, last30, balance


def _parse_earnings_rows(lines: list[str]) -> list[dict]:
    """Video-date-earnings table from the earnings dashboard."""
    rows, n = [], len(lines)
    date_rx = re.compile(
        r"^[A-Z][a-z]{2} \d{1,2}, \d{4}\t" + DOLLAR.replace("$", r"\$")
        + r"\t(\d+)\t([\d.]+)\t(\d+)\t(\d+)$")
    date_rx = re.compile(r"^[A-Z][a-z]{2} \d{1,2}, \d{4}\t\$([0-9,]+\.\d+)"
                         r"\t(\d+)\t([\d.]+)\t(\d+)\t(\d+)$")
    for i, l in enumerate(lines):
        m = date_rx.match(l)
        if not m:
            continue
        title = None
        for back in range(i - 1, max(0, i - 5), -1):
            cand = lines[back]
            if cand in ("Indian History Facts", "Public", "Unlisted",
                        "Private", ""):
                continue
            title = cand
            break
        rows.append({"date": m.group(0).split("\t")[0], "title": (title or "")[:100],
                     "earnings": "$" + m.group(1), "views": int(m.group(2)),
                     "hours": float(m.group(3)), "comments": int(m.group(4)),
                     "likes": int(m.group(5))})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "rumble_stats.json"))
    args = ap.parse_args()
    stats = scrape()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    tmp.replace(out)
    at = stats["all_time"]
    l30 = stats["last30"]
    pend = sum(1 for v in stats["videos"] if v["monetization"] == "pending")
    monetized = sum(1 for v in stats["videos"] if v["monetization"] == "monetized")
    print(f"videos: {monetized} monetized, {pend} pending | "
          f"all-time: {at['views']} views / {at['earnings']} | "
          f"30d: {l30['views']} views / {l30['earnings']} | "
          f"balance: {stats['balance']}")
    print(f"written → {out}")


if __name__ == "__main__":
    main()

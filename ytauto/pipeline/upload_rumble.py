"""Rumble upload pipeline stage.

Rumble has no public upload API, so this drives a real Chrome window with
Playwright. Two realities shape the design:

1. **Cloudflare + bot detection.** The bundled Chromium gets flagged; the
   machine's real Chrome (`channel="chrome"`) does not. A saved session
   (`secrets/rumble_cookies.json`, written after the first successful login)
   skips the auth redirect entirely, but a challenge can still appear — so the
   upload page is *polled* instead of awaited, which gives a human time to
   clear whatever Rumble/Cloudflare shows.
2. **Selectors from the live page** (inspected 2026-10-18): `#Filedata` (video
   file), `#customThumb` (thumbnail), `#title`, `#description`, `#tags`,
   `#submitForm` (the "Upload" button). If Rumble redesigns the upload page,
   these are what stop working — re-dump the page and update here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from ..config import get_settings
from ..db import Database, get_db
from ..logging_setup import get_logger

log = get_logger("pipeline.upload_rumble")

COOKIES_PATH = Path("secrets/rumble_cookies.json")
POLL_SECONDS = 60          # per attempt, enough for one challenge clear
POLL_ATTEMPTS = 3          # headed browsers: a human may need a minute


class RumbleUploader:
    """Upload rendered videos to rumble.com via Playwright + real Chrome."""

    def __init__(self, db: Optional[Database] = None, headless: bool = False):
        self.db = db or get_db()
        self.headless = headless

    def _open_upload_page(self, page):
        """Navigate to the upload page and wait until its file input exists.

        Returns True when the page is usable. Handles login redirects and
        Cloudflare challenges by polling — in a headed browser a human can
        clear them (login once, save cookies, never again).
        """
        get_settings()
        page.goto("https://rumble.com/upload.php", timeout=60000)
        for attempt in range(POLL_ATTEMPTS):
            for _ in range(POLL_SECONDS // 5):
                page.wait_for_timeout(5000)
                try:
                    if page.locator("#Filedata").count() > 0 and "auth." not in page.url \
                            and "moment" not in page.title().lower():
                        return True
                except Exception:  # noqa: BLE001 - mid-navigation DOM access
                    pass
            if self.headless:
                log.warning(
                    "upload page did not appear (%s | %s) — cookies may be "
                    "stale or Cloudflare is blocking headless Chrome; "
                    "re-run with --headed",
                    page.url[:60], page.title()[:40],
                )
                return False
            log.warning(
                "waiting for the upload page… clear the Cloudflare challenge "
                "or log in in the open browser window (attempt %d/%d)",
                attempt + 1, POLL_ATTEMPTS,
            )
        return False

    def upload(self, video_id: str) -> bool:
        from playwright.sync_api import sync_playwright

        v = self.db.get_video(video_id)
        if v is None:
            raise ValueError(f"unknown video {video_id}")
        if v["status"] != "rendered":
            raise ValueError(f"video {video_id} not rendered (status={v['status']})")

        media_path = Path(v["video_path"])
        if not media_path.exists():
            raise FileNotFoundError(f"media missing: {media_path}")

        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(
                    headless=self.headless,
                    channel="chrome",
                    args=["--disable-blink-features=AutomationControlled"],
                )
            except Exception:
                browser = p.chromium.launch(headless=self.headless)
            context = browser.new_context(viewport={"width": 1280, "height": 900})
            if COOKIES_PATH.exists():
                context.add_cookies(json.loads(COOKIES_PATH.read_text()))
            page = context.new_page()

            if not self._open_upload_page(page):
                browser.close()
                raise RuntimeError(
                    f"upload page never became ready (last URL: {page.url})"
                )

            log.info("attaching video via the dropzone (real file chooser)…")
            with page.expect_file_chooser() as fc_info:
                page.locator(".upload-video-placeholder").click()
            fc_info.value.set_files(str(media_path))

            # The upload starts on selection; the progress bar hits 100% when
            # Rumble has the whole file. Only after that is step 1 accepted.
            for waited in range(0, 3600, 5):
                try:
                    width = page.eval_on_selector(
                        ".green_percent",
                        "e => e.style.width", strict=False,
                    )
                except Exception:  # noqa: BLE001 - bar may not exist yet
                    width = None
                if width and str(width).replace("%", "").strip():
                    try:
                        if float(str(width).replace("%", "")) >= 99:
                            log.info("upload 100%% after %ds", waited)
                            break
                    except ValueError:  # noqa: PERF203
                        pass
                page.wait_for_timeout(5000)
                if waited and waited % 300 == 0:
                    log.info("still uploading… (%ds, bar=%s)", waited, width)
            else:  # never completed
                browser.close()
                raise RuntimeError("video never finished uploading to Rumble")

            if page.locator("#title").count():
                page.fill("#title", v["title"] or "Untitled")
            try:
                if v["description"] and page.locator("#description").count():
                    page.fill("#description", v["description"])
            except Exception:  # noqa: BLE001 - description is optional
                log.warning("could not fill description")
            try:
                if v["tags"] and page.locator("#tags").count():
                    tags = v["tags"]
                    if tags.strip().startswith("["):
                        import json as _json

                        tags = ", ".join(_json.loads(tags))
                    page.fill("#tags", tags)
            except Exception:  # noqa: BLE001 - tags are optional
                log.warning("could not fill tags")

            # Primary + secondary category — Rumble refuses the form without
            # at least the primary one ("Please select at least one category").
            # The widget is a search-as-you-type dropdown, not a native select.
            primary = os.getenv("RUMBLE_PRIMARY_CATEGORY", "Education")
            secondary = os.getenv("RUMBLE_SECONDARY_CATEGORY", "News")
            for name, label in (("primary-category", primary),
                                ("secondary-category", secondary)):
                if not label:
                    continue
                try:
                    box = page.locator(f'input[name="{name}"]')
                    box.click()
                    # this widget filters as the user TYPES — fill() sets the
                    # value without firing the key events, so type it out
                    box.press_sequentially(label, delay=40)
                    page.locator(
                        f'.select-option[data-label="{label}"]').first.click(
                        timeout=5000)
                    log.info("category %s: %s", name.split("-")[0], label)
                except Exception:  # noqa: BLE001 - not fatal; errors surface later
                    log.warning("could not set %s category '%s'", name, label)

            # visibility; Rumble defaults to Public
            vis = (os.getenv("RUMBLE_VISIBILITY") or "public").lower()
            radios = {"public": 0, "unlisted": 1, "private": 2, "scheduled": 3}
            if vis in radios:
                try:
                    page.locator("input[type=radio][name=visibility]").nth(
                        radios[vis]
                    ).check(timeout=5000)
                    log.info("visibility: %s", vis)
                except Exception:  # noqa: BLE001 - keep Rumble's default
                    log.warning("could not set visibility %s — leaving default", vis)

            auto = os.getenv("RUMBLE_AUTO_PUBLISH", "1") == "1"

            log.info("clicking Upload (step 1)…")
            page.locator("#submitForm").click(timeout=10000)

            # step 2 (categories/licence) appears after a valid step 1
            step2_ok = False
            for _ in range(24):  # 2 min
                page.wait_for_timeout(5000)
                try:
                    if page.locator(".upload-form-step2").is_visible():
                        step2_ok = True
                        break
                except Exception:  # noqa: BLE001
                    pass
            errs = page.eval_on_selector_all(
                ".error",
                "els => els.filter(e => (e.innerText||'').trim()).map(e =>"
                " (e.id||'?') + ': ' + e.innerText.trim())",
            )
            page.screenshot(path="/tmp/rumble_after_submit.png", full_page=True)
            if errs:
                log.warning("step-1 errors: %s", errs)
            if not step2_ok:
                # bail with page dump; video stays on Rumble's servers but
                # unsubmitted, and the video keeps its rendered status here
                Path("/tmp/rumble_stuck.html").write_text(page.content())
                browser.close()
                raise RuntimeError(f"step 1 failed validation: {errs}")

            log.info("step 2 visible — completing form…")
            if auto:
                try:
                    page.screenshot(path="/tmp/rumble_step2.png", full_page=True)
                    Path("/tmp/rumble_step2.html").write_text(page.content())

                    # The step-2 view collapses the details into an accordion;
                    # title/description/tags/categories/channel all live there.
                    try:
                        page.locator(
                            ".video-more.form-wrap.additional-info .togle-acordion,"
                            " .video-more.form-wrap.additional-info h2",
                        ).first.click(timeout=5000)
                        page.wait_for_timeout(1000)
                        log.info("details accordion expanded")
                    except Exception:  # noqa: BLE001 - maybe already open
                        pass

                    # the step-2 view may have reset the fields; refill
                    try:
                        if page.locator("#title").count() \
                                and not page.input_value("#title"):
                            page.fill("#title", v["title"] or "Untitled")
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        if page.locator("#description").count() \
                                and not page.input_value("#description"):
                            page.fill("#description", v["description"] or "")
                    except Exception:  # noqa: BLE001
                        pass

                    # upload target: pick the labelled channel, not the
                    # "User Profile" that Rumble ticks by default — via JS
                    # because Playwright's actionability checks flake here
                    # (the form re-renders while the video is processed)
                    want = os.getenv("RUMBLE_CHANNEL") or "Indian History Facts"
                    picked = page.evaluate(
                        """label => {
                            const labels = [...document.querySelectorAll('#channelId label, fieldset#channelId label')];
                            const hit = labels.find(l => l.innerText.trim().toLowerCase().includes(label.toLowerCase()));
                            if (!hit) return null;
                            const input = document.getElementById(hit.getAttribute('for'));
                            if (!input || input.disabled) return null;
                            if (!input.checked) input.click();
                            return input.id + '=' + input.value;
                        }""", want)
                    if picked:
                        log.info("upload target: %s (%s)", want, picked)
                    else:
                        log.warning(
                            "channel '%s' not found — video goes to the "
                            "user profile", want)

                    # consent checkboxes — via JS as well; Playwright's check()
                    # times out when the page re-renders mid-action
                    for box_id in ("#crights", "#cterms"):
                        try:
                            page.eval_on_selector(
                                box_id,
                                "e => { if (!e.checked) e.click(); return e.checked }")
                        except Exception:  # noqa: BLE001
                            log.warning("could not tick %s", box_id)

                    page.locator("#submitForm2").click(timeout=10000)
                    page.wait_for_timeout(20000)
                    errs2 = page.eval_on_selector_all(
                        ".error",
                        "els => els.filter(e => (e.innerText||'').trim()).map(e =>"
                        " (e.id||'?') + ': ' + e.innerText.trim())",
                    )
                    page.screenshot(path="/tmp/rumble_final.png", full_page=True)
                    if errs2:
                        log.warning("submit errors: %s", errs2)
                        self.db.set_status(
                            video_id, "rendered",
                            error="rumble step-2: " + "; ".join(errs2),
                        )
                    else:
                        log.info("final submit done — video is going live")
                        self.db.set_status(video_id, "published", error=None)
                except Exception as exc:  # noqa: BLE001
                    log.warning("auto-publish failed: %s", exc)
            else:
                log.info("auto-publish off — finishing the second form manually")

            # persist the session so future uploads skip login + challenge
            COOKIES_PATH.parent.mkdir(parents=True, exist_ok=True)
            COOKIES_PATH.write_text(json.dumps(context.cookies(), indent=2))

            browser.close()
            return True

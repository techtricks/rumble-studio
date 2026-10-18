#!/usr/bin/env python3
"""
Rumble Upload Automation using Playwright
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def upload_to_rumble(video_path, title, description="", tags=""):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("[ERROR] Playwright not installed")
        print("Run: .venv/bin/pip install playwright && .venv/bin/playwright install chromium")
        return False

    video_path = Path(video_path).resolve()
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        return False

    email = os.getenv("RUMBLE_EMAIL")
    password = os.getenv("RUMBLE_PASSWORD")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://rumble.com/upload", timeout=60000)

        if "login" in page.url.lower() or page.locator("input[type='email']").count() > 0:
            if not email or not password:
                print("[ERROR] Login required. Set RUMBLE_EMAIL & RUMBLE_PASSWORD in .env")
                browser.close()
                return False
            print("[INFO] Logging in...")
            page.fill("input[type='email']", email)
            page.fill("input[type='password']", password)
            page.click("button[type='submit']")
            page.wait_for_url("https://rumble.com/upload", timeout=60000)
            print("[INFO] Logged in")

        print(f"[INFO] Uploading {video_path.name}...")
        page.set_input_files("input[type='file']", str(video_path))
        page.wait_for_selector("input[name='title']", timeout=300000)

        if title:
            page.fill("input[name='title']", title)

        if description:
            try:
                page.fill("textarea[name='description']", description)
            except:
                pass

        print("[SUCCESS] Upload ready. Review & publish manually.")
        input("Press Enter to close...")
        browser.close()
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--desc", default="")
    args = parser.parse_args()
    upload_to_rumble(args.file, args.title, args.desc)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

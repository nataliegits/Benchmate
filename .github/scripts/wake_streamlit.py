"""Wake/keep-alive for a Streamlit Community Cloud app.

A plain `curl` does NOT work for this: a sleeping app 303-redirects into
Streamlit's auth/session flow and only "wakes" once a real browser runs the
page's JavaScript and clicks the "get this app back up" button. So we drive a
headless Chromium, click the wake button if the app is napping, and linger a
few seconds so Streamlit registers a genuine session and resets its idle timer.
"""

import os
import re
import sys

from playwright.sync_api import sync_playwright

URL = os.environ.get("APP_URL", "https://benchmate.streamlit.app/").strip()

# Text on the Community Cloud "your app is sleeping" button. Match loosely in
# case Streamlit tweaks the wording.
WAKE_BUTTON = re.compile(r"(get this app back up|wake|back up)", re.IGNORECASE)


def main() -> int:
    print(f"Visiting {URL}")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        # Streamlit holds a websocket open, so "networkidle" may never fire —
        # wait for the DOM and then poll explicitly.
        page.goto(URL, wait_until="domcontentloaded", timeout=90_000)
        page.wait_for_timeout(5_000)

        # If the app was asleep, the wake button is present — click it and wait
        # for the app to boot.
        try:
            btn = page.get_by_role("button", name=WAKE_BUTTON)
            if btn.count() > 0:
                print("App was asleep — clicking the wake button")
                btn.first.click()
                # Cold start can take a while; give it room.
                page.wait_for_timeout(45_000)
            else:
                print("App already awake (no wake button found)")
        except Exception as e:  # noqa: BLE001 — best-effort keep-alive
            print(f"Wake-button step skipped: {e}")

        # Linger so Streamlit counts this as a real session.
        page.wait_for_timeout(10_000)
        print(f"Done. Final URL: {page.url}  Title: {page.title()!r}")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

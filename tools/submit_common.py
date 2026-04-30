"""Shared Playwright + CDP logic for Prosperity portal submit helpers."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

BASE_URL = "https://prosperity.imc.com"

TOOLS_DIR = Path(__file__).resolve().parent
INSPECT_CAPTURE_DIR = TOOLS_DIR / "inspect_captures"


def cdp_url() -> str:
    return os.environ.get("SUBMIT_CDP_URL", "http://localhost:9222").strip() or "http://localhost:9222"


@dataclass(frozen=True)
class BrowserProfile:
    """Human-facing strings for CDP connection errors and login prompts."""

    label: str
    connect_fail_message: str  # use {cdp_url} placeholder


def get_context(pw, profile: BrowserProfile):
    url = cdp_url()
    try:
        browser = pw.chromium.connect_over_cdp(url)
    except Exception:
        print(profile.connect_fail_message.format(cdp_url=url))
        sys.exit(1)

    if browser.contexts:
        ctx = browser.contexts[0]
    else:
        ctx = browser.new_context()

    print(f"  Connected to {profile.label} via CDP ({url})")
    return browser, ctx


def dump_page(page, step: int):
    INSPECT_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    shot = INSPECT_CAPTURE_DIR / f"submit_step{step}.png"
    page.screenshot(path=str(shot), full_page=True)
    print(f"\n  [Step {step}] URL: {page.url}")
    print(f"  Screenshot: {shot.resolve()}")

    print("  Buttons:")
    for btn in page.locator("button").all():
        try:
            txt = btn.inner_text().strip().replace("\n", " ")
            cls = (btn.get_attribute("class") or "")[:80]
            if txt or cls:
                print(f"    text={txt!r:35s}  class={cls!r}")
        except Exception:
            pass

    print("  Inputs/Textareas:")
    for inp in page.locator("input, textarea").all():
        try:
            t = inp.get_attribute("type") or "text"
            n = inp.get_attribute("name") or ""
            ph = inp.get_attribute("placeholder") or ""
            cl = (inp.get_attribute("class") or "")[:60]
            print(f"    type={t!r}  name={n!r}  placeholder={ph!r}  class={cl!r}")
        except Exception:
            pass

    print("  Editor/File elements:")
    for sel in [
        ".CodeMirror",
        ".monaco-editor",
        ".cm-editor",
        "[contenteditable=true]",
        "input[type=file]",
    ]:
        n = page.locator(sel).count()
        if n:
            print(f"    {sel}: {n} found")


def inspect_mode(trader_file: str, profile: BrowserProfile):
    from playwright.sync_api import sync_playwright

    print("\n── INSPECT MODE (step-through) ─────────────────────────────────")
    print("  The script will open the game, then wait for interactive steps.")
    print("  At each step it dumps buttons/inputs and saves a screenshot.\n")

    with sync_playwright() as pw:
        browser, ctx = get_context(pw, profile)
        page = ctx.new_page()

        print(f"  Navigating to {BASE_URL}/game ...")
        page.goto(f"{BASE_URL}/game", wait_until="load", timeout=40_000)

        if "/game" not in page.url:
            print(f"  Redirected to login: {page.url}")
            print(f"  Please log in in {profile.label} ...")
            page.wait_for_url("**/game**", timeout=120_000)

        time.sleep(2)
        dump_page(page, step=1)

        step = 1
        while True:
            print("\n  Type button text to click (or Enter to skip, 'done' to finish):")
            cmd = input("  > ").strip()
            if cmd.lower() == "done":
                break
            if cmd == "":
                continue
            try:
                for locator in [
                    page.locator(f"button:has-text('{cmd}')"),
                    page.locator(f"[class*='btn']:has-text('{cmd}')"),
                    page.locator(f"a:has-text('{cmd}')"),
                ]:
                    if locator.count() > 0:
                        locator.first.click(force=True, timeout=5_000)
                        page.wait_for_load_state("load", timeout=15_000)
                        time.sleep(1)
                        step += 1
                        dump_page(page, step=step)
                        break
                else:
                    print(f"  No element found with text '{cmd}'")
            except Exception as e:
                print(f"  Click failed: {e}")

        print("\n  Press Enter to close the browser connection.")
        input()
        ctx.close()
        browser.close()


def submit_mode(trader_file: str, profile: BrowserProfile):
    from playwright.sync_api import sync_playwright

    code = Path(trader_file).read_text()
    print(f"\n  File: {trader_file}  ({len(code):,} chars)")

    with sync_playwright() as pw:
        browser, ctx = get_context(pw, profile)
        page = ctx.new_page()

        print(f"  Navigating to {BASE_URL}/game ...")
        page.goto(f"{BASE_URL}/game", wait_until="load", timeout=40_000)

        if "/game" not in page.url:
            print(f"  Redirected to: {page.url}")
            print(f"  Please log in in {profile.label} and press Enter here to continue ...")
            input()
            page.wait_for_url("**/game**", timeout=120_000)

        print(f"  On game page: {page.url}")
        time.sleep(2)

        def force_click(selector: str, description: str):
            try:
                el = page.locator(selector).first
                el.wait_for(state="attached", timeout=30_000)
                el.click(force=True, timeout=30_000)
                page.wait_for_load_state("load", timeout=30_000)
                time.sleep(2)
                print(f"  ✓ {description}")
                return True
            except Exception as e:
                print(f"  ✗ {description}: {e}")
                return False

        force_click(".Preloader_loadingCompleteButton__HfZxg", "CONTINUE (preloader)")
        page.keyboard.press("Escape")
        time.sleep(3)

        force_click(".MissionOverviewButton_button__Grzdk", "DASHBOARD MISSION CONTROL")
        page.keyboard.press("Escape")
        time.sleep(3)

        force_click('[aria-label="Open ALGORITHM STATUS challenge"]', "OPEN ALGORITHM STATUS")
        page.keyboard.press("Escape")
        time.sleep(3)

        submitted = False

        for sel in [
            ".monaco-editor textarea",
            ".cm-editor .cm-content",
            ".CodeMirror",
            "textarea[name*='code']",
            "textarea[name*='trader']",
            "textarea[placeholder*='code']",
            "textarea[placeholder*='paste']",
            "textarea",
            "[contenteditable='true']",
        ]:
            try:
                el = page.locator(sel).first
                if el.count() == 0:
                    continue
                el.click(timeout=4_000)
                page.keyboard.press("Meta+a")
                page.keyboard.type(code, delay=0)
                submitted = True
                print(f"  Editor: {sel}")
                break
            except Exception:
                continue

        if not submitted:
            for sel in ["input[type='file']", "input[accept='.py']"]:
                try:
                    el = page.locator(sel).first
                    if el.count() == 0:
                        continue
                    tmp = Path("/tmp/_prosperity_submit.py")
                    tmp.write_text(code)
                    el.set_input_files(str(tmp), timeout=5_000)
                    submitted = True
                    print(f"  File upload: {sel}")
                    break
                except Exception:
                    continue

        if not submitted:
            print("\n  Could not find the code editor on the page.")
            print("  Run with --inspect to dump the page structure.")
            sys.exit(1)

        clicked = False
        for btn_sel in [
            "button:has-text('Submit')",
            "button:has-text('Upload')",
            "button:has-text('Deploy')",
            "button:has-text('Save')",
            "button[type='submit']",
        ]:
            try:
                btn = page.locator(btn_sel).first
                if btn.count() == 0:
                    continue
                btn.click(timeout=5_000)
                print(f"  Clicked: {btn_sel}")
                clicked = True
                break
            except Exception:
                continue

        if not clicked:
            print("  Warning: could not find a submit button — editor was filled but not submitted.")

        time.sleep(3)
        page.wait_for_load_state("load", timeout=15_000)
        print(f"  Done. URL: {page.url}")

        print(f"\n  Submission complete (leave {profile.label} open as usual).")


def run_main(profile: BrowserProfile) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trader")
    parser.add_argument("--inspect", action="store_true", help="dump page structure for debugging")
    args = parser.parse_args()

    if not Path(args.trader).exists():
        print(f"ERROR: {args.trader} not found")
        sys.exit(1)

    if args.inspect:
        inspect_mode(args.trader, profile)
    else:
        submit_mode(args.trader, profile)

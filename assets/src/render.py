"""Render the brand sources to PNG with headless Chromium (Playwright).

    python assets/src/render.py logo.html#mark=../logo-mark.png@1254x1254 ...

Each argument is <html>#<element id>=<output png>@<WxH>; the element is
screenshotted at exactly that CSS size (device scale 1). Deterministic:
local fonts, no network, document.fonts.ready awaited.
"""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent


def main(jobs: list[str]) -> None:
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome")
        for job in jobs:
            src, rest = job.split("#", 1)
            el, rest = rest.split("=", 1)
            out, size = rest.rsplit("@", 1)
            w, h = (int(v) for v in size.split("x"))
            page = b.new_page(viewport={"width": w + 200, "height": h + 200})
            page.goto((HERE / src).as_uri())
            page.evaluate("document.fonts.ready")
            node = page.locator(f"#{el}")
            box = node.bounding_box()
            assert round(box["width"]) == w and round(box["height"]) == h, (job, box)
            node.screenshot(path=str((HERE / out).resolve()))
            page.close()
            print("ok", out, f"{w}x{h}")
        b.close()


if __name__ == "__main__":
    main(sys.argv[1:])

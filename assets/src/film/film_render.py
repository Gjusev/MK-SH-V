"""Capture film.html frame by frame (Playwright, local Chrome) and encode.

    python film_render.py stills 0 120 360 ...     # work/still-<f>.png
    python film_render.py sheet 6 [a-b]            # work/sheet.png, every 6th frame
    python film_render.py full                     # work/frames/*.jpg -> work/picture.mp4
    python film_render.py range 1440 1650          # re-render a range in place, re-encode
"""
import subprocess
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
WORK.mkdir(exist_ok=True)


def page_ctx(p):
    b = p.chromium.launch(channel="chrome", args=["--disable-gpu-vsync", "--force-color-profile=srgb"])
    pg = b.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
    pg.goto((HERE / "film.html").as_uri())
    pg.evaluate("window.ready")
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    return b, pg, errs


def shot(pg, f, path, kind="png"):
    pg.evaluate(f"render({f})")
    pg.screenshot(path=str(path), type=kind, **({"quality": 95} if kind == "jpeg" else {}))


def main(argv):
    mode = argv[0]
    with sync_playwright() as p:
        b, pg, errs = page_ctx(p)
        total = pg.evaluate("window.TOTAL")
        if mode == "stills":
            for f in map(int, argv[1:]):
                shot(pg, f, WORK / f"still-{f:04d}.png")
        elif mode == "sheet":
            from PIL import Image, ImageDraw
            step = int(argv[1])
            a, z = (map(int, argv[2].split("-")) if len(argv) > 2 else (0, total))
            frames = list(range(a, z, step))
            tw, th, cols = 384, 216, 6
            sheet = Image.new("RGB", (cols * tw, ((len(frames) + cols - 1) // cols) * (th + 22)), "#222")
            for i, f in enumerate(frames):
                tmp = WORK / "_t.png"
                shot(pg, f, tmp)
                im = Image.open(tmp).convert("RGB").resize((tw, th), Image.LANCZOS)
                x, y = (i % cols) * tw, (i // cols) * (th + 22)
                sheet.paste(im, (x, y + 22))
                ImageDraw.Draw(sheet).text((x + 6, y + 4), f"f{f}  {f / 60:.2f}s", fill="#fff")
            sheet.save(WORK / f"sheet-{a}-{z}.png")
        elif mode in ("full", "range"):
            fr = WORK / "frames"
            fr.mkdir(exist_ok=True)
            a, z = (int(argv[1]), int(argv[2])) if mode == "range" else (0, total)
            for f in range(a, z):
                shot(pg, f, fr / f"{f:05d}.jpg", "jpeg")
                if f % 120 == 0:
                    print("frame", f, flush=True)
        b.close()
        if errs:
            print("PAGE ERRORS:", errs)
            sys.exit(1)
    if mode in ("full", "range"):
        subprocess.run(["ffmpeg", "-y", "-framerate", "60", "-i", str(WORK / "frames/%05d.jpg"),
                        "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p",
                        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                        "-movflags", "+faststart", str(WORK / "picture.mp4")], check=True)
    print("done", mode)


if __name__ == "__main__":
    main(sys.argv[1:])

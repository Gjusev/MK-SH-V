"""Build the MK-SH-V mark from the family's mark: MK-SH-1's whale shark
(Gjusev/MK-SH-1 assets/logo-mark.png, same author, Apache-2.0), with its tide
band recoloured from blue to the amber of the gate (tau, MK-SH-V's accent).
Antialiased edges keep their blend: every pixel on the ground-to-band line is
moved to the same position on the ground-to-amber line.

    python assets/src/build_mark.py <path to MK-SH-1 logo-mark.png>
Writes assets/logo-mark.png (1254 px) and assets/src/shark-mask.png (the
paper-coloured silhouette as alpha, used by the film's glint).
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
GROUND = np.array([0x1B, 0x2A, 0x41], float)
BAND = np.array([0x4A, 0x6F, 0xA5], float)
AMBER = np.array([0xC9, 0x8B, 0x2E], float)
PAPER = np.array([0xF7, 0xF6, 0xF3], float)


def main(src: str) -> None:
    im = np.asarray(Image.open(src).convert("RGB"), float)
    d = BAND - GROUND
    t = np.clip(((im - GROUND) @ d) / (d @ d), 0, 1)                 # position on ground -> band
    resid = np.linalg.norm(im - (GROUND + t[..., None] * d), axis=-1)  # distance from that line
    on_band = (resid < 14) & (t > 0.04)
    out = im.copy()
    out[on_band] = GROUND + t[on_band][:, None] * (AMBER - GROUND)
    Image.fromarray(out.round().astype(np.uint8)).save(HERE.parent / "logo-mark.png", optimize=True)
    # silhouette mask: closeness to paper (spots count as body)
    pd = PAPER - GROUND
    a = np.clip(((im - GROUND) @ pd) / (pd @ pd), 0, 1)
    a[on_band] = 0
    body = (a > 0.35) | ((np.linalg.norm(im - np.array([0xA8, 0xC3, 0xE8]), axis=-1) < 40))
    mask = np.where(body, 255, (a * 255)).astype(np.uint8)
    rgba = np.dstack([np.full_like(mask, 255)] * 3 + [mask])      # white, silhouette as alpha (CSS mask uses alpha)
    Image.fromarray(rgba).save(HERE / "shark-mask.png", optimize=True)
    print("logo-mark.png + shark-mask.png;", int(on_band.sum()), "band pixels recoloured")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else r"C:/Code Main/mokka-decisions/assets/logo-mark.png")

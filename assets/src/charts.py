"""README charts: real-image accuracy against the field, and the abstention gate
per held-out set. Static PNG (GitHub runs no script), so no hover layer; every
bar is direct-labelled and the README carries the same numbers as tables.

    python assets/src/charts.py      # writes assets/src/charts.html, then render.py

Numbers: MK-SH-V from runs/e41/e41_report.json and runs/e42-calib/e42_summary.json
(read here); comparators as published in README (E39-r2 jev zero-shot, E39 OCR arm,
E39 synthetic-only), measured by us on the same 2,586 rows.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E41 = json.loads((ROOT / "runs/e41/e41_report.json").read_text(encoding="utf-8"))["seeds"]
E42 = json.loads((ROOT / "runs/e42-calib/e42_summary.json").read_text(encoding="utf-8"))["seeds"]

def mk(seed):
    s = E41[seed]; b = s["real_by_slice"]
    return {"gui": b["gui"]["e41"], "cord": b["cord"]["e41"], "real": s["real_acc"]}

ACC = [  # label, sub, values, own
    ("MK-SH-V s42", "260M, trained", mk("42"), True),
    ("MK-SH-V s43", "260M, trained", mk("43"), True),
    ("jev-vlm-4b", "~4B, zero-shot", {"gui": 0.731, "cord": 0.972, "real": 0.785}, False),
    ("OCR + MK-SH-1", "~25M, trained", {"gui": 0.472, "cord": 0.486, "real": 0.475}, False),
    ("Same model, synthetic only", "260M, E39", {"gui": round(E41["42"]["real_by_slice"]["gui"]["e39"] * 2010) / 2010,
                                                 "cord": E41["42"]["real_by_slice"]["cord"]["e39"],
                                                 "real": 0.471}, False),
]
PANELS = [("gui", "Screenshots: is the goal completed?", "GUI-Odyssey, n = 2,010"),
          ("cord", "Receipts: is the total above a threshold?", "CORD, n = 576"),
          ("real", "All real images", "n = 2,586")]

def gate_rows():
    rows = []
    for key, label in (("test_nonfr", "Synthetic test"), ("test_fr", "Synthetic test, French"),
                       ("gui", "Real screenshots"), ("cord", "Real receipts"), ("task", "Text routing task")):
        for seed in ("42", "43"):
            t = E42[seed]["tables"][key]["overall"]
            rows.append((label, seed, t["coverage"], t["risk"], t["cp95"][0], t["cp95"][1]))
    return rows

THEMES = {
    "light": dict(bg="#F7F6F3", ink="#14181F", ink2="#4D5562", muted="#7A808A", grid="#E3E1DC",
                  own="#A86D17", other="#8593A6", budget="#14181F", held="#A86D17", miss="#8593A6"),
    "dark": dict(bg="#0E1724", ink="#EEF1F5", ink2="#B7C2D0", muted="#8E9BAD", grid="#25344A",
                 own="#D6983F", other="#66778E", budget="#EEF1F5", held="#D6983F", miss="#66778E"),
}

def acc_svg(c):
    W, PW, top = 1400, 430, 150
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="560" viewBox="0 0 {W} 560">',
           f'<rect width="{W}" height="560" fill="{c["bg"]}"/>',
           f'<text x="40" y="58" class="h" fill="{c["ink"]}">Accuracy on real images, held out</text>',
           f'<text x="40" y="92" class="s" fill="{c["ink2"]}">MK-SH-V in amber, every comparator neutral. All arms measured by us on the same rows; labels from each source&#8217;s ground truth.</text>']
    for i, (key, title, sub) in enumerate(PANELS):
        x0 = 40 + i * (PW + 25)
        out += [f'<text x="{x0}" y="{top}" class="p" fill="{c["ink"]}">{title}</text>',
                f'<text x="{x0}" y="{top + 24}" class="m" fill="{c["muted"]}">{sub}</text>']
        lx, bx, bw = x0, x0 + 150, PW - 150 - 64
        for j, (lab, sl, vals, own) in enumerate(ACC):
            y = top + 58 + j * 64
            v = vals[key]; col = c["own"] if own else c["other"]
            out += [f'<text x="{lx}" y="{y + 15}" class="l" fill="{c["ink"]}">{lab if len(lab) < 18 else "Synthetic only"}</text>',
                    f'<text x="{lx}" y="{y + 34}" class="m" fill="{c["muted"]}">{sl}</text>',
                    f'<line x1="{bx}" y1="{y - 6}" x2="{bx}" y2="{y + 40}" stroke="{c["grid"]}" stroke-width="2"/>',
                    f'<rect x="{bx}" y="{y + 4}" width="{bw * v:.1f}" height="22" rx="4" fill="{col}"/>',
                    f'<text x="{bx + bw * v + 10:.1f}" y="{y + 21}" class="v" fill="{c["ink"]}">{v:.3f}</text>']
    out.append("</svg>")
    return "\n".join(out)

def gate_svg(c):
    rows = gate_rows(); W, H = 1400, 140 + len(rows) * 40 + 70
    x0, x1, xmax = 400, 1040, 0.30          # axis 0..30 %; wider intervals clip with their value
    X = lambda r: x0 + (x1 - x0) * min(r, xmax) / xmax
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
           f'<rect width="{W}" height="{H}" fill="{c["bg"]}"/>',
           f'<text x="40" y="58" class="h" fill="{c["ink"]}">The refusal gate on held-out sets</text>',
           f'<text x="40" y="92" class="s" fill="{c["ink2"]}">Selective risk at tau = 0.55 with its exact 95 % Clopper-Pearson interval; the budget holds when the whole interval stays left of 5 %.</text>']
    for t in (0, .05, .10, .15, .20, .25, .30):
        x = X(t)
        out.append(f'<line x1="{x}" y1="128" x2="{x}" y2="{H - 56}" stroke="{c["grid"]}" stroke-width="{1.5}"/>')
        out.append(f'<text x="{x}" y="{H - 30}" class="m" text-anchor="middle" fill="{c["muted"]}">{int(t * 100)} %</text>')
    bx = X(0.05)
    out.append(f'<line x1="{bx}" y1="118" x2="{bx}" y2="{H - 56}" stroke="{c["budget"]}" stroke-width="2.5" stroke-dasharray="6 5"/>')
    out.append(f'<text x="{bx + 8}" y="124" class="m" fill="{c["ink"]}">5 % budget</text>')
    for cx, head in ((1105, "risk"), (1240, "CP95 upper"), (1340, "coverage")):
        out.append(f'<text x="{cx}" y="124" class="m" text-anchor="end" fill="{c["muted"]}">{head}</text>')
    prev = None
    for k, (label, seed, cov, risk, lo, hi) in enumerate(rows):
        y = 150 + k * 40
        held = hi <= 0.05; col = c["held"] if held else c["miss"]
        if label != prev:
            out.append(f'<text x="40" y="{y + 26}" class="l" fill="{c["ink"]}">{label}</text>')
            prev = label
        out.append(f'<text x="330" y="{y + 26}" class="m" fill="{c["muted"]}">seed {seed}</text>')
        out.append(f'<line x1="{X(lo):.1f}" y1="{y + 20}" x2="{X(hi):.1f}" y2="{y + 20}" stroke="{col}" stroke-width="6" stroke-linecap="round"/>')
        out.append(f'<circle cx="{X(risk):.1f}" cy="{y + 20}" r="8" fill="{col}" stroke="{c["bg"]}" stroke-width="2"/>')
        if hi > xmax:  # clipped: an arrow at the axis end, the real bound sits in its column
            out.append(f'<path d="M{x1 + 4},{y + 12} l12,8 l-12,8 Z" fill="{col}"/>')
        bold = ' font-weight="700"' if held else ""
        out.append(f'<text x="1105" y="{y + 25}" class="m" text-anchor="end" fill="{c["ink2"]}">{risk * 100:.1f} %</text>')
        out.append(f'<text x="1240" y="{y + 25}" class="m" text-anchor="end"{bold} fill="{c["ink"] if held else c["ink2"]}">{hi * 100:.1f} %</text>')
        out.append(f'<text x="1340" y="{y + 25}" class="m" text-anchor="end" fill="{c["ink2"]}">{cov:.3f}</text>')
    out.append("</svg>")
    return "\n".join(out)

CSS = """@font-face{font-family:Geist;src:url(fonts/geist-400.woff2);font-weight:400}
@font-face{font-family:Geist;src:url(fonts/geist-500.woff2);font-weight:500}
@font-face{font-family:Geist;src:url(fonts/geist-700.woff2);font-weight:700}
@font-face{font-family:"Geist Mono";src:url(fonts/geistmono-400.woff2);font-weight:400}
*{margin:0;padding:0}body{background:#999;padding:20px;display:flex;flex-direction:column;gap:20px;align-items:flex-start}
div>svg{display:block}
.h{font:700 30px Geist;letter-spacing:-.02em}.s{font:400 18px Geist}.p{font:500 20px Geist;letter-spacing:-.01em}
.l{font:500 17px Geist}.m{font:400 14.5px 'Geist Mono'}.v{font:500 16px 'Geist Mono'}"""

if __name__ == "__main__":
    body = []
    for mode, c in THEMES.items():
        body.append(f'<div id="acc-{mode}">{acc_svg(c)}</div>')
        body.append(f'<div id="gate-{mode}">{gate_svg(c)}</div>')
    html = f'<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>{"".join(body)}</body></html>'
    (Path(__file__).parent / "charts.html").write_text(html, encoding="utf-8")
    print("gate rows:", len(gate_rows()), "| height", 140 + len(gate_rows()) * 40 + 70)

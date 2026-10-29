"""E52 kernel: the E44 kernel with the e52 config (criteria docs/E52_CRITERIA_2026-10-29.md).

Init from the E47 checkpoints (output of kernel gjusev/mkshv-e47-train), bind
pool v4 (dataset mkshv-e52-bind: the E47 rows byte-identical plus 12,000
receipt-format rows of own thermal renders, images inside the dataset), E47
recipe, U1024 input rule unchanged, weights re-frozen with the receipt kind at
0.25 (G0 probe dropped the crop arm; full-image everywhere).
"""
from __future__ import annotations

import e44_train as K

if __name__ == "__main__":
    K.EXP = "e52"
    K.main()

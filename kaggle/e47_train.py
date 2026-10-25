"""E47 kernel: the E44 kernel with the e47 config (criteria docs/E47_CRITERIA_2026-10-25.md).

Init from the E46 U1024 checkpoints (output of kernel gjusev/mkshv-e46a-train),
bind pool v3 (dataset mkshv-e47-bind: the E45 rows with the digits kind asking
every digit position), E46 input rule (upscale to 1,024 px) and weights.
"""
from __future__ import annotations

import e44_train as K

if __name__ == "__main__":
    K.EXP = "e47"
    K.main()

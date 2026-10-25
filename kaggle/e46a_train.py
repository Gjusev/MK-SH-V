"""E46 arm U1024: the E44 kernel with the e46a config (criteria docs/E46_CRITERIA_2026-10-25.md).

Init from the E45 checkpoints (output of kernel gjusev/mkshv-e45-train), bind
pool v2 (dataset mkshv-e45-bind) with the E46 weights, every image upscaled to
a longest side of 1,024 px before the processor.
"""
from __future__ import annotations

import e44_train as K

if __name__ == "__main__":
    K.EXP = "e46a"
    K.main()

"""E45 kernel: the E44 kernel with the e45 config (criteria docs/E45_CRITERIA_2026-10-24.md).

Init from the E44 checkpoints (output of kernel gjusev/mkshv-e44-train), bind
pool v2 (dataset mkshv-e45-bind). Workers re-run the shipped e44_train module
with --exp e45.
"""
from __future__ import annotations

import e44_train as K

if __name__ == "__main__":
    K.EXP = "e45"
    K.main()

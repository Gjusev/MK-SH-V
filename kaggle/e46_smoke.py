"""E46 memory smoke (criteria docs/E46_CRITERIA_2026-10-25.md).

20 steps, seed 42, both arms at once: U1024 (e46a) on GPU 0, U1536 (e46b) on
GPU 1. Records peak GPU memory and seconds per step per arm into
e46_smoke_summary.json and starts no full run.
"""
from __future__ import annotations

import e44_train as K

if __name__ == "__main__":
    K.SMOKE_ONLY = ("e46a", "e46b")
    K.main()

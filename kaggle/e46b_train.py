"""E46 arm U1536: the E44 kernel with the e46b config (criteria docs/E46_CRITERIA_2026-10-25.md).

Same as e46a with every image upscaled to (and capped at) a longest side of
1,536 px; train deadline 4.0 h, wall kill 4.8 h.
"""
from __future__ import annotations

import e44_train as K

if __name__ == "__main__":
    K.EXP = "e46b"
    K.main()

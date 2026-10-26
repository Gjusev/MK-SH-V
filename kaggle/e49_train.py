"""E49 kernel: the E44 kernel with the e49 config (criteria docs/E49_CRITERIA_2026-10-26.md).

Init from the E47 checkpoints (output of kernel gjusev/mkshv-e47-train), bind
pool v4 (dataset mkshv-e49-bind: the E47 rows plus goal / goal_match rows of
2,000 new GUI-Odyssey episodes, their final screenshots inside the dataset),
E47 recipe, U1024 input rule and weights unchanged.
"""
from __future__ import annotations

import e44_train as K

if __name__ == "__main__":
    K.EXP = "e49"
    K.main()

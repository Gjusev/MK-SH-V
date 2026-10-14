"""Pure sequence/contract helpers for A1-V (stdlib-only, no torch).

Importable without the modeling stack so the contract tests (option ids,
slot masks, option permutation -> probability permutation) run anywhere.
"""

from __future__ import annotations

import math
import re
from typing import Any, Sequence

from mokka_decisions.contracts import DecisionCase

MAX_MARKERS = 32

# User text may never forge structure: marker patterns AND NeoMME's real
# special tokens (<doc>/<img>/<row>/<eos>/<mask>/<query>) are neutralised
# before tokenization. A literal "<img>" inside a state would otherwise
# inject patch placeholders and break the patch-count contract.
_MARKER_RE = re.compile(r"\[O\d+\]|<(?:img|doc|row|eos|mask|query)>")


def build_tail(
    tokenizer: Any,
    case: DecisionCase,
    marker_ids: Sequence[int],
    *,
    max_state_tokens: int | None = None,
    max_option_tokens: int | None = None,
) -> tuple[list[int], list[int]]:
    """Tokenize ``state <eos> question [O0] opt0 ... [O_{K-1}] opt_{K-1}``.

    Returns ``(tail_ids, marker_positions)`` where positions index into the
    tail. Marker tokens are never truncated; the state is cut first, then
    option descriptions. Marker-looking user text is neutralised so a literal
    ``[O3]`` inside a description can never forge a marker.
    """

    def enc(text: str) -> list[int]:
        return tokenizer(_MARKER_RE.sub(" ", text), add_special_tokens=False)["input_ids"]

    state_ids = enc(case.state)
    if max_state_tokens is not None:
        state_ids = state_ids[:max_state_tokens]
    ids: list[int] = list(state_ids)
    ids.append(tokenizer.eos_token_id)
    ids.extend(enc(case.question))
    markers: list[int] = []
    for i, option in enumerate(case.options):
        if i >= len(marker_ids):
            raise ValueError(
                f"{case.id}: {len(case.options)} options exceed {len(marker_ids)} markers"
            )
        markers.append(len(ids))
        ids.append(marker_ids[i])
        opt = enc(" " + option.description)
        if max_option_tokens is not None:
            opt = opt[:max_option_tokens]
        ids.extend(opt)
    return ids, markers


def masked_softmax_rows(
    score_rows: Sequence[Sequence[float]], num_valid: Sequence[int]
) -> list[list[float]]:
    """Softmax over the first ``num_valid[r]`` entries of each row (pure python)."""
    out = []
    for row, k in zip(score_rows, num_valid):
        vals = [float(v) for v in row[:k]]
        m = max(vals)
        exps = [math.exp(v - m) for v in vals]
        total = math.fsum(exps)
        out.append([e / total for e in exps])
    return out


def rows_to_probabilities(
    cases: Sequence[DecisionCase], score_rows: Sequence[Sequence[float]]
) -> list[dict[str, float]]:
    """Marker scores per case -> contract probabilities keyed by option id."""
    if len(cases) != len(score_rows):
        raise ValueError("cases and score rows must align")
    probs = masked_softmax_rows(score_rows, [len(c.options) for c in cases])
    return [
        {opt.id: p for opt, p in zip(case.options, pv)} for case, pv in zip(cases, probs)
    ]

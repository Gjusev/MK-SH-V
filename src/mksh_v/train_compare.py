"""E39 compared-training helpers (stdlib + PIL only; no torch import).

Pure, testable pieces shared by the E39 kernel: loading E38 rows as decision
cases with image references, the selection metric (macro mean over the 7
families), the paired-by-episode bootstrap used by gate G1, the fixed dev
half used for checkpoint selection, the ablation image transforms and the
gate arithmetic. The training loops themselves live in the kernel
(`kaggle/e39_train.py`) — protocol code, measured not unit-tested.
"""

from __future__ import annotations

import hashlib
import json
import random
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageFilter

from mokka_decisions.contracts import DecisionCase, Option

E38A_FAMILIES = ("doc_type", "field_threshold", "form_complete")
E38B_FAMILIES = ("screen_state", "task_done", "source_conflict", "urgency")
E39_FAMILIES = E38A_FAMILIES + E38B_FAMILIES


def load_e38_rows(root: str | Path, families: Iterable[str] = E39_FAMILIES) -> list[dict]:
    """Read rows.jsonl of every family under an E38 data root (local dir or a
    Hub snapshot). Returns rows verbatim; image paths stay relative."""
    root = Path(root)
    rows = []
    for family in families:
        path = root / family / "rows.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def rows_by_split(rows: Sequence[dict], split: str, drop_fr: bool = True) -> list[dict]:
    out = []
    for r in rows:
        if r.get("split") != split:
            continue
        if drop_fr and r.get("language") == "fr":
            continue
        out.append(r)
    return out


def case_of_row(row: dict) -> DecisionCase:
    return DecisionCase(
        id=row["id"], group_id=row.get("group_id", row["id"]),
        source=row.get("source", "e38"), language=row.get("language", "en"),
        domain=row.get("domain", "documents"), state=row["state"],
        question=row["question"],
        options=tuple(Option(id=o["id"], description=o["description"]) for o in row["options"]),
        target_kind="single", target_option=row["target_option"], split=row.get("split"),
    )


def qtype_of_row(row: dict) -> str:
    """Question type for the julia-arch harness: E35 convention, from the row."""
    return row.get("question_type") or "choice"


def macro_family_accuracy(records: Sequence[dict]) -> float | None:
    """Mean over families of per-family accuracy; records = dicts with
    `family` and `correct` keys (per-case rows from the evaluators)."""
    agg: dict[str, list[int]] = defaultdict(list)
    for rec in records:
        agg[rec["family"]].append(1 if rec["correct"] else 0)
    if not agg:
        return None
    return sum(sum(v) / len(v) for v in agg.values()) / len(agg)


def paired_delta_ci(
    correct_a: Sequence[int],
    correct_b: Sequence[int],
    groups: Sequence[str],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Percentile CI for acc(a) - acc(b), resampling WHOLE groups (episodes).

    Counterfactual variants of one episode are not independent observations;
    the unit of resampling is the group_id (E36/E38 lesson).
    """
    if not (len(correct_a) == len(correct_b) == len(groups)):
        raise ValueError("correct_a, correct_b and groups must align")
    by_group: dict[str, list[int]] = defaultdict(list)
    for a, b, g in zip(correct_a, correct_b, groups):
        by_group[g].append(a - b)
    group_deltas = [tuple(v) for v in by_group.values()]
    total = sum(len(v) for v in group_deltas)
    delta = sum(sum(v) for v in group_deltas) / total
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        acc = 0
        for _ in range(len(group_deltas)):
            acc += sum(group_deltas[rng.randrange(len(group_deltas))])
        stats.append(acc / total)
    stats.sort()
    lo = stats[int(alpha / 2 * n_boot)]
    hi = stats[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    return {"delta": delta, "ci95": [lo, hi], "n_groups": len(group_deltas), "n": total}


def fixed_dev_half(rows: Sequence[dict], *, seed: int = 39, fraction: float = 0.5) -> set[str]:
    """Deterministic id subset of the dev split, stratified by family.

    The SAME half must be used for every selection eval; picking it once and
    freezing the ids keeps checkpoint selection out of the full dev."""
    by_family: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_family[r["family"]].append(r["id"])
    keep: set[str] = set()
    for family in sorted(by_family):
        ids = sorted(by_family[family])
        # crc32, NOT hash(): string hashing is salted per process and the id
        # set must be identical in every run (PYTHONHASHSEED-independent)
        rng = random.Random(seed ^ (zlib.crc32(family.encode()) & 0xFFFF))
        rng.shuffle(ids)
        keep.update(ids[: int(len(ids) * fraction)])
    return keep


# -- ablation helpers ----------------------------------------------------------
def sha_sample(rows: Sequence[dict], per_family: int = 200) -> list[dict]:
    """Deterministic per-family sample: rows sorted by sha256(id), first
    `per_family` per family. Frozen selection rule for zero-shot external
    comparators (the sample ids are evidence)."""
    by_family: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_family[r["family"]].append(r)
    out = []
    for family in sorted(by_family):
        ordered = sorted(by_family[family],
                         key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
        out.extend(ordered[:per_family])
    return out


def accuracy_ci_by_group(correct: Sequence[int], groups: Sequence[str], *,
                         n_boot: int = 1000, alpha: float = 0.05,
                         seed: int = 42) -> dict:
    """Percentile CI for accuracy resampling whole groups (episodes)."""
    by_group: dict[str, list[int]] = defaultdict(list)
    for c, g in zip(correct, groups):
        by_group[g].append(c)
    group_vals = [tuple(v) for v in by_group.values()]
    point = sum(sum(v) for v in group_vals) / sum(len(v) for v in group_vals)
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        acc = cnt = 0
        for _ in range(len(group_vals)):
            pick = group_vals[rng.randrange(len(group_vals))]
            acc += sum(pick)
            cnt += len(pick)
        stats.append(acc / cnt)
    stats.sort()
    return {"accuracy": point,
            "ci95": [stats[int(alpha / 2 * n_boot)],
                     stats[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]]}


def ece_15(confidences: Sequence[float], corrects: Sequence[int],
           bins: int = 15) -> float | None:
    """Equal-width ECE over max-probability confidences (pure python)."""
    if not confidences:
        return None
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for conf, ok in zip(confidences, corrects):
        buckets[min(int(conf * bins), bins - 1)].append((conf, int(ok)))
    n = sum(len(b) for b in buckets)
    ece = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        size = len(bucket)
        ece += size / n * abs(sum(c for c, _ in bucket) / size
                              - sum(o for _, o in bucket) / size)
    return ece


def swap_map_within_blocks(block_keys: Sequence[str]) -> list[int]:
    """Index map dst -> src rolling by one inside each block.

    Chunked evaluation opens image at ``swap_map[i]`` for case ``i``: every
    case receives a foreign image from its own (family x language) block,
    never a fixed point; singletons map to themselves.
    """
    idx_by_block: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(block_keys):
        idx_by_block[k].append(i)
    mapping = list(range(len(block_keys)))
    for idxs in idx_by_block.values():
        if len(idxs) < 2:
            continue
        rolled = idxs[1:] + idxs[:1]
        for dst, src in zip(idxs, rolled):
            mapping[dst] = src
    return mapping


def swap_within_blocks(images: Sequence, block_keys: Sequence[str]) -> list:
    """Convenience image-level swap (tests/small sets): same mapping as above."""
    mapping = swap_map_within_blocks(block_keys)
    return [images[m] for m in mapping]


def blank_like(image: Image.Image) -> Image.Image:
    """Absent-image ablation: same canvas, no content."""
    return Image.new(image.mode, image.size, (255, 255, 255) if image.mode == "RGB" else 255)


def degrade(image: Image.Image) -> Image.Image:
    """Degraded-image ablation: half-res round trip + slight blur."""
    w, h = image.size
    small = image.resize((max(1, w // 2), max(1, h // 2)), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR).filter(ImageFilter.GaussianBlur(0.8))


def gate_report(
    *,
    delta_ci: dict,
    text_sets: dict,  # {set_name: {arm: acc}} for every arm
    text_bars: dict,  # {set_name: absolute_bar} vs frozen MK-SH-1 v1.1
    delta_lo_margin: float = 0.02,
) -> dict:
    """G1/G2 arithmetic with the frozen margins (G3/G4 computed in-kernel).

    G2 is anchored to the FROZEN MK-SH-1 v1.1 references (E35 s42), not to the
    retrained OCR arm: a retrained comparator can degrade too and both could
    slip through a relative gate. Every arm must clear every bar.
    """
    g1 = delta_ci["ci95"][0] >= delta_lo_margin
    g2, detail = {}, {}
    for name, bars_acc in text_sets.items():
        bar = text_bars[name]
        detail[name] = {"bar": bar, "per_arm": dict(bars_acc)}
        g2[name] = all(acc >= bar for acc in bars_acc.values())
    return {
        "G1_visual_advantage": {
            "delta": delta_ci["delta"], "ci95": delta_ci["ci95"],
            "bar": f"lower bound >= +{delta_lo_margin:.2f}", "passed": g1,
        },
        "G2_text_regression": {
            "bar": {n: b for n, b in text_bars.items()},
            "per_set": detail, "passed": all(g2.values()),
        },
    }

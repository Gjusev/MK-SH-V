"""Template-keyed splitting and leakage checks.

Splits are assigned to KEYS (template / app / episode ids) before any variant
is rendered; a case inherits its key's split and can never cross over. The
assignment is a pure hash so it is reproducible without storing an explicit
map, and the checker re-derives everything from the emitted cases.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Sequence


def split_keys_by_hash(keys: Iterable[str], dev_fraction: float) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Deterministically partition keys into (train, dev) by sha256 order.

    The hash order is machine-independent; the first ceil(fraction * N) keys
    in hash order go to dev, the rest to train.
    """
    keys = sorted(set(keys))
    if not keys:
        return (), ()
    n_dev = max(1, min(len(keys) - 1, round(len(keys) * dev_fraction)))
    ordered = sorted(keys, key=lambda k: hashlib.sha256(k.encode()).hexdigest())
    dev = tuple(ordered[:n_dev])
    train = tuple(ordered[n_dev:])
    return train, dev


def stratified_template_split(
    templates: Iterable[str], dev_layouts_per_class: int = 1
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split template ids ``"<class>-<layout>"`` with per-class coverage.

    A plain hash split can starve a class from either side (measured: the
    doc_type hash split put every receipt template in dev). Here each class
    contributes ``dev_layouts_per_class`` layouts (hash-ordered within the
    class) to dev and keeps the rest in train, so every class is present in
    both splits.
    """
    by_class: dict[str, list[str]] = {}
    for t in sorted(set(templates)):
        doc_class, _, layout = t.partition("-")
        by_class.setdefault(doc_class, []).append(t)
    train, dev = [], []
    for doc_class, class_templates in by_class.items():
        if len(class_templates) <= dev_layouts_per_class:
            raise ValueError(
                f"class {doc_class!r} has {len(class_templates)} templates; "
                f"cannot spare {dev_layouts_per_class} for dev"
            )
        ordered = sorted(class_templates, key=lambda k: hashlib.sha256(k.encode()).hexdigest())
        dev.extend(ordered[:dev_layouts_per_class])
        train.extend(ordered[dev_layouts_per_class:])
    return tuple(sorted(train)), tuple(sorted(dev))


def assert_no_leakage(cases: Sequence) -> dict:
    """Raise if any group_id appears under two different splits.

    Returns the {split: {group_id}} map on success so callers can log it.
    """
    seen: dict[str, str] = {}
    by_split: dict[str, set[str]] = {}
    for case in cases:
        split = case.split
        group = case.group_id
        by_split.setdefault(split, set()).add(group)
        if group in seen and seen[group] != split:
            raise AssertionError(
                f"leakage: group {group!r} appears in splits {seen[group]!r} and {split!r}"
            )
        seen[group] = split
    return by_split


def split_of_template(template: str, train_templates: Sequence[str], dev_templates: Sequence[str]) -> str:
    if template in train_templates:
        return "train"
    if template in dev_templates:
        return "dev"
    raise KeyError(f"template {template!r} not assigned to any split")

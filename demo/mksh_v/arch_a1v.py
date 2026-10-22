"""A1-V: one NeoMME pass per decision, options marked inline with marker tokens.

Port of the MK-SH-1 A1 idea (``mokka_decisions.arch_a1.MarkerScorer``) to the
NeoMME-260M single-tower encoder::

    [<doc>] state <eos> question [O0] option_0 [O1] option_1 ...
    [<doc> <img> patch-grid <row> markers] state <eos> question [O0] option_0 ...

Text-only rows start with the document token. Rows with an image reuse the
processor's ``<doc> <img> ...grid...`` prefix and continue with the text tail:
``NeoMMEProcessor`` validates that an image row is *exactly* the doc prefix +
grid (no mixed rows), so image batches are assembled here from processor
outputs (pixel_values, prefix ids, 2-axis positions) plus a tokenized tail
with 1-D positions continuing after the grid.

NeoMME embeddings are factorized (ALBERT-style ``word_embeddings[vocab, rank]
-> projection[hidden]``) plus a separate ``value_embeddings`` table consumed
by the first/last global-attention layers. The transformers implementation
overrides ``_resize_token_embeddings`` to resize BOTH tables, so
``ensure_markers`` relies on ``model.resize_token_embeddings`` and verifies
the vocabulary actually grew (E37 C2 checks gradients reach both tables).

Marker discipline copied from A1: ``n_markers`` is a COUNT (0..k), never a
token id.
"""

from __future__ import annotations

import math
import time
from typing import Any, Sequence

import torch
from torch import nn

from mokka_decisions.contracts import DecisionCase

from .sequence import MAX_MARKERS, build_tail, rows_to_probabilities  # noqa: F401 (re-exported)


# --------------------------------------------------------------------------
# the scorer
# --------------------------------------------------------------------------
class NeoMMEMarkerScorer(nn.Module):
    """NeoMME encoder + shared MLP head over marker-token hidden states."""

    def __init__(self, model_name: str = "Hcompany/NeoMME-260M", dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer
        self.encoder = AutoModel.from_pretrained(model_name, attn_implementation="sdpa")
        self.encoder.float()  # fp32 master weights; fp16 only via autocast (T4)
        hidden = self.encoder.config.hidden_size
        self.n_markers = 0
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    # -- marker management -------------------------------------------------
    def ensure_markers(self, k: int) -> None:
        """Add ``[O_i]`` special tokens (idempotent); resize both embedding tables."""
        k = min(k, MAX_MARKERS)
        if k <= self.n_markers:
            return
        new_tokens = [f"[O{i}]" for i in range(self.n_markers, k)]
        self.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        self.encoder.resize_token_embeddings(len(self.tokenizer))
        vocab = self.encoder.config.vocab_size
        if vocab != len(self.tokenizer):
            raise RuntimeError(
                f"embedding resize mismatch: tokenizer {len(self.tokenizer)} vs config vocab {vocab}"
            )
        # factorized tables must both have grown (word + value embeddings)
        backbone = getattr(self.encoder, self.encoder.base_model_prefix, self.encoder)
        word_rows = backbone.embed_tokens.word_embeddings.weight.shape[0]
        value_rows = backbone.value_embeddings.weight.shape[0]
        if word_rows != vocab or value_rows != vocab:
            raise RuntimeError(
                f"factorized resize incomplete: word {word_rows}, value {value_rows}, expected {vocab}"
            )
        self.n_markers = k

    def marker_ids(self) -> list[int]:
        return [self.tokenizer.convert_tokens_to_ids(f"[O{i}]") for i in range(self.n_markers)]

    # -- encoding ----------------------------------------------------------
    def encode_cases(
        self, cases: Sequence[DecisionCase], *, max_state_tokens=None, max_option_tokens=None
    ) -> list[dict]:
        """Precompute text tails once; reused across epochs by the trainers."""
        self.ensure_markers(max(len(c.options) for c in cases) if cases else 0)
        mids = self.marker_ids()
        out = []
        for case in cases:
            tail, markers = build_tail(
                self.tokenizer, case, mids,
                max_state_tokens=max_state_tokens, max_option_tokens=max_option_tokens,
            )
            target = case.option_ids.index(case.target_option) if case.target_option else -100
            out.append({"tail": tail, "markers": markers, "target": target})
        return out

    def build_batch(
        self,
        encoded: Sequence[dict],
        images: Sequence[Any | None],
        device,
        *,
        max_side: int | None = None,
    ) -> dict:
        """Assemble the padded batch: prefix (doc or doc+grid) + tail per row."""
        if len(encoded) != len(images):
            raise ValueError("encoded cases and images must align")
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = 0
        doc_id = self.tokenizer.document_token_id
        rows, positions, pixel_parts, marker_rows = [], [], [], []
        targets = []
        for r, (enc_row, image) in enumerate(zip(encoded, images)):
            tail = enc_row["tail"]
            if image is not None:
                feats = self.processor(
                    images=[image], max_side=max_side, padding=True, return_tensors="pt"
                )
                prefix = feats["input_ids"][0].tolist()
                prefix_pos = feats["position_ids"][:, 0, :].tolist()  # (2, Lp)
                row_ids = prefix + tail
                n_prefix = len(prefix)
                tail_pos = list(range(n_prefix, n_prefix + len(tail)))
                positions.append([prefix_pos[0] + tail_pos, prefix_pos[1] + tail_pos])
                pixel_parts.append(feats["pixel_values"])
            else:
                row_ids = [doc_id] + tail
                n_prefix = 1
                pos = list(range(len(row_ids)))
                positions.append([pos, pos])
            rows.append(row_ids)
            for slot, m in enumerate(enc_row["markers"]):
                marker_rows.append((r, m + n_prefix, slot))
            targets.append(enc_row["target"])

        maxlen = max(len(r_) for r_ in rows)
        input_ids = torch.full((len(rows), maxlen), pad, dtype=torch.long)
        attention = torch.zeros((len(rows), maxlen), dtype=torch.long)
        position_ids = torch.zeros((2, len(rows), maxlen), dtype=torch.long)
        for i, (row, (pos_a, pos_b)) in enumerate(zip(rows, positions)):
            n = len(row)
            input_ids[i, :n] = torch.tensor(row, dtype=torch.long)
            attention[i, :n] = 1
            position_ids[0, i, :n] = torch.tensor(pos_a, dtype=torch.long)
            position_ids[1, i, :n] = torch.tensor(pos_b, dtype=torch.long)
        max_options = max(len(e["markers"]) for e in encoded)
        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention.to(device),
            "position_ids": position_ids.to(device),
            "pixel_values": torch.cat(pixel_parts, dim=0).to(device) if pixel_parts else None,
            "marker_rows": torch.tensor(marker_rows, dtype=torch.long, device=device),
            "num_decisions": len(encoded),
            "max_options": max_options,
            "targets": torch.tensor(targets, dtype=torch.long, device=device),
        }

    def parameter_groups(self, encoder_lr: float, head_lr: float) -> list[dict]:
        enc = [p for p in self.encoder.parameters() if p.requires_grad]
        rest = [
            p for n, p in self.named_parameters()
            if not n.startswith("encoder.") and p.requires_grad
        ]
        groups = []
        if enc:
            groups.append({"params": enc, "lr": encoder_lr})
        if rest:
            groups.append({"params": rest, "lr": head_lr})
        return groups


def forward_a1v(model: NeoMMEMarkerScorer, batch: dict) -> torch.Tensor:
    """Marker logits ``(num_decisions, max_options)``; empty slots at dtype min."""
    out = model.encoder(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        pixel_values=batch["pixel_values"],
    )
    hidden = out.last_hidden_state
    rows = batch["marker_rows"]
    marker_h = hidden[rows[:, 0], rows[:, 1]]
    scores = model.head(marker_h).squeeze(-1)
    logits = torch.full(
        (batch["num_decisions"], batch["max_options"]),
        torch.finfo(scores.dtype).min,
        dtype=scores.dtype,
        device=scores.device,
    )
    logits[rows[:, 0], rows[:, 2]] = scores
    return logits


@torch.no_grad()
def evaluate_a1v(
    model: NeoMMEMarkerScorer,
    cases: Sequence[DecisionCase],
    images: Sequence[Any | None] | None,
    *,
    device,
    batch_decisions: int = 16,
    max_side: int | None = None,
    max_state_tokens=None,
    max_option_tokens=None,
    collect_rows: bool = False,
) -> dict:
    """Forced-choice metrics (same keys as mokka's ``evaluate_julia_arch``)."""
    if images is None:
        images = [None] * len(cases)
    model.eval()
    encoded_all = model.encode_cases(
        cases, max_state_tokens=max_state_tokens, max_option_tokens=max_option_tokens
    )
    correct = n = 0
    nll_sum = brier_sum = 0.0
    rows = [] if collect_rows else None
    for s in range(0, len(cases), batch_decisions):
        chunk_cases = cases[s : s + batch_decisions]
        chunk_enc = encoded_all[s : s + batch_decisions]
        chunk_imgs = images[s : s + batch_decisions]
        batch = model.build_batch(chunk_enc, chunk_imgs, device, max_side=max_side)
        logits = forward_a1v(model, batch).float()
        probs = logits.softmax(-1)
        for r, case in enumerate(chunk_cases):
            gold = int(batch["targets"][r].item())
            if gold < 0:
                continue
            k = len(case.options)
            pv = probs[r, :k].tolist()
            pred = max(range(k), key=pv.__getitem__)
            ok = int(pred == gold)
            correct += ok
            n += 1
            nll_sum += -math.log(max(pv[gold], 1e-12))
            brier_sum += sum((float(i == gold) - p) ** 2 for i, p in enumerate(pv))
            if rows is not None:
                rows.append(
                    {"id": case.id, "gold_idx": gold, "pred_idx": pred,
                     "probabilities": pv, "correct": ok}
                )
    model.train()
    out = {
        "n": n,
        "accuracy": correct / n if n else None,
        "mean_nll": nll_sum / n if n else None,
        "mean_brier": brier_sum / n if n else None,
    }
    if rows is not None:
        out["rows"] = rows
    return out


def decide_rows(
    model: NeoMMEMarkerScorer,
    cases: Sequence[DecisionCase],
    images: Sequence[Any | None] | None = None,
    *,
    device,
    max_side: int | None = None,
    latency: bool = False,
) -> list[dict]:
    """Contract probabilities per case (dict option_id -> p); optional latency ms."""
    from mokka_decisions.backends.base import normalise_probabilities

    if images is None:
        images = [None] * len(cases)
    model.eval()
    encoded_all = model.encode_cases(cases)
    out = []
    for s in range(0, len(cases), 16):
        chunk = cases[s : s + 16]
        chunk_enc = encoded_all[s : s + 16]
        chunk_imgs = images[s : s + 16]
        if latency:
            start = time.perf_counter()
        batch = model.build_batch(chunk_enc, chunk_imgs, device, max_side=max_side)
        logits = forward_a1v(model, batch).float()
        if latency:
            torch.cuda.synchronize(device) if str(device).startswith("cuda") else None
        # rows_to_probabilities applies the masked softmax itself; raw logits in
        probs = rows_to_probabilities(chunk, logits.tolist())
        for case, p in zip(chunk, probs):
            row = {"id": case.id, "probabilities": normalise_probabilities(p)}
            if latency:
                row["latency_ms"] = (time.perf_counter() - start) * 1000.0 / len(chunk)
            out.append(row)
    return out

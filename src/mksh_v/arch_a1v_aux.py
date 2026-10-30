"""E54 train-time-only auxiliary digit heads on top of the A1-V architecture
(criteria docs/E54_CRITERIA_2026-10-30.md).

Subclass — the frozen A1 decision is untouched (src/mksh_v/arch_a1v.py is not
edited). Adds nine tokens ([DIGIT_1..8] [LEN]) appended after every case tail
and two small heads: a 10-way digit classifier per position and a 9-way length
classifier. Labels ride on the encoded rows as ``aux_digits`` (8 ints, -100
where masked) and ``aux_len`` (-100 masked). ``forward_a1v_aux`` returns the
choice logits plus aux logits; the base ``forward_a1v`` path is unchanged, and
``strip_aux`` exports a pure NeoMMEMarkerScorer state dict (embeddings sliced
back to the base vocabulary).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from mksh_v.arch_a1v import NeoMMEMarkerScorer

AUX_TOKENS = ["[DIGIT%d]" % k for k in range(1, 9)] + ["[LEN]"]
N_POS = 8
DIGIT_CLASSES = 10
LEN_CLASSES = 9  # integer-digit counts 1..9


class NeoMMEAuxScorer(NeoMMEMarkerScorer):
    """A1-V plus train-time-only per-position digit and length heads."""

    aux_vocab = 0  # set by ensure_aux

    def ensure_aux(self) -> None:
        if self.aux_vocab:
            return
        vocab = self.tokenizer.convert_tokens_to_ids("[DIGIT1]")
        if vocab is None or vocab == self.tokenizer.unk_token_id:
            self.encoder.resize_token_embeddings(len(self.tokenizer) + len(AUX_TOKENS))
            new = self.tokenizer.get_added_vocab()
            for i, tok in enumerate(AUX_TOKENS):
                self.tokenizer.add_tokens([tok])
            self.encoder.resize_token_embeddings(len(self.tokenizer))
        hidden = self.encoder.config.hidden_size
        self.aux_digit_head = nn.Linear(hidden, DIGIT_CLASSES)
        self.aux_len_head = nn.Linear(hidden, LEN_CLASSES)
        self.aux_vocab = len(self.tokenizer)

    def aux_token_ids(self) -> list[int]:
        return [self.tokenizer.convert_tokens_to_ids(t) for t in AUX_TOKENS]

    def build_batch(self, encoded, images, device, **kw):
        batch = super().build_batch(encoded, images, device, **kw)
        self.ensure_aux()
        ids = self.aux_token_ids()
        pad = batch["input_ids"].shape[1]
        B = batch["input_ids"].shape[0]
        attn = batch["attention_mask"]
        # append the aux tail after each row's real tokens (right-padding aware)
        dev = batch["input_ids"].device
        grow = torch.full((B, len(ids)), pad, dtype=torch.long, device=dev)
        grow_attn = torch.zeros((B, len(ids)), dtype=torch.long, device=dev)
        ext_ids = torch.cat([batch["input_ids"], grow], dim=1)
        ext_attn = torch.cat([attn, grow_attn], dim=1)
        pos_a, pos_b = batch["position_ids"][0], batch["position_ids"][1]
        next_a = pos_a.max(dim=1, keepdim=True).values + 1
        next_b = pos_b.max(dim=1, keepdim=True).values + 1
        inc_a = torch.arange(1, len(ids) + 1, dtype=torch.long, device=dev).unsqueeze(0)
        ext_a = torch.cat([pos_a, next_a + inc_a - 1], dim=1)
        ext_b = torch.cat([pos_b, next_b + inc_a - 1], dim=1)
        # aux token placement: at the first pad slot of each row
        aux_rows = torch.full((B, len(ids)), -1, dtype=torch.long)
        for r in range(B):
            n = int(attn[r].sum())
            for j, tid in enumerate(ids):
                ext_ids[r, n + j] = tid
                ext_attn[r, n + j] = 1
                aux_rows[r, j] = n + j
        batch["input_ids"] = ext_ids
        batch["attention_mask"] = ext_attn
        batch["position_ids"] = torch.stack([ext_a, ext_b])
        batch["aux_rows"] = aux_rows.to(device)
        batch["aux_digits"] = torch.full((B, N_POS), -100, dtype=torch.long, device=device)
        batch["aux_len"] = torch.full((B,), -100, dtype=torch.long, device=device)
        for r, enc in enumerate(encoded):
            d = enc.get("aux_digits")
            if d:
                batch["aux_digits"][r, : len(d)] = torch.tensor(d, dtype=torch.long)
            L = enc.get("aux_len")
            if L is not None:
                batch["aux_len"][r] = L
        return batch


def forward_a1v_aux(model: NeoMMEAuxScorer, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(choice_logits, digit_logits (B,8,10), len_logits (B,9))."""
    from mksh_v.arch_a1v import forward_a1v

    logits = forward_a1v(model, batch)
    out = model.encoder(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        pixel_values=batch["pixel_values"],
    )
    hidden = out.last_hidden_state
    rows = batch["aux_rows"]
    h = hidden[torch.arange(rows.shape[0]).unsqueeze(1), rows]  # (B, 9, H)
    d_logits = model.aux_digit_head(h[:, :N_POS])  # (B, 8, 10)
    l_logits = model.aux_len_head(h[:, N_POS])  # (B, 9)
    return logits, d_logits, l_logits


def aux_loss(batch: dict, d_logits: torch.Tensor, l_logits: torch.Tensor) -> torch.Tensor:
    ce = torch.nn.functional.cross_entropy
    dl = ce(d_logits.reshape(-1, DIGIT_CLASSES), batch["aux_digits"].reshape(-1), ignore_index=-100)
    ll = ce(l_logits, batch["aux_len"].clamp(min=0), ignore_index=-100)
    parts = [x for x in (dl, ll) if torch.isfinite(x)]
    return torch.stack(parts).mean() if parts else d_logits.sum() * 0.0


def strip_aux(state: dict) -> dict:
    """Export a pure NeoMMEMarkerScorer state dict from an aux checkpoint."""
    from transformers import AutoTokenizer

    base_vocab = len(AutoTokenizer.from_pretrained("Hcompany/NeoMME-260M"))
    out = {}
    for k, v in state.items():
        if k.startswith("aux_"):
            continue
        if "embed_tokens.weight" in k or k == "encoder.embeddings.word_embeddings.weight":
            v = v[:base_vocab + state.get("_markers_added", 32)] if False else v[: v.shape[0] - len(AUX_TOKENS)]
        out[k] = v
    return out

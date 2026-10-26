"""Decision-Index vision engine for MK-SH-V (Jev Decision Index, vision 0.3 suite).

Runs the released checkpoint (E47 seed 43, E48 policy) over the board's public
vision rows through our own inference path: one encoder pass per row, options
marked inline, calibrated softmax (the shipped T). The abstention gate is NOT
applied here: the board scores the engine's per-option probability
distribution and counts unanswered rows as wrong; our gate is a policy layer
on top of these probabilities, documented in the model card.

    python -m decision_index vision run --engine mksh_v.board_engine:MkshvVisionEngine \
        --dir work/vision --out runs/board-s43 --workers 1

Capacity limits (declared, never worked around): our tower reads ONE image per
case, so multi-image rows are `Unsupported`; question types other than
`choice` are `Unsupported`. No truncation, no option dropping, no per-benchmark
prompt adaptation.
"""

from __future__ import annotations

from pathlib import Path

try:  # under the decision-index kit
    from decision_index.engines import Engine, Unsupported
except ImportError:  # standalone (unit tests without the kit installed)

    class Engine:  # noqa: D401 - duck-typed stand-in
        name = "engine"
        provenance: dict = {}
        latency = ""

        def __init__(self, **options):
            self.options = options

    class Unsupported(ValueError):
        pass


# The board's vision rows carry state = {}; the question carries the content.
STATE = "An image and a question."
# The shipped policy's input rule (E46): upscale so the longest side is 1,024 px
# when smaller, then process with max_side 1,024.
SIDE = 1024


def case_of_payload(payload: dict) -> tuple:
    """(DecisionCase, image path or None) for one board payload, or Unsupported."""
    from mokka_decisions.contracts import DecisionCase, Option

    images = payload.get("images") or []
    if len(images) > 1:
        raise Unsupported(f"single-image architecture; got {len(images)} images")
    image = images[0] if images else None
    if image is not None and not isinstance(image, (str, Path)):
        raise Unsupported("image inputs other than paths are not supported")
    questions = payload.get("questions") or {}
    if set(questions) != {"answer"}:
        raise Unsupported(f"expected exactly one 'answer' question, got {sorted(questions)}")
    q = questions["answer"]
    if q.get("type") != "choice":
        raise Unsupported("question type " + str(q.get("type")))
    state = payload.get("state")
    state = state.strip() if isinstance(state, str) and state.strip() else STATE
    criteria = q.get("criteria") or {}
    case = DecisionCase(
        id="board", group_id="board", source="board", language="en", domain="documents",
        state=state, question=q.get("instructions") or "",
        options=tuple(Option(id=str(k), description=str(v)) for k, v in criteria.items()))
    return case, image


class MkshvVisionEngine(Engine):
    name = "mksh-v"
    latency = "one encoder pass per row (batch 16, U1024 input rule, calibrated T); excludes model loading"

    def __init__(self, image_root=None, model_dir=None, device=None, batch=16, threads=None, **options):
        super().__init__(**options)
        import json

        import torch
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file

        from mksh_v.arch_a1v import NeoMMEMarkerScorer, forward_a1v

        self._forward = forward_a1v
        self.root = Path(image_root).resolve() if image_root else None
        pkg = Path(model_dir) if model_dir else Path(
            __import__("os").environ.get("MKSHV_MODEL_DIR") or snapshot_download("Gjusev/MK-SH-V"))
        policy = json.loads((pkg / "inference-policy.json").read_text(encoding="utf-8"))
        self.T = float(policy["temperature"]["value"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            torch.set_num_threads(threads or 2)
        self.batch_size = int(batch)
        self.model = NeoMMEMarkerScorer()
        self.model.ensure_markers(32)
        self.model.load_state_dict(load_file(str(pkg / "model.safetensors")), strict=True)
        self.model.to(self.device).eval()
        self.provenance = {
            "model": "Gjusev/MK-SH-V", "checkpoint": policy["checkpoint"],
            "checkpoint_sha256": policy["checkpoint_sha256"], "code": "github.com/Gjusev/mk-sh-v",
            "policy": "inference-policy.json (E48)", "temperature": self.T,
            "input_rule": f"upscale_to {SIDE} (E46)", "gate": "not applied (board scores probabilities)",
        }

    def _open(self, image) -> object:
        from PIL import Image

        p = Path(image)
        if self.root is not None and not p.is_absolute():
            p = self.root / p
        return Image.open(p).convert("RGB")

    def _respond(self, cases, images) -> list:
        import torch

        enc = self.model.encode_cases(cases, max_state_tokens=700, max_option_tokens=96)
        batch = self.model.build_batch(enc, images, self.device, max_side=SIDE, upscale_to=SIDE)
        with torch.inference_mode():
            probs = (self._forward(self.model, batch) / self.T).softmax(-1).float().cpu()
        out = []
        for r, case in enumerate(cases):
            p = probs[r, : len(case.options)].tolist()
            keys = [o.id for o in case.options]
            top = max(range(len(p)), key=p.__getitem__)
            out.append((
                {"model": self.name, "answers": {"answer": {
                    "type": "choice", "choice": keys[top],
                    "probabilities": {k: round(v, 6) for k, v in zip(keys, p)}}}},
                {"T": self.T, "policy": "E48"}))
        return out

    def batch(self, payloads: list) -> list:
        results: list = [None] * len(payloads)
        cases, images, idx = [], [], []
        for i, p in enumerate(payloads):
            try:
                case, image = case_of_payload(p)
            except Exception as e:  # per-element: one bad row never stops the chunk
                results[i] = e
                continue
            try:
                img = self._open(image) if image is not None else None
            except Exception as e:
                results[i] = e
                continue
            cases.append(case)
            images.append(img)
            idx.append(i)
        for s in range(0, len(cases), self.batch_size):
            for j, resp in enumerate(self._respond(cases[s:s + self.batch_size], images[s:s + self.batch_size])):
                results[idx[s + j]] = resp
        return results

    def __call__(self, state, questions, images=(), prompt=None):
        out = self.batch([{"state": state, "questions": questions, "images": images, "prompt": prompt}])[0]
        if isinstance(out, Exception):
            raise out
        return out

    def close(self):
        del self.model

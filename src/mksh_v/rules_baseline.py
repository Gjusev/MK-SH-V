"""OCR + deterministic rules comparator (E39 fourth arm; frozen with the
criteria, tuned ONLY on the E38a dev split — never on test).

Dev accuracies of these frozen constants (900 dev images, tesseract 5.5.0,
eng/deu/spa, 2026-10-14): field_threshold 0.960, doc_type 0.767,
form_complete 0.440 (majority 0.333). form_complete is weak on purpose: the
label lives in the LAYOUT (which required field is empty), which OCR text
scrambles — that is the family's point.

Claim rule (frozen in docs/E39_CRITERIA_DRAFT.md): in each family where the
CI of (rules - A1-V) has lower bound >= 0, no visual-advantage claim is made
for that family.
"""

from __future__ import annotations

import re

RULE_FAMILIES = ("doc_type", "field_threshold", "form_complete")

RULES_DEV_ACCURACY = {"field_threshold": 0.960, "doc_type": 0.767,
                      "form_complete": 0.440}

_AMOUNT_RE = re.compile(r"\d{1,3}(?:[.,]\d{3})*[.,]\d{2}|\d+[.,]\d{2}")
_THRESHOLD_RE = re.compile(r"(\d{1,3}(?:[.,]\d{3})*[.,]\d{2}|\d+[.,]\d{2})")
_TOTAL_HINT_RE = re.compile(r"total|summe|gesamt|importe|suma|betrag", re.IGNORECASE)

_DOC_TYPE_RULES = (
    ("contract", ("this agreement", "agreement is entered", "vertrag",
                  "contrato no", "undersigned")),
    ("invoice", ("invoice", "billed to", "rechnung")),
    ("receipt", ("subtotal",)),
    ("form", ("form",)),
    ("letter", ("dear ", "estimado", "sehr geehrt", "saludos", "cordial")),
)
_FORM_GARBLED_MARK = "�"  # OCR replacement char
_FORM_CONF_BAR = 85.0          # mean word confidence below this -> illegible
_FORM_WORDS_BAR = 28           # dev-tuned (wcut)
_FORM_EMPTYISH_BAR = 1         # dev-tuned (ecut)


def _parse_amount(raw: str) -> float | None:
    """Currency amounts across en/de/es OCR: strip separators, last two
    digits are decimals ('1.027,73', '1,027.73', '2.656.28' -> x.xx)."""
    digits = re.sub(r"[.,\s]", "", raw)
    if not digits.isdigit() or len(digits) < 3:
        return None
    return float(digits[:-2] + "." + digits[-2:])


def find_amounts(text: str) -> list[float]:
    out = []
    for raw in _AMOUNT_RE.findall(text):
        value = _parse_amount(raw)
        if value is not None:
            out.append(value)
    return out


def parse_question_threshold(question: str) -> float | None:
    m = _THRESHOLD_RE.search(question)
    return _parse_amount(m.group(1)) if m else None


def extract_total(text: str) -> float | None:
    """Amount on a total-hint line if any, else the largest amount."""
    hint_lines = " ".join(l for l in text.splitlines() if _TOTAL_HINT_RE.search(l))
    hint_amounts = find_amounts(hint_lines)
    if hint_amounts:
        return hint_amounts[-1]
    amounts = find_amounts(text)
    return max(amounts) if amounts else None


def rule_field_threshold(question: str, text: str) -> str:
    """Dev-frozen fallback: no amount found -> 'false'."""
    threshold = parse_question_threshold(question)
    total = extract_total(text)
    if threshold is None or total is None:
        return "false"
    return "true" if total >= threshold - 1e-9 else "false"


def rule_doc_type(text: str) -> str:
    low = text.lower()
    for doc_class, words in _DOC_TYPE_RULES:
        if any(w in low for w in words):
            return doc_class
    return "other"


def rule_form_complete(text: str, conf: float) -> str:
    garbled = _FORM_GARBLED_MARK in text or (conf is not None and conf < _FORM_CONF_BAR)
    if garbled:
        return "illegible"
    lines = [l for l in text.splitlines() if l.strip()]
    stars = [l for l in lines if "*" in l]
    emptyish = sum(1 for l in stars if len(l.split()) <= 2)
    if emptyish >= _FORM_EMPTYISH_BAR and len(text.split()) < _FORM_WORDS_BAR:
        return "missing_required"
    return "complete"


def predict_rule(family: str, question: str, ocr_text: str, ocr_conf: float,
                 language: str = "en") -> str | None:
    """Dispatch one frozen rule; None when the family has no rule."""
    if family == "field_threshold":
        return rule_field_threshold(question, ocr_text)
    if family == "doc_type":
        return rule_doc_type(ocr_text)
    if family == "form_complete":
        return rule_form_complete(ocr_text, ocr_conf)
    return None

"""Minimal synthetic doc_type generator (E37; the E38 generator is bigger).

Renders six document classes with PIL. The label is *computed from the
content model* (``derive_doc_type``) and the renderer only draws what the
content says, so the class is recoverable from structure — never read back
from pixels. ``state`` is a constant scan note and carries no label
information: all evidence is visual (this is what the E37 C4 image-swap
check exploits).

Templates are ``"<class>-<layout>"`` ids; splitting happens on template ids
before any variant is rendered, and style perturbations (font, ink colour,
jitter) never change the template, so variants cannot cross splits.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from mokka_decisions.contracts import DecisionCase, Option

DOC_TYPES = ("invoice", "receipt", "contract", "form", "letter", "other")
LAYOUTS = ("classic", "compact", "boxed")
ALL_TEMPLATES = tuple(f"{t}-{l}" for t in DOC_TYPES for l in LAYOUTS)

SCAN_STATE = "Scanned document page."
DOC_QUESTION = "Which document type is this?"

DOC_TYPE_OPTIONS = (
    Option(id="invoice", description="Invoice: an itemized bill with line items, reference number and total."),
    Option(id="receipt", description="Receipt: a proof of purchase with items and total, without invoice reference."),
    Option(id="contract", description="Contract: titled clauses with signature lines for parties."),
    Option(id="form", description="Form: labelled input fields, mostly unfilled."),
    Option(id="letter", description="Letter: sender and recipient blocks, salutation and paragraphs."),
    Option(id="other", description="None of the other document types applies to this page."),
)

_PAGE_SIZES = {
    "invoice": (794, 1000),
    "receipt": (400, 620),
    "contract": (794, 1100),
    "form": (794, 1000),
    "letter": (794, 1050),
    "other": (700, 900),
}

_REG_FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
)
_BOLD_FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)


def _load_font(candidates: Sequence[str], size: int):
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    try:  # Pillow >= 10.1 supports sized default fonts
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


@dataclass
class DocContent:
    """Content model; ``derive_doc_type`` reads only this, never pixels."""

    doc_type: str
    title: str = ""
    sender: str = ""
    recipient: str = ""
    date: str = ""
    reference: str = ""  # invoice number (invoices only, by construction)
    lines: tuple[tuple[str, str, str], ...] = ()  # (description, qty, amount)
    total: str = ""
    payment: str = ""
    paragraphs: tuple[str, ...] = ()
    fields: tuple[tuple[str, str], ...] = ()  # (label, value) — value often empty
    signature_lines: int = 0


def derive_doc_type(c: DocContent) -> str:
    """Content-based label (pure function of the content model)."""
    if c.lines and len(c.lines) >= 2 and c.total and c.reference and c.payment and not c.paragraphs:
        return "invoice"
    if c.lines and c.total and c.payment and not c.reference and not c.paragraphs:
        return "receipt"
    if c.title and len(c.paragraphs) >= 2 and c.signature_lines >= 2:
        return "contract"
    if len(c.fields) >= 3 and not c.paragraphs and not c.lines:
        return "form"
    if c.sender and c.recipient and c.paragraphs and not c.lines and not c.fields and c.signature_lines == 0:
        return "letter"
    return "other"


# -- seeded content ----------------------------------------------------------
_COMPANIES = ("Northwind Traders", "Acme Supplies Ltd", "Baltic Parts OY", "Sierra Logistics",
              "Meridian Foods", "Orion Print GmbH", "Cobalt Systems", "Vela Marine")
_PEOPLE = ("J. Mueller", "A. Petrova", "L. Fontaine", "S. Okonkwo", "M. Alvarez", "K. Bergstrom")
_ITEMS = ("Steel bracket", "Filter cartridge", "Notebook A5", "USB cable 2m", "Ball bearing 6mm",
          "Cleaning concentrate", "Safety goggles", "Thermal paste", "Paper ream A4", "LED bulb E27")
_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December")
_FORM_LABELS = ("Full name", "Date of birth", "Address", "City", "Postal code", "Phone",
                "Email", "Department", "Start date", "Supervisor")
_FORM_VALUES = ("", "", "Mira Lane 12", "Springfield", "41092", "", "", "Operations", "2026-11-02", "")


def _rand_amount(rng, lo=4, hi=480):
    return f"{rng.randint(lo * 100, hi * 100) / 100:.2f}"


def make_content(doc_type: str, rng: random.Random) -> DocContent:
    company = rng.choice(_COMPANIES)
    date = f"{rng.randint(1, 28)} {rng.choice(_MONTHS)} 2026"
    if doc_type == "invoice":
        n = rng.randint(3, 6)
        lines = tuple(
            (f"{rng.choice(_ITEMS)} {chr(65 + i)}", str(rng.randint(1, 12)), _rand_amount(rng))
            for i in range(n)
        )
        c = DocContent(
            doc_type="invoice", title=f"INVOICE", sender=company, recipient=rng.choice(_COMPANIES),
            date=date, reference=f"INV-2026-{rng.randint(1000, 9999)}", lines=lines,
            total=_rand_amount(rng, 80, 2400), payment=rng.choice(("Bank transfer", "Card", "NET 30")),
        )
    elif doc_type == "receipt":
        n = rng.randint(2, 4)
        lines = tuple(
            (f"{rng.choice(_ITEMS)}", str(rng.randint(1, 4)), _rand_amount(rng, 2, 90))
            for _ in range(n)
        )
        c = DocContent(
            doc_type="receipt", title=company.upper(), date=date, lines=lines,
            total=_rand_amount(rng, 6, 320), payment=rng.choice(("Cash", "Card", "Contactless")),
        )
    elif doc_type == "contract":
        c = DocContent(
            doc_type="contract",
            title=f"{rng.choice(('Service Agreement', 'Supply Contract', 'Maintenance Contract', 'Non-Disclosure Agreement'))} No. {rng.randint(10, 99)}/{rng.randint(2024, 2026)}",
            date=date,
            paragraphs=(
                f"This agreement is entered into on {date} between {company} ('the Provider') and {rng.choice(_COMPANIES)} ('the Client').",
                f"1. Scope. The Provider shall deliver the services described in Annex A within {rng.randint(10, 60)} working days of the effective date.",
                f"2. Consideration. The Client shall pay a fixed fee of EUR {_rand_amount(rng, 500, 9000)} within {rng.choice((14, 30))} days of each invoice.",
                f"3. Term. This agreement remains in force for {rng.randint(12, 36)} months and renews automatically unless terminated with {rng.choice((30, 60))} days notice.",
                f"4. Governing law: the laws of the Provider's country of registration. Signed in two originals.",
            ),
            signature_lines=2,
        )
    elif doc_type == "form":
        n = rng.randint(5, 8)
        idx = rng.sample(range(len(_FORM_LABELS)), n)
        c = DocContent(
            doc_type="form", title=rng.choice(("PERSONNEL FORM", "REGISTRATION FORM", "REQUEST FORM")),
            fields=tuple(( _FORM_LABELS[i], _FORM_VALUES[i]) for i in idx),
        )
    elif doc_type == "letter":
        c = DocContent(
            doc_type="letter", sender=f"{rng.choice(_PEOPLE)}, {company}",
            recipient=f"{rng.choice(_PEOPLE)}\n{rng.choice(_COMPANIES)}", date=date,
            paragraphs=(
                f"Dear colleague,",
                f"I am writing to confirm the arrangements discussed on {rng.randint(1, 28)} {rng.choice(_MONTHS)}. The revised schedule is attached and takes effect from the next cycle.",
                f"Should any detail require adjustment, please contact the coordinator before the end of the month so the changes can enter the same revision.",
                f"Thank you for the continued collaboration.",
                "Kind regards,",
            ),
        )
    else:  # other: a poster-like page that matches no class rule
        areas = ("Area A", "Area B", "Loading dock", "Mezzanine", "Archive room", "Workshop", "Courtyard")
        actions = ("closed for maintenance", "restricted for inventory", "unavailable during re-cabling",
                   "reserved for an inspection", "locked after refurbishment")
        deadlines = ("by Friday noon", "before 08:00 on Monday", "by the end of the shift",
                     "before the next working day", "by 17:00 on Wednesday")
        where = ("lockers", "shelves", "cabinets", "workstations", "carts", "racks")
        c = DocContent(
            doc_type="other", title=rng.choice(("STORAGE NOTICE", "INTERNAL MEMO", "NOTICE")),
            paragraphs=(
                f"{rng.choice(areas)} will be {rng.choice(actions)} from {rng.randint(2, 9)}:{rng.choice(('00', '15', '30', '45'))} "
                f"on {rng.choice(_MONTHS).lower()} {rng.randint(1, 28)}.",
                f"Please remove personal items from the {rng.choice(where)} {rng.choice(deadlines)}.",
            ),
        )
    derived = derive_doc_type(c)
    if derived != doc_type:  # generator guard: label must follow from content
        raise AssertionError(f"content derives {derived!r}, intended {doc_type!r}")
    return c


# -- rendering ---------------------------------------------------------------
def render_document(content: DocContent, template: str, style_seed: int) -> Image.Image:
    """Deterministically render one page. Same (content, template, style_seed)
    and same font set -> identical pixels (fonts are machine-local; manifests
    record hashes at generation time)."""
    if not template.startswith(content.doc_type + "-"):
        raise ValueError(f"template {template!r} does not match type {content.doc_type!r}")
    layout = template.split("-", 1)[1]
    rng = random.Random(style_seed)
    width, height = _PAGE_SIZES[content.doc_type]
    img = Image.new("RGB", (width, height), color=(252, 252, 250))
    draw = ImageDraw.Draw(img)
    ink = rng.choice(((24, 24, 28), (18, 22, 34), (30, 26, 24)))
    accent = rng.choice(((60, 90, 160), (120, 60, 60), (40, 100, 80), (90, 90, 90)))
    scale = rng.uniform(0.92, 1.08)
    base = int(15 * scale)
    font = _load_font(_REG_FONTS, base)
    bold = _load_font(_BOLD_FONTS, base + 2)
    title_font = _load_font(_BOLD_FONTS, int(24 * scale))
    margin = {"classic": 54, "compact": 30, "boxed": 44}[layout]
    x0 = margin + rng.randint(-3, 3)

    def text_out(x, y, s, f=font, colour=ink, maxw=None):
        for line in s.split("\n"):
            if maxw is not None:
                while draw.textlength(line, font=f) > maxw and len(line) > 1:
                    line = line[:-1]
            draw.text((x, y), line, font=f, fill=colour)
            y += int(f.size * 1.45)
        return y

    y = margin + rng.randint(0, 6)

    if content.doc_type == "invoice":
        y = text_out(x0, y, content.title, title_font, accent)
        y = text_out(x0, y, f"From: {content.sender}", bold)
        y = text_out(x0, y, f"Billed to: {content.recipient}", font)
        y = text_out(x0, y, f"{content.date}    {content.reference}", font) + 8
        cols = (x0 + 300, x0 + 430, width - margin)
        if layout == "boxed":
            draw.rectangle([x0, y, width - margin, y + 26 * (len(content.lines) + 2)], outline=ink, width=2)
        hdr_y = y
        draw.text((x0 + 4, hdr_y), "Description", font=bold, fill=ink)
        draw.text((cols[0], hdr_y), "Qty", font=bold, fill=ink)
        draw.text((cols[1], hdr_y), "Amount", font=bold, fill=ink)
        y = hdr_y + 24
        draw.line([x0, y, width - margin, y], fill=ink, width=2)
        y += 8
        for desc, qty, amount in content.lines:
            draw.text((x0 + 4, y), desc, font=font, fill=ink)
            draw.text((cols[0], y), qty, font=font, fill=ink)
            draw.text((cols[1], y), amount, font=font, fill=ink)
            y += int(font.size * 1.5)
        y += 10
        draw.line([cols[1] - 20, y, width - margin, y], fill=ink, width=1)
        y = text_out(cols[1], y + 6, f"TOTAL  {content.total}", bold, accent)
        y = text_out(x0, y + 8, f"Payment: {content.payment}", font)
    elif content.doc_type == "receipt":
        y = text_out(x0, y, content.title, title_font, accent)
        y = text_out(x0, y, content.date, font) + 8
        for desc, qty, amount in content.lines:
            row = f"{desc} x{qty}"
            while draw.textlength(row, font=font) + draw.textlength(amount, font=font) > width - 2 * margin and len(row) > 1:
                row = row[:-1]
            draw.text((x0, y), row, font=font, fill=ink)
            draw.text((width - margin - draw.textlength(amount, font=font), y), amount, font=font, fill=ink)
            y += int(font.size * 1.6)
        y += 8
        if layout == "boxed":
            draw.rectangle([x0, y - 4, width - margin, y + 34], outline=ink, width=2)
        y = text_out(x0 + (12 if layout == "boxed" else 0), y + 4, f"TOTAL    {content.total}", bold)
        y = text_out(x0, y + 6, f"Paid by {content.payment}", font)
        dashes = "-" * 18
        y = text_out(x0, y + 10, dashes, font)
        y = text_out(x0, y + 2, f"Thank you", font)
    elif content.doc_type == "contract":
        y = text_out(x0, y, content.title, title_font, accent)
        y = text_out(x0, y + 4, f"Date: {content.date}", font) + 10
        if layout == "boxed":
            draw.rectangle([x0 - 8, y, width - margin + 8, height - margin], outline=(150, 150, 150), width=1)
        for para in content.paragraphs:
            y = text_out(x0, y, para, font, maxw=width - margin - x0) + 8
        y = max(y, height - margin - 90)
        for i in range(content.signature_lines):
            sx = x0 + i * ((width - 2 * margin) // 2)
            draw.line([sx, y, sx + 200, y], fill=ink, width=1)
            draw.text((sx, y + 6), f"Party {i + 1}", font=font, fill=ink)
    elif content.doc_type == "form":
        y = text_out(x0, y, content.title, title_font, accent) + 14
        for label, value in content.fields:
            draw.text((x0, y + 4), label, font=bold, fill=ink)
            if layout == "boxed":
                draw.rectangle([x0 + 170, y, width - margin, y + 28], outline=ink, width=1)
                if value:
                    draw.text((x0 + 178, y + 5), value, font=font, fill=ink)
            else:
                draw.line([x0 + 170, y + 26, width - margin, y + 26], fill=ink, width=1)
                if value:
                    draw.text((x0 + 174, y + 8), value, font=font, fill=ink)
            y += 40
    elif content.doc_type == "letter":
        y = text_out(x0, y, content.sender, font)
        y = text_out(x0 + 140 if layout != "compact" else x0, y, content.recipient, font)
        y = text_out(width - margin - 90 if layout == "classic" else x0, y, content.date, font) + 8
        for para in content.paragraphs:
            y = text_out(x0, y, para, font, maxw=width - margin - x0) + 6
    else:  # other
        y = text_out(x0, y + 20, content.title, title_font, accent) + 10
        for para in content.paragraphs:
            y = text_out(x0, y, para, font, maxw=width - margin - x0) + 10
        for i in range(3):  # decorative blocks: no class rule matches these
            bw = rng.randint(90, 220)
            draw.rectangle([x0, y, x0 + bw, y + 18], outline=accent, width=2)
            y += 34
    return img


# -- case generation ---------------------------------------------------------
def generate_doc_type_cases(
    templates: Sequence[str],
    n: int,
    seed: int,
    split: str,
    variants_per_case: int = 1,
) -> list[dict]:
    """Render ``n`` cases spread over ``templates`` (all of one split!).

    Splitting happens BEFORE variants: every case (and every style variant of
    it) inherits the template's split. ``group_id`` is the template so
    leakage checks can group by it.
    """
    if not templates:
        raise ValueError("templates must not be empty")
    out = []
    for i in range(n):
        template = templates[i % len(templates)]
        doc_type = template.split("-", 1)[0]
        case_seed = seed * 1_000_003 + i
        rng = random.Random(case_seed)
        content = make_content(doc_type, rng)
        for v in range(variants_per_case):
            style_seed = case_seed * 31 + v
            image = render_document(content, template, style_seed)
            case = DecisionCase(
                id=f"doctype-{template}-{i:05d}-v{v}",
                group_id=template,
                source="synthetic-doc-type",
                language="en",
                domain="documents",
                state=SCAN_STATE,
                question=DOC_QUESTION,
                options=DOC_TYPE_OPTIONS,
                target_kind="single",
                target_option=doc_type,
                split=split,
            )
            out.append({"case": case, "image": image, "template": template,
                        "content": content, "style_seed": style_seed})
    return out


def majority_rate(cases: Sequence[DecisionCase]) -> float:
    counts: dict[str, int] = {}
    for c in cases:
        counts[c.target_option] = counts.get(c.target_option, 0) + 1
    return max(counts.values()) / len(cases) if cases else 0.0

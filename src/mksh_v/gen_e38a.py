"""E38a generator: document families as HTML (rendered by Playwright elsewhere).

Pure module (no browser): content models, label rules, case planning and the
HTML body for each case. Criteria: ``docs/E38A_CRITERIA_2026-10-14.md``.

- Labels are computed from the content model, never from pixels.
- Templates (layout skeletons ``h<header>b<body>f<footer>``) are split keys,
  assigned before any variant; themes (font, colours, spacing, offsets) are
  per-case variants.
- Every element the verifier needs carries a ``data-k`` attribute; the
  verifier re-derives labels from the HTML with its own parser.
"""

from __future__ import annotations

import hashlib
import html
import math
import random
from dataclasses import dataclass, replace

from mokka_decisions.contracts import DecisionCase, Option

from .synth_doc_type import DOC_TYPES, DocContent, derive_doc_type

FAMILIES = ("doc_type", "field_threshold", "form_complete")
LANGS = ("en", "es", "de", "fr")
TRAIN_LANGS = ("en", "es", "de")
FT_CLASSES = ("false", "true")
FC_CLASSES = ("complete", "missing_required", "illegible")
CLASSES = {"doc_type": DOC_TYPES, "field_threshold": FT_CLASSES, "form_complete": FC_CLASSES}

# frozen counts: split -> per-language cases (FR only in test)
COUNTS = {"train": 1000, "dev": 100, "cal_temperature": 150, "cal_policy": 150, "test": 200}
HELDOUT_SPLITS = ("dev", "cal_temperature", "cal_policy")

TEMPLATE_SLOTS = [(h, b, f) for h in range(4) for b in range(3) for f in range(2)]  # 24
FONT_FAMILIES = ("DejaVu Sans", "DejaVu Sans Condensed", "DejaVu Serif", "DejaVu Sans Mono")

# ---------------------------------------------------------------- language ---
STATE = {
    "en": "Scanned document page.",
    "es": "Página de documento escaneada.",
    "de": "Gescannte Dokumentseite.",
    "fr": "Page de document numérisée.",
}
QUESTION = {
    "doc_type": {
        "en": "Which document type is this?",
        "es": "¿Qué tipo de documento es este?",
        "de": "Um welche Dokumentart handelt es sich?",
        "fr": "De quel type de document s'agit-il ?",
    },
    "field_threshold": {
        "en": "Is the total on this document at least {thr} EUR?",
        "es": "¿El total de este documento es de al menos {thr} EUR?",
        "de": "Beträgt die Gesamtsumme auf diesem Dokument mindestens {thr} EUR?",
        "fr": "Le total de ce document est-il d'au moins {thr} EUR ?",
    },
    "form_complete": {
        "en": "Is this form complete? Required fields are marked with *.",
        "es": "¿Está completo este formulario? Los campos obligatorios llevan *.",
        "de": "Ist dieses Formular vollständig? Pflichtfelder sind mit * markiert.",
        "fr": "Ce formulaire est-il complet ? Les champs obligatoires sont marqués d'un *.",
    },
}
OPTION_TEXT = {
    "doc_type": {
        "en": ("Invoice: an itemized bill with line items, reference number and total.",
               "Receipt: a proof of purchase with items and total, without invoice reference.",
               "Contract: titled clauses with signature lines for parties.",
               "Form: labelled input fields, mostly unfilled.",
               "Letter: sender and recipient blocks, salutation and paragraphs.",
               "None of the other document types applies to this page."),
        "es": ("Factura: cobro detallado con líneas, número de referencia y total.",
               "Recibo: comprobante de compra con artículos y total, sin referencia de factura.",
               "Contrato: cláusulas con título y líneas de firma para las partes.",
               "Formulario: campos con etiqueta, en su mayoría sin rellenar.",
               "Carta: remitente, destinatario, saludo y párrafos.",
               "Ninguno de los otros tipos de documento corresponde a esta página."),
        "de": ("Rechnung: aufgeschlüsselte Abrechnung mit Positionen, Rechnungsnummer und Summe.",
               "Kassenbon: Kaufbeleg mit Artikeln und Summe, ohne Rechnungsnummer.",
               "Vertrag: betitelte Klauseln mit Unterschriftszeilen der Parteien.",
               "Formular: beschriftete Eingabefelder, meist nicht ausgefüllt.",
               "Brief: Absender, Empfänger, Anrede und Absätze.",
               "Keine der anderen Dokumentarten trifft auf diese Seite zu."),
        "fr": ("Facture : relevé détaillé avec lignes, numéro de référence et total.",
               "Reçu : preuve d'achat avec articles et total, sans référence de facture.",
               "Contrat : clauses titrées avec lignes de signature des parties.",
               "Formulaire : champs étiquetés, pour la plupart non remplis.",
               "Lettre : expéditeur, destinataire, salutation et paragraphes.",
               "Aucun des autres types de document ne s'applique à cette page."),
    },
    "field_threshold": {
        "en": ("No", "Yes"), "es": ("No", "Sí"), "de": ("Nein", "Ja"), "fr": ("Non", "Oui"),
    },
    "form_complete": {
        "en": ("Complete: every required field is filled in and readable.",
               "A required field is empty.",
               "Every required field is filled in, but at least one required value cannot be read."),
        "es": ("Completo: todos los campos obligatorios están rellenos y legibles.",
               "Un campo obligatorio está vacío.",
               "Todos los campos obligatorios están rellenos, pero al menos un valor obligatorio no se puede leer."),
        "de": ("Vollständig: alle Pflichtfelder sind ausgefüllt und lesbar.",
               "Ein Pflichtfeld ist leer.",
               "Alle Pflichtfelder sind ausgefüllt, aber mindestens ein Pflichtwert ist unleserlich."),
        "fr": ("Complet : tous les champs obligatoires sont remplis et lisibles.",
               "Un champ obligatoire est vide.",
               "Tous les champs obligatoires sont remplis, mais au moins une valeur obligatoire est illisible."),
    },
}

W = {  # document vocabulary
    "en": dict(invoice="INVOICE", frm="From", billed="Billed to", date="Date", no="No.", desc="Description",
               qty="Qty", amount="Amount", subtotal="Subtotal", tax="Tax", total="TOTAL", payment="Payment",
               paid="Paid by", thanks="Thank you", tendered="Tendered", change="Change",
               previous="Previous balance", party="Party", page="Page",
               methods=("Bank transfer", "Card", "NET 30", "Cash", "Contactless"),
               items=("Steel bracket", "Filter cartridge", "Notebook A5", "USB cable 2m", "Ball bearing 6mm",
                      "Cleaning concentrate", "Safety goggles", "Paper ream A4", "LED bulb E27", "Thermal paste"),
               months=("January", "February", "March", "April", "May", "June", "July", "August",
                       "September", "October", "November", "December")),
    "es": dict(invoice="FACTURA", frm="De", billed="Facturar a", date="Fecha", no="Nº", desc="Descripción",
               qty="Cant.", amount="Importe", subtotal="Subtotal", tax="IVA", total="TOTAL", payment="Forma de pago",
               paid="Pagado con", thanks="Gracias", tendered="Entregado", change="Cambio",
               previous="Saldo anterior", party="Parte", page="Página",
               methods=("Transferencia", "Tarjeta", "30 días", "Efectivo", "Sin contacto"),
               items=("Soporte de acero", "Cartucho de filtro", "Cuaderno A5", "Cable USB 2m", "Rodamiento 6mm",
                      "Concentrado limpiador", "Gafas de seguridad", "Resma de papel A4", "Bombilla LED E27", "Pasta térmica"),
               months=("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
                       "septiembre", "octubre", "noviembre", "diciembre")),
    "de": dict(invoice="RECHNUNG", frm="Von", billed="Rechnung an", date="Datum", no="Nr.", desc="Beschreibung",
               qty="Menge", amount="Betrag", subtotal="Zwischensumme", tax="MwSt.", total="GESAMT", payment="Zahlungsart",
               paid="Bezahlt mit", thanks="Vielen Dank", tendered="Gegeben", change="Rückgeld",
               previous="Vorheriger Saldo", party="Partei", page="Seite",
               methods=("Überweisung", "Karte", "30 Tage netto", "Bar", "Kontaktlos"),
               items=("Stahlwinkel", "Filterpatrone", "Notizbuch A5", "USB-Kabel 2m", "Kugellager 6mm",
                      "Reinigungskonzentrat", "Schutzbrille", "Papier A4 Paket", "LED-Lampe E27", "Wärmeleitpaste"),
               months=("Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August",
                       "September", "Oktober", "November", "Dezember")),
    "fr": dict(invoice="FACTURE", frm="De", billed="Facturé à", date="Date", no="N°", desc="Description",
               qty="Qté", amount="Montant", subtotal="Sous-total", tax="TVA", total="TOTAL", payment="Paiement",
               paid="Payé par", thanks="Merci", tendered="Reçu", change="Rendu",
               previous="Solde précédent", party="Partie", page="Page",
               methods=("Virement", "Carte", "30 jours", "Espèces", "Sans contact"),
               items=("Équerre en acier", "Cartouche filtrante", "Carnet A5", "Câble USB 2m", "Roulement 6mm",
                      "Concentré nettoyant", "Lunettes de protection", "Rame de papier A4", "Ampoule LED E27", "Pâte thermique"),
               months=("janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
                       "septembre", "octobre", "novembre", "décembre")),
}
COMPANIES = ("Northwind Traders", "Acme Supplies Ltd", "Baltic Parts OY", "Sierra Logistics",
             "Meridian Foods", "Orion Print GmbH", "Cobalt Systems", "Vela Marine")
PEOPLE = ("J. Mueller", "A. Petrova", "L. Fontaine", "S. Okonkwo", "M. Alvarez", "K. Bergstrom")

CONTRACT = {
    "en": (("Service Agreement", "Supply Contract", "Maintenance Contract"),
           ("This agreement is entered into on {date} between {a} (the Provider) and {b} (the Client).",
            "1. Scope. The Provider shall deliver the services in Annex A within {n} working days.",
            "2. Fee. The Client shall pay the agreed fee within {d} days of each invoice.",
            "3. Term. This agreement remains in force for {m} months unless terminated in writing.")),
    "es": (("Contrato de servicios", "Contrato de suministro", "Contrato de mantenimiento"),
           ("El presente contrato se celebra el {date} entre {a} (el Proveedor) y {b} (el Cliente).",
            "1. Objeto. El Proveedor prestará los servicios del Anexo A en un plazo de {n} días hábiles.",
            "2. Precio. El Cliente abonará la tarifa acordada en {d} días desde cada factura.",
            "3. Duración. El contrato estará vigente durante {m} meses salvo rescisión por escrito.")),
    "de": (("Dienstleistungsvertrag", "Liefervertrag", "Wartungsvertrag"),
           ("Dieser Vertrag wird am {date} zwischen {a} (Auftragnehmer) und {b} (Auftraggeber) geschlossen.",
            "1. Umfang. Der Auftragnehmer erbringt die Leistungen aus Anlage A innerhalb von {n} Werktagen.",
            "2. Vergütung. Der Auftraggeber zahlt die vereinbarte Vergütung binnen {d} Tagen nach Rechnung.",
            "3. Laufzeit. Der Vertrag gilt {m} Monate, sofern er nicht schriftlich gekündigt wird.")),
    "fr": (("Contrat de services", "Contrat de fourniture", "Contrat de maintenance"),
           ("Le présent contrat est conclu le {date} entre {a} (le Prestataire) et {b} (le Client).",
            "1. Objet. Le Prestataire fournit les services de l'Annexe A sous {n} jours ouvrés.",
            "2. Prix. Le Client règle le montant convenu sous {d} jours après chaque facture.",
            "3. Durée. Le contrat est conclu pour {m} mois sauf résiliation écrite.")),
}
LETTER = {
    "en": ("Dear colleague,", "I am writing to confirm the arrangements discussed on {date}. The revised schedule is attached.",
           "Please contact the coordinator before the end of the month if any detail needs adjustment.", "Kind regards,"),
    "es": ("Estimado colega:", "Le escribo para confirmar lo acordado el {date}. Adjunto el calendario revisado.",
           "Si algún detalle requiere cambios, contacte con la coordinación antes de fin de mes.", "Atentamente,"),
    "de": ("Sehr geehrte Damen und Herren,", "hiermit bestätige ich die am {date} besprochenen Vereinbarungen. Der überarbeitete Zeitplan liegt bei.",
           "Bitte wenden Sie sich bis Monatsende an die Koordination, falls Anpassungen nötig sind.", "Mit freundlichen Grüßen"),
    "fr": ("Cher collègue,", "Je vous confirme les dispositions convenues le {date}. Le calendrier révisé est joint.",
           "Merci de contacter la coordination avant la fin du mois si un détail doit être modifié.", "Cordialement,"),
}
NOTICE = {
    "en": (("NOTICE", "INTERNAL MEMO"), ("The {area} will be closed for maintenance from {h}:00.",
                                          "Please remove personal items before the end of the shift.")),
    "es": (("AVISO", "NOTA INTERNA"), ("El {area} estará cerrado por mantenimiento desde las {h}:00.",
                                        "Retire sus objetos personales antes del final del turno.")),
    "de": (("HINWEIS", "INTERNE MITTEILUNG"), ("Der Bereich {area} ist ab {h}:00 Uhr wegen Wartung geschlossen.",
                                                "Bitte persönliche Gegenstände bis Schichtende entfernen.")),
    "fr": (("AVIS", "NOTE INTERNE"), ("La zone {area} sera fermée pour maintenance à partir de {h}h00.",
                                       "Merci de retirer vos effets personnels avant la fin du service.")),
}
FORM_TITLES = {"en": ("PERSONNEL FORM", "REGISTRATION FORM", "REQUEST FORM"),
               "es": ("FICHA DE PERSONAL", "FORMULARIO DE REGISTRO", "SOLICITUD"),
               "de": ("PERSONALBOGEN", "ANMELDEFORMULAR", "ANTRAGSFORMULAR"),
               "fr": ("FICHE DU PERSONNEL", "FORMULAIRE D'INSCRIPTION", "FORMULAIRE DE DEMANDE")}
FORM_FIELDS = {  # (label, plausible value)
    "en": (("Full name", "Mira Lane"), ("Date of birth", "1988-04-12"), ("Address", "Harbour St 12"),
           ("City", "Springfield"), ("Postal code", "41092"), ("Phone", "+44 20 7946 0958"),
           ("Email", "m.lane@example.org"), ("Department", "Operations"), ("Start date", "2026-11-02"),
           ("Supervisor", "K. Bergstrom")),
    "es": (("Nombre completo", "Lucía Prieto"), ("Fecha de nacimiento", "12/04/1988"), ("Dirección", "C/ Mayor 12"),
           ("Ciudad", "Valladolid"), ("Código postal", "47001"), ("Teléfono", "+34 983 000 128"),
           ("Correo", "l.prieto@example.org"), ("Departamento", "Operaciones"), ("Fecha de alta", "02/11/2026"),
           ("Responsable", "M. Alvarez")),
    "de": (("Vollständiger Name", "Jonas Keller"), ("Geburtsdatum", "12.04.1988"), ("Adresse", "Hafenstr. 12"),
           ("Ort", "Bremen"), ("Postleitzahl", "28195"), ("Telefon", "+49 421 000 128"),
           ("E-Mail", "j.keller@example.org"), ("Abteilung", "Betrieb"), ("Eintrittsdatum", "02.11.2026"),
           ("Vorgesetzte(r)", "J. Mueller")),
    "fr": (("Nom complet", "Claire Morel"), ("Date de naissance", "12/04/1988"), ("Adresse", "12 rue du Port"),
           ("Ville", "Nantes"), ("Code postal", "44000"), ("Téléphone", "+33 2 40 00 01 28"),
           ("Courriel", "c.morel@example.org"), ("Service", "Exploitation"), ("Date d'entrée", "02/11/2026"),
           ("Responsable", "L. Fontaine")),
}
AREAS = ("A", "B", "C", "D", "E")


def fmt_amount(x: float, lang: str) -> str:
    """EN ``1,234.56``; ES/DE/FR ``1.234,56``."""
    s = f"{x:,.2f}"
    return s if lang == "en" else s.replace(",", "_").replace(".", ",").replace("_", ".")


def _date(rng: random.Random, lang: str) -> str:
    d, m = rng.randint(1, 28), rng.choice(W[lang]["months"])
    return {"en": f"{d} {m} 2026", "es": f"{d} de {m} de 2026", "de": f"{d}. {m} 2026", "fr": f"{d} {m} 2026"}[lang]


def stable_seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:15], 16)


# ----------------------------------------------------------------- content ---
@dataclass
class Doc(DocContent):
    """E37 content model plus amounts used by field_threshold."""

    lang: str = "en"
    subtotal: str = ""
    tax: str = ""
    tax_rate: int = 0
    extra: tuple[tuple[str, str], ...] = ()  # (label, amount) distractors
    total_value: float = 0.0


def doc_content(doc_type: str, rng: random.Random, lang: str) -> Doc:
    """Localized content whose ``derive_doc_type`` equals ``doc_type``."""
    w = W[lang]
    company, date = rng.choice(COMPANIES), _date(rng, lang)
    if doc_type in ("invoice", "receipt"):
        c = amount_doc(doc_type, rng.uniform(20, 2500), rng, lang)
    elif doc_type == "contract":
        titles, paras = CONTRACT[lang]
        c = Doc(doc_type="contract", lang=lang, date=date,
                title=f"{rng.choice(titles)} {w['no']} {rng.randint(10, 99)}/{rng.randint(2024, 2026)}",
                paragraphs=tuple(p.format(date=date, a=company, b=rng.choice(COMPANIES), n=rng.randint(10, 60),
                                          d=rng.choice((14, 30)), m=rng.randint(12, 36)) for p in paras),
                signature_lines=2)
    elif doc_type == "form":
        fields = rng.sample(FORM_FIELDS[lang], rng.randint(5, 8))
        c = Doc(doc_type="form", lang=lang, title=rng.choice(FORM_TITLES[lang]),
                fields=tuple((lab, val if rng.random() < 0.3 else "") for lab, val in fields))
    elif doc_type == "letter":
        c = Doc(doc_type="letter", lang=lang, date=date, sender=f"{rng.choice(PEOPLE)}, {company}",
                recipient=f"{rng.choice(PEOPLE)}, {rng.choice(COMPANIES)}",
                paragraphs=tuple(p.format(date=_date(rng, lang)) for p in LETTER[lang]))
    else:
        titles, paras = NOTICE[lang]
        c = Doc(doc_type="other", lang=lang, title=rng.choice(titles),
                paragraphs=tuple(p.format(area=rng.choice(AREAS), h=rng.randint(6, 18)) for p in paras))
    if derive_doc_type(c) != doc_type:
        raise AssertionError(f"content derives {derive_doc_type(c)!r}, intended {doc_type!r}")
    return c


def amount_doc(kind: str, total: float, rng: random.Random, lang: str, distractor: float | None = None) -> Doc:
    """Invoice/receipt whose lines + tax add up exactly to ``total`` (cents)."""
    w = W[lang]
    lines = tuple((it, str(rng.randint(1, 9)), "") for it in rng.sample(w["items"], rng.randint(2, 5)))
    d = distractor if distractor is not None else total * rng.uniform(0.3, 2.5)
    extra = ((w["previous"] if kind == "invoice" else w["tendered"], fmt_amount(d, lang)),)
    doc = Doc(doc_type=kind, lang=lang, title=w["invoice"] if kind == "invoice" else rng.choice(COMPANIES).upper(),
              sender=rng.choice(COMPANIES) if kind == "invoice" else "",
              recipient=rng.choice(COMPANIES) if kind == "invoice" else "",
              date=_date(rng, lang),
              reference=f"{w['no']} {rng.randint(1000, 9999)}/26" if kind == "invoice" else "",
              lines=lines, payment=rng.choice(w["methods"][:3] if kind == "invoice" else w["methods"][3:]),
              tax_rate=rng.choice((7, 10, 19, 21)), extra=extra)
    return reprice(doc, total, rng)


def reprice(doc: Doc, total: float, rng: random.Random) -> Doc:
    """Set line amounts, subtotal and tax so they add up to ``total``; items,
    quantities, rate and everything else stay (decisive counterfactuals)."""
    total_c = round(total * 100)
    sub_c = round(total_c / (1 + doc.tax_rate / 100))
    n = len(doc.lines)
    cuts = sorted(rng.sample(range(1, sub_c), n - 1)) if sub_c > n else list(range(1, n))
    parts = [b - a for a, b in zip([0] + cuts, cuts + [sub_c])]
    lines = tuple((it, q, fmt_amount(p / 100, doc.lang)) for (it, q, _), p in zip(doc.lines, parts))
    return replace(doc, lines=lines, total=fmt_amount(total_c / 100, doc.lang), total_value=total_c / 100,
                   subtotal=fmt_amount(sub_c / 100, doc.lang), tax=fmt_amount((total_c - sub_c) / 100, doc.lang))


def ft_values(label: str, rng: random.Random) -> tuple[float, float]:
    """(threshold, total). Threshold drawn before the label; total = thr(1±δ),
    never equal to the threshold (criteria G1(f))."""
    thr = math.exp(rng.uniform(math.log(10), math.log(5000)))
    thr = float(round(thr)) if rng.random() < 0.5 else round(thr, 2)
    total = total_for(label, thr, rng)
    return thr, total


def total_for(label: str, thr: float, rng: random.Random) -> float:
    delta = math.exp(rng.uniform(math.log(0.001), math.log(0.5)))
    t = round(thr * (1 + delta) if label == "true" else thr * (1 - delta), 2)
    if label == "true" and t <= thr:
        t = round(thr + 0.01, 2)
    if label == "false" and t >= thr:
        t = round(thr - 0.01, 2)
    return t


def ft_label(total: float, thr: float) -> str:
    return "true" if total >= thr else "false"


@dataclass(frozen=True)
class FormField:
    label: str
    value: str  # "" = empty
    required: bool
    illegible: bool = False


def form_label(fields) -> str:
    req = [f for f in fields if f.required]
    if any(not f.value for f in req):
        return "missing_required"
    if any(f.illegible for f in req):
        return "illegible"
    return "complete"


def form_edit(fields: tuple[FormField, ...], target: str, rng: random.Random) -> tuple[FormField, ...]:
    """Minimal edit of ``fields`` so that ``form_label`` becomes ``target``."""
    fs = list(fields)
    req = [i for i, f in enumerate(fs) if f.required]
    if target in ("complete", "illegible"):
        for i in req:  # fill every empty required field
            if not fs[i].value:
                fs[i] = replace(fs[i], value=_FILL[fs[i].label])
        for i in req:
            if fs[i].illegible and target == "complete":
                fs[i] = replace(fs[i], illegible=False)
        if target == "illegible" and not any(fs[i].illegible for i in req):
            i = rng.choice(req)
            fs[i] = replace(fs[i], illegible=True)
    else:  # missing_required
        if not any(not fs[i].value for i in req):
            i = rng.choice(req)
            fs[i] = replace(fs[i], value="", illegible=False)
    out = tuple(fs)
    assert form_label(out) == target, (form_label(out), target)
    return out


_FILL = {lab: val for lang in FORM_FIELDS for lab, val in FORM_FIELDS[lang]}


def form_content(label: str, rng: random.Random, lang: str) -> tuple[str, tuple[FormField, ...]]:
    picks = rng.sample(FORM_FIELDS[lang], rng.randint(5, 9))
    n_req = rng.randint(3, min(6, len(picks)))
    fields = tuple(FormField(lab, val, required=i < n_req) for i, (lab, val) in enumerate(picks))
    order = list(fields)
    rng.shuffle(order)
    # optional fields: random emptiness; never illegible
    order = [replace(f, value="" if (not f.required and rng.random() < 0.4) else f.value) for f in order]
    fields = form_edit(tuple(order), label, rng)
    if label == "missing_required" and rng.random() < 0.2:  # precedence: missing beats illegible
        filled = [i for i, f in enumerate(fields) if f.required and f.value]
        if filled:
            i = rng.choice(filled)
            fields = fields[:i] + (replace(fields[i], illegible=True),) + fields[i + 1:]
    assert form_label(fields) == label
    return rng.choice(FORM_TITLES[lang]), fields


# --------------------------------------------------------------- templates ---
def family_templates(family: str) -> tuple[str, ...]:
    return tuple(f"{family}-h{h}b{b}f{f}" for h, b, f in TEMPLATE_SLOTS)


def split_templates(family: str) -> dict[str, tuple[str, ...]]:
    """sha256 order: first 5 -> test, next 5 -> heldout, rest (14) -> train.
    Every template renders every class, so each class is in every part."""
    ordered = sorted(family_templates(family), key=lambda t: hashlib.sha256(t.encode()).hexdigest())
    return {"test": tuple(sorted(ordered[:5])), "heldout": tuple(sorted(ordered[5:10])),
            "train": tuple(sorted(ordered[10:]))}


def templates_for_split(family: str, split: str) -> tuple[str, ...]:
    parts = split_templates(family)
    return parts["heldout"] if split in HELDOUT_SPLITS else parts[split]


@dataclass(frozen=True)
class Theme:
    font: str
    base_px: int
    accent: str
    ink: str
    paper: str
    pad: int
    dx: int
    dy: int


def make_theme(seed: int) -> Theme:
    r = random.Random(seed)
    return Theme(font=r.choice(FONT_FAMILIES), base_px=r.randint(14, 17),
                 accent=r.choice(("#3c5aa0", "#784040", "#286450", "#5a5a5a", "#6a3d8f")),
                 ink=r.choice(("#18181c", "#121622", "#1e1a18")),
                 paper=r.choice(("#fcfcfa", "#ffffff", "#f8f6f0")),
                 pad=r.randint(30, 60), dx=r.randint(-4, 4), dy=r.randint(-4, 4))


# ---------------------------------------------------------------- planning ---
def plan_groups(family: str, n: int, rng: random.Random) -> list[list[tuple[str, str]]]:
    """Exactly balanced labels (±1) grouped into episodes: single, decisive
    pair (label flips) or irrelevant pair (label kept). Counts are preserved."""
    classes = CLASSES[family]
    labels = [classes[i % len(classes)] for i in range(n)]
    rng.shuffle(labels)
    p_dec, p_irr = (0.0, 0.2) if family == "doc_type" else (0.25, 0.15)
    groups = []
    while labels:
        a = labels.pop(0)
        u = rng.random()
        if u < p_dec:
            j = next((k for k, b in enumerate(labels) if b != a), None)
            if j is not None:
                groups.append([("base", a), ("decisive", labels.pop(j))])
                continue
        elif u < p_dec + p_irr:
            j = next((k for k, b in enumerate(labels) if b == a), None)
            if j is not None:
                groups.append([("base", a), ("irrelevant", labels.pop(j))])
                continue
        groups.append([("base", a)])
    return groups


@dataclass
class GenCase:
    case: DecisionCase
    family: str
    template: str
    theme_seed: int
    role: str
    base_id: str
    body_html: str
    meta: dict


def options_for(family: str, lang: str) -> tuple[Option, ...]:
    return tuple(Option(id=c, description=d) for c, d in zip(CLASSES[family], OPTION_TEXT[family][lang]))


def generate_split(family: str, split: str, lang: str, seed: int = 38) -> list[GenCase]:
    n = COUNTS[split]
    if lang == "fr" and split != "test":
        raise ValueError("FR is evaluation-only (test)")
    rng = random.Random(stable_seed(seed, family, split, lang))
    templates = templates_for_split(family, split)
    out: list[GenCase] = []
    for g, roles in enumerate(plan_groups(family, n, rng)):
        grng = random.Random(stable_seed(seed, family, split, lang, g))
        template = templates[grng.randrange(len(templates))]
        gid = f"e38a-{family}-{split}-{lang}-{g:05d}"
        theme_seed = stable_seed(gid, "theme")
        base_id = f"{gid}-base"
        state = None  # per-family content carried between base and its variant
        for role, label in roles:
            tseed = stable_seed(gid, "theme-irr") if role == "irrelevant" else theme_seed
            question = QUESTION[family][lang]
            meta: dict = {}
            if family == "doc_type":
                if state is None:
                    state = doc_content(label, grng, lang)
                body = render_doc(state, template, make_theme(tseed))
            elif family == "field_threshold":
                if state is None:
                    thr, total = ft_values(label, grng)
                    kind = grng.choice(("invoice", "receipt"))
                    # distractor on the opposite side of thr in ~50 % of cases
                    opp = grng.random() < 0.5
                    side = (thr * grng.uniform(0.3, 0.95)) if label == "true" else (thr * grng.uniform(1.05, 3.0))
                    dist = side if opp else (total * grng.uniform(1.05, 1.5) if label == "true" else total * grng.uniform(0.3, 0.95))
                    state = (thr, amount_doc(kind, total, grng, lang, distractor=dist))
                elif role == "decisive":
                    thr, doc = state
                    state = (thr, reprice(doc, total_for(label, thr, grng), grng))
                thr, doc = state
                assert ft_label(doc.total_value, thr) == label
                question = question.format(thr=fmt_amount(thr, lang))
                meta = {"threshold": thr, "total": doc.total_value}
                body = render_doc(doc, template, make_theme(tseed))
            else:
                if state is None:
                    state = form_content(label, grng, lang)
                elif role == "decisive":
                    state = (state[0], form_edit(state[1], label, grng))
                elif role == "irrelevant":
                    title, fields = state
                    opt = [i for i, f in enumerate(fields) if not f.required]
                    if opt:
                        i = grng.choice(opt)
                        f = fields[i]
                        fields = fields[:i] + (replace(f, value="" if f.value else _FILL[f.label]),) + fields[i + 1:]
                    state = (title, fields)
                title, fields = state
                assert form_label(fields) == label
                meta = {"n_required": sum(f.required for f in fields)}
                body = render_form(title, fields, lang, template, make_theme(tseed))
            cid = f"{gid}-{role}"
            case = DecisionCase(id=cid, group_id=gid, source="e38a-synthetic", language=lang,
                                domain="documents", state=STATE[lang], question=question,
                                options=options_for(family, lang), target_kind="single",
                                target_option=label, label_origin="e38a-content-model", split=split)
            out.append(GenCase(case, family, template, tseed, role, base_id, body, meta))
    return out


# -------------------------------------------------------------------- HTML ---
def _e(s) -> str:
    return html.escape(str(s))


def _slots(template: str) -> tuple[int, int, int]:
    s = template.rsplit("-", 1)[1]
    return int(s[1]), int(s[3]), int(s[5])


def _css(t: Theme, b: int, width: int) -> str:
    cell = {0: "border-bottom:1px solid #bbb", 1: "border:1px solid " + t.ink, 2: "border:none"}[b]
    return (f"<style>#page{{width:{width}px;box-sizing:border-box;padding:{t.pad + t.dy}px {t.pad - t.dx}px {t.pad - t.dy}px {t.pad + t.dx}px;background:{t.paper};"
            f"color:{t.ink};font-family:'{t.font}';font-size:{t.base_px}px;line-height:1.45}}"
            f"#page h1{{font-size:{t.base_px + 9}px;margin:0 0 8px 0;color:{t.accent}}}"
            f"#page table{{border-collapse:collapse;width:100%;margin:10px 0}}"
            f"#page td,#page th{{{cell};padding:4px 6px;text-align:left}}"
            f"#page .r{{text-align:right}} #page .band{{background:{t.accent};color:#fff;padding:8px 12px}}"
            f"#page .band h1{{color:#fff}} #page .box{{border:1px solid {t.ink};padding:10px}}"
            f"#page .foot{{margin-top:18px;font-size:{t.base_px - 3}px;color:#666;border-top:1px solid #ccc}}"
            f"#page .sig{{display:inline-block;width:40%;margin-right:8%;border-top:1px solid {t.ink};margin-top:40px}}"
            "</style>")


def _header(h: int, title: str, meta: list[str]) -> str:
    t = f'<h1 data-k="title">{_e(title)}</h1>' if title else ""
    m = "".join(f"<div>{x}</div>" for x in meta)
    if h == 0:
        return f'<div style="display:flex;justify-content:space-between"><div>{t}</div><div class="r">{m}</div></div>'
    if h == 1:
        return f'<div style="text-align:center">{t}{m}</div>'
    if h == 2:
        return f'<div class="band">{t}</div><div style="margin-top:8px">{m}</div>'
    return f'<div style="display:flex;justify-content:space-between"><div>{m}</div><div style="font-variant:small-caps">{t}</div></div>'


def _footer(f: int, lang: str) -> str:
    return f'<div class="foot">{_e(W[lang]["page"])} 1/1</div>' if f == 0 else ""


def render_doc(c: Doc, template: str, t: Theme) -> str:
    """HTML body for a doc_type / field_threshold page (``#page`` element)."""
    h, b, f = _slots(template)
    w = W[c.lang]
    width = 380 if c.doc_type == "receipt" else 794
    meta = []
    if c.sender and c.doc_type == "invoice":
        meta.append(f'{_e(w["frm"])}: <span data-k="sender">{_e(c.sender)}</span>')
        meta.append(f'{_e(w["billed"])}: <span data-k="recipient">{_e(c.recipient)}</span>')
    if c.date:
        meta.append(f'{_e(w["date"])}: {_e(c.date)}')
    if c.reference:
        meta.append(f'<span data-k="ref">{_e(c.reference)}</span>')
    body = ""
    if c.doc_type in ("invoice", "receipt"):
        rows = "".join(f'<tr data-k="line"><td>{_e(d)}</td><td class="r">{_e(q)}</td><td class="r">{_e(a)}</td></tr>'
                       for d, q, a in c.lines)
        head = f'<tr><th>{_e(w["desc"])}</th><th class="r">{_e(w["qty"])}</th><th class="r">{_e(w["amount"])}</th></tr>'
        sums = ""
        if c.subtotal:
            sums += f'<tr><td colspan="2" class="r">{_e(w["subtotal"])}</td><td class="r">{_e(c.subtotal)}</td></tr>'
            sums += f'<tr><td colspan="2" class="r">{_e(w["tax"])} {c.tax_rate}%</td><td class="r">{_e(c.tax)}</td></tr>'
        sums += (f'<tr><td colspan="2" class="r"><b>{_e(w["total"])}</b></td>'
                 f'<td class="r"><b data-k="total">{_e(c.total)}</b></td></tr>')
        body = f"<table>{head}{rows}{sums}</table>"
        pay = w["payment"] if c.doc_type == "invoice" else w["paid"]
        body += f'<div>{_e(pay)}: <span data-k="payment">{_e(c.payment)}</span></div>'
        body += "".join(f'<div data-k="extra">{_e(lab)}: {_e(val)}</div>' for lab, val in c.extra)
        if c.doc_type == "receipt":
            body += f'<div style="text-align:center;margin-top:8px">{_e(w["thanks"])}</div>'
    elif c.fields:
        body = _fields_html([FormField(lab, val, False) for lab, val in c.fields], b)
    if c.doc_type == "letter":
        body += f'<div data-k="recipient" style="margin:10px 0 14px 0">{_e(c.recipient)}</div>'
        meta.insert(0, f'<span data-k="sender">{_e(c.sender)}</span>')
    if c.paragraphs:
        paras = "".join(f'<p data-k="para">{_e(p)}</p>' for p in c.paragraphs)
        body += f'<div class="box">{paras}</div>' if b == 1 else paras
    if c.signature_lines:
        body += "".join(f'<span class="sig" data-k="sig">{_e(w["party"])} {i + 1}</span>'
                        for i in range(c.signature_lines))
    return (_css(t, b, width) + f'<div id="page" data-template="{_e(template)}">'
            + _header(h, c.title, meta) + body + _footer(f, c.lang) + "</div>")


def _fields_html(fields, b: int) -> str:
    out = []
    for fld in fields:
        star = " *" if fld.required else ""
        blur = f' style="filter:blur({3.2}px)"' if fld.illegible else ""
        val = f'<span data-k="fval"{blur}>{_e(fld.value)}</span>'
        if b == 0:
            cell = f'<div style="border-bottom:1px solid #444;min-height:1.6em;flex:1;padding-left:6px">{val}</div>'
        elif b == 1:
            cell = f'<div style="border:1px solid #444;min-height:1.6em;flex:1;padding:2px 6px">{val}</div>'
        else:
            cell = f'<div style="border-bottom:1px dotted #444;min-height:1.6em;flex:1;padding-left:6px">{val}</div>'
        req = ' data-req="1"' if fld.required else ""
        out.append(f'<div data-k="field"{req} style="display:flex;gap:10px;margin:10px 0">'
                   f'<div style="width:220px;font-weight:bold" data-k="flabel">{_e(fld.label)}{star}</div>{cell}</div>')
    return "".join(out)


def render_form(title: str, fields, lang: str, template: str, t: Theme) -> str:
    h, b, f = _slots(template)
    return (_css(t, b, 794) + f'<div id="page" data-template="{_e(template)}">'
            + _header(h, title, []) + _fields_html(fields, b) + _footer(f, lang) + "</div>")

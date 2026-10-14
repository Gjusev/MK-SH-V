"""E38b generator: screen families as HTML (rendered by Playwright elsewhere).

Pure module (no browser). Criteria: docs/E38B_CRITERIA_2026-10-14.md.
Labels are computed from the content models below; the verifier (no import of
this module) re-derives labels with its own parser. Split key is the
(app, template) pair, assigned before any variant; themes are per-case
variants.
"""

from __future__ import annotations

import hashlib
import html
import random
import re
from dataclasses import dataclass, replace

from mokka_decisions.contracts import DecisionCase, Option

from .gen_e38a import COUNTS, HELDOUT_SPLITS, Theme, make_theme, stable_seed

FAMILIES = ("screen_state", "task_done", "source_conflict", "urgency")
SS_CLASSES = ("success", "error", "loading", "needs_input", "blocked")
TD_CLASSES = ("false", "true")
SC_CLASSES = ("agree", "conflict")
UR_CLASSES = ("1", "2", "3", "4", "5")
CLASSES = {"screen_state": SS_CLASSES, "task_done": TD_CLASSES,
           "source_conflict": SC_CLASSES, "urgency": UR_CLASSES}
LANGS = ("en", "es", "de", "fr")
TRAIN_LANGS = ("en", "es", "de")

APPS = ("NimbusDesk", "TrackForge", "CloudLedger", "PixelPort")
TEMPLATE_SLOTS = [(c, l, p) for c in range(4) for l in range(3) for p in range(2)]
TEMPLATE = "scr-c{0}l{1}p{2}"

_FORBIDDEN = re.compile(r"<(img|doc|row|eos|mask|query)>|\[O\d+\]")


def assert_clean(*texts: str) -> None:
    for t in texts:
        if _FORBIDDEN.search(t):
            raise AssertionError(f"forbidden marker token in {t!r}")

# --------------------------------------------------------------- language ---
STATE = {
    "screen_state": {"en": "Application screenshot.", "es": "Captura de la aplicación.",
                     "de": "Screenshot der Anwendung.", "fr": "Capture d'écran de l'application."},
    "task_done": {"en": "Goal: {goal}", "es": "Objetivo: {goal}", "de": "Ziel: {goal}", "fr": "Objectif : {goal}"},
    "source_conflict": {"en": "Ticket #{n}. {claims}", "es": "Tique #{n}. {claims}",
                        "de": "Ticket #{n}. {claims}", "fr": "Ticket #{n}. {claims}"},
    "urgency": {"en": "Ticket #{n}. Impact: {impact}. {extra}",
                "es": "Tique #{n}. Impacto: {impact}. {extra}",
                "de": "Ticket #{n}. Auswirkung: {impact}. {extra}",
                "fr": "Ticket #{n}. Impact : {impact}. {extra}"},
}
_QUESTION_SC = {
    "en": "Do the ticket text and the screenshot agree? The screenshot is the text source of truth when they differ.",
    "es": "¿Coinciden el texto del tique y la captura? La captura es la fuente fiable si difieren.",
    "de": "Stimmen Tickettext und Screenshot überein? Der Screenshot ist die verbindliche Quelle bei Abweichung.",
    "fr": "Le texte du ticket et la capture d'écran concordent-ils ? La capture fait foi en cas de divergence.",
}
QUESTION = {
    "screen_state": {"en": "What state is this application screen in?",
                     "es": "¿En qué estado está esta pantalla de la aplicación?",
                     "de": "In welchem Zustand ist dieser Anwendungsbildschirm?",
                     "fr": "Dans quel état se trouve cet écran d'application ?"},
    "task_done": {"en": "Does the screenshot show the goal as completed?",
                  "es": "¿La captura muestra el objetivo como completado?",
                  "de": "Zeigt der Screenshot das Ziel als erledigt?",
                  "fr": "La capture d'écran montre-t-elle l'objectif comme accompli ?"},
    "source_conflict": _QUESTION_SC,
    "urgency": {"en": "How urgent is this ticket on a scale from 1 (lowest) to 5 (highest)?",
                "es": "¿Cuán urgente es este tique en una escala de 1 (menor) a 5 (máxima)?",
                "de": "Wie dringend ist dieses Ticket auf einer Skala von 1 (niedrig) bis 5 (hoch)?",
                "fr": "Quelle est l'urgence de ce ticket sur une échelle de 1 (la plus basse) à 5 (la plus haute) ?"},
}

OPTION_TEXT = {
    "screen_state": {
        "en": ("A confirmation with a check mark: the action finished without problems.",
               "An error message with a code and a retry button.",
               "A loading indicator: a spinner and placeholder rows.",
               "A form asking for input: at least one required field is empty.",
               "A locked notice without a retry button: access or system block."),
        "es": ("Una confirmación con marca: la acción terminó sin problemas.",
               "Un mensaje de error con código y botón de reintento.",
               "Un indicador de carga: indicador giratorio y filas de espera.",
               "Un formulario pidiendo datos: al menos un campo obligatorio está vacío.",
               "Un bloqueo de acceso o del sistema: aviso de bloqueo sin botón de reintento."),
        "de": ("Eine Bestätigung mit Häkchen: die Aktion wurde fehlerfrei abgeschlossen.",
               "Eine Fehlermeldung mit Code und Wiederholen-Schaltfläche.",
               "Ein Ladeindikator: Ladesymbol und Platzhalterzeilen.",
               "Ein Formular zur Eingabe: mindestens ein Pflichtfeld ist leer.",
               "Eine Zugriffs- oder Systemsperre: Sperrhinweis ohne Wiederholen-Schaltfläche."),
        "fr": ("Une confirmation avec coche : l'action s'est terminée sans problème.",
               "Un message d'erreur avec code et bouton Réessayer.",
               "Un indicateur de chargement : roue et lignes provisoires.",
               "De formulaire demandant une saisie : au moins un champ requis est vide.",
               "Un blocage d'accès ou système : avis verrouillé sans bouton de réessayer."),
    },
    "task_done": {"en": ("No", "Yes"), "es": ("No", "Sí"), "de": ("Nein", "Ja"), "fr": ("Non", "Oui")},
    "source_conflict": {
        "en": ("Agree: every claimed detail in the text matches the screen.",
               "Conflict: at least one claimed detail differs on the screen."),
        "es": ("Coinciden: cada dato del texto coincide con la pantalla.",
               "Conflicto: al menos un dato del texto difiere en la pantalla."),
        "de": ("Stimmen überein: jedes genannte Detail im Text stimmt mit dem Bildschirm überein.",
               "Konflikt: mindestens ein genanntes Detail weicht auf dem Bildschirm ab."),
        "fr": ("Concordance : chaque détail cité du texte correspond à l'écran.",
               "Conflit : au moins un détail cité diffère à l'écran."),
    },
}

OPTION_TEXT["urgency"] = {
    "en": ("1 — cosmetic, no user impact.",
           "2 — minor annoyance, workaround exists.",
           "3 — noticeable problem for some users.",
           "4 — major disruption for many users.",
           "5 — critical outage, act immediately."),
    "es": ("1 — cosmético, sin impacto para el usuario.",
           "2 — molestia menor, existe alternativa manual.",
           "3 — problema notable para algunos usuarios.",
           "4 — disrupción mayor para muchos usuarios.",
           "5 — caída crítica, actuar de inmediato."),
    "de": ("1 — kosmetisch, keine Auswirkung für Nutzer.",
           "2 — kleine Belästigung, es gibt einen Workaround.",
           "3 — spürbares Problem für einige Nutzer.",
           "4 — erhebliche Störung für viele Nutzer.",
           "5 — kritischer Ausfall, sofort handeln."),
    "fr": ("1 — cosmétique, aucun impact utilisateur.",
           "2 — petite gêne, une solution de contournement existe.",
           "3 — problème noticeable pour certains utilisateurs.",
           "4 — perturbation majeure pour de nombreux utilisateurs.",
           "5 — panne critique, agir immédiatement."),
}

# ------------------------------------------------------------- UI strings ---
def _w(name, en, es, de, fr):
    """Build the per-language UI string table: _w('nav', 'Dashboard', 'Panel', ...)."""
    return {"en": en, "es": es, "de": de, "fr": fr}


W_TABLE = {
    "nav": _w("", ("Dashboard", "Projects", "Reports", "Settings"),
              ("Panel", "Proyectos", "Informes", "Ajustes"),
              ("Übersicht", "Projekte", "Berichte", "Einstellungen"),
              ("Tableau de bord", "Projets", "Rapports", "Paramètres")),
    "btn_retry": _w("", "Retry", "Reintentar", "Wiederholen", "Réessayer"),
    "btn_submit": _w("", "Submit", "Enviar", "Senden", "Envoyer"),
    "loading": _w("", "Loading…", "Cargando…", "Wird geladen…", "Chargement…"),
    "deadline": _w("", "Deadline", "Plazo", "Frist", "Échéance"),
    "impact": _w("", "Impact", "Impacto", "Auswirkung", "Impact"),
    "ticket": _w("", "Ticket", "Tique", "Ticket", "Ticket"),
    "goal": _w("", "Goal", "Objetivo", "Ziel", "Objectif"),
    "lock": _w("", "LOCKED", "BLOQUEADO", "GESPERRT", "VERROUILLÉ"),
    "tasklist": _w("", "Tasks", "Tareas", "Aufgaben", "Tâches"),
    "record": _w("", "Details", "Detalles", "Details", "Détails"),
}

W_TABLE["status"] = _w("", ("Open", "In progress", "Blocked", "Done"),
                       ("Abierto", "En curso", "Bloqueado", "Hecho"),
                       ("Offen", "In Bearbeitung", "Blockiert", "Erledigt"),
                       ("Ouvert", "En cours", "Bloqué", "Terminé"))

W_TABLE["prio"] = _w("", ("Low", "Medium", "High", "Urgent"),
                     ("Baja", "Media", "Alta", "Urgente"),
                     ("Niedrig", "Mittel", "Hoch", "Dringend"),
                     ("Basse", "Moyenne", "Haute", "Urgente"))
W_TABLE["cat"] = _w("", ("Billing", "Technical", "Access", "Shipping"),
                    ("Facturación", "Técnico", "Acceso", "Envíos"),
                    ("Abrechnung", "Technik", "Zugang", "Versand"),
                    ("Facturation", "Technique", "Accès", "Expédition"))

W_TABLE["chan"] = _w("", ("Email", "Phone", "Chat", "Portal"),
                     ("Correo", "Teléfono", "Chat", "Portal"),
                     ("E-Mail", "Telefon", "Chat", "Portal"),
                     ("E-mail", "Téléphone", "Chat", "Portail"))
OWNERS = ("J. Mueller", "A. Petrova", "L. Fontaine", "S. Okonkwo", "M. Alvarez", "K. Bergstrom")

W_TABLE["impact_vals"] = _w("", ("minor", "moderate", "critical"),
                            ("minor", "moderate", "critical"),
                            ("gering", "mittel", "kritisch"),
                            ("mineur", "modéré", "critique"))
W_TABLE["tasks"] = _w("", ("Export", "Upload", "Rename", "Delete", "Share", "Approve", "Archive",
                           "Validate", "Restore", "Print"),
                      ("Exportar", "Subir", "Renombrar", "Eliminar", "Compartir", "Aprobar", "Archivar",
                       "Validar", "Restaurar", "Imprimir"),
                      ("Exportieren", "Hochladen", "Umbenennen", "Löschen", "Teilen", "Freigeben", "Archivieren",
                       "Prüfen", "Wiederherstellen", "Drucken"),
                      ("Exporter", "Téléverser", "Renommer", "Supprimer", "Partager", "Approuver", "Archiver",
                       "Valider", "Restaurer", "Imprimer"))

# per-language message pools (success / error / loading / blocked / form labels)
MSGS = {
    "en": dict(
        success=("Your request completed successfully.", "All changes have been saved.",
                 "The operation finished without errors."),
        error=("The server could not process the request.", "Something went wrong on our side.",
               "The connection to the service failed."),
        loading=("Fetching the latest data.", "Please wait while we prepare your report.",
                 "Synchronizing with the server."),
        blocked=("Your account does not have access to this area.",
                 "This workspace is locked by an administrator.",
                 "Access denied: the maintenance window is active."),
        form_labels=("Email address", "API key", "Workspace name", "Phone number", "Reason", "Confirmation code"),
    ),
}
MSGS["es"] = dict(
    success=("Tu solicitud se completó correctamente.", "Todos los cambios han sido guardados.",
             "La operación terminó sin errores."),
    error=("El servidor no pudo procesar la solicitud.", "Algo falló de nuestro lado.",
           "La conexión con el servicio falló."),
    loading=("Obteniendo los datos más recientes.", "Espera mientras preparamos tu informe.",
             "Sincronizando con el servidor."),
    blocked=("Tu cuenta no tiene acceso a esta zona.", "Este espacio está bloqueado por un administrador.",
             "Acceso denegado: la ventana de mantenimiento está activa."),
    form_labels=("Dirección de correo", "Clave API", "Nombre del espacio", "Teléfono", "Motivo",
                 "Código de confirmación"),
)

MSGS["de"] = dict(
    success=("Ihre Anfrage wurde erfolgreich abgeschlossen.", "Alle Änderungen wurden gespeichert.",
             "Der Vorgang wurde fehlerfrei beendet."),
    error=("Der Server konnte die Anfrage nicht verarbeiten.", "Auf unserer Seite ist etwas schiefgelaufen.",
           "Die Verbindung zum Dienst ist fehlgeschlagen."),
    loading=("Neueste Daten werden abgerufen.", "Bitte warten Sie, während wir Ihren Bericht vorbereiten.",
             "Synchronisierung mit dem Server."),
    blocked=("Ihr Konto hat keinen Zugriff auf diesen Bereich.",
             "Dieser Arbeitsbereich ist von einem Administrator gesperrt.",
             "Zugang verweigert: das Wartungsfenster ist aktiv."),
    form_labels=("E-Mail-Adresse", "API-Schlüssel", "Arbeitsbereichsname", "Telefonnummer", "Grund",
                 "Bestätigungscode"),
)

MSGS["fr"] = dict(
    success=("Votre demande s'est terminée avec succès.", "Toutes les modifications ont été enregistrées.",
             "L'opération s'est terminée sans erreur."),
    error=("Le serveur n'a pas pu traiter la demande.", "Une erreur est survenue de notre côté.",
           "La connexion au service a échoué."),
    loading=("Récupération des dernières données.", "Veuillez patienter pendant la préparation du rapport.",
             "Synchronisation avec le serveur."),
    blocked=("Votre compte n'a pas accès à cette zone.", "Cet espace est verrouillé par un administrateur.",
             "Accès refusé : la fenêtre de maintenance est active."),
    form_labels=("Adresse e-mail", "Clé API", "Nom de l'espace", "Téléphone", "Motif", "Code de confirmation"),
)

TASK_TOKENS = ("q3_report.xlsx", "invoice_2043.csv", "backup_2026.tar", "user_journey.png", "PAYROLL-8841",
               "audit_trail.log", "orders_nov.csv", "stock_list.xlsx", "SHIP-7712", "weekly_sync.mp4",
               "ledger_2025.db", "access_review.pdf")

GOALS = {
    "en": ('Export "{tok}" to PDF', 'Upload "{tok}" to the archive', 'Rename "{tok}" to final',
           'Delete "{tok}" from the queue', 'Share "{tok}" with the team', 'Approve "{tok}" for release',
           'Archive "{tok}" in cold storage', 'Validate "{tok}" against the schema',
           'Restore "{tok}" from backup', 'Print "{tok}" for the records'),
    "es": ('Exportar "{tok}" a PDF', 'Subir "{tok}" al archivo', 'Renombrar "{tok}" a final',
           'Eliminar "{tok}" de la cola', 'Compartir "{tok}" con el equipo', 'Aprobar "{tok}" para publicación',
           'Archivar "{tok}" en almacenamiento frío', 'Validar "{tok}" contra el esquema',
           'Restaurar "{tok}" desde copia', 'Imprimir "{tok}" para el archivo'),
}

GOALS["de"] = (
    '"{tok}" als PDF exportieren', '"{tok}" ins Archiv hochladen', '"{tok}" zu final umbenennen',
    '"{tok}" aus der Warteschlange löschen', '"{tok}" mit dem Team teilen',
    '"{tok}" zur Veröffentlichung freigeben', '"{tok}" im Kaltstorage archivieren',
    '"{tok}" gegen das Schema prüfen', '"{tok}" aus dem Backup wiederherstellen',
    '"{tok}" für die Unterlagen drucken',
)
GOALS["fr"] = (
    'Exporter "{tok}" en PDF', "Téléverser \"{tok}\" vers l'archive", 'Renommer "{tok}" en final',
    'Supprimer "{tok}" de la file', "Partager \"{tok}\" avec l'équipe", 'Approuver "{tok}" pour publication',
    'Archiver "{tok}" en stockage froid', 'Valider "{tok}" contre le schéma',
    'Restaurer "{tok}" depuis la sauvegarde', 'Imprimer "{tok}" pour les archives',
)

SC_FIELDS = {
    "en": (("priority", "Priority", ("Low", "Medium", "High", "Urgent")),
           ("owner", "Owner", OWNERS),
           ("status", "Status", W_TABLE["status"]["en"]),
           ("category", "Category", W_TABLE["cat"]["en"]),
           ("channel", "Channel", W_TABLE["chan"]["en"])),
    "es": (("priority", "Prioridad", ("Baja", "Media", "Alta", "Urgente")),
           ("owner", "Responsable", OWNERS),
           ("status", "Estado", W_TABLE["status"]["es"]),
           ("category", "Categoría", W_TABLE["cat"]["es"]),
           ("channel", "Canal", W_TABLE["chan"]["es"])),
}

SC_FIELDS["de"] = (
    ("priority", "Priorität", ("Niedrig", "Mittel", "Hoch", "Dringend")),
    ("owner", "Verantwortlich", OWNERS),
    ("status", "Status", W_TABLE["status"]["de"]),
    ("category", "Kategorie", W_TABLE["cat"]["de"]),
    ("channel", "Kanal", W_TABLE["chan"]["de"]),
)
SC_FIELDS["fr"] = (
    ("priority", "Priorité", ("Basse", "Moyenne", "Haute", "Urgente")),
    ("owner", "Responsable", OWNERS),
    ("status", "Statut", W_TABLE["status"]["fr"]),
    ("category", "Catégorie", W_TABLE["cat"]["fr"]),
    ("channel", "Canal", W_TABLE["chan"]["fr"]),
)

HOURS_LATE = (1, 2, 4)
HOURS_OK = (8, 12, 24, 48)
EXTRA_SENTENCES = {
    "en": ("Reference code ZC-{k}. Reported by the on-call engineer.", "Linked to change request CR-{k}.",
           "Duplicate of ticket {k}. Reported twice this week."),
    "es": ("Código de referencia ZC-{k}. Lo reportó el ingeniero de guardia.", "Vinculado a la petición CR-{k}.",
           "Duplicado del tique {k}. Reportado dos veces esta semana."),
    "de": ("Referenzcode ZC-{k}. Vom Bereitschaftsingenieur gemeldet.", "Verknüpft mit Change Request CR-{k}.",
           "Duplikat von Ticket {k}. Diese Woche zweimal gemeldet."),
    "fr": ("Code de référence ZC-{k}. Signalé par l'ingénieur d'astreinte.", "Lié à la demande CR-{k}.",
           "Doublon du ticket {k}. Signalé deux fois cette semaine."),
}
EXTRA_SENTENCES["fr"] = (
    "Code de référence ZC-{k}. Signalé par l'ingénieur d'astreinte.", "Lié à la demande CR-{k}.",
    "Doublon du ticket {k}. Signalé deux fois cette semaine.",
)

BANNER_MSG = {
    "en": dict(warn=("Scheduled maintenance in 30 minutes.", "Your storage is almost full."),
               error=("The service is temporarily unavailable.", "Export failed: file too large."),
               blocked=("This area is locked for your role.", "Access is blocked until the review ends.")),
    "es": dict(warn=("Mantenimiento programado en 30 minutos.", "Tu almacenamiento está casi lleno."),
               error=("El servicio no está disponible temporalmente.",
                      "Exportación fallida: archivo demasiado grande."),
               blocked=("Esta zona está bloqueada para tu rol.",
                        "El acceso está bloqueado hasta el fin de la revisión.")),
}

BANNER_MSG["de"] = dict(warn=("Geplante Wartung in 30 Minuten.", "Ihr Speicher ist fast voll."),
                        error=("Der Dienst ist vorübergehend nicht verfügbar.",
                               "Export fehlgeschlagen: Datei zu groß."),
                        blocked=("Dieser Bereich ist für Ihre Rolle gesperrt.",
                                 "Zugang blockiert bis zum Ende der Prüfung."))
BANNER_MSG["fr"] = dict(warn=("Maintenance planifiée dans 30 minutes.", "Votre stockage est presque plein."),
                        error=("Le service est temporairement indisponible.",
                               "Échec de l'export : fichier trop volumineux."),
                        blocked=("Cette zone est verrouillée pour votre rôle.",
                                 "Accès bloqué jusqu'à la fin de la revue."))

def fmt_ticket(lang: str, n: int) -> str:
    return f"{W_TABLE['ticket'][lang]} #{n}"


# ---------------------------------------------------------------- content ---
def task_done_content(label: str, rng: random.Random, lang: str) -> dict:
    """Goal drawn first, independent of the label; label sets the goal row mark."""
    tok = rng.choice(TASK_TOKENS)
    goal = rng.choice(GOALS[lang]).format(tok=tok)
    rows = [(tok, label == "true")]
    fillers = [t for t in TASK_TOKENS if t != tok]
    for name in rng.sample(fillers, rng.randint(2, 4)):
        rows.append((name, rng.random() < 0.3))
    rng.shuffle(rows)
    return {"goal": goal, "token": tok, "rows": rows}


def task_done_edit(c: dict, label: str, rng: random.Random, role: str) -> dict:
    rows = list(c["rows"])
    if role == "decisive":
        rows = [(n, label == "true" if n == c["token"] else lab) for n, lab in rows]
    else:
        other = [i for i, (n, _) in enumerate(rows) if n != c["token"]]
        i = rng.choice(other)
        rows[i] = (rows[i][0], not rows[i][1])
    return {**c, "rows": rows}


def _pool(lang: str, fid: str) -> tuple[str, ...]:
    return next(p for f, _, p in SC_FIELDS[lang] if f == fid)


def source_conflict_content(label: str, rng: random.Random, lang: str) -> dict:
    """Same text-value pools for both classes; conflict realized only in the image."""
    ids = rng.sample([f for f, _, _ in SC_FIELDS[lang]], rng.randint(3, 4))
    shown = {fid: rng.choice(_pool(lang, fid)) for fid in ids}
    claimed = rng.sample(ids, rng.randint(2, len(ids)))
    text_vals = {fid: shown[fid] for fid in claimed}
    if label == "conflict":
        fid = rng.choice(claimed)
        shown[fid] = rng.choice([v for v in _pool(lang, fid) if v != text_vals[fid]])
    claims = " ".join(f"{next(n for f, n, _ in SC_FIELDS[lang] if f == fid)}: {text_vals[fid]}." for fid in claimed)
    n = rng.randint(1000, 9999)
    return {"n": n, "claimed": claimed, "text_vals": text_vals, "shown": shown,
            "state": f"{fmt_ticket(lang, n)}. {claims}"}


def sc_edit(c: dict, label: str, rng: random.Random, lang: str, role: str) -> dict:
    shown = dict(c["shown"])
    if role == "decisive":
        if label == "conflict":
            fid = rng.choice(c["claimed"])
            shown[fid] = rng.choice([v for v in _pool(lang, fid) if v != c["text_vals"][fid]])
        else:
            for fid in c["claimed"]:
                shown[fid] = c["text_vals"][fid]
        assert (label == "conflict") == any(shown[f] != c["text_vals"][f] for f in c["claimed"])
    else:
        other = [fid for fid in c["shown"] if fid not in c["claimed"]]
        if other:
            fid = rng.choice(other)
            shown[fid] = rng.choice([v for v in _pool(lang, fid) if v != shown[fid]])
    return {**c, "shown": shown}


UR_DECOMPS = {1: [(0, 0)], 2: [(1, 0), (0, 1)], 3: [(2, 0), (1, 1)], 4: [(3, 0), (2, 1)], 5: [(3, 1)]}


def urgency_content(label: str, rng: random.Random, lang: str) -> dict:
    """U drawn first (balanced upstream); decomposition uniform; text independent of U."""
    u = int(label)
    banner, late = rng.choice(UR_DECOMPS[u])
    hours = rng.choice(HOURS_LATE) if late else rng.choice(HOURS_OK)
    impact = rng.choice(W_TABLE["impact_vals"][lang])
    extra = rng.choice(EXTRA_SENTENCES[lang]).format(k=rng.randint(100, 999))
    n = rng.randint(1000, 9999)
    bmsg = ""
    if banner:
        bmsg = rng.choice(BANNER_MSG[lang][("warn", "error", "blocked")[banner - 1]])
    return {"u": u, "banner": banner, "hours": hours, "impact": impact, "extra": extra, "n": n, "bmsg": bmsg,
            "state": STATE["urgency"][lang].format(n=n, impact=impact, extra=extra)}


def urgency_edit(c: dict, label: str, rng: random.Random, lang: str) -> dict:
    """Decisive: fresh draw at the new label (text redrawn independent of U)."""
    return urgency_content(label, rng, lang)


def urgency_irrelevant(c: dict, rng: random.Random, lang: str) -> dict:
    """Change the impact word and distractor sentence; U, banner, deadline keep."""
    impact = rng.choice([v for v in W_TABLE["impact_vals"][lang] if v != c["impact"]])
    extra = rng.choice(EXTRA_SENTENCES[lang]).format(k=rng.randint(100, 999))
    n = c["n"] + rng.randint(1, 9)
    return {**c, "impact": impact, "extra": extra, "n": n,
            "state": STATE["urgency"][lang].format(n=n, impact=impact, extra=extra)}


def ss_content(label: str, rng: random.Random, lang: str) -> dict:
    msgs = MSGS[lang]
    if label == "success":
        return {"kind": "success", "msg": rng.choice(msgs["success"]),
                "sumline": f"{rng.choice(W_TABLE['tasks'][lang])}: OK", "with_sum": rng.random() < 0.5}
    if label == "error":
        return {"kind": "error", "code": rng.choice((403, 500, 502, 504)), "msg": rng.choice(msgs["error"])}
    if label == "loading":
        return {"kind": "loading", "msg": rng.choice(msgs["loading"]), "n_skel": rng.randint(2, 4)}
    if label == "needs_input":
        labels = rng.sample(msgs["form_labels"], rng.randint(2, 3))
        fields = [(lab, "" if i == 0 or rng.random() < 0.3 else "••••") for i, lab in enumerate(labels)]
        return {"kind": "needs_input", "fields": fields}
    return {"kind": "blocked", "msg": rng.choice(msgs["blocked"])}


# ----------------------------------------------------------------- splits ---
def split_pairs() -> dict[str, tuple[str, ...]]:
    """96 (app, template) pairs, sha256 order: 19 test, 19 heldout, 58 train."""
    pairs = sorted(f"{app}|{TEMPLATE.format(c, l, p)}" for app in APPS for c, l, p in TEMPLATE_SLOTS)
    ordered = sorted(pairs, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    return {"test": tuple(sorted(ordered[:19])), "heldout": tuple(sorted(ordered[19:38])),
            "train": tuple(sorted(ordered[38:]))}


def pairs_for_split(split: str) -> tuple[str, ...]:
    parts = split_pairs()
    return parts["heldout"] if split in HELDOUT_SPLITS else parts[split]


def _template_parts(template: str) -> tuple[int, int, int]:
    m = re.match(r"scr-c(\d)l(\d)p(\d)$", template)
    return (int(m[1]), int(m[2]), int(m[3]))


# -------------------------------------------------------------------- HTML ---
def _e(s) -> str:
    return html.escape(str(s))


def _css(t: Theme) -> str:
    return (f"<style>#page{{width:800px;box-sizing:border-box;background:{t.paper};color:{t.ink};"
            f"font-family:'{t.font}';font-size:{t.base_px}px;line-height:1.45}}"
            f"#page .top{{display:flex;justify-content:space-between;align-items:center;"
            f"background:{t.accent};color:#fff;padding:10px 14px}}"
            f"#page .side{{width:150px;background:#0000000d}}"
            f"#page .sideitem{{padding:6px 10px;font-size:{t.base_px - 2}px;border-bottom:1px solid #00000022}}"
            f"#page .tabs{{display:flex;border-bottom:2px solid {t.accent}}}"
            f"#page .tab{{padding:6px 12px;font-size:{t.base_px - 2}px;background:#00000011}}"
            f"#page .tab.on{{background:{t.accent};color:#fff}}"
            f"#page .cardhead{{background:{t.accent};color:#fff;padding:8px 14px}}"
            f"#page .panel{{padding:{t.pad}px}}"
            f"#page .sbox{{border:2px solid {t.ink};padding:14px;margin:0 auto;max-width:520px;text-align:center}}"
            f"#page .glyph{{display:inline-block;width:34px;height:34px;line-height:34px;border-radius:50%;"
            f"background:{t.accent};color:#fff;font-weight:bold}}"
            f"#page .ring{{width:38px;height:38px;margin:6px auto;border:5px solid #00000022;"
            f"border-top-color:{t.accent};border-radius:50%}}"
            f"#page .skel{{height:12px;margin:8px auto;max-width:420px;background:#00000018}}"
            f"#page .inrow{{display:flex;gap:10px;margin:8px 0;text-align:left}}"
            f"#page .fl{{width:200px;font-weight:bold}}"
            f"#page .inbox{{flex:1;border:1px solid {t.ink};min-height:1.6em;padding:2px 6px;text-align:left}}"
            f"#page .btn{{display:inline-block;margin-top:10px;padding:6px 16px;background:{t.accent};color:#fff}}"
            f"#page .taskitem{{display:flex;justify-content:space-between;align-items:center;"
            f"padding:7px 10px;border-bottom:1px solid #00000022;text-align:left}}"
            f"#page .mark{{width:22px;height:22px;line-height:22px;text-align:center;border:1.5px solid {t.ink};"
            f"font-weight:bold}}"
            f"#page .frow{{display:flex;gap:10px;margin:8px 0;text-align:left}}"
            f"#page .frow .k{{width:170px;font-weight:bold}}"
            f"#page .frow .v{{flex:1;border-bottom:1px solid #00000033}}"
            f"#page .banner{{padding:9px 14px;margin:0 0 12px;font-weight:bold}}"
            f"#page .warn{{background:#b98a2a22;border:2px solid #b98a2a}}"
            f"#page .errb{{background:#a33c3c22;border:2px solid #a33c3c}}"
            f"#page .blkb{{background:#44444418;border:2px solid {t.ink}}}"
            f"#page .badge{{display:inline-block;padding:3px 10px;background:{t.ink};color:#fff;"
            f"font-size:{t.base_px - 3}px;letter-spacing:1px}}"
            f"#page h2{{font-size:{t.base_px + 6}px;margin:0 0 12px 0}}"
            "</style>")

def _chrome(app: str, lang: str, c: int, t: Theme) -> str:
    w = W_TABLE
    nav = "".join(f'<span class="sideitem">{_e(x)}</span>' for x in w["nav"][lang])
    top = f'<div class="top"><b>{_e(app)}</b><span>{_e(" · ".join(w["nav"][lang]))}</span></div>'
    if c == 0:
        return top + '<div class="panel">'
    if c == 1:
        return (f'<div style="display:flex"><div class="side"><div style="padding:10px;font-weight:bold">'
                f'{_e(app)}</div>{nav}</div><div class="panel" style="flex:1">')
    if c == 2:
        tabs = "".join(f'<span class="tab{" on" if i == 0 else ""}">{_e(x)}</span>'
                       for i, x in enumerate(w["nav"][lang]))
        return f'<div class="top"><b>{_e(app)}</b></div>{tabs_div(tabs)}' + '<div class="panel">'
    return (f'<div style="padding:{t.pad}px"><div class="cardhead"><b>{_e(app)}</b></div>'
            f'<div class="panel" style="padding:14px">')


def tabs_div(tabs: str) -> str:
    return f'<div class="tabs">{tabs}</div>'


def _chrome_close(c: int) -> str:
    if c in (1, 3):
        return "</div></div>"
    return "</div>"

def ss_panel(c: dict, lang: str, t: Theme) -> str:
    kind = c["kind"]
    if kind == "success":
        inner = f'<span class="glyph" data-k="scheck">✓</span><p data-k="msg">{_e(c["msg"])}</p>'
        if c["with_sum"]:
            inner += f'<p data-k="sumline" style="font-size:{t.base_px - 3}px">{_e(c["sumline"])}</p>'
        box = f'<div class="sbox" data-k="success">{inner}</div>'
    elif kind == "error":
        box = (f'<div class="sbox" data-k="error"><span class="badge" data-k="ecode">Error {_e(c["code"])}</span>'
               f'<p data-k="msg">{_e(c["msg"])}</p><span class="btn" data-k="retry">{_e(W_TABLE["btn_retry"][lang])}</span></div>')
    elif kind == "loading":
        skel = "".join('<div class="skel" data-k="skel"></div>' for _ in range(c["n_skel"]))
        box = (f'<div class="sbox" data-k="loading"><div class="ring"></div>'
               f'<p data-k="msg">{_e(c["msg"])}</p>{skel}</div>')
    elif kind == "needs_input":
        fields = "".join(f'<div class="inrow"><span class="fl">{_e(lab)}</span>'
                         f'<span class="inbox" data-k="fin">{_e(v)}</span></div>' for lab, v in c["fields"])
        box = (f'<div data-k="form"><h2>{_e(W_TABLE["record"][lang])}</h2>{fields}'
               f'<span class="btn" data-k="fsub">{_e(W_TABLE["btn_submit"][lang])}</span></div>')
    else:
        box = (f'<div class="sbox blkb" data-k="blocked"><span class="badge" data-k="lockglyph">'
               f'{_e(W_TABLE["lock"][lang])}</span><p data-k="msg">{_e(c["msg"])}</p></div>')
    return box


def _page(template: str, app: str, lang: str, t: Theme, panel: str, banner: str = "") -> str:
    c, _, _ = _template_parts(template)
    inner = banner + panel
    return (_css(t) + f'<div id="page" data-template="{_e(template)}">'
            + _chrome(app, lang, c, t) + inner + _chrome_close(c) + "</div>")

def td_panel(c: dict, lang: str, t: Theme) -> str:
    rows = []
    for name, done in c["rows"]:
        mark = ('<span class="mark" data-k="done-mark">✓</span>' if done
                else '<span class="mark" data-k="todo-mark">○</span>')
        rows.append(f'<div class="taskitem" data-k="taskitem"><span data-k="tname">{_e(name)}</span>{mark}</div>')
    head = f'<h2>{_e(W_TABLE["tasklist"][lang])}</h2>'
    return head + "".join(rows)


def sc_panel(c: dict, lang: str, t: Theme) -> str:
    rows = []
    for fid, name, _ in SC_FIELDS[lang]:
        if fid in c["shown"]:
            rows.append(f'<div class="frow" data-k="field"><span class="k" data-k="flabel">{_e(name)}</span>'
                        f'<span class="v" data-k="sv" data-f="{_e(fid)}">{_e(c["shown"][fid])}</span></div>')
    head = f'<h2>{_e(fmt_ticket(lang, c["n"]))}</h2>'
    return head + "".join(rows)

def _banner_html(c: dict, lang: str) -> str:
    if c["banner"] == 0:
        return ""
    if c["banner"] == 1:
        return f'<div class="banner warn" data-k="warnbanner">{_e(c["bmsg"])}</div>'
    if c["banner"] == 2:
        return (f'<div class="banner errb" data-k="error"><span class="badge" data-k="ecode">Error 503</span>'
                f'{_e(c["bmsg"])}</div>')
    return (f'<div class="banner blkb" data-k="blocked"><span class="badge" data-k="lockglyph">'
            f'{_e(W_TABLE["lock"][lang])}</span>{_e(c["bmsg"])}</div>')


def ur_panel(c: dict, lang: str, t: Theme) -> str:
    head = f'<h2>{_e(fmt_ticket(lang, c["n"]))}</h2>'
    rows = (
        f'<div class="frow" data-k="field"><span class="k" data-k="flabel">{_e(W_TABLE["impact"][lang])}</span>'
        f'<span class="v" data-k="sv" data-f="impact">{_e(c["impact"])}</span></div>'
        f'<div class="frow" data-k="field"><span class="k" data-k="flabel">{_e(W_TABLE["deadline"][lang])}</span>'
        f'<span class="v" data-k="deadline">{_e(c["hours"])} h</span></div>'
    )
    return head + rows

# --------------------------------------------------------------- planning ---
@dataclass
class GenCase:
    case: DecisionCase
    family: str
    template: str
    app: str
    theme_seed: int
    role: str
    base_id: str
    body_html: str
    meta: dict


def _plan_groups(family: str, n: int, rng: random.Random) -> list[list[tuple[str, str]]]:
    """Balanced labels (±1) grouped into episodes: single, decisive pair (label
    flips) or irrelevant pair (label kept). Counts are preserved."""
    classes = CLASSES[family]
    labels = [classes[i % len(classes)] for i in range(n)]
    rng.shuffle(labels)
    p_dec, p_irr = (0.0, 0.2) if family == "screen_state" else (0.25, 0.15)
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

def _options(family: str, lang: str) -> tuple[Option, ...]:
    return tuple(Option(id=c, description=d)
                 for c, d in zip(CLASSES[family], OPTION_TEXT[family][lang]))


PANEL = {"screen_state": ss_panel, "task_done": td_panel, "source_conflict": sc_panel, "urgency": ur_panel}
BANNER_FAM = ("urgency",)


def generate_split(family: str, split: str, lang: str, seed: int = 38) -> list[GenCase]:
    n = COUNTS[split]
    if lang == "fr" and split != "test":
        raise ValueError("FR is evaluation-only (test)")
    rng = random.Random(stable_seed(38, family, split, lang))
    pairs = pairs_for_split(split)
    groups = _plan_groups(family, n, rng)
    out: list[GenCase] = []
    for g, roles in enumerate(groups):
        grng = random.Random(stable_seed(38, family, split, lang, g))
        app, template = rng.choice(pairs).split("|")
        gid = f"e38b-{family}-{split}-{lang}-{g:05d}"
        theme_seed = stable_seed(gid, "theme")
        base_id = f"{gid}-base"
        state = None
        for role, label in roles:
            tseed = stable_seed(gid, "theme-irr") if role == "irrelevant" else theme_seed
            theme = make_theme(tseed)
            question = QUESTION[family][lang]
            meta: dict = {}
            if family == "screen_state":
                if state is None:
                    state = ss_content(label, grng, lang)
                question_state = STATE["screen_state"][lang]
                body = _page(template, app, lang, theme, ss_panel(state, lang, theme))
            elif family == "task_done":
                if state is None:
                    state = task_done_content(label, grng, lang)
                else:
                    state = task_done_edit(state, label, grng, role)
                goal_row = next(lab for name, lab in state["rows"] if name == state["token"])
                assert goal_row == (label == "true")
                question_state = STATE["task_done"][lang].format(goal=state["goal"])
                body = _page(template, app, lang, theme, td_panel(state, lang, theme))
            elif family == "source_conflict":
                if state is None:
                    state = source_conflict_content(label, grng, lang)
                elif role == "decisive":
                    state = sc_edit(state, label, grng, lang, role)
                else:
                    state = sc_edit(state, label, grng, lang, role)
                body = _page(template, app, lang, theme, sc_panel(state, lang, theme))
                question_state = state["state"]
            else:  # urgency
                if state is None:
                    state = urgency_content(label, grng, lang)
                elif role == "decisive":
                    state = urgency_edit(state, label, grng, lang)
                else:
                    state = urgency_irrelevant(state, grng, lang)
                body = _page(template, app, lang, theme, ur_panel(state, lang, theme),
                             banner=_banner_html(state, lang))
                question_state = state["state"]
            assert_clean(question_state, question)
            cid = f"{gid}-{role}"
            case = DecisionCase(id=cid, group_id=gid, source="e38b-synthetic", language=lang,
                                domain="screens", state=question_state, question=question,
                                options=_options(family, lang), target_kind="single",
                                target_option=label, label_origin="e38b-content-model", split=split)
            out.append(GenCase(case, family, template, app, tseed, role, base_id, body, meta))
    return out

# __APP__

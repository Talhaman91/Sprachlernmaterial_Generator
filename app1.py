"""
Sprachlern-Framework – Streamlit POC (Single File)
- Modus: Alpha 3/4/5/6 oder Ohne Alpha
- Textlänge (Wörter) nur bei Ohne Alpha (0 = unbegrenzt)
- Alpha-Parameter sind gesperrt, wenn Alpha-Modus aktiv ist
- Ohne Alpha: Parameter frei (0 = aus)
- User Prompt:
  - Alpha-Modus: nennt Alpha-Level + verweist auf Algorithmus als RAG-Kontext
  - Ohne Alpha: nennt keinen Alpha-Level, nur Parameter (nur aktive Constraints)
- Validierung: STANZA-only (kein CoreNLP Server)
  Prüft: Max Sätze, Max Wörter/Satz, Max Silben/Token (pyphen),
         Max Dep.-Nebensätze/Satz (UD deps),
         Verbotene Tempora (regelbasiert),
         Max Perfekt/finite Verben (regelbasiert)
  NICHT geprüft: min lexikalische Abdeckung (Parameter bleibt, aber Check ist n/a)
- Zusatzparameter (sekundär) enthalten, ohne die bestehenden Alpha-Mechaniken zu ändern
- Tooltips (help=...) an allen Parametern
- Copy-Funktion entfernt

Install:
  pip install streamlit google-genai stanza pyphen

Erst-Setup (einmalig):
  python -c "import stanza; stanza.download('de')"

Run:
  python -m streamlit run app1.py
"""

from dataclasses import dataclass
from typing import Optional

import streamlit as st

# =========================
# LLM Fix-Konfiguration
# =========================
GEMINI_MODEL = "gemini-3-flash-preview"
#GEMINI_MODEL = "gemini-2.5-flash"

TENSES_ALL = ["Präsens", "Präteritum", "Perfekt", "Plusquamperfekt", "Futur I", "Futur II"]
TEXT_TYPES = ["Erzählung", "Sachtext", "Dialog", "E-Mail", "Nachrichtenberichterstattung"]

# Zusatzparameter: Nebensatz-Typen (vereinfachte Liste)
SUBCLAUSE_TYPES = ["Relativsatz", "Kausalsatz", "Temporalsatz", "Konditionalsatz", "Konzessivsatz", "Finalsatz",
                   "Objektsatz", "Subjektsatz", ]

# Zusatzparameter: Stufen
LEVELS_4 = ["keine Vorgabe", "niedrig", "mittel", "hoch", "sehr hoch"]

# Interne Richtwerte (nur für Prompt)
MTUL_BANDS = {"niedrig": "≤ 8 Wörter pro T-Unit", "mittel": "9–12 Wörter pro T-Unit", "hoch": "13–18 Wörter pro T-Unit",
              "sehr hoch": "> 18 Wörter pro T-Unit", }

ZIPF_BANDS = {"niedrig": "sehr häufige Wörter (Zipf grob ≥ 5.5)", "mittel": "alltagsnah (Zipf grob ≥ 5.0)",
              "hoch": "differenzierter Wortschatz erlaubt (Zipf grob ≥ 4.5)", "sehr hoch": "keine Einschränkung", }

LEXVAR_BANDS = {"niedrig": "geringe lexikalische Vielfalt, mehr Wiederholung", "mittel": "ausgewogene Vielfalt",
                "hoch": "hohe Vielfalt erlaubt", "sehr hoch": "keine Einschränkung", }

CONNECTOR_BANDS = {"niedrig": "wenige Konnektoren (z. B. 0–3)", "mittel": "moderate Konnektoren (z. B. 4–8)",
                   "hoch": "viele Konnektoren erlaubt (z. B. 9–15)", "sehr hoch": "keine Einschränkung", }


# =========================
# Helper
# =========================
def optional_float_or_none(value) -> float | None:
    """
    - None -> None
    - 0.0 -> None (Regel deaktiviert)
    - sonst -> float
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if v == 0.0 else v


def clamp_level(value: str) -> str:
    return value if value in LEVELS_4 else "keine Vorgabe"


# =========================
# Datenmodelle
# =========================
@dataclass
class GeneralParams:
    topic: str
    text_type: str
    target_words: int | None  # nur bei Ohne Alpha; None = keine Begrenzung


@dataclass
class AlphaParams:
    mode: str  # "Alpha 3" | "Alpha 4" | "Alpha 5" | "Alpha 6" | "Ohne Alpha"

    max_sentences: int  # 0 = aus (nur Ohne Alpha)
    max_words_per_sentence: int  # 0 = aus (nur Ohne Alpha)
    max_syllables_per_token: int  # 0 = aus (nur Ohne Alpha)
    max_dep_clauses_per_sentence: float  # 0.0 = aus (nur Ohne Alpha)

    forbidden_tenses: list[str]
    max_perfekt_per_finite_verb: float | None
    min_lexical_coverage: float | None  # bleibt als Parameter, wird aber aktuell nicht validiert


@dataclass
class FineParams:
    enabled: bool
    mtul_level: str
    zipf_level: str
    lexvar_level: str
    connectors_level: str
    forbidden_subclause_types: list[str]
    konjunktiv_mode: str  # "keine Vorgabe" | "erlauben" | "vermeiden"
    coherence_hint: str  # "keine" | "hoch" | "mittel" | "niedrig"


@dataclass
class Params:
    general: GeneralParams
    alpha: AlphaParams
    fine: FineParams


# =========================
# Alpha Presets
# =========================
ALPHA_PRESETS = {"Alpha 3": dict(max_sentences=5, max_words_per_sentence=10, max_syllables_per_token=3,
                                 max_dep_clauses_per_sentence=0.5,
                                 forbidden_tenses=["Plusquamperfekt", "Futur I", "Futur II"],
                                 max_perfekt_per_finite_verb=0.5, min_lexical_coverage=0.95, ),
                 "Alpha 4": dict(max_sentences=10, max_words_per_sentence=10, max_syllables_per_token=5,
                                 max_dep_clauses_per_sentence=1.0,
                                 forbidden_tenses=["Plusquamperfekt", "Futur I", "Futur II"],
                                 max_perfekt_per_finite_verb=None, min_lexical_coverage=None, ),
                 "Alpha 5": dict(max_sentences=15, max_words_per_sentence=12, max_syllables_per_token=5,
                                 max_dep_clauses_per_sentence=1.5, forbidden_tenses=["Plusquamperfekt"],
                                 max_perfekt_per_finite_verb=None, min_lexical_coverage=None, ),
                 "Alpha 6": dict(max_sentences=20, max_words_per_sentence=12, max_syllables_per_token=6,
                                 max_dep_clauses_per_sentence=2.0, forbidden_tenses=[],
                                 max_perfekt_per_finite_verb=None, min_lexical_coverage=None, ), }

FREE_DEFAULTS = dict(max_sentences=8, max_words_per_sentence=18, max_syllables_per_token=6,
                     max_dep_clauses_per_sentence=2.0, forbidden_tenses=[], max_perfekt_per_finite_verb=None,
                     min_lexical_coverage=None, )


# =========================
# Session State Init / Preset Anwendung
# =========================
def ensure_defaults_exist() -> None:
    st.session_state.setdefault("mode", "Alpha 4")
    st.session_state.setdefault("topic", "Alltag")
    st.session_state.setdefault("text_type", "Sachtext")
    st.session_state.setdefault("target_words", 140)  # nur Ohne Alpha

    st.session_state.setdefault("alpha_max_sentences", FREE_DEFAULTS["max_sentences"])
    st.session_state.setdefault("alpha_max_words_per_sentence", FREE_DEFAULTS["max_words_per_sentence"])
    st.session_state.setdefault("alpha_max_syllables_per_token", FREE_DEFAULTS["max_syllables_per_token"])
    st.session_state.setdefault("alpha_max_dep_clauses_per_sentence", FREE_DEFAULTS["max_dep_clauses_per_sentence"])
    st.session_state.setdefault("alpha_forbidden_tenses", FREE_DEFAULTS["forbidden_tenses"])

    st.session_state.setdefault("alpha_max_perfekt_per_finite_verb", 0.0)
    st.session_state.setdefault("alpha_min_lexical_coverage", 0.0)  # bleibt als UI-Parameter (ohne Upload)

    st.session_state.setdefault("fine_enabled", False)
    st.session_state.setdefault("fine_mtul_level", "keine Vorgabe")
    st.session_state.setdefault("fine_zipf_level", "keine Vorgabe")
    st.session_state.setdefault("fine_lexvar_level", "keine Vorgabe")
    st.session_state.setdefault("fine_connectors_level", "keine Vorgabe")
    st.session_state.setdefault("fine_forbidden_subclause_types", [])
    st.session_state.setdefault("fine_konjunktiv_mode", "keine Vorgabe")
    st.session_state.setdefault("fine_coherence_hint", "keine")


def apply_preset_if_alpha(mode: str) -> None:
    st.session_state["mode"] = mode
    if mode in ALPHA_PRESETS:
        preset = ALPHA_PRESETS[mode]
        st.session_state["alpha_max_sentences"] = preset["max_sentences"]
        st.session_state["alpha_max_words_per_sentence"] = preset["max_words_per_sentence"]
        st.session_state["alpha_max_syllables_per_token"] = preset["max_syllables_per_token"]
        st.session_state["alpha_max_dep_clauses_per_sentence"] = preset["max_dep_clauses_per_sentence"]
        st.session_state["alpha_forbidden_tenses"] = preset["forbidden_tenses"]

        st.session_state["alpha_max_perfekt_per_finite_verb"] = (
            0.0 if preset["max_perfekt_per_finite_verb"] is None else float(preset["max_perfekt_per_finite_verb"]))
        st.session_state["alpha_min_lexical_coverage"] = (
            0.0 if preset["min_lexical_coverage"] is None else float(preset["min_lexical_coverage"]))


def on_mode_change() -> None:
    apply_preset_if_alpha(st.session_state["mode"])


# =========================
# Prompts
# =========================
def build_system_prompt() -> str:
    return """Du bist ein erfahrener L2-Lehrer und Sprachexperte für Deutsch. Du erstellst Sprachlernmaterialien in Form zusammenhängender Texte.

Regeln:
- Gib ausschließlich den fertigen Text aus.
- Keine Überschrift, keine Bulletpoints, keine Erklärungen, keine Metakommentare.
- Halte dich an die Vorgaben im User Prompt so gut wie möglich.
- Authentizität: Schreibe natürlich, alltagsnah und plausibel. Die Inhalte sollen realistisch wirken, so als wären sie von Muttersprachlern für Muttersprachler geschrieben.
- Vermeide erfundene, spezifische Fakten (z. B. konkrete Statistiken, Studien, offizielle Zahlen, reale Adressen), außer sie sind für die Aufgabe notwendig oder wurden vom Nutzer vorgegeben.
- Wenn Vorgaben widersprüchlich sind, löse sie sinnvoll auf: Priorisiere Verständlichkeit, Natürlichkeit und realistische Inhalte.
"""


def rag_context_for_alpha(mode: str) -> str:
    if mode == "Alpha 3":
        return ("Retrieved Context (Alpha Readability Level Algorithmus, Regeln):\n"
                "- wordsPerSentence <= 10\n"
                "- nSentences <= 5\n"
                "- syllablesPerToken <= 3\n"
                "- pastPerfectsPerFiniteVerb == 0\n"
                "- future1sPerFiniteVerb == 0\n"
                "- future2sPerFiniteVerb == 0\n"
                "- depClausesPerSentence <= 0.5\n"
                "- presentPerfectsPerFiniteVerb <= 0.5\n"
                "- typesFoundInSubtlexPerLexicalType >= 0.95\n")
    if mode == "Alpha 4":
        return ("Retrieved Context (Alpha Readability Level Algorithmus, Regeln):\n"
                "- wordsPerSentence <= 10\n"
                "- nSentences <= 10\n"
                "- syllablesPerToken <= 5\n"
                "- pastPerfectsPerFiniteVerb == 0\n"
                "- future1sPerFiniteVerb == 0\n"
                "- future2sPerFiniteVerb == 0\n")
    if mode == "Alpha 5":
        return ("Retrieved Context (Alpha Readability Level Algorithmus, Regeln):\n"
                "- wordsPerSentence <= 12\n"
                "- nSentences <= 15\n"
                "- pastPerfectsPerFiniteVerb == 0\n")
    if mode == "Alpha 6":
        return ("Retrieved Context (Alpha Readability Level Algorithmus, Regeln):\n"
                "- wordsPerSentence <= 12\n"
                "- nSentences <= 20\n")
    return ""


def rag_context_for_fine_params() -> str:
    return ("Retrieved Context (Zusatzparameter-Glossar):\n"
            "| Parameter | Bedeutung | Intention |\n"
            "|---|---|---|\n"
            "| MTUL | mittlere Wörter pro T-Unit | kürzere syntaktische Einheiten |\n"
            "| LexVar | lexikalische Vielfalt (TTR/MTLD zusammengefasst) | niedrig = mehr Wiederholung |\n"
            "| Zipf | Wortfrequenz (1–7) | höher = häufigere Wörter |\n"
            "| Konnektoren | z.B. weil, aber, obwohl | weniger explizite Verknüpfungen |\n"
            "| Modus | Konjunktiv (I/II) | vermeiden = morphologisch einfacher |\n"
            "| Nebensatzarten | Relativsatz, Kausalsatz, ... | bestimmte Strukturen vermeiden |\n"
            "| Kohärenz | logischer Zusammenhang | klar nachvollziehbare Bezüge |\n")


def fine_params_to_prompt_lines(f: FineParams) -> list[str]:
    if not f.enabled:
        return []
    lines: list[str] = []

    if f.mtul_level != "keine Vorgabe":
        band = MTUL_BANDS.get(f.mtul_level, f.mtul_level)
        lines.append(f"- MTUL: {f.mtul_level}. Richtwert: {band}. (bestmöglich)")

    if f.forbidden_subclause_types:
        lines.append(f"- Vermeide folgende Nebensatzarten: {', '.join(f.forbidden_subclause_types)}.")

    if f.zipf_level != "keine Vorgabe":
        band = ZIPF_BANDS.get(f.zipf_level, f.zipf_level)
        lines.append(f"- Lexik/Wortfrequenz (Zipf): {f.zipf_level}. Ziel: {band}. (bestmöglich)")

    if f.lexvar_level != "keine Vorgabe":
        band = LEXVAR_BANDS.get(f.lexvar_level, f.lexvar_level)
        lines.append(f"- Lexikalische Vielfalt: {f.lexvar_level}. Ziel: {band}. (bestmöglich)")

    if f.konjunktiv_mode == "erlauben":
        lines.append("- Konjunktiv I/II: erlaubt (wenn passend).")
    elif f.konjunktiv_mode == "vermeiden":
        lines.append("- Konjunktiv I/II: vermeiden.")

    if f.connectors_level != "keine Vorgabe":
        band = CONNECTOR_BANDS.get(f.connectors_level, f.connectors_level)
        lines.append(f"- Konnektoren: {f.connectors_level}. Richtwert: {band}. (bestmöglich)")

    if f.coherence_hint and f.coherence_hint != "keine":
        lines.append(f"- Kohärenz: {f.coherence_hint} (logisch gut nachvollziehbar, klare Bezüge).")

    return lines


def build_user_prompt(p: Params) -> str:
    g = p.general
    a = p.alpha
    f = p.fine

    forbidden = ", ".join(a.forbidden_tenses) if a.forbidden_tenses else "keine"
    fine_lines = fine_params_to_prompt_lines(f)
    fine_ctx = rag_context_for_fine_params() if fine_lines else ""

    if a.mode == "Ohne Alpha":
        lines: list[str] = [f'Thema: "{g.topic}".', f"Textart: {g.text_type}."]

        if g.target_words is not None:
            lines.append(f"Textlänge: ca. {g.target_words} Wörter.")

        lines += ["", "Parameter (bitte so gut wie möglich einhalten):"]

        if a.max_sentences > 0:
            lines.append(f"- Anzahl Sätze: höchstens {a.max_sentences}.")
        if a.max_words_per_sentence > 0:
            lines.append(f"- Wörter pro Satz: höchstens {a.max_words_per_sentence}.")
        if a.max_syllables_per_token > 0:
            lines.append(f"- Silben pro Token: höchstens {a.max_syllables_per_token}.")
        if a.max_dep_clauses_per_sentence > 0.0:
            lines.append(f"- Dependenznebensätze pro Satz: höchstens {a.max_dep_clauses_per_sentence}.")
        lines.append(f"- Verbotene Tempora: {forbidden}.")

        if a.max_perfekt_per_finite_verb is not None:
            lines.append(f"- Perfekt pro finitem Verb: höchstens {a.max_perfekt_per_finite_verb}.")
        if a.min_lexical_coverage is not None:
            lines.append(
                f"- Lexik: Ziel-Abdeckung mindestens {a.min_lexical_coverage}. (Hinweis: später extern geprüft)")

        if fine_ctx:
            lines += ["", "Beziehe dich für die folgenden Zusatzparameter auf das Glossar (Retrieved Context):", "",
                      fine_ctx.strip()]
        if fine_lines:
            lines += ["", "Zusatzparameter (sekundär, ändern kein Alpha-Level):", *fine_lines]

        lines += ["", "Konsistenzregel:",
                  "- Falls Textlänge und Satzanzahl nicht zusammenpassen, wähle eine sinnvolle Satzanzahl, die einen natürlichen Text ermöglicht.",
                  "", "Stil:", "- Klar, alltagsnah, keine Schachtelsätze.", "- Keine Metakommentare.", ]
        return "\n".join(lines)

    rag_ctx = rag_context_for_alpha(a.mode)
    lines = [f"ALPHA-LEVEL: {a.mode}",
             "Beziehe dich auf den folgenden Algorithmus-Kontext (RAG) und halte die Parameter so gut wie möglich ein.",
             "", rag_ctx.strip(), "", f'Thema: "{g.topic}".', f"Textart: {g.text_type}.", "",
             "Parameter (so gut wie möglich einhalten):", f"- Anzahl Sätze: höchstens {a.max_sentences}.",
             f"- Wörter pro Satz: höchstens {a.max_words_per_sentence}.",
             f"- Silben pro Token: höchstens {a.max_syllables_per_token}.",
             f"- Dependenznebensätze pro Satz: höchstens {a.max_dep_clauses_per_sentence}.",
             f"- Verbotene Tempora: {forbidden}.", ]

    if a.max_perfekt_per_finite_verb is not None:
        lines.append(f"- Perfekt pro finitem Verb: höchstens {a.max_perfekt_per_finite_verb}.")
    if a.min_lexical_coverage is not None:
        lines.append(f"- Lexik: Ziel-Abdeckung mindestens {a.min_lexical_coverage}. (Hinweis: später extern geprüft)")

    if fine_ctx:
        lines += ["", "Beziehe dich für die folgenden Zusatzparameter auf das Glossar (Retrieved Context):", "",
                  fine_ctx.strip()]
    if fine_lines:
        lines += ["", "Zusatzparameter (sekundär, ändern dieses Alpha-Level nicht):", *fine_lines]

    lines += ["", "Stil:", "- Klar, alltagsnah, keine Schachtelsätze.", "- Keine Metakommentare."]
    return "\n".join(lines)


# =========================
# Gemini Call
# =========================
def gemini_generate(api_key: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
    if not api_key.strip():
        raise RuntimeError("Kein Gemini API Key gesetzt.")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    cfg = types.GenerateContentConfig(system_instruction=system_prompt, temperature=float(temperature))
    resp = client.models.generate_content(model=GEMINI_MODEL, contents=user_prompt, config=cfg)

    text = getattr(resp, "text", None)
    if not text:
        raise RuntimeError("Gemini hat keinen Text geliefert.")
    return text.strip()


# =========================
# Validation / Analyse (Stanza-only)
# =========================
import pyphen
import stanza

_DIC = pyphen.Pyphen(lang="de_DE")

UD_SUBORD_REL_PREFIXES = ("advcl",  # adverbial clause
                          "ccomp",  # clausal complement
                          "csubj",  # clausal subject
                          "acl:relcl",  # relative clause
                          )


def _syllables_pyphen(word: str) -> int:
    w = (word or "").strip()
    if not w:
        return 0
    hyph = _DIC.inserted(w)
    parts = [p for p in hyph.split("-") if p]
    return max(1, len(parts))


def _ufeats_get(word, key: str) -> Optional[str]:
    u = getattr(word, "feats", None)  # stanza uses .feats
    if not u:
        return None
    # feats is a string like "Case=Nom|Gender=Masc|..."
    for feat in str(u).split("|"):
        if "=" in feat:
            k, v = feat.split("=", 1)
            if k == key:
                return v
    return None


def _is_finite(word) -> bool:
    return _ufeats_get(word, "VerbForm") == "Fin" and getattr(word, "upos", None) in ("VERB", "AUX")


def _is_participle(word) -> bool:
    return _ufeats_get(word, "VerbForm") == "Part" and getattr(word, "upos", None) in ("VERB", "AUX")


def _is_infinitive(word) -> bool:
    return _ufeats_get(word, "VerbForm") == "Inf" and getattr(word, "upos", None) in ("VERB", "AUX")


def _lemma(word) -> str:
    return (getattr(word, "lemma", "") or "").lower()


def _dep_rel(word) -> str:
    return (getattr(word, "deprel", "") or "")


def _head_id(word) -> int:
    return int(getattr(word, "head", 0) or 0)


def _token_is_word(word) -> bool:
    return getattr(word, "upos", None) != "PUNCT"


@dataclass
class TenseCounts:
    praesens: int = 0
    praeteritum: int = 0
    perfekt: int = 0
    plusquamperfekt: int = 0
    futur1: int = 0
    futur2: int = 0


@st.cache_resource
def get_stanza_nlp():
    # assumes user ran stanza.download('de') once; if not, raise a helpful error
    try:
        return stanza.Pipeline("de", processors="tokenize,pos,lemma,depparse", tokenize_no_ssplit=False, use_gpu=False,
                               verbose=False, )
    except Exception as e:
        raise RuntimeError("Stanza Pipeline konnte nicht gestartet werden. "
                           "Hast du vorher 'stanza.download(\"de\")' ausgeführt?\n"
                           f"Originalfehler: {e}")


def _detect_tenses_for_sentence(words) -> tuple[TenseCounts, int, int]:
    """
    Rule-based detection on lemma+feats+dependency.
    Returns:
      - tense counts (by detected constructions/finite verbs)
      - finite verbs count
      - perfekt constructions count (for ratio)
    """
    counts = TenseCounts()
    finite_verbs = [w for w in words if _is_finite(w)]
    finite_count = len(finite_verbs)

    # Build lookup: word ids are 1..n
    by_id = {w.id: w for w in words if getattr(w, "id", None) is not None}
    children = {w.id: [] for w in words if getattr(w, "id", None) is not None}
    for w in words:
        hid = _head_id(w)
        if hid in children and getattr(w, "id", None) is not None:
            children[hid].append(w.id)

    # Perfekt / Plusquamperfekt: finite AUX (haben/sein) + participle in subtree
    perfekt_constructions = 0
    for w in finite_verbs:
        if getattr(w, "upos", None) != "AUX":
            continue
        lem = _lemma(w)
        if lem not in ("haben", "sein"):
            continue
        tense = _ufeats_get(w, "Tense")  # Pres/Past
        sid = w.id

        stack = list(children.get(sid, []))
        seen = set()
        found_part = False
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            ww = by_id.get(nid)
            if ww and _is_participle(ww):
                found_part = True
                break
            stack.extend(children.get(nid, []))

        if found_part:
            perfekt_constructions += 1
            if tense == "Past":
                counts.plusquamperfekt += 1
            else:
                counts.perfekt += 1

    # Futur I / II: finite AUX werden + infinitive; Futur II adds participle + aux infinitive
    for w in finite_verbs:
        if getattr(w, "upos", None) != "AUX":
            continue
        if _lemma(w) != "werden":
            continue
        sid = w.id
        stack = list(children.get(sid, []))
        seen = set()
        subtree = []
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            ww = by_id.get(nid)
            if ww:
                subtree.append(ww)
            stack.extend(children.get(nid, []))

        has_inf = any(_is_infinitive(x) for x in subtree)
        has_part = any(_is_participle(x) for x in subtree)
        has_aux_inf = any(_is_infinitive(x) and _lemma(x) in ("haben", "sein") for x in subtree)

        if has_part and has_aux_inf:
            counts.futur2 += 1
        elif has_inf:
            counts.futur1 += 1

    # Präsens / Präteritum: finite main verbs by Tense feature, excluding auxiliaries
    for w in finite_verbs:
        upos = getattr(w, "upos", None)
        lem = _lemma(w)
        tense = _ufeats_get(w, "Tense")
        if upos == "AUX" and lem in ("haben", "sein", "werden"):
            continue
        if tense == "Past":
            counts.praeteritum += 1
        elif tense == "Pres":
            counts.praesens += 1

    return counts, finite_count, perfekt_constructions


def _words_from_doc(doc) -> list[str]:
    toks: list[str] = []
    for sent in doc.sentences:
        for w in sent.words:
            if _token_is_word(w):
                t = (getattr(w, "text", "") or "").strip()
                if t:
                    toks.append(t)
    return toks


def _ttr(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    low = [t.lower() for t in tokens]
    return len(set(low)) / len(low)


def _mtld(tokens: list[str], ttr_threshold: float = 0.72) -> float:
    low = [t.lower() for t in tokens if t.strip()]
    if len(low) < 10:
        return 0.0

    def mtld_one_pass(seq: list[str]) -> float:
        factors = 0
        types = set()
        tok_count = 0
        for tok in seq:
            tok_count += 1
            types.add(tok)
            cur_ttr = len(types) / tok_count
            if cur_ttr <= ttr_threshold:
                factors += 1
                types = set()
                tok_count = 0
        if tok_count == 0:
            return float(factors) if factors > 0 else 0.0
        cur_ttr = len(types) / tok_count
        partial = (1 - cur_ttr) / (1 - ttr_threshold) if (1 - ttr_threshold) != 0 else 0.0
        return factors + partial

    fwd = mtld_one_pass(low)
    bwd = mtld_one_pass(list(reversed(low)))
    denom = (fwd + bwd) / 2.0
    if denom <= 0:
        return 0.0
    return len(low) / denom


def _zipf_avg(tokens: list[str]) -> float | None:
    try:
        from wordfreq import zipf_frequency  # type: ignore
    except Exception:
        return None
    low = [t.lower() for t in tokens if t.strip()]
    if not low:
        return None
    vals = [zipf_frequency(w, "de") for w in low]
    return sum(vals) / len(vals)


_CONNECTORS = {"und", "aber", "oder", "denn", "sondern", "doch", "jedoch", "weil", "da", "deshalb", "deswegen", "daher",
               "darum", "obwohl", "trotzdem", "dennoch", "als", "wenn", "während", "bevor", "nachdem", "sobald", "seit",
               "dann", "danach", "zuerst", "schließlich", "damit", "falls", "außerdem", "zusätzlich", }

_SUBCLAUSE_MARKERS = {"Relativsatz": {"der", "die", "das", "welcher", "welche", "welches"},
                      "Kausalsatz": {"weil", "da"},
                      "Temporalsatz": {"als", "wenn", "während", "bevor", "nachdem", "sobald", "seit", "bis"},
                      "Konditionalsatz": {"wenn", "falls"},
                      "Konzessivsatz": {"obwohl", "wenngleich", "trotzdem", "dennoch"}, "Finalsatz": {"damit", "um"},
                      "Objektsatz": {"dass", "ob"}, "Subjektsatz": {"dass"}, }


def _count_konjunktiv(words) -> int:
    cnt = 0
    for w in words:
        if getattr(w, "upos", None) in ("VERB", "AUX"):
            mood = _ufeats_get(w, "Mood")
            if mood == "Sub":
                cnt += 1
    return cnt


def _coherence_score(doc) -> float:
    sent_sets: list[set[str]] = []
    for sent in doc.sentences:
        s = set()
        for w in sent.words:
            if getattr(w, "upos", None) in ("NOUN", "PROPN", "PRON"):
                lem = (getattr(w, "lemma", "") or "").lower().strip()
                if lem:
                    s.add(lem)
        sent_sets.append(s)

    if len(sent_sets) < 2:
        return 0.0

    scores = []
    for a, b in zip(sent_sets, sent_sets[1:]):
        if not a and not b:
            scores.append(0.0)
            continue
        inter = len(a & b)
        union = len(a | b) if (a | b) else 1
        scores.append(inter / union)
    return sum(scores) / len(scores) if scores else 0.0


def _mtul_from_sentences(doc) -> float:
    total_words = 0
    total_tunits = 0

    for sent in doc.sentences:
        words = sent.words
        word_tokens = [w for w in words if _token_is_word(w)]
        total_words += len(word_tokens)

        t_units = 0
        for w in words:
            if _is_finite(w):
                rel = _dep_rel(w)
                if rel in ("root", "conj"):
                    t_units += 1
        total_tunits += max(1, t_units)

    return (total_words / total_tunits) if total_tunits > 0 else 0.0


def _subclause_hits(doc) -> dict[str, int]:
    hits = {k: 0 for k in _SUBCLAUSE_MARKERS.keys()}
    for sent in doc.sentences:
        toks = [(getattr(w, "text", "") or "").lower() for w in sent.words if _token_is_word(w)]
        for typ, markers in _SUBCLAUSE_MARKERS.items():
            hits[typ] += sum(1 for t in toks if t in markers)
    return hits


def analyze_text_stanza(text: str) -> dict:
    nlp = get_stanza_nlp()
    doc = nlp(text or "")

    sentences = doc.sentences
    sent_count = len(sentences)

    tokens = _words_from_doc(doc)
    word_count = len(tokens)

    words_per_sentence = []
    max_syllables = 0

    dep_subclause_heads_total = 0
    tense_counts_total = TenseCounts()
    finite_verbs_total = 0
    perfekt_constructions_total = 0

    connectors_count = 0
    konj_cnt = 0

    for sent in sentences:
        words = sent.words
        word_tokens = [w for w in words if _token_is_word(w)]
        words_per_sentence.append(len(word_tokens))

        for w in word_tokens:
            tok = getattr(w, "text", "") or ""
            if tok:
                max_syllables = max(max_syllables, _syllables_pyphen(tok))
                if tok.lower() in _CONNECTORS:
                    connectors_count += 1

        for w in words:
            rel = _dep_rel(w)
            if any(rel == x or rel.startswith(x + ":") for x in UD_SUBORD_REL_PREFIXES):
                if getattr(w, "upos", None) in ("VERB", "AUX"):
                    dep_subclause_heads_total += 1

        tc, fin_cnt, perf_cnt = _detect_tenses_for_sentence(words)
        tense_counts_total.praesens += tc.praesens
        tense_counts_total.praeteritum += tc.praeteritum
        tense_counts_total.perfekt += tc.perfekt
        tense_counts_total.plusquamperfekt += tc.plusquamperfekt
        tense_counts_total.futur1 += tc.futur1
        tense_counts_total.futur2 += tc.futur2

        finite_verbs_total += fin_cnt
        perfekt_constructions_total += perf_cnt
        konj_cnt += _count_konjunktiv(words)

    max_wps = max(words_per_sentence) if words_per_sentence else 0
    avg_wps = (sum(words_per_sentence) / len(words_per_sentence)) if words_per_sentence else 0.0

    dep_per_sentence = (dep_subclause_heads_total / sent_count) if sent_count > 0 else 0.0
    perf_ratio = (perfekt_constructions_total / finite_verbs_total) if finite_verbs_total > 0 else 0.0

    ttr_val = _ttr(tokens)
    mtld_val = _mtld(tokens)
    mtul_val = _mtul_from_sentences(doc)
    zipf_val = _zipf_avg(tokens)

    connectors_per_100 = (connectors_count / word_count * 100.0) if word_count > 0 else 0.0
    coh = _coherence_score(doc)
    sub_hits = _subclause_hits(doc)

    return {"word_count": word_count,

            "sent_count": sent_count, "max_words_per_sentence": max_wps, "avg_words_per_sentence": avg_wps,
            "max_syllables_per_token": max_syllables, "dep_clauses_per_sentence": dep_per_sentence,
            "tense_counts": tense_counts_total, "finite_verbs": finite_verbs_total,
            "perfekt_constructions": perfekt_constructions_total, "perfekt_per_finite_verb": perf_ratio,

            "mtul": mtul_val, "ttr": ttr_val, "mtld": mtld_val, "zipf_avg": zipf_val,
            "connectors_count": connectors_count, "connectors_per_100w": connectors_per_100,
            "konjunktiv_count": konj_cnt, "coherence_score": coh, "subclause_hits": sub_hits, }


def validate_alpha_stanza(params: Params, metrics: dict) -> dict:
    a = params.alpha
    checks = []

    def add(name, ok, value, target=None, note=""):
        checks.append({"name": name, "ok": ok, "value": value, "target": target, "note": note})

    # In Ohne Alpha: 0 = aus
    if a.mode == "Ohne Alpha":
        if a.max_sentences > 0:
            add("Max Sätze", metrics["sent_count"] <= a.max_sentences, metrics["sent_count"], a.max_sentences)
        if a.max_words_per_sentence > 0:
            add("Max Wörter/Satz", metrics["max_words_per_sentence"] <= a.max_words_per_sentence,
                metrics["max_words_per_sentence"], a.max_words_per_sentence)
        if a.max_syllables_per_token > 0:
            add("Max Silben/Token", metrics["max_syllables_per_token"] <= a.max_syllables_per_token,
                metrics["max_syllables_per_token"], a.max_syllables_per_token, note="Silben via Pyphen (de_DE).")
        if a.max_dep_clauses_per_sentence > 0.0:
            add("Dep-Nebensätze/Satz", metrics["dep_clauses_per_sentence"] <= a.max_dep_clauses_per_sentence,
                round(metrics["dep_clauses_per_sentence"], 3), a.max_dep_clauses_per_sentence,
                note="UD: advcl/ccomp/csubj/acl:relcl")

        forbidden = set(a.forbidden_tenses or [])
        if forbidden:
            tc: TenseCounts = metrics["tense_counts"]
            tense_map = {"Präsens": tc.praesens, "Präteritum": tc.praeteritum, "Perfekt": tc.perfekt,
                         "Plusquamperfekt": tc.plusquamperfekt, "Futur I": tc.futur1, "Futur II": tc.futur2, }
            for tense_name in forbidden:
                cnt = tense_map.get(tense_name, 0)
                add(f"Verbotenes Tempus: {tense_name}", cnt == 0, cnt, 0)

        if a.max_perfekt_per_finite_verb is not None:
            ratio = metrics["perfekt_per_finite_verb"]
            add("Perfekt/finite Verben (Ratio)", ratio <= a.max_perfekt_per_finite_verb, round(ratio, 3),
                a.max_perfekt_per_finite_verb,
                note=f"perfekt={metrics['perfekt_constructions']}, finite={metrics['finite_verbs']}")

        # lex coverage: bleibt als Parameter, wird hier nicht geprüft
        if a.min_lexical_coverage is not None:
            add("Lexikalische Abdeckung", True, "n/a", a.min_lexical_coverage,
                note="Check deaktiviert (wird später anders gelöst).")

        overall = all(c["ok"] for c in checks) if checks else None
        return {"overall": overall, "checks": checks}

    # Alpha-Modus: alles prüfen (Presets > 0)
    add("Max Sätze", metrics["sent_count"] <= a.max_sentences, metrics["sent_count"], a.max_sentences)
    add("Max Wörter/Satz", metrics["max_words_per_sentence"] <= a.max_words_per_sentence,
        metrics["max_words_per_sentence"], a.max_words_per_sentence)
    add("Max Silben/Token", metrics["max_syllables_per_token"] <= a.max_syllables_per_token,
        metrics["max_syllables_per_token"], a.max_syllables_per_token, note="Silben via Pyphen (de_DE).")
    add("Dep-Nebensätze/Satz", metrics["dep_clauses_per_sentence"] <= a.max_dep_clauses_per_sentence,
        round(metrics["dep_clauses_per_sentence"], 3), a.max_dep_clauses_per_sentence,
        note="UD: advcl/ccomp/csubj/acl:relcl")

    tc: TenseCounts = metrics["tense_counts"]
    forbidden = set(a.forbidden_tenses or [])
    tense_map = {"Präsens": tc.praesens, "Präteritum": tc.praeteritum, "Perfekt": tc.perfekt,
                 "Plusquamperfekt": tc.plusquamperfekt, "Futur I": tc.futur1, "Futur II": tc.futur2, }
    for tense_name in forbidden:
        cnt = tense_map.get(tense_name, 0)
        add(f"Verbotenes Tempus: {tense_name}", cnt == 0, cnt, 0)

    if a.max_perfekt_per_finite_verb is not None:
        ratio = metrics["perfekt_per_finite_verb"]
        add("Perfekt/finite Verben (Ratio)", ratio <= a.max_perfekt_per_finite_verb, round(ratio, 3),
            a.max_perfekt_per_finite_verb,
            note=f"perfekt={metrics['perfekt_constructions']}, finite={metrics['finite_verbs']}")

    if a.min_lexical_coverage is not None:
        add("Lexikalische Abdeckung", True, "n/a", a.min_lexical_coverage,
            note="Check deaktiviert (wird später anders gelöst).")

    overall = all(c["ok"] for c in checks) if checks else None
    return {"overall": overall, "checks": checks}


def validate_fine_params(params: Params, m: dict) -> dict:
    f = params.fine
    checks = []

    def add(name, ok, value, target=None, note=""):
        checks.append({"name": name, "ok": ok, "value": value, "target": target, "note": note})

    if not f.enabled:
        return {"enabled": False, "overall": None, "checks": []}

    # MTUL
    mtul = m["mtul"]
    if f.mtul_level != "keine Vorgabe":
        if f.mtul_level == "niedrig":
            add("MTUL", mtul <= 8.0, round(mtul, 2), "≤ 8", note="T-Unit approx via finite root|conj.")
        elif f.mtul_level == "mittel":
            add("MTUL", 9.0 <= mtul <= 12.0, round(mtul, 2), "9–12", note="T-Unit approx via finite root|conj.")
        elif f.mtul_level == "hoch":
            add("MTUL", 13.0 <= mtul <= 18.0, round(mtul, 2), "13–18", note="T-Unit approx via finite root|conj.")
        elif f.mtul_level == "sehr hoch":
            add("MTUL", True, round(mtul, 2), "> 18 / keine Einschränkung", note="Keine harte Grenze.")

    # Zu vermeidende Nebensatzarten
    hits = m["subclause_hits"]
    if f.forbidden_subclause_types:
        for typ in f.forbidden_subclause_types:
            cnt = hits.get(typ, 0)
            add(f"Nebensatzart vermeiden: {typ}", cnt == 0, cnt, "0",
                note="Heuristik: Marker-Wörter pro Typ (weil/obwohl/der...).", )

    # Zipf
    zipf = m["zipf_avg"]
    if f.zipf_level != "keine Vorgabe":
        if zipf is None:
            add("Zipf (avg)", False, "n/a", "wordfreq nötig", note="Install: pip install wordfreq")
        else:
            if f.zipf_level == "niedrig":
                add("Zipf (avg)", zipf >= 5.5, round(zipf, 2), "≥ 5.5")
            elif f.zipf_level == "mittel":
                add("Zipf (avg)", zipf >= 5.0, round(zipf, 2), "≥ 5.0")
            elif f.zipf_level == "hoch":
                add("Zipf (avg)", zipf >= 4.5, round(zipf, 2), "≥ 4.5")
            elif f.zipf_level == "sehr hoch":
                add("Zipf (avg)", True, round(zipf, 2), "keine Einschränkung")

    # LexVar (TTR/MTLD) – Heuristik-Schwellen
    ttr = m["ttr"]
    mtld = m["mtld"]
    if f.lexvar_level != "keine Vorgabe":
        if f.lexvar_level == "niedrig":
            add("LexVar (TTR/MTLD)", (ttr <= 0.48) or (mtld <= 60), f"TTR={ttr:.3f}, MTLD={mtld:.1f}",
                "TTR≤0.48 oder MTLD≤60", note="Heuristik-Schwellen.", )
        elif f.lexvar_level == "mittel":
            add("LexVar (TTR/MTLD)", (0.45 <= ttr <= 0.58) or (60 <= mtld <= 90), f"TTR={ttr:.3f}, MTLD={mtld:.1f}",
                "TTR≈0.45–0.58 oder MTLD≈60–90", note="Heuristik-Schwellen.", )
        elif f.lexvar_level == "hoch":
            add("LexVar (TTR/MTLD)", (ttr >= 0.55) or (mtld >= 85), f"TTR={ttr:.3f}, MTLD={mtld:.1f}",
                "TTR≥0.55 oder MTLD≥85", note="Heuristik-Schwellen.", )
        elif f.lexvar_level == "sehr hoch":
            add("LexVar (TTR/MTLD)", True, f"TTR={ttr:.3f}, MTLD={mtld:.1f}", "keine Einschränkung")

    # Konnektoren
    c = m["connectors_count"]
    c100 = m["connectors_per_100w"]
    if f.connectors_level != "keine Vorgabe":
        if f.connectors_level == "niedrig":
            add("Konnektoren", c <= 3, f"{c} ({c100:.2f}/100W)", "0–3", note="Liste vordefinierter Konnektoren.")
        elif f.connectors_level == "mittel":
            add("Konnektoren", 4 <= c <= 8, f"{c} ({c100:.2f}/100W)", "4–8", note="Liste vordefinierter Konnektoren.")
        elif f.connectors_level == "hoch":
            add("Konnektoren", 9 <= c <= 15, f"{c} ({c100:.2f}/100W)", "9–15", note="Liste vordefinierter Konnektoren.")
        elif f.connectors_level == "sehr hoch":
            add("Konnektoren", True, f"{c} ({c100:.2f}/100W)", "keine Einschränkung")

    # Konjunktiv
    k = m["konjunktiv_count"]
    if f.konjunktiv_mode == "vermeiden":
        add("Konjunktiv", k == 0, k, "0", note="Marker: Mood=Sub (UD).")
    elif f.konjunktiv_mode == "erlauben":
        add("Konjunktiv", True, k, "erlaubt", note="Marker: Mood=Sub (UD).")

    # Kohärenz
    coh = m["coherence_score"]
    if f.coherence_hint and f.coherence_hint != "keine":
        if f.coherence_hint == "hoch":
            add("Kohärenz", coh >= 0.20, round(coh, 3), "≥ 0.20", note="Heuristik: Satz-Overlap (Jaccard).")
        elif f.coherence_hint == "mittel":
            add("Kohärenz", coh >= 0.12, round(coh, 3), "≥ 0.12", note="Heuristik: Satz-Overlap (Jaccard).")
        elif f.coherence_hint == "niedrig":
            add("Kohärenz", coh >= 0.05, round(coh, 3), "≥ 0.05", note="Heuristik: Satz-Overlap (Jaccard).")

    overall = all(c["ok"] for c in checks) if checks else None
    return {"enabled": True, "overall": overall, "checks": checks}

def build_validation_report(params: Params, text: str) -> str:
    m = analyze_text_stanza(text)
    a = params.alpha
    f = params.fine

    lines: list[str] = []

    # =========================
    # Allgemein
    # =========================
    lines.append("=== Allgemein ===")

    # Wortanzahl: Soll/Ist nur wenn Ohne Alpha UND Textlänge gesetzt wurde
    if a.mode == "Ohne Alpha" and params.general.target_words is not None:
        lines.append(f"Wortanzahl | Soll: {params.general.target_words} | Ist: {m['word_count']}")
    else:
        lines.append(f"Wortanzahl (Ist): {m['word_count']}")

    lines.append("")

    # =========================
    # Alpha-Parameter (immer Soll vs Ist)
    # =========================
    lines.append("=== Alpha-Parameter ===")

    # Alpha-Status nur sinnvoll im Alpha-Modus (Presets)
    if a.mode in ALPHA_PRESETS:
        alpha_res = validate_alpha_stanza(params, m)
        status = "BESTANDEN ✅" if alpha_res.get("overall") else "NICHT BESTANDEN ❌"
        lines.append(f"Mode: {a.mode} → {status}")
    else:
        lines.append(f"Mode: {a.mode} → (kein Alpha-Status, nur Analyse)")

    def fmt_soll_int(v: int) -> str:
        return "unbegrenzt" if v == 0 else str(v)

    def fmt_soll_float(v: float) -> str:
        return "unbegrenzt" if abs(v - 0.0) < 1e-9 else str(v)

    # Max Sätze
    lines.append(f"- Max Sätze | Soll: {fmt_soll_int(a.max_sentences)} | Ist: {m['sent_count']}")

    # Max Wörter/Satz
    lines.append(
        f"- Max Wörter pro Satz | Soll: {fmt_soll_int(a.max_words_per_sentence)} | Ist: {m['max_words_per_sentence']} (Ø {m['avg_words_per_sentence']:.2f})"
    )

    # Max Silben/Token
    lines.append(
        f"- Max Silben pro Token | Soll: {fmt_soll_int(a.max_syllables_per_token)} | Ist: {m['max_syllables_per_token']} (Pyphen de_DE)"
    )

    # Max Dep-Nebensätze/Satz
    lines.append(
        f"- Max Dep-Nebensätze pro Satz | Soll: {fmt_soll_float(a.max_dep_clauses_per_sentence)} | Ist: {m['dep_clauses_per_sentence']:.3f} (UD: advcl/ccomp/csubj/acl:relcl)"
    )

    # Verbotene Tempora: immer anzeigen, mit Ist-Zählung
    tc: TenseCounts = m["tense_counts"]
    tense_map = {
        "Präsens": tc.praesens,
        "Präteritum": tc.praeteritum,
        "Perfekt": tc.perfekt,
        "Plusquamperfekt": tc.plusquamperfekt,
        "Futur I": tc.futur1,
        "Futur II": tc.futur2,
    }
    forbidden = a.forbidden_tenses or []
    lines.append(f"- Verbotene Tempora | Soll: {', '.join(forbidden) if forbidden else 'keine'}")
    # Ist immer ausgeben:
    lines.append("  Ist (Tempus-Zählung):")
    for tname in ["Präsens", "Präteritum", "Perfekt", "Plusquamperfekt", "Futur I", "Futur II"]:
        lines.append(f"  · {tname}: {tense_map[tname]}")

    # Perfekt/finite Verben Ratio: Soll vs Ist
    if a.max_perfekt_per_finite_verb is None:
        soll_perf = "unbegrenzt"
    else:
        soll_perf = str(a.max_perfekt_per_finite_verb)
    lines.append(
        f"- Max Perfekt/finite Verben | Soll: {soll_perf} | Ist: {m['perfekt_per_finite_verb']:.3f} "
        f"(perfekt={m['perfekt_constructions']}, finite={m['finite_verbs']})"
    )

    # Lexikalische Abdeckung bleibt Parameter, aber Check noch nicht (Soll vs Ist = n/a)
    if a.min_lexical_coverage is None:
        soll_lex = "unbegrenzt"
    else:
        soll_lex = str(a.min_lexical_coverage)
    lines.append(f"- Min lexikalische Abdeckung | Soll: {soll_lex} | Ist: n/a (noch nicht implementiert)")
    lines.append("")

    # =========================
    # Zusatzparameter (immer messen, immer anzeigen)
    # Kein Status, nur Soll vs Ist
    # =========================
    lines.append("=== Zusatzparameter ===")
    lines.append(f"Aktiviert: {f.enabled}")
    lines.append("")

    # Helper: Soll-Ausgabe nur wenn enabled, sonst "keine Vorgabe"
    def soll_if_enabled(val: str) -> str:
        return val if f.enabled else "keine Vorgabe"

    # MTUL
    mtul_soll = "keine Vorgabe"
    if f.mtul_level != "keine Vorgabe":
        band = MTUL_BANDS.get(f.mtul_level, f.mtul_level)
        mtul_soll = f"{f.mtul_level} (Richtwert: {band})"
    lines.append(f"- MTUL-Komplexität | Soll: {soll_if_enabled(mtul_soll)} | Ist: {m['mtul']:.2f}")

    # Nebensatzarten vermeiden
    ns_soll = "keine Vorgabe"
    if f.forbidden_subclause_types:
        ns_soll = ", ".join(f.forbidden_subclause_types)
    lines.append(f"- Zu vermeidende Nebensatzarten | Soll: {soll_if_enabled(ns_soll)}")
    # Ist: immer zeigen (auch wenn keine ausgewählt)
    hits = m["subclause_hits"]
    lines.append("  Ist (Marker-Zählung pro Typ):")
    for typ in SUBCLAUSE_TYPES:
        lines.append(f"  · {typ}: {hits.get(typ, 0)}")

    # Zipf
    zipf_soll = "keine Vorgabe"
    if f.zipf_level != "keine Vorgabe":
        zipf_soll = f"{f.zipf_level} (Ziel: {ZIPF_BANDS.get(f.zipf_level, f.zipf_level)})"
    zipf_ist = "n/a (pip install wordfreq)" if m["zipf_avg"] is None else f"{m['zipf_avg']:.2f}"
    lines.append(f"- Wortfrequenz (Zipf) – Stufe | Soll: {soll_if_enabled(zipf_soll)} | Ist: {zipf_ist}")

    # LexVar (TTR/MTLD)
    lexvar_soll = "keine Vorgabe"
    if f.lexvar_level != "keine Vorgabe":
        lexvar_soll = f"{f.lexvar_level} (Ziel: {LEXVAR_BANDS.get(f.lexvar_level, f.lexvar_level)})"
    lines.append(
        f"- Lexikalische Vielfalt (TTR/MTLD) – Stufe | Soll: {soll_if_enabled(lexvar_soll)} | Ist: TTR={m['ttr']:.3f}, MTLD={m['mtld']:.1f}"
    )

    # Konnektoren
    conn_soll = "keine Vorgabe"
    if f.connectors_level != "keine Vorgabe":
        conn_soll = f"{f.connectors_level} (Richtwert: {CONNECTOR_BANDS.get(f.connectors_level, f.connectors_level)})"
    lines.append(
        f"- Konnektoren-Dichte – Stufe | Soll: {soll_if_enabled(conn_soll)} | Ist: {m['connectors_count']} ({m['connectors_per_100w']:.2f}/100W)"
    )

    # Konjunktiv
    konj_soll = "keine Vorgabe"
    if f.konjunktiv_mode != "keine Vorgabe":
        konj_soll = f.konjunktiv_mode
    lines.append(
        f"- Konjunktiv I/II | Soll: {soll_if_enabled(konj_soll)} | Ist: {m['konjunktiv_count']} (Marker Mood=Sub)"
    )

    # Kohärenz
    coh_soll = "keine Vorgabe"
    if f.coherence_hint and f.coherence_hint != "keine":
        coh_soll = f.coherence_hint
    lines.append(
        f"- Kohärenz-Hinweis | Soll: {soll_if_enabled(coh_soll)} | Ist: {m['coherence_score']:.3f} (Heuristik: Satz-Overlap)"
    )

    return "\n".join(lines)


# =========================
# UI
# =========================
st.set_page_config(page_title="Sprachlernmaterial Generator", layout="wide")
st.title("Sprachlernmaterial Generator")
st.caption(
    "Modus: Alpha 3–6 oder Ohne Alpha · Ohne Alpha: 0 = aus · Validierung: Stanza-only · Lexik-Abdeckung: Parameter bleibt, Check später")

ensure_defaults_exist()

# Sidebar
with st.sidebar:
    st.header("LLM")
    api_key = st.text_input("Gemini API Key", type="password", help="API Key für Google Gemini.")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1,
                            help="Steuert die Zufälligkeit der Generierung: niedrig = stabiler/regelkonformer, hoch = variantenreicher.", )

    st.divider()

    with st.expander("Allgemein", expanded=True):
        st.text_input("Thema", key="topic", help="Übergeordnetes Thema des Textes, z. B. Alltag, Arbeit, Freizeit.")
        st.selectbox("Textart", TEXT_TYPES, index=TEXT_TYPES.index(st.session_state["text_type"]) if st.session_state[
                                                                                                         "text_type"] in TEXT_TYPES else 1,
                     key="text_type",
                     help="Kommunikative Textsorte, die Struktur, Stil und typische Formulierungen beeinflusst.", )

        if st.session_state["mode"] == "Ohne Alpha":
            st.number_input("Textlänge (Wörter, 0 = unbegrenzt)", min_value=0, max_value=2000, key="target_words",
                            step=10, help="Zielwortanzahl. 0 = keine Vorgabe.", )
        else:
            st.caption("Textlänge ist im Alpha-Modus deaktiviert (wird aus Alpha-Constraints abgeleitet).")

    with st.expander("Alpha-Parameter", expanded=True):
        st.selectbox("Modus", ["Alpha 3", "Alpha 4", "Alpha 5", "Alpha 6", "Ohne Alpha"], key="mode",
                     on_change=on_mode_change,
                     help="Alpha 3–6 lädt feste Presets (Parameter gesperrt). »Ohne Alpha« erlaubt freie Alpha-Parameter (0 = aus).", )

        mode = st.session_state["mode"]
        alpha_locked = (mode != "Ohne Alpha")

        st.number_input("Max Sätze (0 = aus)", min_value=0, max_value=100, key="alpha_max_sentences",
                        disabled=alpha_locked, help="Maximale Satzanzahl. 0 deaktiviert (nur »Ohne Alpha«).", )
        st.number_input("Max Wörter pro Satz (0 = aus)", min_value=0, max_value=60, key="alpha_max_words_per_sentence",
                        disabled=alpha_locked,
                        help="Maximale Satzlänge in Wörtern. 0 deaktiviert (nur »Ohne Alpha«).", )
        st.number_input("Max Silben pro Token (0 = aus)", min_value=0, max_value=12,
                        key="alpha_max_syllables_per_token", disabled=alpha_locked,
                        help="Max Silben pro Wort. Silbenzählung via Pyphen (de_DE). 0 deaktiviert (nur »Ohne Alpha«).", )

        current_dep = float(st.session_state.get("alpha_max_dep_clauses_per_sentence", 1.0) or 0.0)
        dep_max = max(10.0, current_dep)
        st.slider("Max Dep-Nebensätze pro Satz (0 = aus)", 0.0, dep_max, key="alpha_max_dep_clauses_per_sentence",
                  step=0.5, disabled=alpha_locked,
                  help="Begrenzung der Subordination. Gemessen über UD-Relationen. 0 deaktiviert (nur »Ohne Alpha«).", )

        st.multiselect("Verbotene Tempora", options=TENSES_ALL, key="alpha_forbidden_tenses", disabled=alpha_locked,
                       help="Tempora, die im Text nicht verwendet werden sollen.", )

        st.number_input("Max Perfekt/finite Verben (0 = aus)", min_value=0.0, max_value=5.0, step=0.1,
                        key="alpha_max_perfekt_per_finite_verb", disabled=alpha_locked,
                        help="Max Verhältnis Perfekt-Konstruktionen / finite Verben. 0 deaktiviert.", )

        # Parameter bleibt, aber kein Upload/kein Check
        st.number_input("Min lexikalische Abdeckung (0 = aus)", min_value=0.0, max_value=1.0, step=0.01,
                        key="alpha_min_lexical_coverage", disabled=alpha_locked,
                        help="Parameter bleibt im Tool (für Prompt/Presets). Die Prüfung erfolgt später extern.", )

    with st.expander("Zusatzparameter (sekundär, ändern Alpha nicht)", expanded=False):
        st.checkbox("Zusatzparameter aktivieren", key="fine_enabled",
                    help="Wenn deaktiviert, werden Zusatzparameter nicht in den Prompt aufgenommen.", )

        fine_disabled = not st.session_state.get("fine_enabled", False)

        st.selectbox("MTUL-Komplexität", LEVELS_4, key="fine_mtul_level", disabled=fine_disabled,
                     help="MTUL = mittlere Wörter pro T-Unit. Stufen sind Richtwerte.", )

        st.multiselect("Zu vermeidende Nebensatzarten", options=SUBCLAUSE_TYPES, key="fine_forbidden_subclause_types",
                       disabled=fine_disabled, help="Bestimmte Nebensatztypen, die im Text vermieden werden sollen.", )

        st.selectbox("Wortfrequenz (Zipf) – Stufe", LEVELS_4, key="fine_zipf_level", disabled=fine_disabled,
                     help="Stufen als Richtwerte.", )

        st.selectbox("Lexikalische Vielfalt (TTR/MTLD) – Stufe", LEVELS_4, key="fine_lexvar_level",
                     disabled=fine_disabled, help="Stufen als Richtwerte.", )

        st.selectbox("Konnektoren-Dichte – Stufe", LEVELS_4, key="fine_connectors_level", disabled=fine_disabled,
                     help="Wie viele Konnektoren genutzt werden sollen.", )

        st.selectbox("Konjunktiv I/II", ["keine Vorgabe", "erlauben", "vermeiden"], key="fine_konjunktiv_mode",
                     disabled=fine_disabled, help="Ob Konjunktivformen genutzt oder vermieden werden sollen.", )

        st.selectbox("Kohärenz-Hinweis", ["keine", "hoch", "mittel", "niedrig"], key="fine_coherence_hint",
                     disabled=fine_disabled, help="Abstrakter Hinweis zur Kohärenz.", )

# Build Params
if st.session_state["mode"] == "Ohne Alpha":
    tw = int(st.session_state["target_words"])
    target_words = None if tw == 0 else tw
else:
    target_words = None

params = Params(general=GeneralParams(topic=st.session_state["topic"], text_type=st.session_state["text_type"],
                                      target_words=target_words, ), alpha=AlphaParams(mode=st.session_state["mode"],
                                                                                      max_sentences=int(
                                                                                          st.session_state[
                                                                                              "alpha_max_sentences"]),
                                                                                      max_words_per_sentence=int(
                                                                                          st.session_state[
                                                                                              "alpha_max_words_per_sentence"]),
                                                                                      max_syllables_per_token=int(
                                                                                          st.session_state[
                                                                                              "alpha_max_syllables_per_token"]),
                                                                                      max_dep_clauses_per_sentence=float(
                                                                                          st.session_state[
                                                                                              "alpha_max_dep_clauses_per_sentence"] or 0.0),
                                                                                      forbidden_tenses=list(
                                                                                          st.session_state[
                                                                                              "alpha_forbidden_tenses"]),
                                                                                      max_perfekt_per_finite_verb=optional_float_or_none(
                                                                                          st.session_state.get(
                                                                                              "alpha_max_perfekt_per_finite_verb")),
                                                                                      min_lexical_coverage=optional_float_or_none(
                                                                                          st.session_state.get(
                                                                                              "alpha_min_lexical_coverage")), ),
                fine=FineParams(enabled=bool(st.session_state.get("fine_enabled", False)),
                                mtul_level=clamp_level(str(st.session_state.get("fine_mtul_level", "keine Vorgabe"))),
                                zipf_level=clamp_level(str(st.session_state.get("fine_zipf_level", "keine Vorgabe"))),
                                lexvar_level=clamp_level(
                                    str(st.session_state.get("fine_lexvar_level", "keine Vorgabe"))),
                                connectors_level=clamp_level(
                                    str(st.session_state.get("fine_connectors_level", "keine Vorgabe"))),
                                forbidden_subclause_types=list(
                                    st.session_state.get("fine_forbidden_subclause_types", [])),
                                konjunktiv_mode=str(st.session_state.get("fine_konjunktiv_mode", "keine Vorgabe")),
                                coherence_hint=str(st.session_state.get("fine_coherence_hint", "keine")), ), )

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Prompts")
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(params)

    with st.expander("System Prompt"):
        st.code(system_prompt, language="text")

    with st.expander("User Prompt"):
        st.code(user_prompt, language="text")

with col2:
    st.subheader("Generierter Text")

    if st.button("Generate", type="primary", use_container_width=True, help="Startet die Textgenerierung."):
        try:
            out = gemini_generate(api_key, system_prompt, user_prompt, temperature)
            st.session_state["last_text"] = out
        except Exception as e:
            st.error(str(e))

    if "last_text" in st.session_state:
        st.text_area("Output", st.session_state["last_text"], height=520, help="Ausgabe des LLMs.")

        try:
            report = build_validation_report(params, st.session_state["last_text"])
            st.text_area("Validierung / Metriken (Stanza-only)", report, height=420,
                         help="Stanza-basierte Auswertung: Token/Sätze/Dep-Parsing + regelbasierte Tempus- und Ratio-Prüfung. "
                              "Lexik-Abdeckung ist aktuell nicht geprüft.", )
        except Exception as e:
            st.error(str(e))
    else:
        st.info("Klick auf Generate.")

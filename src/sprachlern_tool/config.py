"""
Zentrale Konfiguration und Konstanten.

"""

#GEMINI_MODEL = "gemini-3-flash-preview"
GEMINI_MODEL = "gemini-2.5-flash"

TENSES_ALL = ["Präsens", "Präteritum", "Perfekt", "Plusquamperfekt", "Futur I", "Futur II"]
TENSE_WEIGHT_LEVELS = ["keine Vorgabe", "verboten", "wenig", "mittel", "viel"]
TEXT_TYPES = ["Erzählung", "Sachtext", "Dialog", "E-Mail", "Nachrichtenberichterstattung"]

SUBCLAUSE_TYPES = [
    "Relativsatz",
    "Kausalsatz",
    "Temporalsatz",
    "Konditionalsatz",
    "Konzessivsatz",
    "Finalsatz",
    "Objektsatz",
    "Subjektsatz",
]

LEVELS_4 = ["keine Vorgabe", "niedrig", "mittel", "hoch", "sehr hoch"]

MTUL_BANDS = {
    "niedrig": "≤ 8 Wörter pro T-Unit",
    "mittel": "9–12 Wörter pro T-Unit",
    "hoch": "13–18 Wörter pro T-Unit",
    "sehr hoch": "> 18 Wörter pro T-Unit",
}

ZIPF_BANDS = {
    "niedrig": "sehr häufige Wörter (Zipf grob ≥ 5.5)",
    "mittel": "alltagsnah (Zipf grob zwischen 4.5 und 5.5)",
    "hoch": "differenzierter Wortschatz erlaubt (Zipf grob ≤ 4.5)",
    "sehr hoch": "keine Einschränkung",
}

LEXVAR_BANDS = {
    "niedrig": "geringe lexikalische Vielfalt, mehr Wiederholung",
    "mittel": "ausgewogene Vielfalt",
    "hoch": "hohe Vielfalt erlaubt",
    "sehr hoch": "keine Einschränkung",
}

CONNECTOR_BANDS = {
    "niedrig": "wenige Konnektoren (z. B. 0–3)",
    "mittel": "moderate Konnektoren (z. B. 4–8)",
    "hoch": "viele Konnektoren erlaubt (z. B. 9–15)",
    "sehr hoch": "keine Einschränkung",
}


ALPHA_PRESETS = {
    "Alpha 3": dict(
        max_sentences=5,
        max_words_per_sentence=10,
        max_syllables_per_token=3,
        max_dep_clauses_per_sentence=0.5,
        forbidden_tenses=["Plusquamperfekt", "Futur I", "Futur II"],
        max_perfekt_per_finite_verb=0.5,
        min_lexical_coverage=0.95,
    ),
    "Alpha 4": dict(
        max_sentences=10,
        max_words_per_sentence=10,
        max_syllables_per_token=5,
        max_dep_clauses_per_sentence=1.0,
        forbidden_tenses=["Plusquamperfekt", "Futur I", "Futur II"],
        max_perfekt_per_finite_verb=None,
        min_lexical_coverage=None,
    ),
    "Alpha 5": dict(
        max_sentences=15,
        max_words_per_sentence=12,
        max_syllables_per_token=5,
        max_dep_clauses_per_sentence=1.5,
        forbidden_tenses=["Plusquamperfekt"],
        max_perfekt_per_finite_verb=None,
        min_lexical_coverage=None,
    ),
    "Alpha 6": dict(
        max_sentences=20,
        max_words_per_sentence=12,
        max_syllables_per_token=6,
        max_dep_clauses_per_sentence=2.0,
        forbidden_tenses=[],
        max_perfekt_per_finite_verb=None,
        min_lexical_coverage=None,
    ),
}

FREE_DEFAULTS = dict(
    max_sentences=8,
    max_words_per_sentence=18,
    max_syllables_per_token=6,
    max_dep_clauses_per_sentence=2.0,
    forbidden_tenses=[],
    max_perfekt_per_finite_verb=None,
    min_lexical_coverage=None,
)

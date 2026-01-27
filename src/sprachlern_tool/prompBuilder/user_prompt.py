from src.sprachlern_tool.config import MTUL_BANDS, ZIPF_BANDS, LEXVAR_BANDS, CONNECTOR_BANDS
from src.sprachlern_tool.models import Params, FineParams
from src.sprachlern_tool.prompBuilder.context import context_for_alpha


def fine_params_to_prompt_lines(f: FineParams) -> list[str]:
    """
    Formatiert Zusatzparameter als „weiche Ziele“.

    Die Zusatzparameter sollen ein Alpha-Level nicht ersetzen, sondern innerhalb des Levels fein steuern.
    """
    if not f.enabled:
        return []

    lines: list[str] = []

    weight_text = {
        "verboten": "nicht verwenden",
        "wenig": "selten verwenden",
        "mittel": "in moderater Häufigkeit verwenden",
        "viel": "häufig verwenden",
    }

    selected = [
        (t, w)
        for t, w in (f.tense_weights or {}).items()
        if w and w != "keine Vorgabe"
    ]

    if selected:
        parts = []
        for t, w in selected:
            parts.append(f"{t}: {weight_text.get(w, w)}")
        lines.append("- Tempora-Gewichtung (Präferenz): " + "; ".join(parts) + ".")

    if f.mtul_level != "keine Vorgabe":
        band = MTUL_BANDS.get(f.mtul_level, f.mtul_level)
        lines.append(f"- MTUL: {f.mtul_level}. Richtwert: {band}. (bestmöglich)")

    if f.forbidden_subclause_types:
        lines.append(f"- Vermeide folgende Nebensatzarten: {', '.join(f.forbidden_subclause_types)}. Versuche die erlaubten Nebensatzarten einzubauen sofern es Sinn ergibt")

    if f.zipf_level != "keine Vorgabe":
        band = ZIPF_BANDS.get(f.zipf_level, f.zipf_level)
        lines.append(f"- Lexik/Wortfrequenz (Zipf): {f.zipf_level}. Ziel: {band}. (bestmöglich)")

    if f.lexvar_level != "keine Vorgabe":
        band = LEXVAR_BANDS.get(f.lexvar_level, f.lexvar_level)
        lines.append(f"- Lexikalische Vielfalt: {f.lexvar_level}. Ziel: {band}. (bestmöglich)")

    if f.konjunktiv_mode == "soll vorkommen":
        lines.append("- Konjunktiv I/II: soll vorkommen (wenn passend).")
    elif f.konjunktiv_mode == "vermeiden":
        lines.append("- Konjunktiv I/II: vermeiden.")

    if f.connectors_level != "keine Vorgabe":
        band = CONNECTOR_BANDS.get(f.connectors_level, f.connectors_level)
        lines.append(f"- Konnektoren: {f.connectors_level}. Richtwert: {band}. (bestmöglich)")

    if f.coherence_hint and f.coherence_hint != "keine":
        lines.append(f"- Kohärenz: {f.coherence_hint} (logisch gut nachvollziehbar, klare Bezüge).")

    return lines


def build_user_prompt(p: Params) -> str:
    """
    User Prompt: konkrete Aufgabenbeschreibung und Constraints.

    - Im Alpha-Modus wird das Alpha-Level explizit genannt und der RAG-Kontext eingebunden.
    - Ohne Alpha werden nur aktive Constraints ausgegeben (0 = aus / None = aus).
    """
    g = p.general
    a = p.alpha
    f = p.fine

    forbidden = ", ".join(a.forbidden_tenses) if a.forbidden_tenses else "keine"
    fine_lines = fine_params_to_prompt_lines(f)

    if f.enabled and a.forbidden_tenses:
        forbidden_set = set(a.forbidden_tenses)

        cleaned_weights = {
            t: w
            for t, w in (f.tense_weights or {}).items()
            if t not in forbidden_set
        }

        f_clean = FineParams(
            enabled=f.enabled,
            mtul_level=f.mtul_level,
            zipf_level=f.zipf_level,
            lexvar_level=f.lexvar_level,
            connectors_level=f.connectors_level,
            tense_weights=cleaned_weights,
            forbidden_subclause_types=f.forbidden_subclause_types,
            konjunktiv_mode=f.konjunktiv_mode,
            coherence_hint=f.coherence_hint,
        )

        fine_lines = fine_params_to_prompt_lines(f_clean)

    if a.mode == "Ohne Alpha":
        lines: list[str] = [f'Thema: "{g.topic}".', f"Textart: {g.text_type}."]

        if g.target_words is not None:
            lines.append(f"Textlänge: ca. {g.target_words} Wörter.")

        lines += ["", "Parameter (bitte so gut wie möglich einhalten):"]

        if a.max_sentences > 0:
            lines.append(f"- Anzahl Sätze: höchstens {a.max_sentences} (weniger ist ok).")
        if a.max_words_per_sentence > 0:
            lines.append(f"- Wörter pro Satz: höchstens {a.max_words_per_sentence} (weniger ist ok).")
        if a.max_syllables_per_token > 0:
            lines.append(f"- Silben pro Token: höchstens {a.max_syllables_per_token} (weniger ist ok).")
        if a.max_dep_clauses_per_sentence > 0.0:
            lines.append(f"- Dependenznebensätze pro Satz: höchstens {a.max_dep_clauses_per_sentence}.")
        lines.append(f"- Verbotene Tempora: {forbidden}. Versuche alle erlaubten Tempora zu verwenden sofern es Sinn ergibt.")

        if a.max_perfekt_per_finite_verb is not None:
            lines.append(f"- Perfekt pro finitem Verb: höchstens {a.max_perfekt_per_finite_verb}.Versuche bis zu {a.max_dep_clauses_per_sentence} einzubauen wenn möglich.")
        if a.min_lexical_coverage is not None:
            lines.append(f"- Lexikalische Abdeckung: Ziel-Abdeckung mindestens {a.min_lexical_coverage}. Orientiere dich an SUBTLEX_DE")

        if fine_lines:
            lines += ["", "Zusatzparameter (sekundär, ändern kein Alpha-Level):", *fine_lines]

        lines += [
            "",
            "Konsistenzregel:",
            "- Falls Textlänge und Satzanzahl nicht zusammenpassen, wähle eine sinnvolle Satzanzahl, die einen natürlichen Text ermöglicht.",
            "",
            "Stil:",
            "- Klar, alltagsnah, keine Schachtelsätze.",
            "- Keine Metakommentare.",
        ]
        return "\n".join(lines)

    rag_ctx = context_for_alpha(a.mode)
    lines = [
        f"ALPHA-LEVEL: {a.mode}",
        "Beziehe dich auf den folgenden Algorithmus-Kontext und halte die Parameter so gut wie möglich ein.",
        "",
        rag_ctx.strip(),
        "",
        f'Thema: "{g.topic}".',
        f"Textart: {g.text_type}.",
        "",
        "Parameter (so gut wie möglich einhalten):",
        f"- Anzahl Sätze: höchstens {a.max_sentences} (weniger ist ok).",
        f"- Wörter pro Satz: höchstens {a.max_words_per_sentence} (weniger ist ok).",
        f"- Silben pro Token: höchstens {a.max_syllables_per_token} (weniger ist ok).",
        f"- Dependenznebensätze pro Satz: höchstens {a.max_dep_clauses_per_sentence}. Versuche bis zu {a.max_dep_clauses_per_sentence} einzubauen wenn möglich.",
        f"- Verbotene Tempora: {forbidden}. Versuche alle erlaubten Tempora zu verwenden sofern es Sinn ergibt.",
    ]

    if a.max_perfekt_per_finite_verb is not None:
        lines.append(f"- Perfekt pro finitem Verb: höchstens {a.max_perfekt_per_finite_verb}.")
    if a.min_lexical_coverage is not None:
        lines.append(f"- Lexikalische Abdeckung: Ziel-Abdeckung mindestens {a.min_lexical_coverage}. Orientiere dich an SUBTLEX_DE")

    if fine_lines:
        lines += ["", "Zusatzparameter (sekundär, ändern dieses Alpha-Level nicht):", *fine_lines]

    lines += ["", "Stil:", "- Klar, alltagsnah, keine Schachtelsätze.", "- Keine Metakommentare."]
    return "\n".join(lines)

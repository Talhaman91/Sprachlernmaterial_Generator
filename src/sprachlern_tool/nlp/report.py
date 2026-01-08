from src.sprachlern_tool.config import ALPHA_PRESETS, SUBCLAUSE_TYPES, MTUL_BANDS, ZIPF_BANDS, LEXVAR_BANDS, CONNECTOR_BANDS
from src.sprachlern_tool.models import Params
from src.sprachlern_tool.nlp.analysis import analyze_text_stanza, TenseCounts
from src.sprachlern_tool.nlp.alpha_validation import validate_alpha_stanza


def build_validation_report(params: Params, text: str) -> str:
    """
    Erstellt den Reporttext für das Validierungsfenster.

    Format:
    - Allgemein
    - Alpha-Parameter (Soll/Ist + optionaler Bestanden-Status im Alpha-Modus)
    - Zusatzparameter (Soll/Ist, unabhängig von der Aktivierung)
    """
    m = analyze_text_stanza(text)
    a = params.alpha
    f = params.fine

    lines: list[str] = []

    lines.append("=== Allgemein ===")
    if a.mode == "Ohne Alpha" and params.general.target_words is not None:
        lines.append(f"Wortanzahl | Soll: {params.general.target_words} | Ist: {m['word_count']}")
    else:
        lines.append(f"Wortanzahl (Ist): {m['word_count']}")
    lines.append("")

    lines.append("=== Alpha-Parameter ===")
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

    lines.append(f"- Max Sätze | Soll: {fmt_soll_int(a.max_sentences)} | Ist: {m['sent_count']}")
    lines.append(
        f"- Max Wörter pro Satz | Soll: {fmt_soll_int(a.max_words_per_sentence)} | Ist: {m['max_words_per_sentence']} "
        f"(Ø {m['avg_words_per_sentence']:.2f})"
    )
    lines.append(
        f"- Max Silben pro Token | Soll: {fmt_soll_int(a.max_syllables_per_token)} | Ist: {m['max_syllables_per_token']} "
        f"(Pyphen de_DE)"
    )
    lines.append(
        f"- Max Dep-Nebensätze pro Satz | Soll: {fmt_soll_float(a.max_dep_clauses_per_sentence)} | Ist: {m['dep_clauses_per_sentence']:.3f} "
        f"(UD: advcl/ccomp/csubj/acl:relcl)"
    )

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
    lines.append("  Ist (Tempus-Zählung):")
    for tname in ["Präsens", "Präteritum", "Perfekt", "Plusquamperfekt", "Futur I", "Futur II"]:
        lines.append(f"  · {tname}: {tense_map[tname]}")

    soll_perf = "unbegrenzt" if a.max_perfekt_per_finite_verb is None else str(a.max_perfekt_per_finite_verb)
    lines.append(
        f"- Max Perfekt/finite Verben | Soll: {soll_perf} | Ist: {m['perfekt_per_finite_verb']:.3f} "
        f"(perfekt={m['perfekt_constructions']}, finite={m['finite_verbs']})"
    )

    soll_lex = "unbegrenzt" if a.min_lexical_coverage is None else str(a.min_lexical_coverage)
    lines.append(f"- Min lexikalische Abdeckung | Soll: {soll_lex} | Ist: n/a (noch nicht implementiert)")
    lines.append("")

    lines.append("=== Zusatzparameter ===")
    lines.append(f"Aktiviert: {f.enabled}")
    lines.append("")

    def soll_if_enabled(val: str) -> str:
        return val if f.enabled else "keine Vorgabe"

    mtul_soll = "keine Vorgabe"
    if f.mtul_level != "keine Vorgabe":
        mtul_soll = f"{f.mtul_level} (Richtwert: {MTUL_BANDS.get(f.mtul_level, f.mtul_level)})"
    lines.append(f"- MTUL-Komplexität | Soll: {soll_if_enabled(mtul_soll)} | Ist: {m['mtul']:.2f}")

    ns_soll = "keine Vorgabe"
    if f.forbidden_subclause_types:
        ns_soll = ", ".join(f.forbidden_subclause_types)
    lines.append(f"- Zu vermeidende Nebensatzarten | Soll: {soll_if_enabled(ns_soll)}")
    hits = m["subclause_hits"]
    lines.append("  Ist (Marker-Zählung pro Typ):")
    for typ in SUBCLAUSE_TYPES:
        lines.append(f"  · {typ}: {hits.get(typ, 0)}")

    zipf_soll = "keine Vorgabe"
    if f.zipf_level != "keine Vorgabe":
        zipf_soll = f"{f.zipf_level} (Ziel: {ZIPF_BANDS.get(f.zipf_level, f.zipf_level)})"
    zipf_ist = "n/a (pip install wordfreq)" if m["zipf_avg"] is None else f"{m['zipf_avg']:.2f}"
    lines.append(f"- Wortfrequenz (Zipf) – Stufe | Soll: {soll_if_enabled(zipf_soll)} | Ist: {zipf_ist}")

    lexvar_soll = "keine Vorgabe"
    if f.lexvar_level != "keine Vorgabe":
        lexvar_soll = f"{f.lexvar_level} (Ziel: {LEXVAR_BANDS.get(f.lexvar_level, f.lexvar_level)})"
    lines.append(
        f"- Lexikalische Vielfalt (TTR/MTLD) – Stufe | Soll: {soll_if_enabled(lexvar_soll)} | Ist: "
        f"TTR={m['ttr']:.3f}, MTLD={m['mtld']:.1f}"
    )

    conn_soll = "keine Vorgabe"
    if f.connectors_level != "keine Vorgabe":
        conn_soll = f"{f.connectors_level} (Richtwert: {CONNECTOR_BANDS.get(f.connectors_level, f.connectors_level)})"
    lines.append(
        f"- Konnektoren-Dichte – Stufe | Soll: {soll_if_enabled(conn_soll)} | Ist: "
        f"{m['connectors_count']} ({m['connectors_per_100w']:.2f}/100W)"
    )

    konj_soll = "keine Vorgabe"
    if f.konjunktiv_mode != "keine Vorgabe":
        konj_soll = f.konjunktiv_mode
    lines.append(
        f"- Konjunktiv I/II | Soll: {soll_if_enabled(konj_soll)} | Ist: {m['konjunktiv_count']} (Marker Mood=Sub)"
    )

    coh_soll = "keine Vorgabe"
    if f.coherence_hint and f.coherence_hint != "keine":
        coh_soll = f.coherence_hint
    lines.append(
        f"- Kohärenz-Hinweis | Soll: {soll_if_enabled(coh_soll)} | Ist: {m['coherence_score']:.3f} (Heuristik: Satz-Overlap)"
    )

    return "\n".join(lines)

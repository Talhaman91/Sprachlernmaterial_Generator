from src.sprachlern_tool.models import Params
from src.sprachlern_tool.nlp.analysis import TenseCounts


def validate_alpha_stanza(params: Params, metrics: dict) -> dict:
    """
    Prüft Alpha-Constraints.

    Im Alpha-Modus wird der Status „bestanden/nicht bestanden“ verwendet.
    Im Ohne-Alpha-Modus werden nur aktive Constraints geprüft (0 = aus).
    """
    a = params.alpha
    checks = []

    def add(name, ok, value, target=None, note=""):
        checks.append({"name": name, "ok": ok, "value": value, "target": target, "note": note})

    if a.mode == "Ohne Alpha":
        if a.max_sentences > 0:
            add("Max Sätze", metrics["sent_count"] <= a.max_sentences, metrics["sent_count"], a.max_sentences)

        if a.max_words_per_sentence > 0:
            add(
                "Max Wörter/Satz",
                metrics["max_words_per_sentence"] <= a.max_words_per_sentence,
                metrics["max_words_per_sentence"],
                a.max_words_per_sentence,
            )

        if a.max_syllables_per_token > 0:
            add(
                "Max Silben/Token",
                metrics["max_syllables_per_token"] <= a.max_syllables_per_token,
                metrics["max_syllables_per_token"],
                a.max_syllables_per_token,
                note="Silben via Pyphen (de_DE).",
            )

        if a.max_dep_clauses_per_sentence > 0.0:
            add(
                "Dep-Nebensätze/Satz",
                metrics["dep_clauses_per_sentence"] <= a.max_dep_clauses_per_sentence,
                round(metrics["dep_clauses_per_sentence"], 3),
                a.max_dep_clauses_per_sentence,
                note="UD: advcl/ccomp/csubj/acl:relcl",
            )

        forbidden = set(a.forbidden_tenses or [])
        if forbidden:
            tc: TenseCounts = metrics["tense_counts"]
            tense_map = {
                "Präsens": tc.praesens,
                "Präteritum": tc.praeteritum,
                "Perfekt": tc.perfekt,
                "Plusquamperfekt": tc.plusquamperfekt,
                "Futur I": tc.futur1,
                "Futur II": tc.futur2,
            }
            for tense_name in forbidden:
                cnt = tense_map.get(tense_name, 0)
                add(f"Verbotenes Tempus: {tense_name}", cnt == 0, cnt, 0)

        if a.max_perfekt_per_finite_verb is not None:
            ratio = metrics["perfekt_per_finite_verb"]
            add(
                "Perfekt/finite Verben (Ratio)",
                ratio <= a.max_perfekt_per_finite_verb,
                round(ratio, 3),
                a.max_perfekt_per_finite_verb,
                note=f"perfekt={metrics['perfekt_constructions']}, finite={metrics['finite_verbs']}",
            )

        if a.min_lexical_coverage is not None:
            add(
                "Lexikalische Abdeckung",
                True,
                "n/a",
                a.min_lexical_coverage,
                note="Prüfung derzeit nicht implementiert.",
            )

        overall = all(c["ok"] for c in checks) if checks else None
        return {"overall": overall, "checks": checks}

    # Alpha-Modus: alle Checks (Presets sind > 0)
    add("Max Sätze", metrics["sent_count"] <= a.max_sentences, metrics["sent_count"], a.max_sentences)
    add(
        "Max Wörter/Satz",
        metrics["max_words_per_sentence"] <= a.max_words_per_sentence,
        metrics["max_words_per_sentence"],
        a.max_words_per_sentence,
    )
    add(
        "Max Silben/Token",
        metrics["max_syllables_per_token"] <= a.max_syllables_per_token,
        metrics["max_syllables_per_token"],
        a.max_syllables_per_token,
        note="Silben via Pyphen (de_DE).",
    )
    add(
        "Dep-Nebensätze/Satz",
        metrics["dep_clauses_per_sentence"] <= a.max_dep_clauses_per_sentence,
        round(metrics["dep_clauses_per_sentence"], 3),
        a.max_dep_clauses_per_sentence,
        note="UD: advcl/ccomp/csubj/acl:relcl",
    )

    tc: TenseCounts = metrics["tense_counts"]
    forbidden = set(a.forbidden_tenses or [])
    tense_map = {
        "Präsens": tc.praesens,
        "Präteritum": tc.praeteritum,
        "Perfekt": tc.perfekt,
        "Plusquamperfekt": tc.plusquamperfekt,
        "Futur I": tc.futur1,
        "Futur II": tc.futur2,
    }
    for tense_name in forbidden:
        cnt = tense_map.get(tense_name, 0)
        add(f"Verbotenes Tempus: {tense_name}", cnt == 0, cnt, 0)

    if a.max_perfekt_per_finite_verb is not None:
        ratio = metrics["perfekt_per_finite_verb"]
        add(
            "Perfekt/finite Verben (Ratio)",
            ratio <= a.max_perfekt_per_finite_verb,
            round(ratio, 3),
            a.max_perfekt_per_finite_verb,
            note=f"perfekt={metrics['perfekt_constructions']}, finite={metrics['finite_verbs']}",
        )

    if a.min_lexical_coverage is not None:
        add(
            "Lexikalische Abdeckung",
            True,
            "n/a",
            a.min_lexical_coverage,
            note="Prüfung derzeit nicht implementiert.",
        )

    overall = all(c["ok"] for c in checks) if checks else None
    return {"overall": overall, "checks": checks}

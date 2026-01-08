def rag_context_for_alpha(mode: str) -> str:
    """
    Kontextblock für Alpha-Level.

    Dieser Block wird im User Prompt als „Retrieved Context“ dargestellt,
    um die Regeln kompakt und wiederholbar einzubinden.
    """
    if mode == "Alpha 3":
        return (
            "Retrieved Context (Alpha Readability Level Algorithmus, Regeln):\n"
            "- wordsPerSentence <= 10\n"
            "- nSentences <= 5\n"
            "- syllablesPerToken <= 3\n"
            "- pastPerfectsPerFiniteVerb == 0\n"
            "- future1sPerFiniteVerb == 0\n"
            "- future2sPerFiniteVerb == 0\n"
            "- depClausesPerSentence <= 0.5\n"
            "- presentPerfectsPerFiniteVerb <= 0.5\n"
            "- typesFoundInSubtlexPerLexicalType >= 0.95\n"
        )
    if mode == "Alpha 4":
        return (
            "Retrieved Context (Alpha Readability Level Algorithmus, Regeln):\n"
            "- wordsPerSentence <= 10\n"
            "- nSentences <= 10\n"
            "- syllablesPerToken <= 5\n"
            "- pastPerfectsPerFiniteVerb == 0\n"
            "- future1sPerFiniteVerb == 0\n"
            "- future2sPerFiniteVerb == 0\n"
        )
    if mode == "Alpha 5":
        return (
            "Retrieved Context (Alpha Readability Level Algorithmus, Regeln):\n"
            "- wordsPerSentence <= 12\n"
            "- nSentences <= 15\n"
            "- pastPerfectsPerFiniteVerb == 0\n"
        )
    if mode == "Alpha 6":
        return (
            "Retrieved Context (Alpha Readability Level Algorithmus, Regeln):\n"
            "- wordsPerSentence <= 12\n"
            "- nSentences <= 20\n"
        )
    return ""


def rag_context_for_fine_params() -> str:
    """
    Glossarblock für Zusatzparameter.

    Der Block vereinheitlicht Begriffe (MTUL, Zipf, Konnektoren …), ohne das Modell
    mit langen Erklärungen zu überladen.
    """
    return (
        "Retrieved Context (Zusatzparameter-Glossar):\n"
        "| Parameter | Bedeutung | Intention |\n"
        "|---|---|---|\n"
        "| MTUL | mittlere Wörter pro T-Unit | kürzere syntaktische Einheiten |\n"
        "| LexVar | lexikalische Vielfalt (TTR/MTLD zusammengefasst) | niedrig = mehr Wiederholung |\n"
        "| Zipf | Wortfrequenz (1–7) | höher = häufigere Wörter |\n"
        "| Konnektoren | z.B. weil, aber, obwohl | weniger explizite Verknüpfungen |\n"
        "| Modus | Konjunktiv (I/II) | vermeiden = morphologisch einfacher |\n"
        "| Nebensatzarten | Relativsatz, Kausalsatz, ... | bestimmte Strukturen vermeiden |\n"
        "| Kohärenz | logischer Zusammenhang | klar nachvollziehbare Bezüge |\n"
    )

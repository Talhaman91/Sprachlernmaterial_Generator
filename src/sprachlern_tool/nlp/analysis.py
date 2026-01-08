from dataclasses import dataclass
from typing import Optional

import pyphen

from src.sprachlern_tool.nlp.stanza_pipeline import get_stanza_nlp


_DIC = pyphen.Pyphen(lang="de_DE")

UD_SUBORD_REL_PREFIXES = (
    "advcl",      # adverbial clause
    "ccomp",      # clausal complement
    "csubj",      # clausal subject
    "acl:relcl",  # relative clause
)

_CONNECTORS = {
    "und", "aber", "oder", "denn", "sondern", "doch", "jedoch",
    "weil", "da", "deshalb", "deswegen", "daher", "darum",
    "obwohl", "trotzdem", "dennoch",
    "als", "wenn", "während", "bevor", "nachdem", "sobald", "seit",
    "dann", "danach", "zuerst", "schließlich",
    "damit", "falls",
    "außerdem", "zusätzlich",
}

_SUBCLAUSE_MARKERS = {
    "Relativsatz": {"der", "die", "das", "welcher", "welche", "welches"},
    "Kausalsatz": {"weil", "da"},
    "Temporalsatz": {"als", "wenn", "während", "bevor", "nachdem", "sobald", "seit", "bis"},
    "Konditionalsatz": {"wenn", "falls"},
    "Konzessivsatz": {"obwohl", "wenngleich", "trotzdem", "dennoch"},
    "Finalsatz": {"damit", "um"},
    "Objektsatz": {"dass", "ob"},
    "Subjektsatz": {"dass"},
}


def _syllables_pyphen(word: str) -> int:
    """
    Silbenzählung über Pyphen-Hyphenation.

    Pyphen liefert Trennstellen, die hier als Silben-Näherung verwendet werden.
    """
    w = (word or "").strip()
    if not w:
        return 0
    hyph = _DIC.inserted(w)
    parts = [p for p in hyph.split("-") if p]
    return max(1, len(parts))


def _ufeats_get(word, key: str) -> Optional[str]:
    """
    Stanza speichert UD-Features als String: "Case=Nom|Gender=Masc|...".
    Diese Funktion liest einen gewünschten Feature-Key aus.
    """
    feats = getattr(word, "feats", None)
    if not feats:
        return None
    for feat in str(feats).split("|"):
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


def _detect_tenses_for_sentence(words) -> tuple[TenseCounts, int, int]:
    """
    Regelbasierte Erkennung von Tempus-Konstruktionen.

    Rückgabe:
    - Zählungen für Präsens/Präteritum/Perfekt/Plusquamperfekt/Futur I/Futur II
    - Anzahl finiter Verben (für Ratio)
    - Anzahl Perfekt-/Plusquamperfekt-Konstruktionen (für Ratio)
    """
    counts = TenseCounts()
    finite_verbs = [w for w in words if _is_finite(w)]
    finite_count = len(finite_verbs)

    by_id = {w.id: w for w in words if getattr(w, "id", None) is not None}
    children = {w.id: [] for w in words if getattr(w, "id", None) is not None}
    for w in words:
        hid = _head_id(w)
        if hid in children and getattr(w, "id", None) is not None:
            children[hid].append(w.id)

    perfekt_constructions = 0

    # Perfekt/Plusquamperfekt: AUX (haben/sein) + Partizip in Subtree
    for w in finite_verbs:
        if getattr(w, "upos", None) != "AUX":
            continue
        if _lemma(w) not in ("haben", "sein"):
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

    # Futur I/II: AUX werden + Infinitiv; Futur II zusätzlich Partizip + AUX-Infinitiv
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

    # Präsens/Präteritum: finite Vollverben nach Tense-Feature, ohne zentrale Hilfsverben
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
    """
    Optionaler Zipf-Wert über wordfreq.

    Falls wordfreq nicht installiert ist, liefert die Funktion None.
    """
    try:
        from wordfreq import zipf_frequency  # type: ignore
    except Exception:
        return None

    low = [t.lower() for t in tokens if t.strip()]
    if not low:
        return None
    vals = [zipf_frequency(w, "de") for w in low]
    return sum(vals) / len(vals)


def _count_konjunktiv(words) -> int:
    """
    Zählt Konjunktiv-Marker über UD-Feature Mood=Sub.
    """
    cnt = 0
    for w in words:
        if getattr(w, "upos", None) in ("VERB", "AUX"):
            if _ufeats_get(w, "Mood") == "Sub":
                cnt += 1
    return cnt


def _coherence_score(doc) -> float:
    """
    Grober Kohärenzindikator: lexikalischer Overlap von NOUN/PROPN/PRON (Lemma)
    zwischen aufeinanderfolgenden Sätzen (Jaccard-Ähnlichkeit).
    """
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
    """
    MTUL-Näherung:
    - Wörter werden pro Satz gezählt.
    - T-Units werden als finite Verben mit DepRel root|conj gezählt.
    """
    total_words = 0
    total_tunits = 0

    for sent in doc.sentences:
        words = sent.words
        word_tokens = [w for w in words if _token_is_word(w)]
        total_words += len(word_tokens)

        t_units = 0
        for w in words:
            if _is_finite(w) and _dep_rel(w) in ("root", "conj"):
                t_units += 1
        total_tunits += max(1, t_units)

    return (total_words / total_tunits) if total_tunits > 0 else 0.0


def _subclause_hits(doc) -> dict[str, int]:
    """
    Marker-basierte Zählung von Nebensatztypen.

    Diese Zählung ist eine Heuristik, um „zu vermeidende Nebensatzarten“ im Report sichtbar zu machen.
    """
    hits = {k: 0 for k in _SUBCLAUSE_MARKERS.keys()}
    for sent in doc.sentences:
        toks = [(getattr(w, "text", "") or "").lower() for w in sent.words if _token_is_word(w)]
        for typ, markers in _SUBCLAUSE_MARKERS.items():
            hits[typ] += sum(1 for t in toks if t in markers)
    return hits


def analyze_text_stanza(text: str) -> dict:
    """
    Analysiert den Text mit Stanza und berechnet alle Metriken, die im Report angezeigt werden.
    """
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

    return {
        "word_count": word_count,

        "sent_count": sent_count,
        "max_words_per_sentence": max_wps,
        "avg_words_per_sentence": avg_wps,
        "max_syllables_per_token": max_syllables,
        "dep_clauses_per_sentence": dep_per_sentence,
        "tense_counts": tense_counts_total,
        "finite_verbs": finite_verbs_total,
        "perfekt_constructions": perfekt_constructions_total,
        "perfekt_per_finite_verb": perf_ratio,

        "mtul": mtul_val,
        "ttr": ttr_val,
        "mtld": mtld_val,
        "zipf_avg": zipf_val,
        "connectors_count": connectors_count,
        "connectors_per_100w": connectors_per_100,
        "konjunktiv_count": konj_cnt,
        "coherence_score": coh,
        "subclause_hits": sub_hits,
    }

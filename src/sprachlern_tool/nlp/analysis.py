from dataclasses import dataclass
from typing import Optional
import re

import pyphen

from src.sprachlern_tool.nlp.stanza_pipeline import get_stanza_nlp


_DIC = pyphen.Pyphen(lang="de_DE")

# UD-Relationen, die typisch für untergeordnete Sätze sind.
# (Für "Nebensatz pro Satz" usw.)
UD_SUBORD_REL_PREFIXES = (
    "advcl",      # adverbial clause
    "ccomp",      # clausal complement
    "csubj",      # clausal subject
    "acl:relcl",  # relative clause
)

# Sehr einfache Liste häufiger Konnektoren.
# Das ist bewusst pragmatisch und dient als grober Indikator, nicht als perfekte Linguistik.
_CONNECTORS = {
    "und", "aber", "oder", "denn", "sondern", "doch", "jedoch",
    "weil", "da", "deshalb", "deswegen", "daher", "darum",
    "obwohl", "trotzdem", "dennoch",
    "als", "wenn", "während", "bevor", "nachdem", "sobald", "seit",
    "dann", "danach", "zuerst", "schließlich",
    "damit", "falls",
    "außerdem", "zusätzlich",
}

# Heuristik-Marker für Nebensatzarten.
# Wichtig: "Relativsatz" NICHT über Artikel zählen, weil das extrem viele False Positives erzeugt.
_SUBCLAUSE_MARKERS = {
    # Relativsatz wird UD-basiert gezählt (acl:relcl). Marker nur als Fallback (optional).
    "Relativsatz": {"welcher", "welche", "welches"},
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
    """
    Definition "Worttoken" für Zählungen:
    - schließt reine Satzzeichen / Symbole aus
    - zählt nur Tokens, die mindestens einen Buchstaben oder eine Ziffer enthalten
    """
    upos = getattr(word, "upos", None)
    if upos in ("PUNCT", "SYM"):
        return False
    tok = (getattr(word, "text", "") or "").strip()
    return bool(tok) and bool(re.search(r"[0-9A-Za-zÄÖÜäöüß]", tok))


# Regex-basierte Wortzählung: stabiler für UI (inkl. Bindestrich-Wörter).
_WORD_RE = re.compile(r"[0-9A-Za-zÄÖÜäöüß]+(?:-[0-9A-Za-zÄÖÜäöüß]+)*", re.UNICODE)


def _count_words_regex(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _split_sentences_regex(text: str) -> list[str]:
    """
    Sehr einfache Satzsegmentierung für UI-Zählwerte.
    Ziel: Satzanzahl soll "so zählen wie ein Mensch" (robust gegen LLM-Zeilenumbrüche etc.).
    """
    t = (text or "").strip()
    if not t:
        return []

    # Abkürzungen entschärfen, damit z.B. "z.B." nicht als Satzende zählt.
    abbr = ["z.B.", "u.a.", "d.h.", "bzw.", "etc."]
    for a in abbr:
        t = t.replace(a, a.replace(".", "§"))

    chunks = re.split(r"\n+", t)
    sents: list[str] = []
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        parts = re.split(r"(?<=[.!?])\s+", c)
        for p in parts:
            p = p.strip()
            if p:
                sents.append(p)

    return [s.replace("§", ".") for s in sents]


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
    Regelbasierte Erkennung von Tempus-Konstruktionen (UD-basiert).

    WICHTIGER FIX:
    - "sein/haben/werden" sind nicht immer Hilfsverben.
      Beispiel: "Die Landschaft war schön."
      In UD ist "war" oft AUX mit deprel=cop (Kopula) -> das ist ein echtes Tempus-Signal
      und darf NICHT pauschal rausgefiltert werden.

    Grundidee:
    1) Perfekt/Plusquamperfekt nur zählen, wenn:
       - es ein Partizip gibt UND
       - ein AUX (haben/sein) mit deprel=aux direkt an diesem Partizip hängt (head=Partizip)
    2) Futur I/II über "werden" + Infinitiv / + Partizip + haben/sein-Infinitiv
    3) Präsens/Präteritum über finite Verben (inkl. cop), aber:
       - AUX, die wirklich als "aux" fürs Perfekt benutzt wurden, zählen wir nicht nochmal.
    """
    counts = TenseCounts()

    # Alle finiten Verbformen in diesem Satz (VERB oder AUX)
    finite_verbs = [w for w in words if _is_finite(w)]
    finite_count = 0  # wird weiter unten sauber gezählt (nach Ausschlüssen)

    # Kleine Indexstruktur
    by_id = {w.id: w for w in words if getattr(w, "id", None) is not None}

    # ------------------------------------------------------------
    # 1) Perfekt / Plusquamperfekt sauber erkennen
    # ------------------------------------------------------------
    # Wir merken uns, welche AUX-Tokens wirklich als Perfekt-Aux genutzt wurden,
    # damit sie später nicht nochmal als Präsens/Präteritum zählen.
    aux_ids_used_for_perfekt: set[int] = set()
    perfekt_constructions = 0

    for w in words:
        # Wir suchen Partizipien (meist der "Kern" im Perfekt)
        if not _is_participle(w):
            continue
        part_id = getattr(w, "id", None)
        if not part_id:
            continue

        # Suche AUX-Kinder, die per UD als aux am Partizip hängen
        perf_aux = []
        for a in words:
            if getattr(a, "upos", None) != "AUX":
                continue
            if _dep_rel(a) != "aux":
                continue
            if _head_id(a) != part_id:
                continue
            if _lemma(a) in ("haben", "sein"):
                perf_aux.append(a)

        if not perf_aux:
            continue

        # Wenn wir hier sind, ist das sehr wahrscheinlich ein echtes Perfekt/Plusquamperfekt
        perfekt_constructions += 1
        for a in perf_aux:
            aux_ids_used_for_perfekt.add(a.id)

        # Past am AUX -> Plusquamperfekt, Pres -> Perfekt
        # (Ist eine Heuristik, aber deutlich stabiler als "haben/sein irgendwo im Satz")
        aux_tense = _ufeats_get(perf_aux[0], "Tense")  # Pres/Past
        if aux_tense == "Past":
            counts.plusquamperfekt += 1
        else:
            counts.perfekt += 1

    # ------------------------------------------------------------
    # 2) Futur I / Futur II erkennen
    # ------------------------------------------------------------
    # (Hier bleibt deine Logik erhalten, weil sie pragmatisch funktioniert.)
    # Futur I: werden + Infinitiv
    # Futur II: werden + Partizip + haben/sein-Infinitiv
    children = {w.id: [] for w in words if getattr(w, "id", None) is not None}
    for w in words:
        hid = _head_id(w)
        if hid in children and getattr(w, "id", None) is not None:
            children[hid].append(w.id)

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

    # ------------------------------------------------------------
    # 3) Präsens / Präteritum zählen
    # ------------------------------------------------------------
    # Hier ist der Kern-Fix: "sein/haben/werden" nur dann rausfiltern,
    # wenn sie wirklich AUX mit deprel=aux sind (Hilfsverbfunktion).
    #
    # Kopula ("war", "ist") hat oft deprel=cop -> das zählt als Tempus!
    for w in finite_verbs:
        upos = getattr(w, "upos", None)
        lem = _lemma(w)
        rel = _dep_rel(w)
        tense = _ufeats_get(w, "Tense")  # Pres / Past / None

        wid = getattr(w, "id", None)
        if wid is None:
            continue

        # AUX, die fürs Perfekt benutzt wurden, zählen wir nicht noch als Präsens/Präteritum.
        if wid in aux_ids_used_for_perfekt:
            continue

        # Hilfsverben wirklich nur dann skippen, wenn UD sie als "aux" markiert.
        # Dadurch bleibt "war" (cop) drin und wird korrekt als Präteritum gezählt.
        if upos == "AUX" and rel == "aux" and lem in ("haben", "sein", "werden"):
            continue

        finite_count += 1

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


def _lexical_coverage_wordfreq(doc) -> float | None:
    """
    Lexikalische Abdeckung auf Basis von wordfreq:
    - lexikalische Types = unique Lemmata von NOUN/VERB/ADJ/ADV
    - "known" wenn zipf_frequency(lemma, 'de') > 0
    """
    try:
        from wordfreq import zipf_frequency  # type: ignore
    except Exception:
        return None

    lexical_upos = {"NOUN", "VERB", "ADJ", "ADV"}

    types: set[str] = set()
    for sent in doc.sentences:
        for w in sent.words:
            upos = getattr(w, "upos", None)
            if upos not in lexical_upos:
                continue

            lem = (getattr(w, "lemma", "") or "").lower().strip()
            if not lem:
                continue
            types.add(lem)

    if not types:
        return 1.0

    known = sum(1 for t in types if zipf_frequency(t, "de") > 0.0)
    return known / len(types)


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


def _count_relative_clauses_ud(doc) -> int:
    """
    Zählt Relativsätze über UD-Relation 'acl:relcl'.

    Vorteil:
    - Artikel wie "der/die/das" werden NICHT fälschlich als Relativsatz gezählt.
    """
    cnt = 0
    for sent in doc.sentences:
        for w in sent.words:
            if (getattr(w, "deprel", "") or "") == "acl:relcl":
                cnt += 1
    return cnt


def _subclause_hits(doc) -> dict[str, int]:
    """
    Zählung von Nebensatztypen für den Report.

    - Relativsatz: UD-basiert über 'acl:relcl'
    - Andere Typen: marker-basiert (Heuristik)
    """
    hits = {k: 0 for k in _SUBCLAUSE_MARKERS.keys()}

    # Marker-basierte Zählung für die übrigen Typen
    for sent in doc.sentences:
        toks = [(getattr(w, "text", "") or "").lower() for w in sent.words if _token_is_word(w)]
        for typ, markers in _SUBCLAUSE_MARKERS.items():
            if typ == "Relativsatz":
                continue
            hits[typ] += sum(1 for t in toks if t in markers)

    # Relativsatz: UD-basiert
    hits["Relativsatz"] = _count_relative_clauses_ud(doc)

    return hits


def analyze_text_stanza(text: str) -> dict:
    """
    Analysiert den Text mit Stanza.
    Wort- und Satzanzahl werden regex-basiert ermittelt (UI-stabil),
    alle linguistischen Metriken weiterhin mit Stanza.
    """
    text = text or ""

    # ------------------------------------------------------------
    # UI-stabile Zählwerte (sollen mit sichtbarem Text übereinstimmen)
    # ------------------------------------------------------------
    ui_sentences = _split_sentences_regex(text)
    sent_count_ui = len(ui_sentences)
    word_count_ui = _count_words_regex(text)

    words_per_sentence_ui = [_count_words_regex(s) for s in ui_sentences]
    max_wps = max(words_per_sentence_ui) if words_per_sentence_ui else 0
    avg_wps = (
        sum(words_per_sentence_ui) / len(words_per_sentence_ui)
        if words_per_sentence_ui
        else 0.0
    )

    # ------------------------------------------------------------
    # Stanza-Analyse für syntaktische / linguistische Metriken
    # ------------------------------------------------------------
    nlp = get_stanza_nlp()
    doc = nlp(text)
    sentences = doc.sentences
    sent_count_stanza = len(sentences)

    tokens = _words_from_doc(doc)

    max_syllables = 0
    lexcov = _lexical_coverage_wordfreq(doc)

    dep_subclause_heads_total = 0
    tense_counts_total = TenseCounts()
    finite_verbs_total = 0
    perfekt_constructions_total = 0

    connectors_count = 0
    konj_cnt = 0

    for sent in sentences:
        words = sent.words
        word_tokens = [w for w in words if _token_is_word(w)]

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

    # ------------------------------------------------------------
    # Aggregierte Kennzahlen
    # ------------------------------------------------------------
    dep_per_sentence = (
        dep_subclause_heads_total / sent_count_stanza
        if sent_count_stanza > 0
        else 0.0
    )

    perf_ratio = (
        perfekt_constructions_total / finite_verbs_total
        if finite_verbs_total > 0
        else 0.0
    )

    ttr_val = _ttr(tokens)
    mtld_val = _mtld(tokens)
    mtul_val = _mtul_from_sentences(doc)
    zipf_val = _zipf_avg(tokens)

    connectors_per_100 = (
        connectors_count / word_count_ui * 100.0
        if word_count_ui > 0
        else 0.0
    )

    coh = _coherence_score(doc)
    sub_hits = _subclause_hits(doc)

    return {
        "word_count": word_count_ui,
        "sent_count": sent_count_ui,
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
        "lexical_coverage_wordfreq": lexcov,
        "connectors_count": connectors_count,
        "connectors_per_100w": connectors_per_100,
        "konjunktiv_count": konj_cnt,
        "coherence_score": coh,
        "subclause_hits": sub_hits,
    }

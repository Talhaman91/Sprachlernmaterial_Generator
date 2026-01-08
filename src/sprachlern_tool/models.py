from dataclasses import dataclass


@dataclass
class GeneralParams:
    topic: str
    text_type: str
    target_words: int | None


@dataclass
class AlphaParams:
    mode: str

    max_sentences: int
    max_words_per_sentence: int
    max_syllables_per_token: int
    max_dep_clauses_per_sentence: float

    forbidden_tenses: list[str]
    max_perfekt_per_finite_verb: float | None
    min_lexical_coverage: float | None


@dataclass
class FineParams:
    enabled: bool
    mtul_level: str
    zipf_level: str
    lexvar_level: str
    connectors_level: str
    forbidden_subclause_types: list[str]
    konjunktiv_mode: str
    coherence_hint: str


@dataclass
class Params:
    general: GeneralParams
    alpha: AlphaParams
    fine: FineParams

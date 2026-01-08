import streamlit as st

from src.sprachlern_tool.config import FREE_DEFAULTS, ALPHA_PRESETS
from src.sprachlern_tool.models import Params, GeneralParams, AlphaParams, FineParams
from src.sprachlern_tool.utils import optional_float_or_none, clamp_level


def ensure_defaults_exist() -> None:
    """
    Legt Standardwerte im Streamlit Session State an.

    """
    st.session_state.setdefault("mode", "Alpha 4")
    st.session_state.setdefault("topic", "Alltag")
    st.session_state.setdefault("text_type", "Sachtext")
    st.session_state.setdefault("target_words", 140)

    st.session_state.setdefault("alpha_max_sentences", FREE_DEFAULTS["max_sentences"])
    st.session_state.setdefault("alpha_max_words_per_sentence", FREE_DEFAULTS["max_words_per_sentence"])
    st.session_state.setdefault("alpha_max_syllables_per_token", FREE_DEFAULTS["max_syllables_per_token"])
    st.session_state.setdefault("alpha_max_dep_clauses_per_sentence", FREE_DEFAULTS["max_dep_clauses_per_sentence"])
    st.session_state.setdefault("alpha_forbidden_tenses", FREE_DEFAULTS["forbidden_tenses"])

    st.session_state.setdefault("alpha_max_perfekt_per_finite_verb", 0.0)
    st.session_state.setdefault("alpha_min_lexical_coverage", 0.0)

    st.session_state.setdefault("fine_enabled", False)
    st.session_state.setdefault("fine_mtul_level", "keine Vorgabe")
    st.session_state.setdefault("fine_zipf_level", "keine Vorgabe")
    st.session_state.setdefault("fine_lexvar_level", "keine Vorgabe")
    st.session_state.setdefault("fine_connectors_level", "keine Vorgabe")
    st.session_state.setdefault("fine_forbidden_subclause_types", [])
    st.session_state.setdefault("fine_konjunktiv_mode", "keine Vorgabe")
    st.session_state.setdefault("fine_coherence_hint", "keine")


def apply_preset_if_alpha(mode: str) -> None:
    """
    Lädt Presets bei Alpha 3–6 in den Session State und sperrt die Editierbarkeit in der UI.
    """
    st.session_state["mode"] = mode
    if mode in ALPHA_PRESETS:
        preset = ALPHA_PRESETS[mode]
        st.session_state["alpha_max_sentences"] = preset["max_sentences"]
        st.session_state["alpha_max_words_per_sentence"] = preset["max_words_per_sentence"]
        st.session_state["alpha_max_syllables_per_token"] = preset["max_syllables_per_token"]
        st.session_state["alpha_max_dep_clauses_per_sentence"] = preset["max_dep_clauses_per_sentence"]
        st.session_state["alpha_forbidden_tenses"] = preset["forbidden_tenses"]

        st.session_state["alpha_max_perfekt_per_finite_verb"] = (
            0.0 if preset["max_perfekt_per_finite_verb"] is None else float(preset["max_perfekt_per_finite_verb"])
        )
        st.session_state["alpha_min_lexical_coverage"] = (
            0.0 if preset["min_lexical_coverage"] is None else float(preset["min_lexical_coverage"])
        )


def on_mode_change() -> None:
    apply_preset_if_alpha(st.session_state["mode"])


def build_params_from_state() -> Params:
    """
    Baut das interne Params-Objekt aus dem Session State.

    Dieses Objekt wird in Prompt-Building, LLM-Call und Report verwendet.
    """
    if st.session_state["mode"] == "Ohne Alpha":
        tw = int(st.session_state["target_words"])
        target_words = None if tw == 0 else tw
    else:
        target_words = None

    return Params(
        general=GeneralParams(
            topic=st.session_state["topic"],
            text_type=st.session_state["text_type"],
            target_words=target_words,
        ),
        alpha=AlphaParams(
            mode=st.session_state["mode"],
            max_sentences=int(st.session_state["alpha_max_sentences"]),
            max_words_per_sentence=int(st.session_state["alpha_max_words_per_sentence"]),
            max_syllables_per_token=int(st.session_state["alpha_max_syllables_per_token"]),
            max_dep_clauses_per_sentence=float(st.session_state["alpha_max_dep_clauses_per_sentence"] or 0.0),
            forbidden_tenses=list(st.session_state["alpha_forbidden_tenses"]),
            max_perfekt_per_finite_verb=optional_float_or_none(st.session_state.get("alpha_max_perfekt_per_finite_verb")),
            min_lexical_coverage=optional_float_or_none(st.session_state.get("alpha_min_lexical_coverage")),
        ),
        fine=FineParams(
            enabled=bool(st.session_state.get("fine_enabled", False)),
            mtul_level=clamp_level(str(st.session_state.get("fine_mtul_level", "keine Vorgabe"))),
            zipf_level=clamp_level(str(st.session_state.get("fine_zipf_level", "keine Vorgabe"))),
            lexvar_level=clamp_level(str(st.session_state.get("fine_lexvar_level", "keine Vorgabe"))),
            connectors_level=clamp_level(str(st.session_state.get("fine_connectors_level", "keine Vorgabe"))),
            forbidden_subclause_types=list(st.session_state.get("fine_forbidden_subclause_types", [])),
            konjunktiv_mode=str(st.session_state.get("fine_konjunktiv_mode", "keine Vorgabe")),
            coherence_hint=str(st.session_state.get("fine_coherence_hint", "keine")),
        ),
    )

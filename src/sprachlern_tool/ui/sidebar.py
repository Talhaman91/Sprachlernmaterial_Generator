import streamlit as st

from src.sprachlern_tool.config import TEXT_TYPES, TENSES_ALL, LEVELS_4, SUBCLAUSE_TYPES
from src.sprachlern_tool.ui.state import on_mode_change


def render_sidebar() -> tuple[str, float]:
    """
    Rendert die Sidebar und liefert die LLM-Einstellungen zurück.
    """
    with st.sidebar:
        st.header("LLM")
        api_key = st.text_input("Gemini API Key", type="password", help="API Key für Google Gemini.")
        temperature = st.slider(
            "Temperature",
            0.0,
            1.5,
            0.7,
            0.1,
            help="Niedrig = stabiler/regelkonformer, hoch = variantenreicher.",
        )

        st.divider()

        with st.expander("Allgemein", expanded=True):
            st.text_input("Thema", key="topic", help="Übergeordnetes Thema des Textes, z. B. Alltag, Arbeit, Freizeit.")
            st.selectbox(
                "Textart",
                TEXT_TYPES,
                index=TEXT_TYPES.index(st.session_state["text_type"]) if st.session_state["text_type"] in TEXT_TYPES else 1,
                key="text_type",
                help="Textsorte beeinflusst Struktur, Stil und typische Formulierungen.",
            )

            if st.session_state["mode"] == "Ohne Alpha":
                st.number_input(
                    "Textlänge (Wörter, 0 = unbegrenzt)",
                    min_value=0,
                    max_value=2000,
                    key="target_words",
                    step=10,
                    help="Zielwortanzahl. 0 = keine Vorgabe.",
                )
            else:
                st.caption("Textlänge ist im Alpha-Modus deaktiviert (wird aus Alpha-Constraints abgeleitet).")

        with st.expander("Alpha-Parameter", expanded=True):
            st.selectbox(
                "Modus",
                ["Alpha 3", "Alpha 4", "Alpha 5", "Alpha 6", "Ohne Alpha"],
                key="mode",
                on_change=on_mode_change,
                help="Alpha 3–6 lädt Presets (Parameter gesperrt). Ohne Alpha erlaubt freie Werte (0 = aus).",
            )

            mode = st.session_state["mode"]
            alpha_locked = (mode != "Ohne Alpha")

            st.number_input(
                "Max Sätze (0 = aus)",
                min_value=0,
                max_value=100,
                key="alpha_max_sentences",
                disabled=alpha_locked,
                help="Maximale Satzanzahl. 0 deaktiviert (nur Ohne Alpha).",
            )
            st.number_input(
                "Max Wörter pro Satz (0 = aus)",
                min_value=0,
                max_value=60,
                key="alpha_max_words_per_sentence",
                disabled=alpha_locked,
                help="Maximale Satzlänge in Wörtern. 0 deaktiviert (nur Ohne Alpha).",
            )
            st.number_input(
                "Max Silben pro Token (0 = aus)",
                min_value=0,
                max_value=12,
                key="alpha_max_syllables_per_token",
                disabled=alpha_locked,
                help="Max Silben pro Wort. Silbenzählung via Pyphen (de_DE). 0 deaktiviert (nur Ohne Alpha).",
            )

            current_dep = float(st.session_state.get("alpha_max_dep_clauses_per_sentence", 1.0) or 0.0)
            dep_max = max(10.0, current_dep)
            st.slider(
                "Max Dep-Nebensätze pro Satz (0 = aus)",
                0.0,
                dep_max,
                key="alpha_max_dep_clauses_per_sentence",
                step=0.5,
                disabled=alpha_locked,
                help="Begrenzung der Subordination. Gemessen über UD-Relationen. 0 deaktiviert (nur Ohne Alpha).",
            )

            st.multiselect(
                "Verbotene Tempora",
                options=TENSES_ALL,
                key="alpha_forbidden_tenses",
                disabled=alpha_locked,
                help="Tempora, die im Text nicht verwendet werden sollen.",
            )

            st.number_input(
                "Max Perfekt/finite Verben (0 = aus)",
                min_value=0.0,
                max_value=5.0,
                step=0.1,
                key="alpha_max_perfekt_per_finite_verb",
                disabled=alpha_locked,
                help="Max Verhältnis Perfekt-Konstruktionen / finite Verben. 0 deaktiviert.",
            )

            st.number_input(
                "Min lexikalische Abdeckung (0 = aus)",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                key="alpha_min_lexical_coverage",
                disabled=alpha_locked,
                help="Parameter bleibt im Tool. Die Prüfung erfolgt später extern.",
            )

        with st.expander("Zusatzparameter (sekundär, ändern Alpha nicht)", expanded=False):
            st.checkbox(
                "Zusatzparameter aktivieren",
                key="fine_enabled",
                help="Wenn deaktiviert, werden Zusatzparameter nicht in den Prompt aufgenommen.",
            )

            fine_disabled = not st.session_state.get("fine_enabled", False)

            st.selectbox(
                "MTUL-Komplexität",
                LEVELS_4,
                key="fine_mtul_level",
                disabled=fine_disabled,
                help="MTUL = mittlere Wörter pro T-Unit. Stufen sind Richtwerte.",
            )

            st.multiselect(
                "Zu vermeidende Nebensatzarten",
                options=SUBCLAUSE_TYPES,
                key="fine_forbidden_subclause_types",
                disabled=fine_disabled,
                help="Bestimmte Nebensatztypen, die im Text vermieden werden sollen.",
            )

            st.selectbox(
                "Wortfrequenz (Zipf) – Stufe",
                LEVELS_4,
                key="fine_zipf_level",
                disabled=fine_disabled,
                help="Stufen als Richtwerte.",
            )

            st.selectbox(
                "Lexikalische Vielfalt (TTR/MTLD) – Stufe",
                LEVELS_4,
                key="fine_lexvar_level",
                disabled=fine_disabled,
                help="Stufen als Richtwerte.",
            )

            st.selectbox(
                "Konnektoren-Dichte – Stufe",
                LEVELS_4,
                key="fine_connectors_level",
                disabled=fine_disabled,
                help="Wie viele Konnektoren genutzt werden sollen.",
            )

            st.selectbox(
                "Konjunktiv I/II",
                ["keine Vorgabe", "erlauben", "vermeiden"],
                key="fine_konjunktiv_mode",
                disabled=fine_disabled,
                help="Ob Konjunktivformen genutzt oder vermieden werden sollen.",
            )

            st.selectbox(
                "Kohärenz-Hinweis",
                ["keine", "hoch", "mittel", "niedrig"],
                key="fine_coherence_hint",
                disabled=fine_disabled,
                help="Abstrakter Hinweis zur Kohärenz.",
            )

    return api_key, float(temperature)

import streamlit as st

from src.sprachlern_tool.prompBuilder.system_prompt import build_system_prompt
from src.sprachlern_tool.prompBuilder.user_prompt import build_user_prompt
from src.sprachlern_tool.llm.gemini_client import gemini_generate
from src.sprachlern_tool.nlp.report import build_validation_report
from src.sprachlern_tool.ui.state import ensure_defaults_exist, build_params_from_state
from src.sprachlern_tool.ui.sidebar import render_sidebar


def main() -> None:
    st.set_page_config(page_title="Sprachlernmaterial Generator", layout="wide")
    st.title("Sprachlernmaterial Generator")

    ensure_defaults_exist()
    api_key, temperature = render_sidebar()

    if "text_input" not in st.session_state:
        st.session_state["text_input"] = ""

    if "analysis_report" not in st.session_state:
        st.session_state["analysis_report"] = ""


    params = build_params_from_state()
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(params)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Prompts")
        with st.expander("System Prompt"):
            st.code(system_prompt, language="text")
        with st.expander("User Prompt"):
            st.code(user_prompt, language="text")

    with col2:
        st.subheader("Text")

        # Generierung: schreibt direkt ins Textfeld, das der Nutzer auch manuell bearbeiten kann
        if st.button("Generate", type="primary", use_container_width=True, help="Startet die Textgenerierung."):
            try:
                out = gemini_generate(api_key, system_prompt, user_prompt, temperature)
                st.session_state["text_input"] = out
                st.session_state["analysis_report"] = ""  # optional: alten Report leeren
            except Exception as e:
                st.error(str(e))

        # Textfeld ist IMMER sichtbar und editierbar
        st.text_area(
            "Text (generiert oder manuell einfügen)",
            key="text_input",
            height=520,
            help="Hier steht der generierte Text. Du kannst aber auch deinen eigenen Text einfügen.",
        )

        # Analyse-Button direkt unter dem Textfeld
        if st.button("Analyze", use_container_width=True,
                     help="Analysiert den Text mit den aktuell gewählten Parametern."):
            try:
                current_text = (st.session_state.get("text_input") or "").strip()
                if not current_text:
                    st.warning("Bitte erst Text generieren oder einfügen.")
                else:
                    st.session_state["analysis_report"] = build_validation_report(params, current_text)
            except Exception as e:
                st.error(str(e))

        # Report-Feld ebenfalls immer sichtbar (auch wenn noch leer)
        st.text_area(
            "Validierung / Metriken",
            st.session_state.get("analysis_report", ""),
            height=420,
            help=(
                "Stanza-basierte Auswertung: Token/Sätze/Dep-Parsing + regelbasierte Tempus- und Ratio-Prüfung. "
                "Lexik-Abdeckung ist aktuell nicht geprüft."
            ),
        )


if __name__ == "__main__":
    main()

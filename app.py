import streamlit as st

from src.sprachlern_tool.prompts.system_prompt import build_system_prompt
from src.sprachlern_tool.prompts.user_prompt import build_user_prompt
from src.sprachlern_tool.llm.gemini_client import gemini_generate
from src.sprachlern_tool.nlp.report import build_validation_report
from src.sprachlern_tool.ui.state import ensure_defaults_exist, build_params_from_state
from src.sprachlern_tool.ui.sidebar import render_sidebar


def main() -> None:
    st.set_page_config(page_title="Sprachlernmaterial Generator", layout="wide")
    st.title("Sprachlernmaterial Generator")
    st.caption(
        "Modus: Alpha 3–6 oder Ohne Alpha · Ohne Alpha: 0 = aus · Validierung: Stanza-only · "
        "Lexik-Abdeckung: Parameter bleibt, Prüfung später"
    )

    ensure_defaults_exist()
    api_key, temperature = render_sidebar()

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
        st.subheader("Generierter Text")

        if st.button("Generate", type="primary", use_container_width=True, help="Startet die Textgenerierung."):
            try:
                out = gemini_generate(api_key, system_prompt, user_prompt, temperature)
                st.session_state["last_text"] = out
            except Exception as e:
                st.error(str(e))

        if "last_text" in st.session_state:
            st.text_area("Output", st.session_state["last_text"], height=520, help="Ausgabe des LLMs.")

            try:
                report = build_validation_report(params, st.session_state["last_text"])
                st.text_area(
                    "Validierung / Metriken (Stanza-only)",
                    report,
                    height=420,
                    help=(
                        "Stanza-basierte Auswertung: Token/Sätze/Dep-Parsing + regelbasierte Tempus- und Ratio-Prüfung. "
                        "Lexik-Abdeckung ist aktuell nicht geprüft."
                    ),
                )
            except Exception as e:
                st.error(str(e))
        else:
            st.info("Klick auf Generate.")


if __name__ == "__main__":
    main()

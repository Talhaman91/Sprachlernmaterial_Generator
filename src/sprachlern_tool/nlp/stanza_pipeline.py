import streamlit as st
import stanza


@st.cache_resource
def get_stanza_nlp():
    """
    Initialisiert die Stanza-Pipeline für Deutsch.

    """
    try:
        return stanza.Pipeline(
            "de",
            processors="tokenize,pos,lemma,depparse",
            tokenize_no_ssplit=False,
            use_gpu=False,
            verbose=False,
        )
    except Exception as e:
        raise RuntimeError(
            "Stanza Pipeline konnte nicht gestartet werden. "
            "Hast du vorher 'python -c \"import stanza; stanza.download(\\\"de\\\")\"' ausgeführt?\n"
            f"Originalfehler: {e}"
        )

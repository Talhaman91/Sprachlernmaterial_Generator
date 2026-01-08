from src.sprachlern_tool.config import GEMINI_MODEL


def gemini_generate(api_key: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
    """
    Führt einen einzelnen Generierungs-Call gegen Gemini aus.

    Die Funktion ist bewusst klein gehalten, um sie bei Bedarf leicht austauschen/testen zu können.
    """
    if not api_key.strip():
        raise RuntimeError("Kein Gemini API Key gesetzt.")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    cfg = types.GenerateContentConfig(system_instruction=system_prompt, temperature=float(temperature))
    resp = client.models.generate_content(model=GEMINI_MODEL, contents=user_prompt, config=cfg)

    text = getattr(resp, "text", None)
    if not text:
        raise RuntimeError("Gemini hat keinen Text geliefert.")
    return text.strip()

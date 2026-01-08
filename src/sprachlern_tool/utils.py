from src.sprachlern_tool.config import LEVELS_4


def optional_float_or_none(value) -> float | None:
    """
    Normalisiert UI-Eingaben.

    Streamlit number_input liefert in der Praxis float/int.
    Im Tool bedeutet 0.0: Regel deaktiviert -> None.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if v == 0.0 else v


def clamp_level(value: str) -> str:
    """
    Schützt vor ungültigen Stufenwerten, z. B. bei Session-State Migrations.
    """
    return value if value in LEVELS_4 else "keine Vorgabe"

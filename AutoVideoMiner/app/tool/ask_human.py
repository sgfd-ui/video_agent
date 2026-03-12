"""Human-in-the-loop interrupt primitive."""


def ask_human(reason: str) -> dict[str, str]:
    return {"status": "interrupted", "ask_human": reason}

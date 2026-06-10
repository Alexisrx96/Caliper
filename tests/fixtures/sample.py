"""Sample module used as indexer test corpus."""


class Greeter:
    """Greets people in several languages."""

    def __init__(self, lang: str = "es") -> None:
        self.lang = lang

    def greet(self, name: str) -> str:
        """Return a greeting for `name`."""
        templates = {"es": f"Hola, {name}", "en": f"Hello, {name}"}
        return templates[self.lang]


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

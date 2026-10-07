"""Event sources for shadow mode. A source turns some upstream (a bot's database, an export file, ...) into the
unified, pseudonymized event frame. Raw identifiers / raw text must never leave the source module."""
import importlib
def load(name: str):
    mod = name if "." in name else f"tidal.shadow.sources.{name}"
    return importlib.import_module(mod).Source()

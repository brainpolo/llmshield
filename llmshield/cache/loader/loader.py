"""Corpus loader that works with separate language packages."""

import importlib


def get_available_languages() -> list[str]:
    """Get list of available language corpus packages.

    Returns:
        List of available language codes

    """
    available = []
    common_languages = ["spanish"]  # TODO: Add more languages as needed

    for lang in common_languages:
        try:
            importlib.import_module(f"llmshield_corpus_{lang}")
            available.append(lang)
        except ImportError:
            continue

    return available

"""Test cases for language corpus loading.

Makes sure the necessary language
packages are installed and loaded.
"""

import importlib
import unittest

from llmshield.cache.loader.loader import get_available_languages

SPANISH_LANGUAGE_DEPENDENCY_PACKAGE_NAME = "llmshield_spanish_corpus"


class TestLanguageCorpus(unittest.TestCase):
    """Test cases for language corpus loading.

    Args:
        unittest (module): The unittest module.

    """

    def test_spanish_corpus_installed(self):
        """Test if the Spanish corpus dependency is installed."""
        try:
            importlib.import_module(SPANISH_LANGUAGE_DEPENDENCY_PACKAGE_NAME)
        except ImportError:
            self.fail("Spanish corpus dependency is not installed.")

    def test_spanish_corpus_loaded(self):
        """Test if the Spanish corpus is available in the loader."""
        available_languages = get_available_languages()
        self.assertIn(
            "spanish", available_languages, "Spanish corpus is not loaded."
        )


if __name__ == "__main__":
    unittest.main()

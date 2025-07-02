"""Test package installation and resource accessibility.

Description:
    This test module verifies that the package can be built, installed, and
    that all required data files are properly included and accessible. This
    catches packaging configuration issues that would only surface after
    installation.

Test Classes:
    - TestPackageInstallation: Tests package build and resource access

Author: LLMShield by brainpolo, 2025
"""

import subprocess
import sys
import unittest
from importlib import resources
from pathlib import Path

from llmshield.entity_detector import EntityDetector


class TestPackageInstallation(unittest.TestCase):
    """Test package building and installation to catch packaging issues."""

    def test_package_build_succeeds(self) -> None:
        """Test that the package builds without errors."""
        result = subprocess.run(
            [sys.executable, "-m", "build", "--wheel"],
            check=False,
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            result.returncode,
            0,
            f"Package build failed:\nSTDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}",
        )
        self.assertIn("Successfully built", result.stdout)

    def test_data_files_accessible_after_import(self) -> None:
        """Test that all dictionary files are accessible after import."""
        # Create detector instance which loads all data files
        detector = EntityDetector()

        # Verify all expected data structures are populated
        self.assertGreater(
            len(detector.cache.cities),
            0,
            "Cities list should not be empty after loading cities.txt",
        )
        self.assertGreater(
            len(detector.cache.countries),
            0,
            "Countries list should not be empty after loading countries.txt",
        )
        self.assertGreater(
            len(detector.cache.organisations),
            0,
            "Organisations list should not be empty after loading "
            "organisations.txt",
        )
        self.assertGreater(
            len(detector.cache.english_corpus),
            0,
            "English corpus should not be empty after loading "
            "corpus/english.txt",
        )

    def test_corpus_file_specifically_accessible(self) -> None:
        """Test that the corpus/english.txt file is specifically accessible.

        This was the file that was missing in the original packaging issue.
        """
        # Test direct access to the problematic corpus file
        corpus_path = resources.files("llmshield.matchers.dicts").joinpath(
            "corpus/english.txt"
        )

        self.assertTrue(
            corpus_path.is_file(),
            "corpus/english.txt should be accessible via importlib.resources",
        )

        # Test that we can actually read content from it
        with corpus_path.open("r") as f:
            content = f.read()

        self.assertGreater(
            len(content.strip()),
            0,
            "corpus/english.txt should contain actual content",
        )

        # Verify it contains expected English words
        words = {
            word.strip().lower()
            for word in content.splitlines()
            if word.strip()
        }

        # Test for some common English words that should be in the corpus
        expected_words = {"the", "and", "or", "but", "in", "on", "at", "to"}
        found_words = expected_words.intersection(words)

        self.assertGreater(
            len(found_words),
            0,
            f"Expected to find common English words in corpus, "
            f"but none of {expected_words} were found in the loaded words",
        )

    def test_entity_detection_works_with_packaged_resources(self) -> None:
        """Test that entity detection works with packaged resource files."""
        detector = EntityDetector()

        # Test text that should trigger different types of entity detection
        test_text = "John Smith from London works at Microsoft."

        entities = detector.detect_entities(test_text)

        # Should detect at least some entities
        self.assertGreater(
            len(entities),
            0,
            f"Should detect entities in '{test_text}' but found none. "
            f"This suggests resource files are not loading properly.",
        )

    def test_all_dict_files_accessible(self) -> None:
        """Test that all dictionary files are accessible via importlib."""
        dict_files = [
            "cities.txt",
            "countries.txt",
            "organisations.txt",
            "corpus/english.txt",
        ]

        for dict_file in dict_files:
            with self.subTest(dict_file=dict_file):
                file_path = resources.files(
                    "llmshield.matchers.dicts"
                ).joinpath(dict_file)

                self.assertTrue(
                    file_path.is_file(),
                    f"Dictionary file {dict_file} should be accessible "
                    f"via importlib.resources",
                )

                # Verify we can read the file
                with file_path.open("r") as f:
                    content = f.read()

                self.assertGreater(
                    len(content.strip()),
                    0,
                    f"Dictionary file {dict_file} should contain content",
                )


if __name__ == "__main__":
    unittest.main()

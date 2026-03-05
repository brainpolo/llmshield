"""Test allowlist functionality for excluding terms from PII detection.

Author:
    LLMShield by brainpolo, 2025-2026
"""

import unittest

from llmshield import LLMShield


class TestAllowlist(unittest.TestCase):
    """Test allowlist at instance and per-call levels."""

    def test_instance_level_allowlist(self):
        """Test that allowlisted terms are not cloaked."""
        shield = LLMShield(allowlist=["John"])
        cloaked, entity_map = shield.cloak("Contact John at john@example.com")
        # "John" should pass through, email should be cloaked
        self.assertIn("John", cloaked)
        self.assertNotIn("john@example.com", cloaked)
        self.assertTrue(any("EMAIL" in k for k in entity_map))
        self.assertFalse(any(v == "John" for v in entity_map.values()))

    def test_per_call_allowlist_via_cloak(self):
        """Test per-call allowlist on cloak() method."""
        shield = LLMShield()
        cloaked, _ = shield.cloak(
            "Contact John at john@example.com",
            allowlist=["John"],
        )
        self.assertIn("John", cloaked)
        self.assertNotIn("john@example.com", cloaked)

    def test_per_call_allowlist_via_ask(self):
        """Test per-call allowlist on ask() method."""
        captured = []

        def capture(**kwargs):
            """Capture LLM call arguments."""
            captured.append(kwargs.get("messages", []))
            return "OK"

        shield = LLMShield(llm_func=capture)
        shield.ask(
            messages=[
                {
                    "role": "user",
                    "content": ("Contact John at john@example.com"),
                }
            ],
            allowlist=["John"],
        )
        sent_content = captured[0][0]["content"]
        self.assertIn("John", sent_content)
        self.assertNotIn("john@example.com", sent_content)

    def test_merged_allowlist(self):
        """Test instance + per-call allowlists are merged."""
        shield = LLMShield(allowlist=["John"])
        cloaked, _ = shield.cloak(
            "John and Sarah at john@example.com",
            allowlist=["Sarah"],
        )
        self.assertIn("John", cloaked)
        self.assertIn("Sarah", cloaked)
        self.assertNotIn("john@example.com", cloaked)

    def test_case_insensitive_matching(self):
        """Test allowlist matching is case-insensitive."""
        shield = LLMShield(allowlist=["john"])
        cloaked, entity_map = shield.cloak("Contact John about the project")
        self.assertIn("John", cloaked)
        self.assertFalse(any(v == "John" for v in entity_map.values()))

    def test_case_insensitive_uppercase(self):
        """Test allowlist with uppercase input."""
        shield = LLMShield(allowlist=["JOHN"])
        cloaked, _ = shield.cloak("Contact John about the project")
        self.assertIn("John", cloaked)

    def test_non_allowlisted_entities_still_cloaked(self):
        """Test that entities not in allowlist are still cloaked."""
        shield = LLMShield(allowlist=["John"])
        cloaked, entity_map = shield.cloak(
            "John and Sarah at sarah@example.com"
        )
        self.assertIn("John", cloaked)
        self.assertNotIn("sarah@example.com", cloaked)
        self.assertTrue(any("EMAIL" in k for k in entity_map))

    def test_chaining_preserves_allowlist(self):
        """Test that chaining methods preserve the allowlist."""
        shield = LLMShield(allowlist=["John"]).without_contacts()
        cloaked, _ = shield.cloak("John at john@example.com called 555-1234")
        # John allowlisted, contacts disabled — nothing should be cloaked
        # except possibly other entity types
        self.assertIn("John", cloaked)
        # Email and phone should pass through (contacts disabled)
        self.assertIn("john@example.com", cloaked)

    def test_empty_allowlist_no_effect(self):
        """Test that empty allowlist has no effect."""
        shield = LLMShield(allowlist=[])
        cloaked, _ = shield.cloak("Contact John at john@example.com")
        # Everything should be cloaked as normal
        self.assertNotIn("john@example.com", cloaked)

    def test_none_allowlist_no_effect(self):
        """Test that None allowlist has no effect."""
        shield = LLMShield(allowlist=None)
        cloaked, _ = shield.cloak("Contact John at john@example.com")
        self.assertNotIn("john@example.com", cloaked)

    def test_allowlist_with_email(self):
        """Test allowlisting an email address."""
        shield = LLMShield(allowlist=["info@company.com"])
        cloaked, _ = shield.cloak("Email info@company.com or john@example.com")
        self.assertIn("info@company.com", cloaked)
        self.assertNotIn("john@example.com", cloaked)

    def test_allowlist_preserved_in_cache_size_change(self):
        """Test with_cache_size preserves allowlist."""
        shield = LLMShield(allowlist=["John"]).with_cache_size(500)
        cloaked, _ = shield.cloak("Contact John about the project")
        self.assertIn("John", cloaked)

    def test_allowlist_in_factory_method(self):
        """Test allowlist works with factory classmethods."""
        shield = LLMShield.disable_locations(
            allowlist=["John"],
        )
        cloaked, _ = shield.cloak("John lives in London")
        self.assertIn("John", cloaked)
        # London should pass through (locations disabled)
        self.assertIn("London", cloaked)

    def test_per_call_allowlist_multi_turn(self):
        """Test per-call allowlist applies to all messages."""
        captured = []

        def capture(**kwargs):
            """Capture LLM call arguments."""
            captured.append(kwargs.get("messages", []))
            return "OK"

        shield = LLMShield(llm_func=capture)
        shield.ask(
            messages=[
                {
                    "role": "user",
                    "content": "My friend John called me.",
                },
                {
                    "role": "assistant",
                    "content": "Tell me more about John.",
                },
                {
                    "role": "user",
                    "content": "John likes pizza.",
                },
            ],
            allowlist=["John"],
        )
        # All messages should have "John" preserved
        for msg in captured[0]:
            if msg["content"]:
                self.assertIn("John", msg["content"])


if __name__ == "__main__":
    unittest.main()

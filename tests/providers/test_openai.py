"""Tests for OpenAI standard API to ensure all functionality is fully supported.

@see https://platform.openai.com/docs/api-reference/chat/create
"""

# Standard Library Imports
import os
from unittest import TestCase

from openai import OpenAI

# Third-Party Imports
from pydantic import BaseModel

# Local Imports
from llmshield import LLMShield

# Constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SKIP_OPENAI_TESTS = not OPENAI_API_KEY


class TestModel(BaseModel):
    """Test model."""

    name: str
    age: int


class TestOpenAI(TestCase):
    """Test suite for OpenAI standard API."""

    def setUp(self):
        """Set up the test environment."""
        if SKIP_OPENAI_TESTS:
            self.skipTest("OpenAI API key not provided")

        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.shield = LLMShield(llm_func=self.openai_client.chat.completions.create)
        self.messages = [
            {
                "role": "user",
                "content": "Hello, how are you?",
            },
        ]
        self.structured_messages = [
            {
                "role": "user",
                "content": (
                    "Format the following text into a JSON object: 'John Doe is 30 years old.'"
                ),
            },
        ]
        self.structured_response = TestModel(name="John Doe", age=30)

    def test_openai_standard_api(self):
        """Test OpenAI standard API."""
        response = self.shield.ask(
            model="gpt-4o-mini",
            messages=self.messages,
            temperature=0,
        )
        print(response)
        self.assertIsNotNone(response)

    def test_openai_standard_api_stream(self):
        """Test OpenAI standard API with streaming."""
        response = self.shield.ask(
            model="gpt-4o-mini",
            messages=self.messages,
            temperature=0,
            stream=True,
        )

        for chunk in response:
            self.assertIsNotNone(chunk)
            print(chunk, end="", flush=True)

    def test_openai_beta_api_structured_output(self):
        """Test OpenAI beta API with structured output using parse() - automatic detection."""
        # Use beta API directly - library should automatically detect and handle parameters
        beta_shield = LLMShield(llm_func=self.openai_client.beta.chat.completions.parse)

        response = beta_shield.ask(
            model="gpt-4o-mini",
            messages=self.structured_messages,
            temperature=0,
            response_format=TestModel,
        )

        # Beta API returns ParsedChatCompletion, access the parsed model
        self.assertTrue(hasattr(response, "choices"))
        parsed_model = response.choices[0].message.parsed
        self.assertIsInstance(parsed_model, TestModel)
        self.assertEqual(parsed_model.name, "John")  # Note: LLM may shorten names
        self.assertEqual(parsed_model.age, 30)

    def test_openai_chat_completion_object_integrity(self):
        """Test that ChatCompletion objects maintain their structure after uncloaking."""
        messages = [
            {
                "role": "user",
                "content": (
                    "My name is Alice Smith and I live in New York. Tell me about the weather."
                ),
            }
        ]

        response = self.shield.ask(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
        )

        # Verify full ChatCompletion object structure is intact
        self.assertTrue(hasattr(response, "id"))
        self.assertTrue(hasattr(response, "object"))
        self.assertTrue(hasattr(response, "created"))
        self.assertTrue(hasattr(response, "model"))
        self.assertTrue(hasattr(response, "choices"))
        self.assertTrue(hasattr(response, "usage"))

        # Verify choice structure
        choice = response.choices[0]
        self.assertTrue(hasattr(choice, "index"))
        self.assertTrue(hasattr(choice, "message"))
        self.assertTrue(hasattr(choice, "finish_reason"))

        # Verify message structure
        message = choice.message
        self.assertTrue(hasattr(message, "role"))
        self.assertTrue(hasattr(message, "content"))
        self.assertEqual(message.role, "assistant")
        self.assertIsNotNone(message.content)

        # Verify usage structure
        self.assertTrue(hasattr(response.usage, "prompt_tokens"))
        self.assertTrue(hasattr(response.usage, "completion_tokens"))
        self.assertTrue(hasattr(response.usage, "total_tokens"))
        self.assertGreater(response.usage.total_tokens, 0)

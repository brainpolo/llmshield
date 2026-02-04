"""OpenAI provider integration tests with real API.

Description:
    This test module provides integration tests for the OpenAI provider,
    testing real API interactions including chat completions, streaming,
    tool calls, and structured outputs when API credentials are available.

Test Classes:
    - TestOpenAIBasic: Basic chat completion tests
    - TestOpenAIStreaming: Streaming response tests
    - TestOpenAIBeta: Beta features and structured outputs
    - TestOpenAITools: Tool/function calling tests

Author:
    LLMShield by brainpolo, 2025-2026
"""

# Standard Library Imports
import json
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
        self.shield = LLMShield(
            llm_func=self.openai_client.chat.completions.create,
        )
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
                    "Format the following text into a JSON object: "
                    "'John Doe is 30 years old.'"
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
        # Verify response is not None
        assert response is not None

    def test_openai_standard_api_stream(self):
        """Test OpenAI standard API with streaming."""
        response = self.shield.ask(
            model="gpt-4o-mini",
            messages=self.messages,
            temperature=0,
            stream=True,
        )

        # Collect all chunks
        chunks = list(response)
        assert all(chunk is not None for chunk in chunks)

    def test_openai_beta_api_structured_output(self):
        """Test OpenAI beta API with structured output using parse().

        Tests automatic detection of beta API functionality.
        """
        # Use beta API directly - library should automatically detect and
        # handle parameters
        beta_shield = LLMShield(
            llm_func=self.openai_client.beta.chat.completions.parse,
        )

        response = beta_shield.ask(
            model="gpt-4o-mini",
            messages=self.structured_messages,
            temperature=0,
            response_format=TestModel,
        )

        # Beta API returns ParsedChatCompletion, access the parsed model
        assert hasattr(response, "choices")
        parsed_model = response.choices[0].message.parsed
        assert isinstance(parsed_model, TestModel)
        assert parsed_model.name == "John"  # Note: LLM may shorten names
        expected_age = 30
        assert parsed_model.age == expected_age

    def test_openai_chat_completion_object_integrity(self):
        """Test ChatCompletion object structure preservation.

        Validates that ChatCompletion objects maintain their structure after
        uncloaking.
        """
        messages = [
            {
                "role": "user",
                "content": (
                    "My name is Alice Smith and I live in New York. "
                    "Tell me about the weather."
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

    def test_openai_tool_calls_weather_function(self):
        """Test OpenAI tool/function calling with weather example."""
        # Define the weather tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": (
                        "Get the current weather in a given location"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": (
                                    "The city/state, e.g. San Francisco, CA"
                                ),
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        # Create shield with tool-enabled function
        shield = LLMShield(llm_func=self.openai_client.chat.completions.create)

        # Ask about weather - should trigger tool call
        messages = [
            {
                "role": "user",
                "content": "What's the weather like in London?",
            }
        ]

        response = shield.ask(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            temperature=0,
        )

        # Verify response structure
        self.assertTrue(hasattr(response, "choices"))
        choice = response.choices[0]
        self.assertTrue(hasattr(choice, "message"))

        # Check if tool call was made
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            # Verify tool call structure
            tool_call = choice.message.tool_calls[0]
            self.assertEqual(tool_call.type, "function")
            self.assertEqual(tool_call.function.name, "get_current_weather")

            # Verify arguments contain location
            args = json.loads(tool_call.function.arguments)
            self.assertIn("location", args)
            self.assertIn("London", args["location"])

            # Simulate tool response and continue conversation
            tool_response_messages = messages + [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": choice.message.tool_calls,
                },
                {
                    "role": "tool",
                    "content": "Temperature: 15°C, Partly cloudy",
                    "tool_call_id": tool_call.id,
                },
            ]

            # Get final response with tool result
            final_response = shield.ask(
                model="gpt-4o-mini",
                messages=tool_response_messages,
                tools=tools,
                temperature=0,
            )

            # Verify final response mentions the weather
            final_content = final_response.choices[0].message.content
            self.assertIsNotNone(final_content)
            self.assertIn("15", final_content)
        else:
            # If no tool call, response should still mention weather/London
            content = choice.message.content
            self.assertIsNotNone(content)
            self.assertTrue(
                "weather" in content.lower() or "London" in content
            )

    def test_openai_tool_calls_multi_turn(self):
        """Test multi-turn conversation with tool calls."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform basic arithmetic calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": [
                                    "add",
                                    "subtract",
                                    "multiply",
                                    "divide",
                                ],
                            },
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["operation", "a", "b"],
                    },
                },
            }
        ]

        shield = LLMShield(llm_func=self.openai_client.chat.completions.create)

        # Start conversation
        messages = [{"role": "user", "content": "What is 15 plus 27?"}]

        response = shield.ask(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            temperature=0,
        )

        # Check for tool call
        if (
            hasattr(response.choices[0].message, "tool_calls")
            and response.choices[0].message.tool_calls
        ):
            tool_call = response.choices[0].message.tool_calls[0]

            # Verify calculation request
            args = json.loads(tool_call.function.arguments)
            self.assertEqual(args["operation"], "add")
            self.assertEqual(args["a"], 15)
            self.assertEqual(args["b"], 27)

            # Continue with tool response
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": response.choices[0].message.content,
                        "tool_calls": response.choices[0].message.tool_calls,
                    },
                    {
                        "role": "tool",
                        "content": "42",
                        "tool_call_id": tool_call.id,
                    },
                ]
            )

            # Get final answer
            final_response = shield.ask(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools,
                temperature=0,
            )

            # Verify answer mentions 42
            self.assertIn("42", final_response.choices[0].message.content)

    def test_openai_tool_calls_with_pii(self):
        """Test tool calls with PII in the conversation."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "send_email",
                    "description": "Send an email to a recipient",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {
                                "type": "string",
                                "description": "Email address of recipient",
                            },
                            "subject": {"type": "string"},
                            "body": {"type": "string"},
                        },
                        "required": ["to", "subject", "body"],
                    },
                },
            }
        ]

        shield = LLMShield(llm_func=self.openai_client.chat.completions.create)

        # Message with PII
        messages = [
            {
                "role": "user",
                "content": (
                    "Send an email to john.doe@example.com about the "
                    "meeting with Sarah Johnson tomorrow at 3 PM."
                ),
            }
        ]

        response = shield.ask(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            temperature=0,
        )

        # Verify response structure is maintained
        self.assertTrue(hasattr(response, "choices"))
        choice = response.choices[0]

        # If tool call was made, verify PII was properly handled
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            tool_call = choice.message.tool_calls[0]
            self.assertEqual(tool_call.function.name, "send_email")

            # Check that the email and name were preserved in the response
            args = json.loads(tool_call.function.arguments)

            # The email should be restored in the tool call arguments
            self.assertEqual(args["to"], "john.doe@example.com")

            # Body should mention the person's name
            self.assertIn("Sarah Johnson", args["body"])

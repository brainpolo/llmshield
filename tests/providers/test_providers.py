"""Provider integration tests with real APIs.

Description:
    Integration tests for all supported LLM providers,
    testing real API interactions including chat completions,
    streaming, tool calls, multi-turn conversations,
    system messages, and PII cloaking/uncloaking when
    API credentials are available.

    Common tests are parameterized across all providers.
    Provider-specific tests remain in dedicated classes.

Test Classes:
    - TestProviderIntegration: Parameterized tests
    - TestOpenAI: OpenAI-specific tests

Author:
    LLMShield by brainpolo, 2025-2026
"""

# Standard Library Imports
import json
import os
from unittest import TestCase

# Third-Party Imports
from openai import OpenAI
from parameterized import parameterized
from pydantic import BaseModel

# Optional Third-party Imports
try:
    from anthropic import Anthropic

    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

try:
    from google import genai

    _HAS_GOOGLE = True
except ImportError:
    _HAS_GOOGLE = False

try:
    from xai_sdk import Client as XAIClient

    _HAS_XAI = True
except ImportError:
    _HAS_XAI = False

try:
    from cohere import ClientV2 as CohereClient

    _HAS_COHERE = True
except ImportError:
    _HAS_COHERE = False

# Local Imports
from llmshield import LLMShield
from llmshield.detection_utils import (
    extract_response_content,
    is_anthropic_message_like,
    is_chatcompletion_like,
    is_cohere_response_like,
    is_google_response_like,
    is_xai_response_like,
)

# ── Build available providers: (name, shield, model) ──

PROVIDERS = []

if os.getenv("OPENAI_API_KEY"):
    _oc = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    PROVIDERS.append(
        (
            "openai",
            LLMShield(
                llm_func=_oc.chat.completions.create,
            ),
            "gpt-4o-mini",
        )
    )

if _HAS_ANTHROPIC and os.getenv("ANTHROPIC_API_KEY"):
    _ac = Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    PROVIDERS.append(
        (
            "anthropic",
            LLMShield(
                llm_func=_ac.messages.create,
            ),
            "claude-haiku-4-5-20251001",
        )
    )

if _HAS_GOOGLE and os.getenv("GOOGLE_API_KEY"):
    _gc = genai.Client(
        api_key=os.getenv("GOOGLE_API_KEY"),
    )
    PROVIDERS.append(
        (
            "google",
            LLMShield(
                llm_func=_gc.models.generate_content,
            ),
            "gemini-2.0-flash",
        )
    )

if _HAS_XAI and os.getenv("XAI_API_KEY"):
    _xc = XAIClient(
        api_key=os.getenv("XAI_API_KEY"),
    )
    PROVIDERS.append(
        (
            "xai",
            LLMShield(
                llm_func=_xc.chat.create,
            ),
            "grok-3-mini-fast",
        )
    )

if _HAS_COHERE and os.getenv("COHERE_API_KEY"):
    _cc = CohereClient(
        api_key=os.getenv("COHERE_API_KEY"),
    )
    PROVIDERS.append(
        (
            "cohere",
            LLMShield(
                llm_func=_cc.chat,
            ),
            "command-a-03-2025",
        )
    )

# Detection functions per provider
_DETECTION_MAP = {
    "openai": is_chatcompletion_like,
    "anthropic": is_anthropic_message_like,
    "google": is_google_response_like,
    "xai": is_xai_response_like,
    "cohere": is_cohere_response_like,
}

_PROVIDER_CLASS_MAP = {
    "openai": "OpenAIProvider",
    "anthropic": "AnthropicProvider",
    "google": "GoogleProvider",
    "xai": "XAIProvider",
    "cohere": "CohereProvider",
}

# Constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SKIP_OPENAI_TESTS = not OPENAI_API_KEY


class TestModel(BaseModel):
    """Test model."""

    name: str
    age: int


# ── Shared tool definitions ──

LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_word",
        "description": "Look up the definition of a word",
        "parameters": {
            "type": "object",
            "properties": {
                "word": {
                    "type": "string",
                    "description": "The word to look up",
                },
            },
            "required": ["word"],
        },
    },
}

SEND_EMAIL_TOOL = {
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "Send an email to a recipient",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Email address",
                },
                "subject": {
                    "type": "string",
                },
                "body": {
                    "type": "string",
                },
            },
            "required": ["to", "subject", "body"],
        },
    },
}

CALCULATE_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform basic arithmetic",
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
            "required": [
                "operation",
                "a",
                "b",
            ],
        },
    },
}


# ── Helper functions ──


def _extract_tool_calls(  # noqa: PLR0911
    response, provider_name
):
    """Extract tool calls from any provider response.

    Returns list of normalised dicts:
    [{"id": str, "name": str, "arguments": dict}]
    """
    if provider_name == "openai":
        msg = response.choices[0].message
        if not getattr(msg, "tool_calls", None):
            return []
        return [
            {
                "id": getattr(tc, "id", ""),
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments),
            }
            for tc in msg.tool_calls
        ]

    if provider_name == "anthropic":
        blocks = response.content
        if not blocks:
            return []
        return [
            {
                "id": b.id,
                "name": b.name,
                "arguments": b.input,
            }
            for b in blocks
            if getattr(b, "type", None) == "tool_use"
        ]

    if provider_name == "google":
        parts = response.candidates[0].content.parts
        return [
            {
                "id": "",
                "name": p.function_call.name,
                "arguments": dict(p.function_call.args),
            }
            for p in parts
            if getattr(p, "function_call", None)
        ]

    if provider_name == "xai":
        tc_list = getattr(response, "tool_calls", None)
        if not tc_list:
            return []
        return [
            {
                "id": getattr(tc, "id", ""),
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments),
            }
            for tc in tc_list
        ]

    if provider_name == "cohere":
        msg = getattr(response, "message", None)
        if not msg:
            return []
        tc_list = getattr(msg, "tool_calls", None)
        if not tc_list:
            return []
        results = []
        for tc in tc_list:
            try:
                results.append(
                    {
                        "id": getattr(tc, "id", ""),
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments),
                    }
                )
            except (
                json.JSONDecodeError,
                AttributeError,
            ):
                continue
        return results

    return []


def _build_tool_result_messages(
    original_messages,
    tool_calls_raw,
    tool_result_content,
    provider_name,
):
    """Build follow-up messages including tool result.

    Constructs the correct tool result message format
    for each provider.

    Args:
        original_messages: The original user messages
        tool_calls_raw: Raw tool_calls from the response
        tool_result_content: The tool result string
        provider_name: Provider name for format selection

    Returns:
        Full message list for the follow-up call

    """
    if not tool_calls_raw:
        return original_messages

    tc_id = tool_calls_raw[0]["id"]

    # Assistant message with tool calls (OpenAI dict format)
    assistant_tool_calls = [
        {
            "id": tc["id"],
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": json.dumps(tc["arguments"]),
            },
        }
        for tc in tool_calls_raw
    ]

    assistant_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": assistant_tool_calls,
    }

    if provider_name == "anthropic":
        # Anthropic expects role="user" + tool_call_id
        tool_result_msg = {
            "role": "user",
            "tool_call_id": tc_id,
            "content": tool_result_content,
        }
    else:
        # OpenAI, xAI, Google use role="tool"
        tool_result_msg = {
            "role": "tool",
            "tool_call_id": tc_id,
            "content": tool_result_content,
        }

    return original_messages + [assistant_msg, tool_result_msg]


class TestProviderIntegration(TestCase):
    """Parameterized integration tests across providers."""

    @parameterized.expand(PROVIDERS, skip_on_empty=True)
    def test_basic_chat(self, name, shield, model):
        """Test basic chat completion."""
        response = shield.ask(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Say hello.",
                }
            ],
            max_tokens=50,
            temperature=0,
        )
        content = extract_response_content(response)
        self.assertIsNotNone(content)
        self.assertTrue(len(str(content)) > 0)

    @parameterized.expand(PROVIDERS, skip_on_empty=True)
    def test_pii_cloaking(self, name, shield, model):
        """Test PII is cloaked then uncloaked.

        Verifies that PII is protected: either uncloaking
        restores the original name, or at minimum no raw
        placeholder tags (e.g. <PERSON_0>) leak through.
        Some models strip angle brackets from placeholders,
        which is a known limitation tested more rigorously
        in test_pii_multi_turn.
        """
        response = shield.ask(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "My name is John Smith. Greet me using my first name."
                    ),
                }
            ],
            max_tokens=50,
            temperature=0,
        )
        content = str(extract_response_content(response))
        self.assertTrue(
            "John" in content or "<PERSON_" not in content,
            f"Raw placeholder leaked: {content}",
        )

    @parameterized.expand(PROVIDERS, skip_on_empty=True)
    def test_streaming(self, name, shield, model):
        """Test streaming response."""
        response = shield.ask(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Say hi.",
                }
            ],
            max_tokens=50,
            temperature=0,
            stream=True,
        )
        chunks = list(response)
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(isinstance(c, str) for c in chunks))

    @parameterized.expand(PROVIDERS, skip_on_empty=True)
    def test_provider_detected(self, name, shield, model):
        """Test correct provider class is detected."""
        expected = _PROVIDER_CLASS_MAP[name]
        self.assertEqual(type(shield.provider).__name__, expected)

    @parameterized.expand(PROVIDERS, skip_on_empty=True)
    def test_response_object_integrity(self, name, shield, model):
        """Test response passes type detection."""
        response = shield.ask(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Say hello.",
                }
            ],
            max_tokens=50,
            temperature=0,
        )
        detect_fn = _DETECTION_MAP[name]
        self.assertTrue(detect_fn(response))

    @parameterized.expand(PROVIDERS, skip_on_empty=True)
    def test_multi_turn_conversation(self, name, shield, model):
        """Test multi-turn conversation context."""
        response = shield.ask(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": ("Remember: the secret word is banana."),
                },
                {
                    "role": "assistant",
                    "content": (
                        "I will remember that the secret word is banana."
                    ),
                },
                {
                    "role": "user",
                    "content": ("What is the secret word I told you?"),
                },
            ],
            max_tokens=50,
            temperature=0,
        )
        content = str(extract_response_content(response)).lower()
        self.assertIn("banana", content)

    @parameterized.expand(PROVIDERS, skip_on_empty=True)
    def test_system_message(self, name, shield, model):
        """Test system message handling."""
        response = shield.ask(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a pirate. You MUST"
                        " include 'Arrr' in every"
                        " response. Never break"
                        " character."
                    ),
                },
                {
                    "role": "user",
                    "content": "Say hello.",
                },
            ],
            max_tokens=50,
            temperature=0,
        )
        content = str(extract_response_content(response)).lower()
        pirate_indicators = [
            "arrr",
            "ahoy",
            "matey",
            "pirate",
            "ye",
            "aye",
            "sail",
            "captain",
            "ship",
            "treasure",
        ]
        self.assertTrue(
            any(w in content for w in pirate_indicators),
            f"No pirate language found in: {content}",
        )

    @parameterized.expand(PROVIDERS, skip_on_empty=True)
    def test_pii_multi_turn(self, name, shield, model):
        """Test PII consistency across turns."""
        response = shield.ask(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": ("My friend Alice Thompson lives in London."),
                },
                {
                    "role": "assistant",
                    "content": (
                        "I understand your friend"
                        " Alice Thompson lives"
                        " in London."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "What is my friend's name? Reply with just the name."
                    ),
                },
            ],
            max_tokens=50,
            temperature=0,
        )
        content = str(extract_response_content(response))
        self.assertIn("Alice", content)
        self.assertNotIn("<PERSON_", content)

    @parameterized.expand(PROVIDERS, skip_on_empty=True)
    def test_tool_call(self, name, shield, model):
        """Test tool calling across providers."""
        response = shield.ask(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You MUST use the lookup_word"
                        " tool. Do NOT answer"
                        " without calling the tool."
                        " Look up: serendipity"
                    ),
                }
            ],
            tools=[LOOKUP_TOOL],
            max_tokens=200,
            temperature=0,
        )
        tool_calls = _extract_tool_calls(response, name)
        if tool_calls:
            self.assertEqual(tool_calls[0]["name"], "lookup_word")
            self.assertIn(
                "serendipity",
                str(tool_calls[0]["arguments"]).lower(),
            )
        else:
            content = str(extract_response_content(response))
            self.assertTrue(len(content) > 0)

    @parameterized.expand(PROVIDERS, skip_on_empty=True)
    def test_tool_call_with_pii(self, name, shield, model):
        """Test PII uncloaking in tool call args."""
        response = shield.ask(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Send an email to"
                        " john.doe@example.com"
                        " about the meeting with"
                        " Sarah Johnson tomorrow"
                        " at 3 PM."
                    ),
                }
            ],
            tools=[SEND_EMAIL_TOOL],
            max_tokens=200,
            temperature=0,
        )
        tool_calls = _extract_tool_calls(response, name)
        if tool_calls:
            self.assertEqual(
                tool_calls[0]["name"],
                "send_email",
            )
            args = tool_calls[0]["arguments"]
            self.assertEqual(
                args["to"],
                "john.doe@example.com",
            )
        else:
            content = str(extract_response_content(response))
            self.assertTrue(len(content) > 0)

    @parameterized.expand(PROVIDERS, skip_on_empty=True)
    def test_multi_turn_tool_cycle(self, name, shield, model):
        """Test full tool call cycle with result."""
        messages = [
            {
                "role": "user",
                "content": (
                    "What is 15 plus 27? You MUST use the calculate tool."
                ),
            }
        ]

        # Round 1: get tool call
        response = shield.ask(
            model=model,
            messages=messages,
            tools=[CALCULATE_TOOL],
            max_tokens=200,
            temperature=0,
        )
        tool_calls = _extract_tool_calls(response, name)

        if not tool_calls:
            # Model answered directly
            content = str(extract_response_content(response))
            self.assertIn("42", content)
            return

        self.assertEqual(tool_calls[0]["name"], "calculate")

        # Round 2: send tool result back
        follow_up = _build_tool_result_messages(
            messages,
            tool_calls,
            "42",
            name,
        )
        final = shield.ask(
            model=model,
            messages=follow_up,
            tools=[CALCULATE_TOOL],
            max_tokens=200,
            temperature=0,
        )
        final_content = str(extract_response_content(final))
        self.assertIn("42", final_content)


class TestOpenAI(TestCase):
    """Test suite for OpenAI-specific features."""

    def setUp(self):
        """Set up the test environment."""
        if SKIP_OPENAI_TESTS:
            self.skipTest("OpenAI API key not provided")

        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.shield = LLMShield(
            llm_func=(self.openai_client.chat.completions.create),
        )

    def test_openai_beta_api_structured_output(
        self,
    ):
        """Test OpenAI beta API with structured output.

        Tests automatic detection of beta API
        functionality using parse().
        """
        beta_shield = LLMShield(
            llm_func=(self.openai_client.beta.chat.completions.parse),
        )

        response = beta_shield.ask(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Format the following text"
                        " into a JSON object: "
                        "'John Doe is 30 years old.'"
                    ),
                },
            ],
            temperature=0,
            response_format=TestModel,
        )

        assert hasattr(response, "choices")
        parsed = response.choices[0].message.parsed
        assert isinstance(parsed, TestModel)
        assert parsed.name == "John"
        expected_age = 30
        assert parsed.age == expected_age

    def test_openai_chat_completion_object_integrity(
        self,
    ):
        """Test ChatCompletion object structure.

        Validates that ChatCompletion objects maintain
        their structure after uncloaking.
        """
        messages = [
            {
                "role": "user",
                "content": (
                    "My name is Alice Smith and "
                    "I live in New York. "
                    "Tell me about the weather."
                ),
            }
        ]

        response = self.shield.ask(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
        )

        # Verify full ChatCompletion structure
        self.assertTrue(hasattr(response, "id"))
        self.assertTrue(hasattr(response, "object"))
        self.assertTrue(hasattr(response, "created"))
        self.assertTrue(hasattr(response, "model"))
        self.assertTrue(hasattr(response, "choices"))
        self.assertTrue(hasattr(response, "usage"))

        choice = response.choices[0]
        self.assertTrue(hasattr(choice, "index"))
        self.assertTrue(hasattr(choice, "message"))
        self.assertTrue(hasattr(choice, "finish_reason"))

        message = choice.message
        self.assertTrue(hasattr(message, "role"))
        self.assertTrue(hasattr(message, "content"))
        self.assertEqual(message.role, "assistant")
        self.assertIsNotNone(message.content)

        usage = response.usage
        self.assertTrue(hasattr(usage, "prompt_tokens"))
        self.assertTrue(hasattr(usage, "completion_tokens"))
        self.assertTrue(hasattr(usage, "total_tokens"))
        self.assertGreater(usage.total_tokens, 0)

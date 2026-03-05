"""Google GenAI provider for handling Google Gemini SDK.

Description:
    This module provides specialised handling for the Google GenAI
    SDK (google-genai package). It manages the Content/Part message
    format and GenerateContentConfig required by Google's API.

    The user should pass client.models.generate_content as the
    llm_func.

Classes:
    GoogleProvider: Specialised provider for Google GenAI integration

Author:
    LLMShield by brainpolo, 2025-2026
"""

# Standard Library Imports
import json
from collections.abc import Callable, Generator
from typing import Any

# Optional Third-party Import (google-genai)
try:
    from google.genai import types as _google_types

    _HAS_GOOGLE_GENAI = True
except ImportError:  # pragma: no cover
    _google_types = None
    _HAS_GOOGLE_GENAI = False

# Local Imports
from .base import BaseLLMProvider


class GoogleProvider(BaseLLMProvider):
    """Provider for Google GenAI SDK (google-genai).

    Handles the Content/Part message format:
    types.Content(role, parts=[types.Part.from_text(text)])

    System messages go into GenerateContentConfig as
    system_instruction. Role 'assistant' maps to 'model'.

    The llm_func should be client.models.generate_content.
    """

    def execute_single_message(
        self,
        cloaked_text: str,
        input_param: str,
        stream: bool,
        **kwargs,
    ) -> tuple[Any, bool]:
        """Execute single-message Google call."""
        messages = [{"role": "user", "content": cloaked_text}]
        return self._execute(messages, stream, **kwargs)

    def execute_multi_message(
        self,
        cloaked_messages: list[dict],
        stream: bool,
        **kwargs,
    ) -> tuple[Any, bool]:
        """Execute multi-message Google call."""
        return self._execute(cloaked_messages, stream, **kwargs)

    def execute_raw(self, **kwargs) -> Any:
        """Execute Google call without cloaking."""
        stream = kwargs.pop("stream", False)
        prompt = kwargs.pop("prompt", None)
        message = kwargs.pop("message", None)
        messages = kwargs.pop("messages", None)

        if messages is None:
            text = prompt or message or ""
            messages = [{"role": "user", "content": str(text)}]

        response, _ = self._execute(messages, stream, **kwargs)
        return response

    def _execute(
        self,
        messages: list[dict],
        stream: bool,
        **kwargs,
    ) -> tuple[Any, bool]:
        """Build Google request and execute.

        Converts messages to Content/Part format, builds
        GenerateContentConfig from kwargs, and calls
        llm_func (client.models.generate_content).

        Args:
            messages: List of message dicts with role/content
            stream: Whether to stream the response
            **kwargs: Params for generate_content (model,
                temperature, max_output_tokens, etc.)

        Returns:
            Tuple of (response, actual_stream_flag)

        """
        # Clean kwargs not for generate_content
        kwargs.pop("messages", None)
        kwargs.pop("prompt", None)
        kwargs.pop("message", None)
        kwargs.pop("stream", None)

        # Extract model (direct param to generate_content)
        model = kwargs.pop("model", None)

        # Convert messages to Google Content format
        contents, system_text = self._convert_messages(messages)

        # Convert OpenAI-style tools to Google format
        if "tools" in kwargs:
            kwargs["tools"] = self._convert_tools(kwargs["tools"])

        # Build or use provided config
        config = kwargs.pop("config", None)
        if config is None:
            config = self._build_config(system_text, kwargs)

        # Build call arguments
        call_kwargs: dict[str, Any] = {
            "contents": contents,
        }
        if config is not None:
            call_kwargs["config"] = config
        if model is not None:
            call_kwargs["model"] = model

        # Execute
        if stream:
            stream_func = self._get_stream_func()
            if stream_func is not None:
                return (
                    self._stream(stream_func, call_kwargs),
                    True,
                )

        return self.llm_func(**call_kwargs), False

    @staticmethod
    def _convert_tool_calls(
        msg: dict,
        content: str,
        tool_call_names: dict[str, str],
    ) -> _google_types.Content:
        """Convert assistant tool call message to Google Content."""
        parts = []
        if content:
            parts.append(_google_types.Part.from_text(text=content))
        for tc in msg["tool_calls"]:
            func = tc.get("function", {})
            tc_id = tc.get("id", "")
            name = func.get("name", "")
            tool_call_names[tc_id] = name
            args = func.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            parts.append(
                _google_types.Part.from_function_response(
                    name=name,
                    response={"call": args},
                )
            )
        return _google_types.Content(role="model", parts=parts)

    @staticmethod
    def _convert_tool_result(
        msg: dict,
        content: str,
        tool_call_names: dict[str, str],
    ) -> _google_types.Content:
        """Convert tool result message to Google Content."""
        tc_id = msg.get("tool_call_id", "")
        name = msg.get("name") or tool_call_names.get(tc_id, "tool")
        return _google_types.Content(
            role="user",
            parts=[
                _google_types.Part.from_function_response(
                    name=name,
                    response={"result": content},
                )
            ],
        )

    @staticmethod
    def _convert_messages(
        messages: list[dict],
    ) -> tuple[list, str | None]:
        """Convert message dicts to Google Content objects.

        Extracts system messages as system_instruction and
        converts user/assistant messages to Content objects.
        Handles OpenAI-style tool call and tool result
        messages by converting to Google's format.

        Args:
            messages: List of message dicts with role/content

        Returns:
            Tuple of (contents list, system_instruction)

        """
        contents = []
        system_parts: list[str] = []
        tool_call_names: dict[str, str] = {}

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content") or ""

            if isinstance(content, list):
                content = str(content)

            if role == "system":
                if content:
                    system_parts.append(content)
                continue

            if role == "assistant" and msg.get("tool_calls"):
                contents.append(
                    GoogleProvider._convert_tool_calls(
                        msg, content, tool_call_names
                    )
                )
                continue

            if role == "tool":
                contents.append(
                    GoogleProvider._convert_tool_result(
                        msg, content, tool_call_names
                    )
                )
                continue

            google_role = "model" if role == "assistant" else role
            contents.append(
                _google_types.Content(
                    role=google_role,
                    parts=[_google_types.Part.from_text(text=content)],
                )
            )

        system_text = "\n".join(system_parts) if system_parts else None
        return contents, system_text

    @staticmethod
    def _build_config(
        system_text: str | None,
        kwargs: dict,
    ) -> Any | None:
        """Build GenerateContentConfig from kwargs.

        Extracts config-relevant params from kwargs and
        builds a GenerateContentConfig. All remaining kwargs
        after model/messages/stream/config are popped are
        treated as config params.

        Args:
            system_text: System instruction from messages
            kwargs: Remaining kwargs to extract config from

        Returns:
            GenerateContentConfig or None

        """
        config_kwargs: dict[str, Any] = {}

        if system_text:
            config_kwargs["system_instruction"] = system_text

        # Translate max_tokens -> max_output_tokens
        max_tokens = kwargs.pop("max_tokens", None)
        if max_tokens is not None:
            kwargs.setdefault("max_output_tokens", max_tokens)

        # All remaining kwargs are config params
        config_kwargs.update(kwargs)
        kwargs.clear()

        if not config_kwargs:
            return None

        return _google_types.GenerateContentConfig(**config_kwargs)

    @staticmethod
    def _convert_tools(tools: list) -> list:
        """Convert OpenAI-style tools to Google format.

        OpenAI: [{"type": "function", "function":
                  {"name": ..., "parameters": ...}}]
        Google: [{"function_declarations":
                  [{"name": ..., "parameters": ...}]}]
        """
        declarations = []
        passthrough = []
        for tool in tools:
            if (
                isinstance(tool, dict)
                and tool.get("type") == "function"
                and "function" in tool
            ):
                func = tool["function"]
                declarations.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {}),
                    }
                )
            else:
                passthrough.append(tool)
        if declarations:
            passthrough.append({"function_declarations": declarations})
        return passthrough

    def _get_stream_func(self) -> Callable | None:
        """Get the streaming variant of generate_content.

        Accesses generate_content_stream on the same
        models object via the bound method's __self__.

        Returns:
            Stream function or None if unavailable

        """
        models_obj = getattr(self.llm_func, "__self__", None)
        return getattr(
            models_obj,
            "generate_content_stream",
            None,
        )

    @staticmethod
    def _stream(
        stream_func: Callable,
        call_kwargs: dict,
    ) -> Generator[str, None, None]:
        """Wrap Google streaming to yield string chunks."""
        for chunk in stream_func(**call_kwargs):
            if chunk.text:
                yield chunk.text

    @classmethod
    def can_handle(cls, llm_func: Callable) -> bool:
        """Check if this is a Google GenAI function."""
        if not _HAS_GOOGLE_GENAI:
            return False
        func_module = getattr(llm_func, "__module__", "")
        return "google.genai" in func_module

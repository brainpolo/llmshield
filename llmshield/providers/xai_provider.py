"""xAI provider for handling xAI SDK specifics.

Description:
    This module provides specialised handling for xAI SDK functions,
    using the official xai-sdk package (gRPC-based). It manages the
    stateful chat builder pattern required by xAI's API.

    The user should pass client.chat.create as the llm_func.

Classes:
    XAIProvider: Specialised provider for xAI SDK integration

Author:
    LLMShield by brainpolo, 2025-2026
"""

# Standard Library Imports
import json
from collections.abc import Callable, Generator
from typing import Any

# Optional Third-party Import (xai-sdk)
try:
    from xai_sdk import chat as _xai_chat
    from xai_sdk.chat import chat_pb2 as _xai_pb2

    _HAS_XAI_SDK = True
except ImportError:  # pragma: no cover
    _xai_chat = None
    _xai_pb2 = None
    _HAS_XAI_SDK = False

# Local Imports
from .base import BaseLLMProvider


class XAIProvider(BaseLLMProvider):
    """Provider for xAI SDK (gRPC-based).

    Handles the stateful chat builder pattern:
    client.chat.create() -> chat.append() -> chat.sample()

    The llm_func should be client.chat.create.
    """

    def execute_single_message(
        self,
        cloaked_text: str,
        input_param: str,
        stream: bool,
        **kwargs,
    ) -> tuple[Any, bool]:
        """Execute single-message xAI call."""
        messages = [{"role": "user", "content": cloaked_text}]
        return self._execute(messages, stream, **kwargs)

    def execute_multi_message(
        self,
        cloaked_messages: list[dict],
        stream: bool,
        **kwargs,
    ) -> tuple[Any, bool]:
        """Execute multi-message xAI call."""
        return self._execute(cloaked_messages, stream, **kwargs)

    def execute_raw(self, **kwargs) -> Any:
        """Execute xAI call without cloaking."""
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
        """Build xAI chat and execute.

        Extracts xAI-specific params from kwargs, creates a
        chat via llm_func (client.chat.create), appends
        messages, and calls sample() or stream().

        Args:
            messages: List of message dicts with role/content
            stream: Whether to stream the response
            **kwargs: Params for client.chat.create (model,
                temperature, tools, etc.)

        Returns:
            Tuple of (response, actual_stream_flag)

        """
        # Clean kwargs that are not for chat.create
        kwargs.pop("messages", None)
        kwargs.pop("prompt", None)
        kwargs.pop("message", None)
        kwargs.pop("stream", None)

        # Remove token limits (not supported by chat.create)
        kwargs.pop("max_tokens", None)
        kwargs.pop("max_completion_tokens", None)

        # Convert OpenAI-style tools to xAI format
        if "tools" in kwargs:
            kwargs["tools"] = self._convert_tools(kwargs["tools"])

        # Create chat via llm_func (client.chat.create)
        chat = self.llm_func(**kwargs)

        # Append messages using xai_sdk helpers
        self._append_messages(chat, messages)

        # Execute
        if stream:
            return self._stream(chat), True
        return chat.sample(), False

    @staticmethod
    def _append_messages(chat: Any, messages: list[dict]) -> None:
        """Append messages to xAI chat object."""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content") or ""

            if isinstance(content, list):
                content = str(content)

            if role == "system":
                chat.append(_xai_chat.system(content))
            elif role == "user":
                chat.append(_xai_chat.user(content))
            elif role == "assistant" and msg.get("tool_calls"):
                # Build protobuf Message with tool_calls
                content_parts = []
                if content:
                    content_parts.append(_xai_pb2.Content(text=content))
                tool_calls_pb = []
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    args = func.get("arguments", "")
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    tool_calls_pb.append(
                        _xai_pb2.ToolCall(
                            id=tc.get("id", ""),
                            function=_xai_pb2.FunctionCall(
                                name=func.get("name", ""),
                                arguments=args,
                            ),
                        )
                    )
                chat.append(
                    _xai_pb2.Message(
                        role=_xai_pb2.MessageRole.ROLE_ASSISTANT,
                        content=content_parts,
                        tool_calls=tool_calls_pb,
                    )
                )
            elif role == "assistant":
                chat.append(_xai_chat.assistant(content))
            elif role == "tool":
                tc_id = msg.get("tool_call_id", "")
                chat.append(_xai_chat.tool_result(content, tool_call_id=tc_id))

    @staticmethod
    def _stream(
        chat: Any,
    ) -> Generator[str, None, None]:
        """Wrap xAI streaming to yield string chunks."""
        for _response, chunk in chat.stream():
            if chunk.content:
                yield chunk.content

    @staticmethod
    def _convert_tools(tools: list) -> list:
        """Convert OpenAI-style tools to xAI format.

        OpenAI: {"type": "function", "function":
                 {"name": ..., "parameters": {dict}}}
        xAI:    {"function":
                 {"name": ..., "parameters": "json_str"}}

        xAI protobuf Tool has no "type" field, and the
        Function.parameters field is a JSON string.
        """
        converted = []
        for tool in tools:
            if (
                isinstance(tool, dict)
                and tool.get("type") == "function"
                and "function" in tool
            ):
                func = dict(tool["function"])
                if "parameters" in func and isinstance(
                    func["parameters"], dict
                ):
                    func["parameters"] = json.dumps(func["parameters"])
                converted.append({"function": func})
            else:
                converted.append(tool)
        return converted

    @classmethod
    def can_handle(cls, llm_func: Callable) -> bool:
        """Check if this is an xAI SDK function."""
        if not _HAS_XAI_SDK:
            return False
        func_module = getattr(llm_func, "__module__", "")
        func_qualname = getattr(llm_func, "__qualname__", "")
        return "xai_sdk" in func_module or "xai_sdk" in func_qualname

"""Streaming response uncloaking module.

Description:
    This module handles the restoration of original sensitive data in
    streaming LLM responses. It uses an intelligent buffering approach to
    handle placeholders that may be split across multiple stream chunks.

Functions:
    stream_uncloak_response: Restore entities in streaming LLM responses

Note:
    This module is intended for internal use only. Users should interact
    with the LLMShield class rather than calling these functions directly.

Author:
    LLMShield by brainpolo, 2025

"""

# Standard library Imports
from collections.abc import Generator


def uncloak_stream_response(
    stream: Generator[str, None, None],
    entity_map: dict[str, str] | None = None,
    start_delimiter: str = "<",
    end_delimiter: str = ">",
) -> Generator[str, None, None]:
    """Uncloaks a stream response by replacing placeholders with values.

    Args:
        stream: The stream of cloaked responses.
        entity_map: A mapping of placeholders to original values.
        start_delimiter: The start delimiter for placeholders.
        end_delimiter: The end delimiter for placeholders.

    Yields:
        str: The uncloaked response chunks.

    """
    buffer = ""
    for chunk in stream:
        # Extract actual text content if possible (for OpenAI
        # ChatCompletionChunk)
        content = getattr(chunk, "choices", None)
        if content and hasattr(chunk.choices[0].delta, "content"):
            text = chunk.choices[0].delta.content or ""
        else:
            text = str(chunk)
        buffer += text

        # Inner function to check if buffer has content
        def is_buffer_used(buffer: str) -> bool:
            """Check if buffer has content."""
            return bool(buffer and buffer.strip())

        while is_buffer_used(buffer):
            # Find the next potential placeholder start
            start_pos = buffer.find(start_delimiter)
            if start_pos == -1:
                # No more placeholders in buffer, yield everything and break
                if is_buffer_used(buffer):
                    yield buffer
                buffer = ""
                break
            if start_pos > 0:
                # Yield text before placeholder and update buffer
                yield buffer[:start_pos]
                buffer = buffer[start_pos:]
            # Look for placeholder end
            end_pos = buffer.find(end_delimiter)
            if end_pos == -1:
                # Incomplete placeholder, wait for more chunks
                break
            # Extract and uncloak complete placeholder
            placeholder = buffer[: end_pos + len(end_delimiter)]
            yield entity_map.get(placeholder, placeholder)  # type: ignore
            buffer = buffer[end_pos + len(end_delimiter) :]
    # Yield any remaining buffer content
    if buffer:
        yield buffer

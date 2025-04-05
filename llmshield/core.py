"""
Core module for llmshield.

This module provides the main LLMShield class for protecting sensitive information
in Large Language Model (LLM) interactions. It handles cloaking of sensitive entities
in prompts before sending to LLMs, and uncloaking of responses to restore the
original information.

Key features:
- Entity detection and protection (names, emails, numbers, etc.)
- Configurable delimiters for entity placeholders
- Direct LLM function integration
- Zero dependencies

Example:
    >>> shield = LLMShield()
    >>> safe_prompt, entities = shield.cloak("Hi, I'm John (john@example.com)")
    >>> response = shield.uncloak(llm_response, entities)
"""


# Python imports
from typing import Callable, Any

# Local imports
from .utils import is_valid_delimiter, PydanticLike
from .cloak_prompt import _cloak_prompt
from .uncloak_response import _uncloak_response


DEFAULT_START_DELIMITER = '<'
DEFAULT_END_DELIMITER = '>'


class LLMShield:
    """
    Main class for LLMShield - protects sensitive information in LLM interactions.

    Example:
        >>> from llmshield import LLMShield
        >>> shield = LLMShield()
        >>> cloaked_prompt, entity_map = shield.cloak("Hi, I'm John Doe (john.doe@example.com)")
        >>> print(cloaked_prompt)
        "Hi, I'm <PERSON_0> (<EMAIL_1>)"
        >>> llm_response = get_llm_response(cloaked_prompt)  # Your LLM call
        >>> original = shield.uncloak(llm_response, entity_map)
    """

    def __init__(self,
                 start_delimiter: str = DEFAULT_START_DELIMITER,
                 end_delimiter: str = DEFAULT_END_DELIMITER,
                 llm_func: Callable[[str], str] | None = None):
        """
        Initialise LLMShield.

        Args:
            start_delimiter: Character(s) to wrap entity placeholders (default: '<')
            end_delimiter: Character(s) to wrap entity placeholders (default: '>')
            llm_func: Optional function that calls your LLM (enables direct usage)
        """
        if not is_valid_delimiter(start_delimiter):
            raise ValueError("Invalid start delimiter")
        if not is_valid_delimiter(end_delimiter):
            raise ValueError("Invalid end delimiter")
        if llm_func and not callable(llm_func):
            raise ValueError("llm_func must be a callable")

        self.start_delimiter = start_delimiter
        self.end_delimiter = end_delimiter
        self._llm_func = llm_func
        self._last_entity_map = None

    def cloak(self, prompt: str) -> tuple[str, dict[str, str]]:
        """
        Cloak sensitive information in the prompt.

        Args:
            prompt: The original prompt containing sensitive information.

        Returns:
            Tuple of (cloaked_prompt, entity_mapping)
        """
        cloaked, entity_map = _cloak_prompt(
            prompt,
            self.start_delimiter,
            self.end_delimiter
        )
        self._last_entity_map = entity_map
        return cloaked, entity_map

    def uncloak(
            self,
            response: str | list[Any] | dict[str, Any] | PydanticLike,
            entity_map: dict[str, str] | None = None
    ) -> str | list[Any] | dict[str, Any] | PydanticLike:
        """
        Restore original entities in the LLM response. It supports strings and
        structured outputs consisting of any combination of strings, lists, and
        dictionaries.

        Limitations:
            - Does not support streaming.
            - Does not support tool calls.

        Args:
            response: The LLM response containing placeholders. Supports both
            strings and structured outputs (dicts).
            entity_map: Mapping of placeholders to original values
                        (if empty, uses mapping from last cloak call)

        Returns:
            Response with original entities restored

        Raises:
            TypeError: If response parameters of invalid type.
            ValueError: If no entity mapping is provided and no previous cloak call.s
        """
        # Validate inputs
        if not response:
            raise ValueError("Response cannot be empty")

        if not isinstance(response, (str, list, dict, PydanticLike)):
            raise TypeError(f"Response must be in [str, list, dict] or a Pydantic model, but got: {type(response)}!")

        if entity_map is None:
            if self._last_entity_map is None:
                raise ValueError("No entity mapping provided or stored from previous cloak!")
            entity_map = self._last_entity_map

        if isinstance(response, PydanticLike):
            model_class = response.__class__
            uncloaked_dict = _uncloak_response(response.model_dump(), entity_map)
            return model_class.model_validate(uncloaked_dict)
        else:
            return _uncloak_response(response, entity_map)

    def ask(self, **kwargs) -> str:
        """
        Complete end-to-end LLM interaction with automatic protection.

        NOTE: If you are using a structured output, ensure that your keys
        do not contain PII and that any keys that may contain PII are either
        string, lists, or dicts. Other types like int, float, are unable to be
        cloaked and will be returned as is.

        Limitations:
            - Does not support streaming.
            - Does not support multiple messages (multi-shot requests).

        Args:
            prompt/message: Original prompt with sensitive information. This will be cloaked
                   and passed to your LLM function. Do not pass both, and do not use any other
                   parameter names as they are unrecognised by the shield.
            **kwargs: Additional arguments to pass to your LLM function, such as:
                    - model: The model to use (e.g., "gpt-4")
                    - system_prompt: System instructions
                    - temperature: Sampling temperature
                    - max_tokens: Maximum tokens in response
                    etc.
        ! The arguments do not have to be in any specific order!

        Returns:
            str: Uncloaked LLM response

        Raises:
            ValueError: If no LLM function was provided during initialization,
                       if prompt is invalid, or if both prompt and message are provided
        """
        # * 1. Validate inputs
        if self._llm_func is None:
            raise ValueError("No LLM function provided. Either provide llm_func in constructor "
                           "or use cloak/uncloak separately.")

        if 'prompt' not in kwargs and 'message' not in kwargs:
            raise ValueError("Either 'prompt' or 'message' must be provided!")


        if 'prompt' in kwargs and 'message' in kwargs:
            raise ValueError("Do not provide both 'prompt' and 'message'. Use only 'prompt' "
                           "parameter - it will be passed to your LLM function.")

        # * 2. Get the input text and determine parameter name
        input_param = 'message' if 'message' in kwargs else 'prompt'
        input_text = kwargs[input_param]

        # * 3. Cloak the input text
        cloaked_text, entity_map = self.cloak(input_text)

        # * 4. Pass the cloaked text under the correct parameter name for the LLM function
        func_preferred_param = 'message' if 'message' in self._llm_func.__code__.co_varnames else 'prompt'

        # Remove the original parameter and add under the LLM's preferred name
        del kwargs[input_param]
        kwargs[func_preferred_param] = cloaked_text

        # * 5. Get response from LLM
        llm_response = self._llm_func(**kwargs)

        # * 6. Uncloak and return
        return self.uncloak(llm_response, entity_map)

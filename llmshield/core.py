"""
Core module for llmshield.
"""

from typing import Tuple, Dict, Optional, Callable

from .utils import is_valid_delimiter
from .cloak_prompt import _cloak_prompt
from .uncloak_response import _uncloak_response


DEFAULT_START_DELIMITER = '<'
DEFAULT_END_DELIMITER = '>'


class LLMShield:
    """
    Main class for LLMShield - protects sensitive information in LLM interactions.

    Example:
        >>> shield = LLMShield(start_delimiter='<', end_delimiter='>')
        >>> cloaked_prompt, entity_map = shield.cloak("Hi, I'm John Doe (john.doe@example.com)")
        >>> print(cloaked_prompt)
        "Hi, I'm <PERSON_0> (<EMAIL_0>)"
        >>> llm_response = get_llm_response(cloaked_prompt)  # Your LLM call
        >>> original = shield.uncloak(llm_response, entity_map)
    """

    def __init__(self,
                 start_delimiter: str = DEFAULT_START_DELIMITER,
                 end_delimiter: str = DEFAULT_END_DELIMITER,
                 llm_func: Optional[Callable[[str], str]] = None):
        """
        Initialise LLMShield.

        @param start_delimiter: Character(s) to wrap entity placeholders (default: '<')
        @param end_delimiter: Character(s) to wrap entity placeholders (default: '>')
        @param llm_func: Optional function that calls your LLM (enables direct usage)
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


    def cloak(self, prompt: str) -> Tuple[str, Dict[str, str]]:
        """
        Cloak sensitive information in the prompt.

        @param prompt: The original prompt containing sensitive information

        @return: Tuple of (cloaked_prompt, entity_mapping)
        """
        cloaked, entity_map = _cloak_prompt(prompt, self.start_delimiter, self.end_delimiter)
        self._last_entity_map = entity_map
        return cloaked, entity_map


    def uncloak(self, response: str, entity_map: Optional[Dict[str, str]] = None) -> str:
        """
        Restore original entities in the LLM response.

        @param response: The LLM response containing placeholders
        @param entity_map: Mapping of placeholders to original values
                          (if None, uses mapping from last cloak call)

        @return: Response with original entities restored
        """
        # * Validate inputs
        if not response or not isinstance(response, str):
            raise ValueError("Response must be a string")
        if entity_map is None:
            if self._last_entity_map is None:
                raise ValueError("No entity mapping provided or stored from previous cloak")
            entity_map = self._last_entity_map
        return _uncloak_response(response, entity_map)


    def ask(self, prompt: str, **kwargs) -> str:
        """
        Complete end-to-end LLM interaction with automatic protection.

        @param prompt: Original prompt with sensitive information
        @param **kwargs: Additional arguments to pass to your LLM function if
                         provided during initialisation.

        @return: Uncloaked LLM response

        @raise ValueError: If no LLM function was provided during initialization
        """
        # * Validate inputs
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a string!")
        if self._llm_func is None:
            raise ValueError("No LLM function provided. Either provide llm_func in constructor "
                           "or use cloak/uncloak separately.")

        cloaked_prompt, entity_map = self.cloak(prompt)
        llm_response = self._llm_func(cloaked_prompt, **kwargs)
        return self.uncloak(llm_response, entity_map)
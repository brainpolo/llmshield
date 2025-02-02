"""
LLMShield: Protect sensitive information in LLM interactions.

Basic usage:
    >>> from llmshield import LLMShield
    >>> shield = LLMShield()
    >>> safe_prompt, entities = shield.cloak("Hi, I'm John (john@example.com)")
    >>> response = shield.uncloak(llm_response, entities)

Direct usage with LLM:
    >>> def my_llm(prompt: str) -> str:
    ...     # Your LLM API call here
    ...     return response

    >>> shield = LLMShield(llm_func=my_llm)
    >>> response = shield.ask(prompt="Hi, I'm John (john@example.com)")
"""

from .core import LLMShield

__all__ = ['LLMShield']

def create_shield(**kwargs):
    """
    Create a new LLMShield instance with the given configuration.

    Args:
        **kwargs: Arguments to pass to LLMShield constructor
            - start_delimiter: Character(s) to wrap entities (default: '<')
            - end_delimiter: Character(s) to wrap entities (default: '>')
            - llm_func: Optional function to call LLM

    Returns:
        LLMShield instance
    """
    return LLMShield(**kwargs)

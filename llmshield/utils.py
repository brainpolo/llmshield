"""
Module for utility functions for the llmshield library.
"""


import re

from llmshield.entity_detector import EntityType


def split_fragments(text: str) -> list[str]:
    """Split the text into fragments based on the following rules:
    - Split on sentence boundaries (punctuation / new line)
    - Remove any empty fragments
    """
    fragments = re.split(r'[.!?]+\s+|\n+', text)
    return [f.strip() for f in fragments if f.strip()]


def is_valid_delimiter(delimiter: str) -> bool:
    """
    Validate a delimiter based on the following rules:
    - Must be a string.
    - Must be at least 1 character long.

    @param delimiter: The delimiter to validate.

    @return: True if the delimiter is valid, False otherwise.
    """
    return isinstance(delimiter, str) and len(delimiter) > 0


def wrap_entity(
        entity_type: EntityType,
        suffix: int,
        start_delimiter: str,
        end_delimiter: str) -> str:
    """
    Wrap an entity in a start and end delimiter.

    The wrapper works as follows:
    - The value will be wrapped with START_DELIMETER and END_DELIMETER.
    - The suffix will be appended to the entity.

    @param entity: The entity to wrap.
    @param start_delimiter: The start delimiter.
    @param end_delimiter: The end delimiter.

    @return: The wrapped entity.
    """
    return f"{start_delimiter}{entity_type.name}_{suffix}{end_delimiter}"


def normalise_spaces(text: str) -> str:
    """Normalise spaces in the text by replacing multiple spaces with a single space."""
    return re.sub(r'\s+', ' ', text).strip()

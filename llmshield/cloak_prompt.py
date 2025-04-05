"""
Module objectives:
- Cloak the prompt before sending it to the LLM.
- Return the cloaked prompt and a mapping of placeholders to original values.

! Module is intended for internal use only.
"""

import re

from .entity_detector import Entity, EntityDetector
from .utils import wrap_entity


# pylint: disable=too-many-locals
def _cloak_prompt(prompt: str, start_delimiter: str, end_delimiter: str) -> tuple[str, dict]:
    """
    Rewritten cloaking function:
    - Collects all match positions from the original prompt.
    - Sorts matches in descending order by their start index.
    - Replaces the matches in one pass.
    """
    detector = EntityDetector()
    entities: set[Entity] = detector.detect_entities(prompt)

    # Collect all match positions and the corresponding placeholder
    matches = []  # Each item will be a tuple: (start, end, placeholder, entity_value)
    counter = 0
    for entity in entities:
        escaped = re.escape(entity.value)
        for match in re.finditer(escaped, prompt):
            placeholder = wrap_entity(entity.type, counter, start_delimiter, end_delimiter)
            matches.append((match.start(), match.end(), placeholder, entity.value))
            counter += 1

    # Sort matches in descending order by the match start index
    matches.sort(key=lambda m: m[0], reverse=True)

    cloaked_prompt = prompt
    entity_map = {}
    for start, end, placeholder, value in matches:
        cloaked_prompt = cloaked_prompt[:start] + placeholder + cloaked_prompt[end:]
        entity_map[placeholder] = value

    return cloaked_prompt, entity_map


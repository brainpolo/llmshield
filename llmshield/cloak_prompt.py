"""Module objectives:
- Cloak the prompt before sending it to the LLM.
- Return the cloaked prompt and a mapping of placeholders to original values.

! Module is intended for internal use only.
"""

import re
from collections import OrderedDict

from .entity_detector import Entity, EntityDetector
from .utils import wrap_entity


# pylint: disable=too-many-locals
def _cloak_prompt(
    prompt: str,
    start_delimiter: str,
    end_delimiter: str,
    entity_map: dict[str, str] | None = None,
) -> tuple[str, dict[str, str]]:
    """
    Rewritten cloaking function:
    - Collects all match positions from the original prompt.
    - Sorts matches in descending order by their start index.
    - Replaces the matches in one pass.
    - Accepts an existing entity_map to maintain placeholder consistency.
    """
    if entity_map is None:
        entity_map = OrderedDict()

    # Create a reverse map for quick lookups of existing values
    reversed_entity_map = {v: k for k, v in entity_map.items()}
    if len(entity_map.keys()) > 0:
        print("Cached Reverse entity map", reversed_entity_map)

    detector = EntityDetector()
    entities: set[Entity] = detector.detect_entities(prompt)

    matches = []
    # The counter should start from the current size of the entity map
    # to ensure new placeholders are unique.
    counter = len(entity_map)

    for entity in entities:
        # If the entity value is already in our map, use the existing placeholder
        if entity.value in reversed_entity_map:
            placeholder = reversed_entity_map[entity.value]
        else:
            # Otherwise, create a new placeholder
            placeholder = wrap_entity(entity.type, counter, start_delimiter, end_delimiter)
            # Add the new entity to the maps
            entity_map[placeholder] = entity.value
            reversed_entity_map[entity.value] = placeholder
            counter += 1

        # Find all occurrences of the entity value in the prompt
        escaped = re.escape(entity.value)
        for match in re.finditer(escaped, prompt):
            matches.append((match.start(), match.end(), placeholder, entity.value))

    # Sort matches in descending order by the match start index to avoid shifting
    matches.sort(key=lambda m: m[0], reverse=True)

    cloaked_prompt = prompt
    # We don't need to build the entity map here again, just replace the text
    for start, end, placeholder, _ in matches:
        cloaked_prompt = cloaked_prompt[:start] + placeholder + cloaked_prompt[end:]

    return cloaked_prompt, entity_map

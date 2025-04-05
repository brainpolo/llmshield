"""
Module for detecting and classifying different types of entities in text.

Uses a combination of rule-based and pattern-based approaches with:
- Regex patterns for structured data (emails, URLs, etc.)
- Dictionary lookups for known entities (cities, countries, organizations)
- Heuristic rules for proper nouns and other entities
"""


import re

from dataclasses import dataclass
from enum import Enum
from importlib import resources


ENT_REPLACEMENT = "\n" # Use to void overlap with another entity

SPACE = " "


class EntityType(str, Enum):
    """Primary classification of entity types"""
    # Proper Nouns
    PERSON = "PERSON"
    ORGANISATION = "ORGANISATION"
    PLACE = "PLACE"
    CONCEPT = "CONCEPT"

    # Numbers
    PHONE_NUMBER = "PHONE_NUMBER"
    CREDIT_CARD = "CREDIT_CARD"

    # Locators
    EMAIL = "EMAIL"
    URL = "URL"
    IP_ADDRESS = "IP_ADDRESS"

class EntityGroup(str, Enum):
    """Groups of related entity types"""
    PNOUN = "PNOUN"
    NUMBER = "NUMBER"
    LOCATOR = "LOCATOR"

    def get_types(self) -> set[EntityType]:
        """Get all entity types belonging to this group"""
        group_map = {
            self.PNOUN: {
                EntityType.PERSON,
                EntityType.ORGANISATION,
                EntityType.PLACE,
                EntityType.CONCEPT
            },
            self.NUMBER: {
                EntityType.PHONE_NUMBER,
                EntityType.CREDIT_CARD
            },
            self.LOCATOR: {
                EntityType.EMAIL,
                EntityType.URL,
                EntityType.IP_ADDRESS
            }
        }
        return group_map[self]

@dataclass(frozen=True)
class Entity:
    """Represents a detected entity in text"""
    type: EntityType
    value: str

    @property
    def group(self) -> EntityGroup:
        """Get the group this entity belongs to"""
        for group in EntityGroup:
            if self.type in group.get_types():
                return group
        raise ValueError(f"Unknown entity type: {self.type}")


class EntityDetector:
    """
    Main entity detection system that combines rule-based and pattern-based
    approaches to identify sensitive information in text.

    Uses a waterfall approach to detection, where each detection method is
    tried in order, and the text is reduced as each entity is found. This
    eliminates the potential of overlapping entities being detected, and
    improves the accuracy of the detection of entity type.
    """

    def __init__(self):
        """
        Initialise large lists of entities.
        """
        from llmshield.utils import split_fragments, normalise_spaces

        from llmshield.matchers.lists import (
            EN_PERSON_INITIALS,
            EN_ORG_COMPONENTS,
            EN_PLACE_COMPONENTS,
            EN_COMMON_WORDS,
            EN_PUNCTUATION
        )

        from llmshield.matchers.regex import (
            EMAIL_ADDRESS_PATTERN,
            CREDIT_CARD_PATTERN,
            IP_ADDRESS_PATTERN,
            URL_PATTERN,
            PHONE_NUMBER_PATTERN
        )

        from llmshield.matchers.functions import _luhn_check

        self.split_fragments = split_fragments
        self.normalise_spaces = normalise_spaces

        self.cities = self._load_cities()
        self.countries = self._load_countries()
        self.organisations = self._load_organisations()
        self.en_person_initials = EN_PERSON_INITIALS
        self.en_org_components = EN_ORG_COMPONENTS
        self.en_place_components = EN_PLACE_COMPONENTS
        self.en_common_words = EN_COMMON_WORDS
        self.en_punctuation = EN_PUNCTUATION

        self.email_pattern = EMAIL_ADDRESS_PATTERN
        self.credit_card_pattern = CREDIT_CARD_PATTERN
        self.ip_address_pattern = IP_ADDRESS_PATTERN
        self.url_pattern = URL_PATTERN
        self.phone_number_pattern = PHONE_NUMBER_PATTERN

        self.luhn_check = _luhn_check


    @staticmethod
    def _load_cities() -> list[str]:
        """Load cities from lists/cities.txt"""
        with resources.files('llmshield.matchers.dicts').joinpath('cities.txt').open('r') as f:
            return [city.strip() for city in f.read().splitlines() if city.strip()]

    @staticmethod
    def _load_countries() -> list[str]:
        """Load countries from lists/countries.txt"""
        with resources.files('llmshield.matchers.dicts').joinpath('countries.txt').open('r') as f:
            return [country.strip() for country in f.read().splitlines() if country.strip()]

    @staticmethod
    def _load_organisations() -> list[str]:
        """Load organisations from lists/organisations.txt"""
        with resources.files('llmshield.matchers.dicts').joinpath('organisations.txt').open('r') as f:
            return [org.strip() for org in f.read().splitlines() if org.strip()]


    def detect_entities(self, text: str) -> set[Entity]:
        """Main entry point for entity detection using waterfall methodology."""
        detection_methods = [
            (self._detect_locators, EntityGroup.LOCATOR),
            (self._detect_numbers, EntityGroup.NUMBER),
            (self._detect_proper_nouns, EntityGroup.PNOUN)
        ]

        entities: set[Entity] = set()
        working_text: str = text

        for method, _ in detection_methods:  # Use _ for unused group variable
            new_entities, working_text = method(working_text)
            entities.update(new_entities)

        return entities

    def _detect_proper_nouns(self, text: str) -> tuple[set[Entity], str]:
        """
        Umbrella method for proper noun detection.
        It collects candidate proper nouns from the text, then classifies each.
        If an entity is classified (and potentially cleaned), it is added as an Entity.
        """
        entities = set()
        reduced_text = text

        # Step 1: Collect sequential proper nouns.
        sequential_pnouns = self._collect_proper_nouns(text)

        # Step 2: Process each proper noun.
        for p_noun in sequential_pnouns:
            if not p_noun or p_noun not in reduced_text:
                continue

            result = self._classify_proper_noun(p_noun)
            if result is None:
                continue

            cleaned_value, entity_type = result
            entities.add(Entity(type=entity_type, value=cleaned_value))
            reduced_text = reduced_text.replace(p_noun, ENT_REPLACEMENT)

        return entities, reduced_text

    def _collect_proper_nouns(self, text: str) -> list[str]:
        """
        Collect sequential proper nouns from text.
        """
        sequential_pnouns = []
        normalised_text = self.normalise_spaces(text)
        fragments = self.split_fragments(normalised_text)

        for fragment in fragments:
            # Split on common contractions first
            for split_word in ["I'm", "I've", "I'll"]:
                if split_word in fragment:
                    fragment = fragment.replace(split_word, f"{split_word} ")

            fragment_words = fragment.split(SPACE)
            pending_p_noun = ""
            skip_next = False

            for i, word in enumerate(fragment_words):
                if skip_next:
                    skip_next = False
                    continue

                if not word:
                    continue

                # Skip personal pronouns and their contractions
                if word in {"I'm", "I've", "I'll", "I"}:
                    if pending_p_noun:
                        sequential_pnouns.append(pending_p_noun.strip())
                        pending_p_noun = ""
                    continue

                # Look ahead for potential name after "I'm", etc.
                if i < len(fragment_words) - 1 and word in {"I'm", "I've", "I'll"}:
                    next_word = fragment_words[i + 1]
                    if next_word[0].isupper():
                        pending_p_noun = next_word
                        skip_next = True
                    continue

                normalized_word = word.strip(".,!?;:")
                is_honorific = normalized_word in self.en_person_initials
                is_capitalised = word and word[0].isupper()

                if is_honorific or (not any(c in word for c in self.en_punctuation if c != ".") and is_capitalised):
                    pending_p_noun = (pending_p_noun + SPACE + word if pending_p_noun else word)
                elif pending_p_noun:
                    sequential_pnouns.append(pending_p_noun.strip())
                    pending_p_noun = ""

            if pending_p_noun:
                sequential_pnouns.append(pending_p_noun.strip())

        return sorted([p for p in sequential_pnouns if p], key=len, reverse=True)

    def _clean_person_name(self, p_noun: str) -> str:
        """
        Remove a leading honorific (if any) from a person proper noun.
        For example "Dr. John Doe" becomes "John Doe".
        """
        words = p_noun.split()
        if not words:
            return p_noun
        first_word_norm = words[0].strip(".,!?;:")
        if first_word_norm in self.en_person_initials and len(words) > 1:
            return " ".join(words[1:]).strip()
        return p_noun

    def _classify_proper_noun(self, p_noun: str) -> tuple[str, EntityType] | None:
        """
        Classify a proper noun into its entity type, and clean it if necessary.

        Returns a tuple (modified_value, EntityType) or None if no classification is made.
        In the PERSON case, if a honorific is detected at the start, it is removed.
        All punctuation is stripped from the final value.
        """
        if not p_noun:
            return None

        def clean_value(value: str) -> str:
            """Remove all punctuation from the value."""
            return ''.join(c for c in value if c not in self.en_punctuation).strip()

        # 1. Check for organizations first.
        if self._is_organization(p_noun):
            return (clean_value(p_noun), EntityType.ORGANISATION)

        # 2. Check for places.
        if self._is_place(p_noun):
            return (clean_value(p_noun), EntityType.PLACE)

        # 3. Check for persons.
        if self._is_person(p_noun):
            cleaned = self._clean_person_name(p_noun)
            final_value = clean_value(cleaned)
            return (final_value, EntityType.PERSON)

        # 4. Check for concepts (e.g. all uppercase, one word, no punctuation).
        if self._is_concept(p_noun):
            return (clean_value(p_noun), EntityType.CONCEPT)

        # 5. Default to None.
        return None

    def _is_concept(self, p_noun: str) -> bool:
        """Check if proper noun is a concept."""
        return (
            all(word.isupper() for word in p_noun.split()) and
            len(p_noun.split()) == 1 and
            not any(c in p_noun for c in self.en_punctuation)
        )

    def _is_organization(self, p_noun: str) -> bool:
        """Check if proper noun is an organization."""
        # Case-insensitive check for organization names
        p_noun_lower = p_noun.lower()

        # Add checks for organizations with numbers
        if any(char.isdigit() for char in p_noun):
            # Check if it matches known patterns like "3M", "7-Eleven"
            if re.match(r'^\d+[A-Z].*|.*-.*\d+.*', p_noun):
                return True

        # Check for multi-word organizations like "New York Times"
        if len(p_noun.split()) > 2:
            last_word = p_noun.split()[-1]
            if last_word in {"Times", "News", "Corporation", "Inc", "Corp", "Co"}:
                return True

        return (
            any(org.lower() == p_noun_lower for org in self.organisations) or
            any(comp in p_noun for comp in self.en_org_components)
        )

    def _is_place(self, p_noun: str) -> bool:
        """Check if proper noun is a place."""
        return (
                any(city.lower() == p_noun.lower() for city in self.cities) or
                any(country.lower() == p_noun.lower() for country in self.countries) or
                any(comp in p_noun.split() for comp in self.en_place_components))

    def _is_person(self, p_noun: str) -> bool:
        """Check if proper noun is a person."""
        words = p_noun.split()

        # Handle possessives
        words = [w.rstrip("'s") for w in words]

        # Must have at least one word after cleaning
        if not words:
            return False

        # Skip honorifics at start
        if words[0].strip(".,!?;:") in self.en_person_initials:
            words = words[1:]

        # Must have remaining words after removing honorifics
        if not words:
            return False

        # Check each word
        for word in words:
            clean_word = word.strip(".,!?;:")

            # Skip empty words
            if not clean_word:
                continue

            # Allow hyphenated names
            if '-' in clean_word:
                parts = clean_word.split('-')
                if not all(part and part[0].isupper() for part in parts):
                    return False
                continue

            # Each word must:
            # 1. Start with capital letter
            # 2. Not be in common words
            # 3. Not contain digits
            if (not clean_word[0].isupper() or
                clean_word.lower() in (w.lower() for w in self.en_common_words) or
                any(c.isdigit() for c in clean_word)):
                return False

        return True

    def _detect_numbers(self, text: str) -> tuple[set[Entity], str]:
        """Detect numbers in the text"""
        entities = set()
        reduced_text = text

        # * 1. Split on sentence boundaries (punctuation / new line)
        emails = self.email_pattern.finditer(text)
        for email in emails:
            entities.add(Entity(
                type=EntityType.EMAIL,
                value=email.group(),
            ))
            reduced_text = reduced_text.replace(email.group(), ENT_REPLACEMENT)

        # * 2. Detect credit cards
        credit_cards = self.credit_card_pattern.finditer(text)
        for credit_card in credit_cards:
            if self.luhn_check(credit_card.group()):
                entities.add(Entity(
                    type=EntityType.CREDIT_CARD,
                    value=credit_card.group(),
                ))
                reduced_text = reduced_text.replace(credit_card.group(), ENT_REPLACEMENT)

        # * 3. Detect phone numbers
        phone_numbers = self.phone_number_pattern.finditer(text)
        for phone_number in phone_numbers:
            entities.add(Entity(
                type=EntityType.PHONE_NUMBER,
                value=phone_number.group(),
            ))
            reduced_text = reduced_text.replace(phone_number.group(), ENT_REPLACEMENT)


        # * 4. Return the reduced text and entities found
        return entities, reduced_text


    def _detect_locators(self, text: str) -> tuple[set[Entity], str]:
        """Detect locators in the text."""
        entities = set()
        reduced_text = text

        # * 1. Detect URLs
        urls = self.url_pattern.finditer(text)
        for url in urls:
            entities.add(Entity(
                type=EntityType.URL,
                value=url.group(),
            ))
            reduced_text = reduced_text.replace(url.group(), ENT_REPLACEMENT)

        # * 2. Detect emails
        emails = self.email_pattern.finditer(text)
        for email in emails:
            entities.add(Entity(
                type=EntityType.EMAIL,
                value=email.group(),
            ))
            reduced_text = reduced_text.replace(email.group(), ENT_REPLACEMENT)


        # * 3. Detect IP addresses
        ip_addresses = self.ip_address_pattern.finditer(text)
        for ip_address in ip_addresses:
            entities.add(Entity(
                type=EntityType.IP_ADDRESS,
                value=ip_address.group(),
            ))
            reduced_text = reduced_text.replace(ip_address.group(), ENT_REPLACEMENT)

        # * 4. Return the reduced text and entities found
        return entities, reduced_text


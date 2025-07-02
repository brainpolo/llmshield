"""List of list-based matchers used to detect entities.

Description:
    This module contains lists of keywords and components used for entity
    detection. These lists include person titles, organisation identifiers,
    place components, and punctuation markers that help identify and classify
    different types of entities in text.

Lists:
    EN_PUNCTUATION: Common English punctuation marks
    EN_PERSON_INITIALS: Titles and honorifics for person detection
    EN_ORG_COMPONENTS: Corporate suffixes and identifiers
    EN_PLACE_COMPONENTS: Street and place name components

Author:
    LLMShield by brainpolo, 2025
"""

# * Punctuation
# ----------------------------------------------------------------------------
EN_PUNCTUATION = ["!", ",", ".", "?", "\\'", "\\'"]


# * PERSON
# ----------------------------------------------------------------------------
EN_PERSON_INITIALS = [
    "Mr.",
    "Mrs.",
    "Ms.",
    "Dr.",
    "Prof.",
    "Professor",
    "Sir",
    "Lady",
    "Lord",
    "Duke",
    "Duchess",
    "Prince",
    "Princess",
    "King",
    "Queen",
    "CEO",
    "VP",
    "CFO",
    "COO",
    "CTO",
]


# * ORGANISATION
# ----------------------------------------------------------------------------
EN_ORG_COMPONENTS = [
    "Holdings",
    "Group",
    "LLP",
    "Ltd",
    "Corp",
    "Corporation",
    "Inc",
    "Industries",
    "Company",
    "Co",
    "LLC",
    "GmbH",
    "AG",
    "Pty",
    "L.P.",
]

# * PLACES
# ----------------------------------------------------------------------------

EN_PLACE_COMPONENTS = ["St", "St.", "Street", "Road", "Avenue", "Ave", "Rd"]

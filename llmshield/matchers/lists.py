"""
List of list-based matchers used to detect entities.

@see entity_detector.py for the entity detection system that uses these matchers.
"""

# * Punctuation
# ------------------------------------------------------------------------------
EN_PUNCTUATION = ["!", ",", ".", "?", "\\'", "\\’"]

# * Common words
# ------------------------------------------------------------------------------
EN_COMMON_WORDS = [
    "I", "A", "An", "The", "Of", "In", "On", "At", "To", "From", "By",
    "With", "As", "But", "If", "Or", "Because", "As", "Until",
    "While", "As", "Until", "While", "As", "Until", "While",
    "If", "You", "They", "So", "Hello", "Hi", "Hey", "Goodbye", "Bye",
    "Good", "My", "Its"
]

# * PERSON
# ------------------------------------------------------------------------------
EN_PERSON_INITIALS = [
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Professor", "Sir", "Lady", "Lord", "Duke", "Duchess",
    "Prince", "Princess", "King", "Queen", "CEO", "VP", "CFO", "COO", "CTO",
]

ES_PERSON_SINITIALS = [
    "Sr", "Sra", "Srta", "Dr", "Prof", "Sra", "Srta", "Srta", "Srta", "Srta",
    "Srta", "Srta", "Srta", "Srta"
]

# * ORGANISATION
# ------------------------------------------------------------------------------
EN_ORG_COMPONENTS = [
    "Holdings", "Group", "LLP", "Ltd", "Corp", "Corporation", "Inc", "Industries",
    "Company", "Co", "LLC", "GmbH", "AG", "Pty", "L.P."
]

ES_ORG_COMPONENTS = [
    "Holdings", "Grupo", "LLP", "Ltd", "Corp", "Corporación", "Inc", "Industrias",
    "Empresa", "Co", "LLC", "GmbH", "AG", "Pty", "L.P."
]


# * PLACES
# ------------------------------------------------------------------------------

EN_PLACE_COMPONENTS = ["St", "St.", "Street", "Road", "Avenue", "Ave", "Rd"]
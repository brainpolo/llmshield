"""
List of list-based matchers used to detect entities.

@see entity_detector.py for the entity detection system that uses these matchers.
"""

# * Punctuation
# ------------------------------------------------------------------------------
EN_PUNCTUATION = ["!", ",", ".", "?", "\\'", "\\'"]

# * Common words (to be ignored from PNOUN detection)
# ------------------------------------------------------------------------------
EN_COMMON_WORDS = [
    # Articles and basic prepositions
    "I", "A", "An", "The", "Of", "In", "On", "At", "To", "From", "By",
    "With", "As", "But", "If", "Or", "For", "Into", "Onto", "Upon",

    # Pronouns
    "You", "He", "She", "It", "We", "They", "Me", "Him", "Her", "Us", "Them",
    "My", "Your", "His", "Its", "Our", "Their", "Mine", "Yours", "Hers", "Ours",
    "Theirs", "This", "That", "These", "Those",

    # Common verbs
    "Is", "Are", "Was", "Were", "Be", "Been", "Have", "Has", "Had",
    "Do", "Does", "Did", "Will", "Would", "Can", "Could", "Should", "May",
    "Might", "Must", "Shall", "Going", "Gone", "Get", "Got", "Getting",

    # Conjunctions and connectors
    "And", "But", "Or", "Because", "While", "Until", "Unless", "Though",
    "Although", "However", "Therefore", "Thus", "Hence", "So", "Since",
    "Whether", "Where", "When", "What", "Who", "Why", "How",

    # Common adverbs
    "Very", "Really", "Quite", "Rather", "Too", "Also", "Just", "Only",
    "Now", "Then", "Here", "There", "Today", "Tomorrow", "Yesterday",
    "Always", "Never", "Sometimes", "Often", "Rarely",

    # Greetings and common expressions
    "Hello", "Hi", "Hey", "Goodbye", "Bye", "Good", "Bad", "Yes", "No",
    "Please", "Thank", "Thanks", "Sorry", "Excuse", "Welcome", "Dear",

    # Numbers and quantities
    "One", "Two", "Three", "First", "Second", "Third", "Last",
    "Many", "Much", "More", "Most", "Some", "Any", "All", "None", "Few",
    "Several", "Every", "Each", "Both", "Either", "Neither",

    # Time-related
    "Day", "Week", "Month", "Year", "Time", "Date", "Morning", "Evening",
    "Night", "Today", "Tomorrow", "Yesterday", "Soon", "Later", "Now",

    # Other common words
    "Way", "Thing", "Things", "Something", "Anything", "Nothing", "Everything",
    "Someone", "Anyone", "Everyone", "Nobody", "Everybody", "Anyone", "Someone",
    "Like", "About", "Over", "Under", "Between", "Among", "Through"
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


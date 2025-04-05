"""
List of functions-based matchers used to detect entities.
"""


def _luhn_check(card_number: str) -> bool:
    """Validate card number using Luhn algorithm."""
    digits = [int(d) for d in card_number if d.isdigit()]
    checksum = digits.pop()
    digits.reverse()
    doubled = [(d * 2) if i % 2 == 0 else d for i, d in enumerate(digits)]
    total = sum(sum(divmod(d, 10)) for d in doubled) + checksum
    return total % 10 == 0


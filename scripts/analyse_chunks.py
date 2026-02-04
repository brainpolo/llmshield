#!/usr/bin/env python3
"""Analyse text files for entity detection in chunks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llmshield import LLMShield  # skipcq: FLK-E402


def analyse_in_chunks(file_path: str, chunk_size: int = 50000):
    """Analyse text file in chunks."""
    print(f"Reading: {file_path}")
    with open(file_path, encoding="utf-8") as f:  # skipcq: PTC-W6004
        text = f.read()

    print(f"File size: {len(text):,} characters")
    print(f"Processing in chunks of {chunk_size:,} characters")
    print()

    shield = LLMShield()
    all_entities = {}

    # Process in chunks
    num_chunks = (len(text) + chunk_size - 1) // chunk_size

    for i in range(0, len(text), chunk_size):
        chunk_num = i // chunk_size + 1
        chunk = text[i : i + chunk_size]

        print(f"Processing chunk {chunk_num}/{num_chunks}...", end="\r")

        _, entity_map = shield.cloak(chunk)

        # Merge entities
        for placeholder, value in entity_map.items():
            if value not in all_entities:
                entity_type = placeholder.split("_")[0].replace("<", "")
                all_entities[value] = entity_type

    print(f"\nFound {len(all_entities)} unique entities across all chunks")

    # Group by type
    entity_types = {}
    for value, entity_type in all_entities.items():
        if entity_type not in entity_types:
            entity_types[entity_type] = []
        entity_types[entity_type].append(value)

    # Print summary
    print("\n" + "=" * 80)
    print("DETECTION SUMMARY")
    print("=" * 80)
    for entity_type, values in sorted(entity_types.items()):
        print(f"{entity_type:15s}: {len(values):5d} unique entities")

    # Show all entities by type
    print("\n" + "=" * 80)
    print("ALL ENTITIES BY TYPE")
    print("=" * 80)

    for entity_type, values in sorted(entity_types.items()):
        print(f"\n{entity_type} ({len(values)} unique):")
        print("-" * 80)
        for value in sorted(values):
            print(f"  {value}")


if __name__ == "__main__":
    MIN_ARGS = 2
    if len(sys.argv) < MIN_ARGS:
        print("Usage: python scripts/analyse_chunks.py <text_file>")
        print()
        print("Example:")
        print(
            "  python scripts/analyse_chunks.py "
            "tests/text_samples/KingJamesBible.txt"
        )
        sys.exit(1)

    input_arg = sys.argv[1]
    input_path = Path(input_arg).resolve()

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    analyse_in_chunks(str(input_path))

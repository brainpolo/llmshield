# Changelog

## [2.0.0] - 2026-02-04

**Improved API, Reduced False Positives**

This release makes major improvements to reducing false positives across all entity classes and introduces a more flexible chaining API.

View this release on [PyPI](https://pypi.org/project/llmshield/2.0.0/).

### Improvements
- **Reduced False Positives** - Overhaul of detection logic to prevent cloaking common words
- **Opt-In Concept Detection** - `CONCEPT` detection now disabled by default for technical acronyms
- **Chaining API** - New `without_*` methods for specific PII configuration
- **Scalable Caching** - Default `max_cache_size` increased to 10,000 message turns
- **State-Preserving Chaining** - Methods preserve configurations instead of resetting them

## [1.0.0] - 2025-07-03

**Initial Production Release**

We're excited to announce the first stable release of LLMShield, a production-ready, zero-dependency Python library for protecting sensitive information in LLM interactions.

View this release on [PyPI](https://pypi.org/project/llmshield/1.0.0/).

### Key Features

#### Core Functionality
- **Zero Dependencies** - Pure Python implementation requiring no external packages
- **Universal LLM Compatibility** - Works with OpenAI, Anthropic, xAI, and any OpenAI-compatible provider
- **Automatic PII Detection** - Multi-layered detection system for 9 entity types
- **Streaming Support** - Real-time processing for streaming LLM responses
- **Conversation Memory** - Maintains entity consistency across multi-turn conversations
- **Selective Protection** - Fine-grained control over which entity types to protect

#### Entity Detection Types
- **PERSON** - Names and titles (John Doe, Dr. Smith)
- **ORGANISATION** - Companies and institutions (Microsoft, NHS)
- **EMAIL** - Email addresses with validation
- **PHONE** - International phone number formats
- **CREDIT_CARD** - Credit card numbers with Luhn validation
- **PLACE** - Locations and addresses
- **URL** - Web addresses
- **IP_ADDRESS** - IPv4 addresses
- **CONCEPT** - Uppercase technical terms (API, SQL)

#### Provider Support
Full support includes text chat completions, image inputs (for supported models), structured outputs, and tool calling.

- **OpenAI** - Full support
- **Anthropic** - Full support
- **xAI Grok** - Full support
- **OpenAI-Compatible** - Full support
- Google - Unsupported
- Cohere - Unsupported

#### Advanced Features
- **LRU Cache System** - Configurable caching for optimal performance
- **Custom Delimiters** - Flexible placeholder formatting
- **Tool Call Protection** - Automatic PII protection in function/tool calls
- **High Performance** - O(1) entity lookups, 99.9% faster initialization
- **Thread-Safe** - Singleton pattern with proper locking
- **Comprehensive Testing** - 98.42% code coverage with 707 tests
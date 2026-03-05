# Changelog

## [2.1.0] - 2026-03-04

**New Providers, Provider Transparency, and Bug Fixes**

This release adds native support for Google Gemini and Cohere, exposes the detected provider for transparency, and fixes several correctness issues.

### New Features
- **Allowlist** - Instance-level and per-call allowlist for excluding specific terms from PII detection with case-insensitive matching
- **Google Gemini Provider** - Native support for the Google GenAI API including function calls
- **Cohere Provider** - Native support for the Cohere Chat API with closure-based client detection
- **xAI Responses API Provider** - Native support for the xAI SDK, replacing OpenAI-compatible workaround with dedicated provider handling
- **Provider Property** - `shield.provider` readonly property exposes the detected LLM provider for transparency and testability

### Bug Fixes
- **Conversation Hash Ordering** - Fixed `conversation_hash` using `frozenset` which ignored message order, causing potential cache collisions for reordered conversations
- **Unicode-Escaped Placeholders** - Fixed uncloaking failure when providers (e.g. Cohere) return unicode-escaped delimiters (`\u003c` instead of `<`) in tool call arguments

### Improvements
- **Provider Detection Ordering** - Module-based detectors now run before duck-typing detectors, preventing false-positive provider matching from Pydantic `extra='allow'` attributes
- **Shared Tool Call Uncloaking** - Extracted common `function.arguments` uncloaking pattern into shared helper, reducing duplication across providers
- **Provider Caching** - Provider is now detected once at initialisation instead of on every `ask()` call

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
# Contributing to llmshield

Thank you for your interest in contributing to llmshield! We welcome contributions from the community and are pleased to have you join us in making LLM interactions more secure and privacy-focused.

## Table of Contents

- [Development Principles](#development-principles)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Issue Reporting](#issue-reporting)

## Development Principles

Before contributing, please understand our core principles:

### 🔒 **Zero Dependencies**

- **Non-negotiable**: llmshield must remain dependency-free for production use
- Development dependencies are allowed (testing, formatting, etc.)
- All functionality must use only Python standard library

### 🚀 **Performance First**

- Optimize for production workloads and minimal latency
- Profile performance-critical code paths
- Consider memory efficiency and caching strategies

### 🛡️ **Security Focus**

- Prioritize data protection and privacy above convenience
- Assume all inputs contain sensitive information
- Follow secure coding practices

### 🌍 **Universal Compatibility**

- Support all major LLM providers
- Maintain backward compatibility when possible
- Design for extensibility

### ✨ **Code Quality**

- Follow British English in all naming and documentation
- Write comprehensive tests for new features
- Maintain clear, self-documenting code

## Getting Started

### Prerequisites

- **Python 3.10+** (required)
- **Git** for version control
- **Make** for build automation (optional but recommended)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/llmshield.git
cd llmshield
```

3. Add the upstream repository:

```bash
git remote add upstream https://github.com/ORIGINAL_OWNER/llmshield.git
```

## Development Environment

### Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
make dev-dependencies
```

### Development Dependencies

The following tools are used for development:

- **black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **pytest**: Testing framework
- **coverage**: Test coverage reporting

### Recommended IDE Setup

**VS Code Extensions:**

- Python
- Black Formatter
- isort
- Markdownlint

**PyCharm:**

- Enable Black as external formatter
- Configure isort for import optimization

## Code Standards

### Formatting

We use **Black** for code formatting with the following configuration:

```bash
# Format all Python files
black llmshield/ tests/

# Check formatting without making changes
black --check llmshield/ tests/
```

### Import Sorting

Use **isort** to organize imports:

```bash
# Sort imports
isort llmshield/ tests/

# Check import sorting
isort --check-only llmshield/ tests/
```

### Linting

We use **flake8** for linting:

```bash
# Run linting
flake8 llmshield/ tests/
```

### Code Style Guidelines

1. **Function/Variable Names**: Use `snake_case`
2. **Class Names**: Use `PascalCase`
3. **Constants**: Use `UPPER_SNAKE_CASE`
4. **Private Methods**: Prefix with single underscore `_method_name`
5. **Documentation**: Use British English spelling
6. **Type Hints**: Required for all public functions
7. **Docstrings**: Follow Google style for public methods

### Example Code Style

```python
from typing import Dict, List, Optional, Tuple

class EntityDetector:
    """Detects and classifies sensitive entities in text.

    This class provides methods for identifying various types of
    personally identifiable information (PII) within text inputs.
    """

    def __init__(self, config: Optional[Dict[str, str]] = None) -> None:
        """Initialise the entity detector.

        Args:
            config: Optional configuration dictionary for customisation.
        """
        self._config = config or {}

    def detect_entities(self, text: str) -> List[Tuple[str, str, int]]:
        """Detect entities in the provided text.

        Args:
            text: Input text to analyse for sensitive entities.

        Returns:
            List of tuples containing (entity_text, entity_type, position).

        Raises:
            ValueError: If text is empty or invalid.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")

        return self._extract_entities(text)

    def _extract_entities(self, text: str) -> List[Tuple[str, str, int]]:
        """Internal method for entity extraction."""
        # Implementation details...
        pass
```

## Testing

### Test Requirements

- **100% test coverage** for new features (95% minimum overall coverage threshold)
- **Maintain existing coverage** - don't break existing tests
- **Test both happy and error paths**
- **Include edge cases and boundary conditions**

### Running Tests

```bash
# Run all tests
make tests

# Run specific test file
python -m unittest tests/test_core.py

# Run with verbose output
python -m unittest discover tests -v

# Run tests with coverage
make coverage
```

### Test Categories

1. **Unit Tests**: Test individual functions and methods
2. **Integration Tests**: Test component interactions
3. **Provider Tests**: Test LLM provider integrations (requires API keys)
4. **Performance Tests**: Benchmark critical paths

### Writing Tests

```python
import unittest
from llmshield import LLMShield

class TestEntityDetection(unittest.TestCase):
    """Test entity detection functionality."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.shield = LLMShield()

    def test_person_detection(self) -> None:
        """Test detection of person entities."""
        text = "Hello, I'm John Doe"
        cloaked, entity_map = self.shield.cloak(text)

        self.assertIn("PERSON_0", cloaked)
        self.assertIn("John Doe", entity_map.values())

    def test_invalid_input(self) -> None:
        """Test handling of invalid inputs."""
        with self.assertRaises(ValueError):
            self.shield.cloak("")
```

### Provider Testing

For tests requiring API keys:

```bash
# OpenAI provider tests
OPENAI_API_KEY=your-key python -m unittest tests/providers/test_openai.py

# Skip API tests if no key provided
python -m unittest tests/providers/test_openai.py  # Will skip gracefully
```

## Submitting Changes

### Before Submitting

1. **Run the full test suite**:

   ```bash
   make tests
   ```

2. **Check code formatting**:

   ```bash
   black --check llmshield/ tests/
   isort --check-only llmshield/ tests/
   ```

3. **Run linting**:

   ```bash
   flake8 llmshield/ tests/
   ```

4. **Update documentation** if needed

### Pull Request Process

1. **Create a feature branch**:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code standards

3. **Add tests** for new functionality

4. **Update documentation** as needed

5. **Commit your changes**:

   ```bash
   git add .
   git commit -m "(feat|fix|docs|test): Brief description of changes"
   ```

6. **Push to your fork**:

   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request** on GitHub

### Pull Request Guidelines

- **Clear title** describing the change
- **Detailed description** explaining what and why
- **Link related issues** using `Fixes #123` or `Closes #123`
- **Include test results** if applicable
- **Update CHANGELOG** for user-facing changes

### Commit Message Format

Use conventional commit format:

```
type(scope): Brief description

Detailed explanation of the change (if needed)

Fixes #123
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

1. **Python version** and operating system
2. **llmshield version**
3. **Minimal reproducible example**
4. **Expected vs actual behaviour**
5. **Error messages and stack traces**
6. **LLM provider** (if applicable)

### Feature Requests

For new features, please provide:

1. **Clear use case** and motivation
2. **Proposed API** (if applicable)
3. **Backward compatibility** considerations
4. **Performance implications**

### Security Issues

**Do not report security vulnerabilities publicly.**

Please email security issues privately to a project maintainer.

## Getting Help

- **GitHub Discussions**: General questions and ideas
- **GitHub Issues**: Bug reports and feature requests
- **Code Review**: PR feedback and suggestions

---

Thank you for contributing to llmshield! Together, we're making LLM interactions more secure and privacy-focused. 🛡️

# llmshield

## Overview

llmshield is a lightweight and dependency-free Python library designed for high-performance cloaking and uncloaking of sensitive information in prompts and responses from Large Language Models (LLMs). It provides robust entity detection and protection where data privacy and security are paramount.

The aim is to be extremely accurate, using a combination of list-based, rule-based,
pattern-based, and probabilistic approaches.

## Key Features

- 🔒 **Secure Entity Detection**: Identifies and protects sensitive information including:
  - Proper nouns (Persons, Places, Organisations, Concepts)
  - Locators (Email addresses, URLs)
  - Numbers (Phone numbers, Credit card numbers)

_Additional PII types are in development._

- 🚀 **High Performance**: Optimised for minimal latency in LLM interactions
- 🔌 **Zero Dependencies**: Pure Pythonic implementation with no external requirements
- 🛡️ **End-to-End Protection**: Cloaks and uncloaks both prompts and responses
- 🎯 **Flexible Integration**: Works directly with your existing LLM function.

## Installation

```bash
pip install llmshield
```

## Quick Start

```python
from llmshield import LLMShield

# Basic usage - Manual LLM integration
shield = LLMShield()

# Cloak sensitive information
cloaked_prompt, entity_map = shield.cloak("Hi, I'm John Doe (john.doe@example.com)")
print(cloaked_prompt)  # "Hi, I'm <PERSON_0> (<EMAIL_0>)"

# Send to your LLM...
llm_response = your_llm_function(cloaked_prompt)

# Uncloak the response
original_response = shield.uncloak(llm_response, entity_map)

# Direct LLM integration
def my_llm_function(prompt: str) -> str:
    # Your LLM API call here
    return response

shield = LLMShield(llm_func=my_llm_function)
response = shield.ask(prompt="Hi, I'm John Doe (john.doe@example.com)")
```

## Configuration

### Delimiters

You can customise the delimiters used to wrap protected entities:

```python
shield = LLMShield(
    start_delimiter='[[',  # Default: '<'
    end_delimiter=']]'     # Default: '>'
)
```

The choice of delimiters should align with your LLM provider's training. Different providers may perform better with different delimiter styles.

### LLM Function Integration

Provide your LLM function during initialization for streamlined usage:

```python
shield = LLMShield(llm_func=your_llm_function)
```

## Best Practices

1. **Consistent Delimiters**: Use the same delimiters across your entire application
2. **Error Handling**: Always handle potential ValueError exceptions
3. **Entity Mapping**: Store entity maps securely if needed for later uncloaking
4. **Input Validation**: Ensure prompts are well-formed and grammatically correct

## Requirements

- Python 3.7+
- No additional dependencies
- Officially supports English and Spanish texts only.
- May work with other languages with lower accuracy and potential PII leakage.

## Support

- GitHub Issues: [Report bugs or request features](https://github.com/yourusername/llmshield/issues)

## Contributing

Contributions are welcome! Please follow these guidelines:

0. **Recommended IDE Development Packages**:
   - Black
   - Isort
   - Markdownlint

1. **Getting Started**:

   - Ensure you have a virtual environment with Python 3.7+
   - Install all development dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

   - Install other dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. **Code Quality and Formatting Guidelines**:
   - Follow black and isort rules
   - Add tests for new features
   - Do not break existing tests (unless justifying the change)
   - Maintain zero (non-development) dependencies (non-negotiable)
   - Use British English in all naming and documentation

2. **Testing**:

   ```bash
   python -m unittest discover -v
   ```

3. **Documentation**:
   - Update docstrings
   - Keep README.md current
   - Add examples for new features

4. **Build for development**

   ```bash
   pip install -e ".[dev]"
   ```

5. **Build and publish**

   ```bash
   python -m build
   python -m twine upload dist/*
   ```

Note: You will need to have a PyPI account and be authenticated.

## License

GNU APGLv3 License - See LICENSE.txt file for details

## Notable Uses

llmshield is currently used by:

- [brainful.ai](https://brainful.ai)

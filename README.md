# llmshield

## Overview

llmshield is a lightweight and dependency-free Python library designed for high-performance cloaking and uncloaking of sensitive information in prompts and responses from Large Language Models (LLMs). It provides robust entity detection and protection where data privacy and security are paramount.

The aim is to be extremely accurate, using a combination of list-based, rule-based,
pattern-based, and probabilistic approaches.

## Key Features

- 🔒 **Secure Entity Detection**: Identifies and protects sensitive information including:
  - Personal names
  - Email addresses
  - Phone numbers
  - IP addresses
  - URLs
  - Credit card numbers
  - Reference/Order numbers
  - And more...

- 🚀 **High Performance**: Optimised for minimal latency in LLM interactions
- 🔌 **Zero Dependencies**: Pure Python implementation with no external requirements
- 🛡️ **Bidirectional Protection**: Secures both prompts and responses
- 🎯 **Flexible Integration**: Works with any LLM provider

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
response = shield.ask("Hi, I'm John Doe (john.doe@example.com)")
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

## Performance

llmshield is optimized for minimal latency:

- Compiled regex patterns
- Efficient entity detection algorithms
- No external API calls
- Minimal memory footprint

## Security Considerations

- Entity maps contain sensitive information and should be handled securely
- Consider your LLM provider's security guidelines when choosing delimiters
- Regular updates are recommended for the latest security features

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Code Quality**:
   - Follow black and isort formatting
   - Add tests for new features
   - Maintain zero dependencies
   - Use British English across naming and documentation

2. **Testing**:

   ```bash
   python -m unittest discover -v
   ```

3. **Documentation**:
   - Update docstrings
   - Keep README.md current
   - Add examples for new features

## License

GNU APGLv3 License - See LICENSE.txt file for details

## Support

- GitHub Issues: [Report bugs or request features](https://github.com/yourusername/llmshield/issues)
- Documentation: [Full documentation](https://llmshield.readthedocs.io/)

## Notable Uses

llmshield is currently used by:

- [brainful.ai](https://brainful.ai)


## Building

```bash
rm -rf build/ dist/ *.egg-info/
python setup.py build
python setup.py sdist bdist_wheel
pip uninstall llmshield -y
pip install -e .
pip install dist/llmshield-*.whl

```

# Contributing to MiniAI

Thank you for your interest in contributing to MiniAI! This document provides guidelines and steps for contributing.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/agdhruv/miniai.git
   cd miniai
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Code Style

- Follow PEP 8 guidelines
- Use type hints
- Add docstrings for all public functions and classes
- Keep functions focused and small
- Write clear, descriptive commit messages

## Testing

- Write tests for new features
- Ensure all tests pass:
  ```bash
  pytest
  ```
- Maintain or improve test coverage

## Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Run tests:
   ```bash
   pytest
   ```
4. Update documentation if needed
5. Push your changes:
   ```bash
   git push origin feature/your-feature-name
   ```
6. Create a Pull Request

## Adding New Providers

To add a new AI provider:

1. Create a new provider file in `miniai/providers/` with the following structure:
   ```python
   from miniai.providers.base import BaseProvider

   class ProviderName(BaseProvider):
       def ask(self, question: str, **kwargs) -> str:
           pass
   ```
2. Implement required methods:
   - `ask()`
   - `embedding()`
3. Add provider to `PROVIDERS` dictionary in `miniai/providers/__init__.py`
4. Test the new provider using the examples in `examples/`

## Documentation

- Update README.md for new features
- Add examples to `examples/`
- Add docstrings for new functions/classes

## Questions?

Feel free to open an issue for any questions or suggestions! 
# Contributing to Researchify

First off, thank you for considering contributing to Researchify! It's people like you that make Researchify such a great tool for researchers and academics.

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct:
- Use welcoming and inclusive language
- Be respectful of different viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

- Your operating system name and version
- Python version and relevant package versions
- Detailed steps to reproduce the issue
- Any specific error messages
- A code sample or test case that demonstrates the issue

### Suggesting Enhancements

If you have a suggestion for a new feature or enhancement:

1. Check the existing issues to see if it's already been suggested
2. Provide a clear description of the enhancement
3. Include examples of how the feature would be used
4. If possible, outline how the enhancement might be implemented

### Pull Requests

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code lints
6. Issue the pull request

## Development Setup

1. Fork and clone the repository
```bash
git clone https://github.com/kazkozdev/researchify.git
cd researchify
```

2. Create a virtual environment and activate it
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

4. Set up pre-commit hooks
```bash
pre-commit install
```

## Style Guide

We use the following tools to maintain code quality:

- Black for code formatting
- Flake8 for style guide enforcement
- MyPy for type checking
- isort for import sorting

Configuration for these tools is in `pyproject.toml` and `.flake8`.

### Code Style

- Use Python type hints
- Write docstrings for functions and classes (Google style)
- Keep functions focused and under 50 lines where possible
- Use meaningful variable names
- Comment complex logic, but prefer readable code

### Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

Example:
```
Add support for PDF table extraction

- Implement table detection algorithm
- Add table-to-DataFrame conversion
- Update documentation with table processing examples

Fixes #123
```

## Testing

- Write tests for new features
- Update tests when changing existing functionality
- Run the test suite before submitting:
```bash
pytest
```

## Documentation

- Update the README.md if needed
- Add docstrings to new functions and classes
- Update the docs/ folder with any new features
- Include code examples for new functionality

## Making a Release

1. Update CHANGELOG.md
2. Update version in `__init__.py`
3. Create a new tag
4. Push to GitHub
5. Create a GitHub release

## Getting Help

If you need help, you can:
- Open an issue with your question
- Comment on the relevant issue or pull request
- Reach out to the maintainers

## Recognition

Contributors are recognized in several ways:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in relevant documentation

Thank you for contributing to Researchify!
# Contributing to Alzheimer MRI Processing Pipeline

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Issues

- Use the GitHub issue tracker to report bugs or suggest features
- Include a clear description, steps to reproduce (for bugs), and expected vs. actual behavior
- For ADNI-related issues, ensure you comply with ADNI Data Use Agreement requirements

### Pull Requests

1. **Fork the repository** and create a feature branch from `main`
2. **Make your changes** following the coding standards below
3. **Test your changes** to ensure they work as expected
4. **Update documentation** if your changes affect user-facing features
5. **Submit a pull request** with a clear description of changes

### Code Style

- **Python**: Follow PEP 8 style guide
- **Formatting**: Use **Black** for code formatting (if configured)
- **Linting**: Use **Ruff** for linting (if configured)
- **Type hints**: Use type hints where appropriate
- **Docstrings**: Follow Google or NumPy docstring style

### Testing

- Write tests for new features using **pytest**
- Ensure all existing tests pass before submitting
- Aim for reasonable test coverage

### Documentation

- Update README.md if adding new features or changing behavior
- Add docstrings to new functions and classes
- Update CHANGELOG.md for user-facing changes

### Adding References

When adding new methods or algorithms:

1. Add the reference to `00_CITATION_GUIDE.md` (internal reference file)
2. Include proper citation in code comments if applicable
3. Update README References section if it's a major component

### Commit Messages

- Use clear, descriptive commit messages
- Follow conventional commit format when possible:
  - `feat:` for new features
  - `fix:` for bug fixes
  - `docs:` for documentation changes
  - `refactor:` for code refactoring
  - `test:` for test additions/changes

### Review Process

- All pull requests require review before merging
- Maintainers will review for code quality, tests, and documentation
- Address review comments promptly

## Development Setup

1. Clone your fork: `git clone https://github.com/YOUR_USERNAME/alzheimer-mri-processing-pipeline.git`
2. Create a virtual environment: `python -m venv venv`
3. Install dependencies: `pip install -r requirements.txt`
4. Install development dependencies (if any): `pip install -r requirements-dev.txt`
5. Run tests: `pytest`

## Questions?

Feel free to open an issue for questions or reach out to the maintainers.

Thank you for contributing!


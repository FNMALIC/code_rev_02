# Contributing to AI-Powered Automated Code Review System

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or bug fix
4. Make your changes
5. Submit a pull request

## Development Setup

1. **Clone and setup:**
```bash
git clone https://github.com/FNMALIC/code_rev_02.git
cd code_rev_02
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Code Style Guidelines

- Follow PEP 8 style guidelines
- Use Black for code formatting: `black src/ tests/`
- Use type hints for all function parameters and return values
- Write comprehensive docstrings for all public functions and classes
- Keep line length under 88 characters (Black default)

## Commit Guidelines

- Use clear, descriptive commit messages
- Start commit messages with a verb (Add, Fix, Update, etc.)
- Keep the first line under 50 characters
- Reference issues when applicable: "Fix #123: Description"

## Pull Request Process

1. Ensure your code follows the style guidelines
2. Update documentation if you're changing functionality
3. Add tests for new features
4. Ensure all existing tests pass
5. Update the README.md if needed
6. Submit your pull request with a clear description


## Reporting Issues

When reporting issues, please include:
- Python version and operating system
- Steps to reproduce the issue
- Expected vs actual behavior
- Error messages and stack traces
- Sample code that demonstrates the issue

## Feature Requests

For feature requests, please:
- Check if the feature already exists or is planned
- Describe the use case and benefits
- Provide examples of how it would work
- Consider implementation complexity





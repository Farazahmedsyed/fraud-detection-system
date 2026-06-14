# Contributing Guide

## How to Contribute

Thank you for your interest in contributing to the Fraud Detection System! Here's how you can help.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a new branch** for your feature
4. **Make your changes**
5. **Test your changes**
6. **Push to your fork**
7. **Submit a pull request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/fraud-detection-system.git
cd fraud-detection-system

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8
```

## Code Style

We follow PEP 8 style guide. Format your code before committing:

```bash
# Format code with black
black src/

# Check for linting issues
flake8 src/
```

## Testing

Ensure all tests pass:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src

# Run specific test
pytest tests/test_models.py::TestDataProcessor -v
```

## Commit Messages

Use clear, descriptive commit messages:

```
# Good
git commit -m "Add SHAP explainability for individual predictions"

# Bad
git commit -m "fix stuff"
```

## Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new functionality
3. **Update README.md** if needed
4. **Ensure all tests pass**
5. **Keep PRs focused** - one feature per PR
6. **Write clear PR description**

## Types of Contributions

### Bug Reports
- Use GitHub Issues
- Include steps to reproduce
- Provide expected vs actual behavior
- Include Python version and environment info

### Feature Requests
- Clearly describe the feature
- Explain the use case
- Provide examples if possible

### Code Improvements
- Performance optimizations
- Code refactoring
- Documentation improvements
- Test coverage improvements

## Areas for Contribution

- [ ] API endpoint development
- [ ] Database integration
- [ ] Additional ML models
- [ ] Performance optimization
- [ ] Documentation
- [ ] Unit tests
- [ ] Docker support
- [ ] Kubernetes deployment
- [ ] CI/CD pipeline
- [ ] Additional visualizations

## Code Review Process

All submissions require review. Maintainers will:
1. Check code quality
2. Verify tests pass
3. Review for security issues
4. Provide feedback

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

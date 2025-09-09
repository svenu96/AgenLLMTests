# Contributing to LLM Study & Agent Development Platform

We welcome contributions to the LLM Study & Agent Development Platform! This document provides guidelines for contributing.

## 🚀 Getting Started

1. **Fork the Repository**
   ```bash
   git fork https://github.com/svenu96/AgenLLMTests.git
   cd AgenLLMTests
   ```

2. **Set Up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Install in development mode
   pip install -e .
   ```

3. **Run Tests**
   ```bash
   python -m pytest tests/ -v
   ```

## 🎯 Types of Contributions

### 📚 Educational Content
- Add new LLM tutorials and examples
- Improve existing documentation
- Create step-by-step guides
- Add interactive Jupyter notebooks

### 🤖 Agent Framework
- Develop new agent types
- Create additional tools
- Improve memory systems
- Add multi-agent capabilities

### 📄 PDF Q&A System
- Enhance document processing
- Improve retrieval algorithms
- Add new embedding models
- Optimize answer generation

### 🚀 Deployment & Infrastructure
- Docker improvements
- Kubernetes configurations
- Cloud deployment scripts
- Performance optimizations

## 📝 Development Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings to all public functions
- Keep functions focused and well-named

### Testing
- Write tests for new functionality
- Ensure existing tests pass
- Aim for good test coverage
- Include both unit and integration tests

### Documentation
- Update README.md if needed
- Add docstrings to new modules
- Create examples for new features
- Update API documentation

## 🔧 Pull Request Process

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Write clean, well-documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run tests
   python -m pytest tests/ -v
   
   # Run the demo
   python demo.py
   
   # Test specific modules
   python -m agents.examples.simple_agent
   ```

4. **Submit Pull Request**
   - Provide a clear description
   - Reference any related issues
   - Include screenshots for UI changes
   - Request review from maintainers

## 📋 Issue Guidelines

### Bug Reports
Include:
- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Error messages or logs
- Minimal code example

### Feature Requests
Include:
- Clear description of the feature
- Use case and motivation
- Proposed implementation (if any)
- Examples of how it would be used

## 🏷️ Labels and Categories

- `bug`: Something isn't working
- `enhancement`: New feature or improvement
- `documentation`: Documentation improvements
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `question`: Further information requested

## 📚 Resources

- [Python Style Guide (PEP 8)](https://pep8.org/)
- [Type Hints (PEP 484)](https://pep.python.org/pep-0484/)
- [Docstring Conventions (PEP 257)](https://pep.python.org/pep-0257/)
- [Pytest Documentation](https://docs.pytest.org/)

## 🤝 Community

- Be respectful and inclusive
- Help newcomers get started
- Share knowledge and learn from others
- Follow the project's code of conduct

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- Special mentions for outstanding contributions

Thank you for contributing to the LLM Study & Agent Development Platform! 🎉
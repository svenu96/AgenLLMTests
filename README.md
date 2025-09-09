# AgenLLMTests - Learn LLMs from Basics

A comprehensive repository to learn Large Language Models (LLMs) and AI Agents from the ground up with practical code examples and detailed step-by-step tutorials.

## 🎯 Learning Objectives

By the end of this guide, you will:
- Understand the fundamentals of Large Language Models
- Know how to work with pre-trained models
- Learn prompt engineering techniques
- Understand fine-tuning concepts
- Build simple AI agents
- Implement practical LLM applications

## 📚 Table of Contents

1. [Getting Started](#getting-started)
2. [LLM Basics](#llm-basics)
3. [Setting Up Your Environment](#setting-up-your-environment)
4. [Basic Examples](#basic-examples)
5. [Advanced Topics](#advanced-topics)
6. [Practical Projects](#practical-projects)
7. [Resources](#resources)

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Basic understanding of Python programming
- Basic knowledge of machine learning concepts (helpful but not required)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/svenu96/AgenLLMTests.git
cd AgenLLMTests

# Install dependencies
pip install -r requirements.txt

# Run your first LLM example
python examples/01_first_llm.py
```

## 🧠 LLM Basics

### What are Large Language Models?

Large Language Models (LLMs) are AI systems trained on vast amounts of text data to understand and generate human-like text. They work by:

1. **Training on massive datasets**: Learning patterns from billions of text examples
2. **Understanding context**: Using attention mechanisms to focus on relevant parts of input
3. **Generating predictions**: Predicting the next most likely word or token
4. **Transfer learning**: Applying learned knowledge to new tasks

### Key Concepts

#### 1. Tokens
- **Definition**: The basic units that LLMs process (words, subwords, or characters)
- **Example**: "Hello world" might be tokenized as ["Hello", " world"] or ["Hel", "lo", " wor", "ld"]

#### 2. Context Window
- **Definition**: The maximum number of tokens an LLM can process at once
- **Importance**: Determines how much information the model can "remember"

#### 3. Temperature
- **Definition**: Controls randomness in text generation
- **Range**: 0.0 (deterministic) to 1.0+ (more creative/random)

#### 4. Prompts
- **Definition**: The input text that guides the LLM's response
- **Quality**: Better prompts lead to better outputs

## 🛠 Setting Up Your Environment

We'll use Python with popular libraries for working with LLMs:

- **Transformers**: Hugging Face library for pre-trained models
- **OpenAI**: API for GPT models
- **Torch**: PyTorch for deep learning
- **Langchain**: Framework for building LLM applications

See `requirements.txt` for the complete list of dependencies.

## 📖 Learning Path

### Beginner Level
1. [Understanding Tokens and Tokenization](docs/01_tokenization.md)
2. [Your First Text Generation](examples/01_first_llm.py)
3. [Basic Prompt Engineering](examples/02_prompt_engineering.py)
4. [Working with Different Models](examples/03_model_comparison.py)

### Intermediate Level
1. [Advanced Prompt Techniques](examples/04_advanced_prompts.py)
2. [Building a Simple Chatbot](examples/05_simple_chatbot.py)
3. [Text Classification with LLMs](examples/06_text_classification.py)
4. [Working with APIs](examples/07_api_usage.py)

### Advanced Level
1. [Fine-tuning Concepts](docs/advanced/fine_tuning.md)
2. [Building AI Agents](examples/08_simple_agent.py)
3. [RAG (Retrieval Augmented Generation)](examples/09_rag_system.py)
4. [Custom Applications](projects/)

## 🎮 Hands-on Examples

Each example is designed to be self-contained and educational:

- **Beginner-friendly**: Clear explanations and simple code
- **Progressive difficulty**: Each example builds on previous concepts
- **Practical focus**: Real-world applications and use cases
- **Well-documented**: Extensive comments explaining each step

## 🔗 Resources

### Documentation
- [Official Documentation](docs/)
- [API References](docs/api/)
- [Troubleshooting Guide](docs/troubleshooting.md)

### External Resources
- [Hugging Face Documentation](https://huggingface.co/docs)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Awesome LLM Resources](https://github.com/Hannibal046/Awesome-LLM)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Happy Learning! 🚀**

*Start with the basics and gradually work your way up to building sophisticated LLM applications.*

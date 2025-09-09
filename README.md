# LLM Study & Agent Development Platform

A comprehensive codespace for studying Large Language Models (LLMs), creating AI agents, and deploying PDF question-answering systems.

## 🚀 Quick Start

### Using GitHub Codespaces
1. Click "Code" → "Codespaces" → "Create codespace on main"
2. Wait for the environment to set up automatically
3. Start exploring the tutorials and examples!

### Local Development
```bash
git clone https://github.com/svenu96/AgenLLMTests.git
cd AgenLLMTests
pip install -r requirements.txt
```

## 📚 What You'll Learn

### 1. LLM Fundamentals (`llm_study/`)
- **Transformers Architecture**: Understanding attention mechanisms, encoders, and decoders
- **Pre-training & Fine-tuning**: Hands-on experience with model training
- **Prompt Engineering**: Crafting effective prompts for various tasks
- **Model Evaluation**: Metrics and benchmarking techniques

### 2. Agent Development (`agents/`)
- **Agent Architecture**: Building autonomous AI systems
- **Tool Integration**: Connecting agents with external APIs and tools
- **Memory Systems**: Implementing conversation memory and context retention
- **Multi-agent Systems**: Coordinating multiple agents for complex tasks

### 3. PDF Q&A Systems (`pdf_qa/`)
- **Document Processing**: Extracting and chunking text from PDFs
- **Vector Embeddings**: Creating searchable document representations
- **Retrieval-Augmented Generation (RAG)**: Combining retrieval with generation
- **Production Deployment**: Scalable deployment strategies

## 🛠 Project Structure

```
AgenLLMTests/
├── llm_study/                 # Educational modules for LLM learning
│   ├── fundamentals/          # Basic concepts and tutorials
│   ├── architectures/         # Different model architectures
│   ├── training/              # Training and fine-tuning examples
│   └── evaluation/            # Model evaluation and benchmarking
├── agents/                    # Agent development framework
│   ├── core/                  # Base agent classes and utilities
│   ├── tools/                 # Tool integrations (APIs, databases, etc.)
│   ├── memory/                # Memory management systems
│   └── examples/              # Example agent implementations
├── pdf_qa/                    # PDF question-answering system
│   ├── processing/            # Document processing utilities
│   ├── embeddings/            # Vector embedding management
│   ├── retrieval/             # Document retrieval systems
│   └── generation/            # Answer generation modules
├── deployment/                # Deployment configurations and scripts
│   ├── docker/                # Docker containers
│   ├── kubernetes/            # Kubernetes manifests
│   └── cloud/                 # Cloud deployment scripts
├── examples/                  # End-to-end examples and demos
├── config/                    # Configuration files
└── tests/                     # Test suites
```

## 🎯 Learning Path

### Beginner Track
1. **Start Here**: `examples/01_hello_llm.ipynb` - Your first LLM interaction
2. **Understand Basics**: `llm_study/fundamentals/` - Core concepts
3. **Build First Agent**: `agents/examples/simple_agent.py` - Basic agent
4. **Process Your First PDF**: `examples/02_pdf_chat.ipynb` - PDF Q&A

### Intermediate Track
1. **Fine-tune a Model**: `llm_study/training/fine_tuning.ipynb`
2. **Advanced Agent Tools**: `agents/examples/tool_agent.py`
3. **Custom Embeddings**: `pdf_qa/embeddings/custom_embeddings.py`
4. **Multi-document RAG**: `examples/03_multi_doc_rag.ipynb`

### Advanced Track
1. **Custom Architecture**: `llm_study/architectures/custom_transformer.py`
2. **Multi-agent Systems**: `agents/examples/multi_agent_system.py`
3. **Production Deployment**: `deployment/` configurations
4. **Performance Optimization**: Advanced techniques and scaling

## 🔧 Key Features

- **Interactive Jupyter Notebooks**: Step-by-step tutorials with executable code
- **Modular Architecture**: Reusable components for quick prototyping
- **Production-Ready**: Deployment-ready configurations and best practices
- **Comprehensive Examples**: Real-world use cases and implementations
- **Testing Framework**: Robust testing for reliability and quality

## 🌐 Deployment Options

### Web Interface (Streamlit)
```bash
cd deployment
streamlit run app.py
```

### API Server (FastAPI)
```bash
cd deployment
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Docker Container
```bash
docker build -t llm-agent-platform .
docker run -p 8000:8000 llm-agent-platform
```

## 📖 Documentation

- [Getting Started Guide](docs/getting_started.md)
- [LLM Study Curriculum](docs/llm_curriculum.md)
- [Agent Development Guide](docs/agent_development.md)
- [PDF Q&A Tutorial](docs/pdf_qa_guide.md)
- [Deployment Guide](docs/deployment.md)
- [API Reference](docs/api_reference.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for the Transformers library
- LangChain for agent development tools
- OpenAI for pioneering LLM research
- The open-source community for continuous innovation

---

**Ready to start your LLM journey?** Open `examples/01_hello_llm.ipynb` and begin exploring! 🚀

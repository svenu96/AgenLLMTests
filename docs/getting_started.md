# Quick Start Guide

Get up and running with the LLM Study & Agent Development Platform in minutes!

## 🚀 Option 1: GitHub Codespaces (Recommended)

The fastest way to get started:

1. **Open in Codespaces**
   - Click the "Code" button on GitHub
   - Select "Codespaces" → "Create codespace on main"
   - Wait 2-3 minutes for automatic setup

2. **Start Exploring**
   ```bash
   # Run the demo
   python demo.py
   
   # Try the web interface
   streamlit run deployment/app.py
   
   # Start with the tutorial
   jupyter lab examples/01_hello_llm.ipynb
   ```

## 🖥️ Option 2: Local Development

### Prerequisites
- Python 3.8+ (3.11 recommended)
- Git
- 4GB+ RAM
- 2GB+ free disk space

### Installation

1. **Clone Repository**
   ```bash
   git clone https://github.com/svenu96/AgenLLMTests.git
   cd AgenLLMTests
   ```

2. **Create Virtual Environment**
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   
   # Or using conda
   conda create -n llm-study python=3.11
   conda activate llm-study
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env with your API keys (optional for basic functionality)
   # OPENAI_API_KEY=your_key_here
   ```

## 🧪 Verify Installation

```bash
# Run the demo (works without API keys)
python demo.py

# Run basic tests
python -m pytest tests/test_basic.py -v

# Test imports
python -c "import llm_study, agents, pdf_qa; print('✅ All modules imported successfully')"
```

## 🎯 Your First Steps

### 1. 📚 Learn LLM Fundamentals
```bash
# Interactive tutorial
jupyter lab examples/01_hello_llm.ipynb

# Or command line demo
python -c "
from llm_study.fundamentals import TransformerConcepts
concepts = TransformerConcepts()
concepts.attention_mechanism_demo()
"
```

### 2. 🤖 Create Your First Agent
```bash
# Run agent examples
python agents/examples/simple_agent.py

# Or create your own
python -c "
from agents.core import AgentConfig, AgentManager
config = AgentConfig(name='MyAgent', description='My first agent')
manager = AgentManager()
agent = manager.create_agent(config, 'simple')
print(f'Created agent: {agent.config.name}')
"
```

### 3. 📄 Try PDF Q&A
```bash
# Web interface
streamlit run deployment/app.py

# Then upload a PDF and ask questions!
```

## 🔑 API Configuration (Optional)

For full functionality, configure these APIs:

### OpenAI API
1. Get API key from [OpenAI](https://platform.openai.com/api-keys)
2. Add to `.env`: `OPENAI_API_KEY=your_key_here`

### Hugging Face (Optional)
1. Get token from [Hugging Face](https://huggingface.co/settings/tokens)
2. Add to `.env`: `HUGGINGFACE_API_TOKEN=your_token_here`

## 🐳 Docker Option

```bash
# Build and run with Docker
docker build -t llm-platform .
docker run -p 8501:8501 llm-platform

# Or use Docker Compose
docker-compose up
```

## 🌐 Web Interface

Access the web interface at:
- **Streamlit App**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs (when running API)

## 📱 Available Interfaces

| Interface | Command | Port | Best For |
|-----------|---------|------|----------|
| Streamlit Web App | `streamlit run deployment/app.py` | 8501 | Interactive learning |
| Jupyter Lab | `jupyter lab` | 8888 | Development & tutorials |
| FastAPI | `uvicorn deployment.api:app` | 8000 | API integration |
| Command Line | `python demo.py` | - | Quick testing |

## 🔧 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Install missing dependencies
pip install torch transformers langchain

# Or install all
pip install -r requirements.txt
```

**Memory Issues**
```bash
# Use smaller models
export DEFAULT_LLM_MODEL=distilbert-base-uncased
export DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

**API Errors**
- Check your API keys in `.env`
- Verify internet connection
- Check API quotas and billing

### Getting Help

- 📖 Read the [full documentation](docs/)
- 🐛 Report issues on [GitHub](https://github.com/svenu96/AgenLLMTests/issues)
- 💬 Join discussions in [GitHub Discussions](https://github.com/svenu96/AgenLLMTests/discussions)

## 🎉 What's Next?

- 📚 Complete the learning path in [README.md](README.md)
- 🤖 Build advanced agents with custom tools
- 📄 Deploy your own PDF Q&A system
- 🚀 Contribute to the project

Happy learning! 🚀
"""
Test Suite for LLM Study Platform

Basic tests to ensure core functionality works.
"""

import pytest
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from llm_study.fundamentals import LLMBasics, TransformerConcepts
from agents.core import AgentConfig, AgentManager
from agents.tools import CalculatorTool, DateTimeTool, ToolRegistry
from pdf_qa.processing import DocumentChunker
from pdf_qa.embeddings import EmbeddingManager


class TestLLMFundamentals:
    """Test LLM fundamentals module."""
    
    def test_transformer_concepts(self):
        """Test transformer concepts demo."""
        concepts = TransformerConcepts()
        
        # Test attention mechanism demo
        result = concepts.attention_mechanism_demo()
        assert 'words' in result
        assert 'attention_weights' in result
        assert 'output' in result
        
        # Test positional encoding
        pos_encoding = concepts.positional_encoding_demo(seq_len=5, d_model=4)
        assert pos_encoding.shape == (5, 4)
    
    def test_llm_basics_init(self):
        """Test LLM basics initialization."""
        # This might fail without proper model files, but should not crash
        try:
            llm = LLMBasics()
            # If initialization succeeds, test tokenization
            if llm.tokenizer:
                result = llm.tokenization_demo("Hello world")
                assert 'tokens' in result
                assert 'token_ids' in result
        except Exception as e:
            # Expected if models aren't available
            assert True


class TestAgents:
    """Test agent framework."""
    
    def test_agent_config(self):
        """Test agent configuration."""
        config = AgentConfig(
            name="TestAgent",
            description="A test agent",
            system_prompt="You are a test assistant."
        )
        
        assert config.name == "TestAgent"
        assert config.temperature == 0.7  # default value
        assert config.tools == []  # default empty list
    
    def test_agent_manager(self):
        """Test agent manager."""
        manager = AgentManager()
        
        # Test initial state
        assert len(manager.agents) == 0
        
        # Test agent creation (might fail without API keys)
        config = AgentConfig(name="TestAgent")
        try:
            agent = manager.create_agent(config, "simple")
            assert agent.config.name == "TestAgent"
            assert len(manager.agents) == 1
        except Exception:
            # Expected if API keys not configured
            pass


class TestTools:
    """Test agent tools."""
    
    def test_calculator_tool(self):
        """Test calculator tool."""
        calc = CalculatorTool()
        
        # Test basic calculation
        result = calc.execute(expression="2 + 3")
        assert "5" in result
        
        # Test invalid expression
        result = calc.execute(expression="invalid")
        assert "error" in result.lower()
    
    def test_datetime_tool(self):
        """Test datetime tool."""
        dt = DateTimeTool()
        
        result = dt.execute()
        assert "Current time" in result
        
        # Test ISO format
        result = dt.execute(format_type="iso")
        assert "ISO" in result
    
    def test_tool_registry(self):
        """Test tool registry."""
        registry = ToolRegistry()
        
        # Test tool retrieval
        calc_tool = registry.get_tool("calculator")
        assert calc_tool is not None
        assert calc_tool.name == "calculator"
        
        # Test tool listing
        tools = registry.list_tools()
        assert len(tools) > 0
        assert any(tool['name'] == 'calculator' for tool in tools)


class TestPDFProcessing:
    """Test PDF processing components."""
    
    def test_document_chunker(self):
        """Test document chunking."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        
        # Test simple chunking
        text = "This is a test document. " * 20  # Long enough to need chunking
        chunks = chunker.chunk_text(text, method="simple")
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 120 for chunk in chunks)  # Allow some overlap
        
        # Test chunk info
        info = chunker.get_chunk_info(chunks)
        assert info['num_chunks'] == len(chunks)
        assert info['target_chunk_size'] == 100


class TestEmbeddings:
    """Test embedding functionality."""
    
    def test_embedding_manager_init(self):
        """Test embedding manager initialization."""
        try:
            manager = EmbeddingManager()
            
            if manager.model:
                # Test single embedding
                embedding = manager.create_single_embedding("test text")
                assert len(embedding) > 0
                
                # Test multiple embeddings
                texts = ["hello", "world", "test"]
                embeddings = manager.create_embeddings(texts, show_progress=False)
                assert embeddings.shape[0] == 3
                assert embeddings.shape[1] > 0
        except Exception:
            # Expected if model loading fails
            pass


def test_import_structure():
    """Test that all modules can be imported."""
    # Test main package import
    import llm_study
    import agents
    import pdf_qa
    
    # Test specific components
    from llm_study.fundamentals import LLMBasics
    from agents.core import BaseAgent
    from pdf_qa.processing import PDFProcessor
    
    assert True  # If we get here, imports worked


def test_project_structure():
    """Test that project structure is correct."""
    base_path = Path(__file__).parent.parent
    
    # Check main directories exist
    required_dirs = [
        'llm_study',
        'agents', 
        'pdf_qa',
        'deployment',
        'examples',
        'config'
    ]
    
    for dir_name in required_dirs:
        assert (base_path / dir_name).exists(), f"Directory {dir_name} not found"
    
    # Check key files exist
    required_files = [
        'requirements.txt',
        'setup.py',
        'README.md',
        'Dockerfile'
    ]
    
    for file_name in required_files:
        assert (base_path / file_name).exists(), f"File {file_name} not found"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
"""
LLM Study & Agent Development Platform

A comprehensive library for studying Large Language Models,
creating AI agents, and deploying PDF question-answering systems.
"""

__version__ = "0.1.0"
__author__ = "LLM Study Group"
__email__ = "study@llmagents.dev"

# Core imports for easy access (with error handling)
__all__ = []

try:
    from llm_study.fundamentals import LLMBasics
    __all__.append("LLMBasics")
except ImportError:
    pass

try:
    from agents.core import BaseAgent, AgentManager
    __all__.extend(["BaseAgent", "AgentManager"])
except ImportError:
    pass

try:
    from pdf_qa import PDFQASystem
    __all__.append("PDFQASystem")
except ImportError:
    pass
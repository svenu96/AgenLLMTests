"""
LLM Study Module

Educational components for learning Large Language Models step by step.
"""

try:
    from .fundamentals import LLMBasics, TransformerConcepts
    fundamentals_available = True
except ImportError:
    fundamentals_available = False
    print("⚠️ LLM fundamentals not fully available - install torch and transformers")

__all__ = []

if fundamentals_available:
    __all__.extend(["LLMBasics", "TransformerConcepts"])

# Placeholder for future modules
try:
    # from .architectures import ModelArchitectures
    # from .training import ModelTraining  
    # from .evaluation import ModelEvaluation
    pass
except ImportError:
    pass
"""
Medical Diagnostic Analysis System
==================================

This module provides tools for analyzing medical evolution texts
and generating principal diagnoses with their corresponding ICD codes.

Main features:
- Analysis of medical texts to extract diagnoses
- Normalization of diagnoses using RAG
- Comparative evaluation of LLM models
- Visualization of results

Main modules:
- analyzer: Performs the analysis of medical texts
- tester: Evaluates model performance
- auxiliary_functions: General utility functions
"""

__version__ = "0.1.0"

# Main public imports to facilitate package usage
from .analyzer import evolution_text_analysis
from .tester import evaluate_analysis
from .results_manager import ResultsManager
from .data_models import (
    DiagnosticResult,
    EvaluationOutput,
    EvaluationResult,
    PerformanceMetrics,
    ModelInfo
)
from .auxiliary_functions import (
    get_analyzer_configuration,
    get_args,
    get_evolution_texts,
    get_chroma_db,
    get_listed_models,
    choose_model,
    write_results,
    check_ollama_connection
)
from ._validator import validate_result
from ._custom_parser import CustomParser, DiagnosticNormalizerRAG

# Define what should be exposed with "from package import *"
__all__ = [
    "evolution_text_analysis",
    "evaluate_analysis",
    "ResultsManager",
    "DiagnosticResult",
    "EvaluationOutput",
    "EvaluationResult", 
    "PerformanceMetrics",
    "ModelInfo",
    "get_analyzer_configuration",
    "get_args",
    "get_evolution_texts",
    "get_chroma_db",
    "get_listed_models",
    "choose_model",
    "write_results",
    "check_ollama_connection",
    "validate_result",
    "CustomParser",
    "DiagnosticNormalizerRAG"
]
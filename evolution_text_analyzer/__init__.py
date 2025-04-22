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

# Import only the functions used in main.py
from .analyzer import evolution_text_analysis
from .tester import evaluate_analysis
from .auxiliary_functions import (
    check_ollama_connection,
    get_analyzer_configuration,
    get_args,
    model_installed,
    get_evolution_texts,
    write_results
)

# Define what should be exposed with "from package import *"
__all__ = [
    # Functions used in main.py
    "evolution_text_analysis",
    "evaluate_analysis",
    "check_ollama_connection",
    "get_analyzer_configuration",
    "get_args",
    "model_installed",
    "get_evolution_texts",
    "write_results"
]
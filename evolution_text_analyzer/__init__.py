
"""
Medical Diagnostic Analysis System
==================================

This module provides tools for analyzing medical evolution texts
and generating principal diagnoses with their corresponding ICD codes.

Main features:
- Analysis of medical texts to extract diagnoses
- Normalization of diagnoses using a SNOMED based dataset.
- Comparative evaluation of LLM models
- Visualization of results

Main modules:
- analyzer: Performs the analysis of medical texts
- tester: Evaluates model performance
- auxiliary_functions: General utility functions
"""
from .auxiliary_functions import (
    check_ollama_connection,
    get_analyzer_configuration,
    get_args,
    model_installed,
    get_evolution_texts,
    write_results
)
from .tester import evaluate_analysis
from .analyzer import evolution_text_analysis
import os

__version__ = "0.1.0"

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import only the functions used in main.py

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

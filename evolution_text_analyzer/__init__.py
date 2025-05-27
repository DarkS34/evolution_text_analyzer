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

from .utils import (
    check_ollama_connection,
    get_analyzer_configuration,
    get_args,
    model_installed,
    get_evolution_texts,
    write_results
)
from .tester import AnalyzerTester
from .analyzer import Analyzer
import os

__version__ = "0.1.0"

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


__all__ = [
    "Analyzer",
    "AnalyzerTester",
    "check_ollama_connection",
    "get_analyzer_configuration",
    "get_args",
    "model_installed",
    "get_evolution_texts",
    "write_results"
]

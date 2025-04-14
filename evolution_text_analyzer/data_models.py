"""
Data models for the medical diagnostic analysis system.
This module contains all the Pydantic models used throughout the system.
"""

from typing import Dict, Optional
from pydantic import BaseModel


class DiagnosticResult(BaseModel):
    """Result of a diagnostic analysis for a single medical text."""
    icd_code: Optional[str]
    principal_diagnostic: Optional[str]
    validation_error: Optional[str] = None
    processing_error: Optional[str] = None


class EvaluationOutput(BaseModel):
    """Output of the evaluation process for a single diagnostic result."""
    valid: bool
    processed_output: DiagnosticResult
    correct_diagnostic: str


class PerformanceMetrics(BaseModel):
    """Performance metrics for a model evaluation."""
    accuracy: float
    incorrect_outputs: float
    errors: float
    hits: int
    total_texts: int
    duration: float
    start_time: str
    end_time: str
    num_batches: int
    prompt_index: int
    normalized: bool
    expanded: bool


class EvaluationResult(BaseModel):
    """Complete result of a model evaluation including metrics and detailed outputs."""
    model_info: dict
    performance: PerformanceMetrics
    evaluated_texts: Dict[str, EvaluationOutput]
    
class ModelInfo(BaseModel):
    model_name: str
    installed: bool = False
    size: str | None = None
    parameter_size: str | None = None
    quantization_level: str | None = None
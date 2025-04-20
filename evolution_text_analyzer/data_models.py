"""
Data models for the medical diagnostic analysis system.

This module defines the Pydantic models used throughout the system to ensure
data consistency and provide type validation. These models represent various
aspects of the diagnostic process including results, evaluation metrics,
and model information.
"""
from typing import Dict, Optional
from pydantic import BaseModel, Field


class DiagnosticResult(BaseModel):
    """
    Result of a diagnostic analysis for a single medical text.

    Contains the extracted principal diagnosis and ICD code,
    along with any error information.
    """
    icd_code: Optional[str] = Field(
        description="The extracted ICD code for the diagnosis"
    )
    principal_diagnostic: Optional[str] = Field(
        description="The extracted principal diagnosis text"
    )
    validation_error: Optional[str] = Field(
        None, description="Error encountered during validation, if any"
    )
    processing_error: Optional[str] = Field(
        None, description="Error encountered during processing, if any"
    )


class EvaluationOutput(BaseModel):
    """
    Output of the evaluation process for a single diagnostic result.

    Contains the validation result comparing the extracted diagnosis
    with the correct reference diagnosis.
    """
    valid: bool = Field(
        description="Whether the processed output matches the correct diagnosis"
    )
    processed_output: DiagnosticResult = Field(
        description="The processed diagnostic result from the model"
    )
    correct_diagnostic: str = Field(
        description="The correct reference diagnosis for comparison"
    )


class PerformanceMetrics(BaseModel):
    """
    Performance metrics for a model evaluation.

    Contains various metrics to assess model performance including
    accuracy, error rates, and processing time.
    """
    accuracy: float = Field(
        description="Percentage of correctly identified diagnoses"
    )
    incorrect_outputs: float = Field(
        description="Percentage of incorrect diagnoses"
    )
    errors: float = Field(
        description="Percentage of processing or validation errors"
    )
    hits: int = Field(
        description="Number of correctly identified diagnoses"
    )
    total_texts: int = Field(
        description="Total number of texts processed"
    )
    duration: float = Field(
        description="Total processing time in seconds"
    )
    start_time: str = Field(
        description="Start time of the evaluation"
    )
    end_time: str = Field(
        description="End time of the evaluation"
    )
    num_batches: int = Field(
        description="Number of batches used for processing"
    )
    normalized: bool = Field(
        description="Whether diagnosis normalization was applied"
    )


class ModelInfo(BaseModel):
    """
    Information about a language model.

    Contains details about a model including its name, installation status,
    and technical specifications.
    """
    model_name: str = Field(
        description="Name of the model"
    )
    installed: bool = Field(
        False, description="Whether the model is installed locally"
    )
    size: Optional[str] = Field(
        None, description="Size of the model in GB"
    )
    parameter_size: Optional[str] = Field(
        None, description="Number of parameters in the model"
    )
    quantization_level: Optional[str] = Field(
        None, description="Level of quantization applied to the model"
    )


class EvaluationResult(BaseModel):
    """
    Complete result of a model evaluation.

    Contains all information about a model evaluation including
    model details, performance metrics, and individual text results.
    """
    model_info: ModelInfo = Field(
        description="Information about the evaluated model"
    )
    performance: PerformanceMetrics = Field(
        description="Performance metrics for the evaluation"
    )
    evaluated_texts: Dict[str, EvaluationOutput] = Field(
        description="Detailed results for each processed text"
    )

"""
Testing module for medical diagnostic analysis.
Evaluates model performance on medical text diagnosis.
"""

import time
from pathlib import Path

from ._validator import validate_result
from .analyzer import evolution_text_analysis
from .auxiliary_functions import check_model, print_evaluated_results
from .data_models import (
    DiagnosticResult,
    EvaluationOutput,
    EvaluationResult,
    PerformanceMetrics,
)
from .results_manager import ResultsManager


def process_record(processed: dict, expected: str) -> EvaluationOutput:
    """
    Process an individual diagnostic result and evaluate its correctness.

    Args:
        model_name: Name of the model that produced the result
        processed: The processed diagnostic result
        expected: The expected correct diagnostic

    Returns:
        An EvaluationOutput object with the validation result
    """
    try:
        if not processed.get("processing_error"):
            valid = validate_result(
                processed["principal_diagnostic"], expected)
        else:
            valid = False
        result = DiagnosticResult(
            icd_code=processed.get("icd_code"),
            principal_diagnostic=processed.get("principal_diagnostic"),
            processing_error=processed.get("processing_error"),
        )
    except Exception as e:
        valid = False
        result = DiagnosticResult(
            icd_code=None,
            principal_diagnostic=None,
            validation_error=str(e),
        )

    return EvaluationOutput(
        valid=valid,
        processed_output=result,
        correct_diagnostic=expected
    )


def calculate_metrics(
    evaluated: dict[str, EvaluationOutput],
    total_texts: int,
    num_batches: int,
    prompt_index: int,
    start: float,
    end: float,
    date_format: str,
    normalized: bool,
    expanded: bool
) -> PerformanceMetrics:
    """
    Calculate performance metrics based on evaluation results.

    Args:
        evaluated: dictionary of evaluation outputs
        total_texts: Total number of texts processed
        num_batches: Number of batches used
        prompt_index: Index of the prompt used
        start: Start time of the evaluation
        end: End time of the evaluation
        date_format: Format string for date/time
        normalized: Whether normalization was used
        expanded: Whether expansion was used

    Returns:
        PerformanceMetrics object with calculated metrics
    """
    valid = sum(1 for e in evaluated.values() if e.valid)
    errors = sum(
        1 for e in evaluated.values()
        if not e.valid and (e.processed_output.validation_error or e.processed_output.processing_error)
    )
    incorrect = total_texts - valid - errors

    return PerformanceMetrics(
        accuracy=round((valid / total_texts) * 100, 2),
        incorrect_outputs=round((incorrect / total_texts) * 100, 2),
        errors=round((errors / total_texts) * 100, 2),
        hits=valid,
        total_texts=total_texts,
        duration=round(end - start, 2),
        start_time=time.strftime(date_format, time.localtime(start)),
        end_time=time.strftime(date_format, time.localtime(end)),
        num_batches=num_batches,
        prompt_index=prompt_index,
        normalized=normalized,
        expanded=expanded
    )


def evaluate_model(
    model_info: dict,
    prompt: dict,
    evolution_texts: list[dict],
    chroma_db,
    expansion_mode: bool,
    num_batches: int,
    num_texts: int,
    prompt_index: int,
    date_format: str = "%H:%M:%S %d-%m-%Y"
) -> EvaluationResult:
    """
    Evaluate a single model with the given parameters.

    Args:
        model_info: Information about the model to evaluate
        prompt: Prompt to use for the evaluation
        evolution_texts: List of medical texts to analyze
        chroma_db: Chroma database for normalization
        expansion_mode: Whether to use expansion mode
        num_batches: Number of batches to process
        num_texts: Number of texts to process
        prompt_index: Index of the prompt used
        date_format: Format string for date/time

    Returns:
        EvaluationResult object with the evaluation results
    """
    start = time.time()
    processed = evolution_text_analysis(
        model_info["model_name"],
        prompt,
        evolution_texts,
        chroma_db,
        expansion_mode,
        num_batches,
        num_texts,
    )
    end = time.time()

    evaluated = {
        key: process_record(result, evolution_texts[i]["principal_diagnostic"])
        for i, (key, result) in enumerate(processed.items())
    }

    metrics = calculate_metrics(
        evaluated,
        total_texts=num_texts,
        num_batches=num_batches,
        prompt_index=prompt_index,
        start=start,
        end=end,
        date_format=date_format,
        normalized=chroma_db is not None,
        expanded=expansion_mode
    )

    return EvaluationResult(
        model_info=model_info,
        performance=metrics,
        evaluated_texts=evaluated
    )


def evaluate_analysis(
    models: list[dict],
    prompts: dict,
    evolution_texts: list[dict],
    testing_results_dir: Path,
    chroma_db,
    expansion_mode: bool,
    num_batches: int,
    num_texts: int,
    verbose: bool
):
    """
    Main function to evaluate multiple models or prompts.

    Args:
        models: List of models to evaluate
        prompts_info: Tuple of (test_prompts flag, prompts list)
        opt_prompt: Index of the optimal prompt
        evolution_texts: List of medical texts to analyze
        testing_results_dir: Directory to store test results
        chroma_db: Chroma database for normalization
        expansion_mode: Whether to use expansion mode
        num_batches: Number of batches to process
        num_texts: Number of texts to process
        verbose: Whether to print verbose output
    """
    eval_mode = len(models) == 1
    
    # Initialize the results manager
    results_manager = ResultsManager(testing_results_dir, eval_mode)
    
    # Single model evaluation
    if eval_mode:
        evaluation_result = evaluate_model(
            models[0], prompts, evolution_texts, chroma_db,
            expansion_mode, num_batches, num_texts, 0
        )

        # Print detailed results if verbose
        print_evaluated_results(models[0], evaluation_result, verbose)

        # Add to results manager
        results_manager.add_result(evaluation_result)
    # Multiple models evaluation
    else:
        # Process all models
        for i, model in enumerate(models):
            if check_model(model):
                # Test with a single prompt
                    evaluation_result = evaluate_model(
                        model, prompts, evolution_texts, chroma_db,
                        expansion_mode, num_batches, num_texts, 0
                    )

                    # Add to results manager
                    results_manager.add_result(evaluation_result)

    # Generate comprehensive report
    report_path = results_manager.generate_comprehensive_report()
    print(
        f"\nEvaluation complete. Comprehensive report available at:\n{report_path}", end="\n\n")

"""
Testing module for medical diagnostic analysis.
Evaluates model performance on medical text diagnosis.
"""

from argparse import Namespace
import time
from pathlib import Path

from langchain_chroma import Chroma

from ._validator import validate_result
from .analyzer import evolution_text_analysis
from .auxiliary_functions import choose_model, get_context_window_length, get_listed_models_info, model_installed, print_evaluated_results
from .data_models import (
    DiagnosticResult,
    EvaluationOutput,
    EvaluationResult,
    ModelInfo,
    PerformanceMetrics,
)
from .results_manager import ResultsManager


def process_text(processed: dict, expected: str) -> EvaluationOutput:
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
    start: float,
    end: float,
    date_format: str,
    normalized: bool,
) -> PerformanceMetrics:
    """
    Calculate performance metrics based on evaluation results.

    Args:
        evaluated: dictionary of evaluation outputs
        total_texts: Total number of texts processed
        num_batches: Number of batches used
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
        normalized=normalized,
    )


def evaluate_model(
    model_info: ModelInfo,
    prompt: dict,
    evolution_texts: list[dict],
    chroma_db,
    num_batches: int,
    num_texts: int,
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
        date_format: Format string for date/time

    Returns:
        EvaluationResult object with the evaluation results
    """
    ctx_len = get_context_window_length(model_info.model_name)

    start = time.time()
    processed = evolution_text_analysis(
        model_info.model_name,
        prompt,
        ctx_len,
        evolution_texts,
        chroma_db,
        num_batches,
        num_texts,
    )
    end = time.time()

    evaluated = {
        key: process_text(result, evolution_texts[i]["principal_diagnostic"])
        for i, (key, result) in enumerate(processed.items())
    }

    metrics = calculate_metrics(
        evaluated,
        total_texts=num_texts,
        num_batches=num_batches,
        start=start,
        end=end,
        date_format=date_format,
        normalized=chroma_db is not None,
    )

    return EvaluationResult(
        model_info=model_info,
        performance=metrics,
        evaluated_texts=evaluated
    )


def evaluate_analysis(
    models: list[str],
    prompts: dict,
    evolution_texts: list[dict],
    testing_results_dir: Path,
    chroma_db: Chroma,
    args: Namespace
):
    """
    Main function to evaluate multiple models or prompts.

    Args:
        models: List of model names to evaluate
        prompts_info: Tuple of (test_prompts flag, prompts list)
        opt_prompt: Index of the optimal prompt
        evolution_texts: List of medical texts to analyze
        testing_results_dir: Directory to store test results
        chroma_db: Chroma database for normalization
        args: all arguments to control the flow of the program
    """
    # Initialize the results manager
    results_manager = ResultsManager(testing_results_dir, args.eval_mode == 2)

    # Multiple models evaluation
    if args.eval_mode == 1:
        models = get_listed_models_info(
            models, args.only_installed_models_mode)
        for i, model_info in enumerate(models):
            if model_installed(model_info.model_name):
                evaluation_result = evaluate_model(
                    model_info, prompts, evolution_texts, chroma_db, args.num_batches, args.num_texts
                )

                results_manager.add_result(evaluation_result)
        results_manager.generate_comprehensive_report()

    # Single model evaluation
    elif args.eval_mode == 2:
        model_info = choose_model(models, args.only_installed_models_mode)
        evaluation_result = evaluate_model(
            model_info, prompts, evolution_texts, chroma_db, args.num_batches, args.num_texts
        )

        print_evaluated_results(
            model_info, evaluation_result, args.verbose_mode)
        results_manager.add_result(evaluation_result)

"""This is the example module.

This module does stuff.
"""

from pydantic import BaseModel
import time
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from ._validator import validate_result
from .analyzer import evolution_text_analysis
from .auxiliary_functions import check_model, print_evaluated_results, update_results, write_results


class DiagnosticResult(BaseModel):
    icd_code: Optional[str]
    principal_diagnostic: Optional[str]
    validation_error: Optional[str] = None
    processing_error: Optional[str] = None


class EvaluationOutput(BaseModel):
    valid: bool
    processed_output: DiagnosticResult
    correct_diagnostic: str


class PerformanceMetrics(BaseModel):
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


class EvaluationResult(BaseModel):
    model_info: dict
    performance: PerformanceMetrics
    evaluated_texts: Dict[str, EvaluationOutput]


def process_record(model_name: str, processed: dict, expected: str) -> EvaluationOutput:
    try:
        if not processed.get("processing_error"):
            valid = validate_result(
                model_name, processed["principal_diagnostic"], expected)
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
    evaluated: Dict[str, EvaluationOutput],
    total_texts: int,
    num_batches: int,
    prompt_index: int,
    start: float,
    end: float,
    date_format: str
) -> PerformanceMetrics:
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
        key: process_record(
            model_info["model_name"], result, evolution_texts[i]["principal_diagnostic"])
        for i, (key, result) in enumerate(processed.items())
    }

    metrics = calculate_metrics(
        evaluated,
        total_texts=num_texts,
        num_batches=num_batches,
        prompt_index=prompt_index,
        start=start,
        end=end,
        date_format=date_format
    )

    return EvaluationResult(
        model_info=model_info,
        performance=metrics,
        evaluated_texts=evaluated
    )


def evaluate_analysis(
    models: list[dict],
    prompts_info: tuple[bool, list[dict]],
    opt_prompt: int,
    evolution_texts: list[dict],
    testing_results_dir: Path,
    chroma_db,
    expansion_mode: bool,
    num_batches: int,
    num_texts: int,
    verbose: bool
):
    test_prompts, prompts = prompts_info
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    normalization_mode = chroma_db is not None
    aditional_info = "_N" if normalization_mode else "" + "_E" if expansion_mode else ""
    if len(models) == 1:
        if not test_prompts or len(prompts) == 1:
            evaluation_results = evaluate_model(
                models[0], prompts[opt_prompt], evolution_texts, chroma_db, expansion_mode, num_batches, num_texts, 0)

            print_evaluated_results(models[0], evaluation_results, verbose)

            write_results(
                testing_results_dir / f"{timestamp}_{models[0]['model_name'].replace(':', '-')}{aditional_info}.json", evaluation_results)
        else:
            all_evaluations_results = []
            for prompt_index, prompt in enumerate(prompts):
                evaluation_results = evaluate_model(
                    models[0], prompt, evolution_texts, chroma_db, expansion_mode, num_batches, num_texts, prompt_index)

                update_results(
                    testing_results_dir / f"{timestamp}_{models[0]['model_name'].replace(':', '-')}_all_prompts{aditional_info}.json", evaluation_results, all_evaluations_results)
    else:
        all_evaluations_results = []
        testing_results_dir = testing_results_dir / "all_listed_models"
        testing_results_dir.mkdir(parents=True, exist_ok=True)
        for model in models:
            if check_model(model):
                if not test_prompts or len(prompts) == 1:
                    evaluation_results = evaluate_model(
                        model, prompts[opt_prompt], evolution_texts, chroma_db, expansion_mode, num_batches, num_texts, 0)
                    update_results(
                        testing_results_dir /
                        f"{timestamp}_{timestamp}{aditional_info}.json",
                        evaluation_results,
                        all_evaluations_results
                    )
                else:
                    for prompt_index, prompt in enumerate(prompts):
                        evaluation_results = evaluate_model(
                            model, prompt, evolution_texts, chroma_db, expansion_mode, num_batches, num_texts, prompt_index)
                        update_results(
                            testing_results_dir /
                            f"{timestamp}_all_prompts{aditional_info}.json",
                            evaluation_results,
                            all_evaluations_results
                        )

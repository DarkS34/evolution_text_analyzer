"""This is the example module.

This module does stuff.
"""

from pydantic import BaseModel
import time
from pathlib import Path
from typing import Dict, List, Optional

from ._validator import validate_result
from .analyzer import evolution_text_analysis
from .auxiliary_functions import check_model, print_evaluated_results, update_results, write_results


class DiagnosticResult(BaseModel):
    icdCode: Optional[str]
    principalDiagnostic: Optional[str]
    validationError: Optional[str] = None
    processingError: Optional[str] = None


class EvaluationOutput(BaseModel):
    valid: bool
    processedOutput: DiagnosticResult
    correctOutput: Dict[str, str]


class PerformanceMetrics(BaseModel):
    accuracy: float
    incorrectOutputs: float
    errors: float
    hits: int
    totalTexts: int
    duration: float
    startTime: str
    endTime: str
    numBatches: int
    promptIndex: int


class EvaluationResult(BaseModel):
    modelInfo: dict
    performance: PerformanceMetrics
    evaluatedTexts: Dict[str, EvaluationOutput]


def process_record(modelName: str, processed: dict, expected: str) -> EvaluationOutput:
    try:
        if not processed.get("processing_error"):
            valid = validate_result(modelName, processed["principal_diagnostic"], expected)
        else:
            valid = False
        result = DiagnosticResult(
            icdCode=processed.get("icd_code"),
            principalDiagnostic=processed.get("principal_diagnostic"),
            processingError=processed.get("processing_error"),
        )
    except Exception as e:
        valid = False
        result = DiagnosticResult(
            icdCode=None,
            principalDiagnostic=None,
            validationError=str(e),
        )

    return EvaluationOutput(
        valid=valid,
        processedOutput=result,
        correctOutput={"principal_diagnostic": expected},
    )


def calculate_metrics(
    evaluated: Dict[str, EvaluationOutput],
    totalTexts: int,
    numBatches: int,
    promptIndex: int,
    start: float,
    end: float,
    dateFormat: str
) -> PerformanceMetrics:
    valid = sum(1 for e in evaluated.values() if e.valid)
    errors = sum(
        1 for e in evaluated.values()
        if not e.valid and (e.processedOutput.validationError or e.processedOutput.processingError)
    )
    incorrect = totalTexts - valid - errors

    return PerformanceMetrics(
        accuracy=round((valid / totalTexts) * 100, 2),
        incorrectOutputs=round((incorrect / totalTexts) * 100, 2),
        errors=round((errors / totalTexts) * 100, 2),
        hits=valid,
        totalTexts=totalTexts,
        duration=round(end - start, 2),
        startTime=time.strftime(dateFormat, time.localtime(start)),
        endTime=time.strftime(dateFormat, time.localtime(end)),
        numBatches=numBatches,
        promptIndex=promptIndex,
    )


def evaluate_model(
    modelInfo: dict,
    evolutionTexts: List[dict],
    prompt: str,
    formatPrompt: str,
    numBatches: int,
    numTexts: int,
    promptIndex: int,
    dateFormat: str = "%H:%M:%S %d-%m-%Y"
) -> EvaluationResult:

    start = time.time()
    processed = evolution_text_analysis(
        modelInfo["modelName"],
        evolutionTexts,
        prompt + formatPrompt,
        numBatches,
        numTexts,
    )
    end = time.time()

    evaluated = {
        key: process_record(modelInfo["modelName"], result, evolutionTexts[i]["principal_diagnostic"])
        for i, (key, result) in enumerate(processed.items())
    }

    metrics = calculate_metrics(
        evaluated,
        totalTexts=numTexts,
        numBatches=numBatches,
        promptIndex=promptIndex,
        start=start,
        end=end,
        dateFormat=dateFormat
    )

    return EvaluationResult(
        modelInfo=modelInfo,
        performance=metrics,
        evaluatedTexts=evaluated
    )


def evaluate_analysis(
    models: List[dict],
    evolutionTexts: List[dict],
    systemPrompts: List[str],
    formatPrompt: str,
    numBatches: int,
    numTexts: int,
    testingResultsDir: Path,
    verbose: bool
):
    if len(models) == 1:
        if len(systemPrompts) == 1:
            evaluationResults = evaluate_model(models[0], evolutionTexts, systemPrompts[0], formatPrompt, numBatches, numTexts, 0)
            print_evaluated_results(evaluationResults, verbose)
            write_results(
                testingResultsDir / f"results_{models[0]['modelName'].replace(':', '_')}.json",
                evaluationResults
            )
        else:
            allEvaluationsResults = []
            for promptIndex, prompt in enumerate(systemPrompts):
                evaluationResults = evaluate_model(models[0], evolutionTexts, prompt, formatPrompt, numBatches, numTexts, promptIndex)
                update_results(
                    testingResultsDir / f"results_{models[0]['modelName'].replace(':', '_')}_all_prompts.json",
                    evaluationResults,
                    allEvaluationsResults
                )
    else:
        if len(systemPrompts) == 1:
            allEvaluationsResults = []
            for model in models:
                if check_model:
                    evaluationResults = evaluate_model(model, evolutionTexts, systemPrompts[0], formatPrompt, numBatches, numTexts, 0)
                    update_results(
                        testingResultsDir / "results_allListedModels.json",
                        evaluationResults,
                        allEvaluationsResults
                    )
        else:
            allEvaluationsResults = []
            for promptIndex, prompt in enumerate(systemPrompts):
                for model in models:
                    if check_model:
                        evaluationResults = evaluate_model(model, evolutionTexts, prompt, formatPrompt, numBatches, numTexts, promptIndex)
                        update_results(
                            testingResultsDir / "results_allListedModels_all_prompts.json",
                            evaluationResults,
                            allEvaluationsResults
                        )

"""This is the example module.

This module does stuff.
"""

import time
from pathlib import Path

from ._validator import validate_result
from .analyzer import evolutionTextAnalysis
from .auxiliary_functions import check_model, print_evaluated_results, update_results, write_results



def evaluateAnalysis(
    model_s: list,
    evolutionTexts: list,
    systemPrompt: str,
    argsBatches: int,
    argsNumEvolutionTexts: int,
    testingResultsDir: Path
):
    dateFormat = "%H:%M:%S %d-%m-%Y"

    def evaluate(modelInfo: dict):
        def _process_record(processedEvolutionText: dict, correctDiagnostic: str):
            try:
                isValid = validate_result(
                    processedEvolutionText["principal_diagnostic"], correctDiagnostic
                )

                # Validar resultados
                return {
                    "valid": isValid if not processedEvolutionText.get("error") else False,
                    "processedOutput": processedEvolutionText,
                    "correctOutput": {
                        "principal_diagnostic": correctDiagnostic,
                    },
                }
            except Exception as e:
                errorOutput = {
                    "icd_code": None,
                    "principal_diagnostic": None,
                    "validation_error": str(e),
                }

                return {
                    "valid": False,
                    "processedOutput": errorOutput,
                    "correctOutput": {
                        "principal_diagnostic": correctDiagnostic,
                    },
                }

        def _calculate_metrics(evaluatedEvolutionTexts):
            auxList = list(evaluatedEvolutionTexts.values())
            total = len(evaluatedEvolutionTexts)
            validCount = sum(1 for eet in auxList if eet["valid"])
            errorCount = sum(
                1
                for eet in auxList
                if not eet["valid"]
                and (
                    eet["processedOutput"].get("validation_error")
                    or eet["processedOutput"].get("processing_error")
                )
            )
            return {
                "accuracy": round(validCount / total * 100, 2),
                "incorrectOutputs": round(
                    100.00 - ((validCount + errorCount) / total * 100), 2
                ),
                "errors": round(errorCount / total * 100, 2),
            }

        startTime = {"startDuration": time.time(
        ), "startDate": time.localtime()}
        processedEvolutionTexts = evolutionTextAnalysis(
            modelInfo["modelName"],
            evolutionTexts,
            systemPrompt,
            argsBatches,
            argsNumEvolutionTexts,
        )
        endTime = {"endDuration": time.time(), "endDate": time.localtime()}

        evaluatedEvolutionTexts = {}
        processedEvolutionTexts = list(processedEvolutionTexts.items())

        for i in range(argsNumEvolutionTexts):
            evaluatedEvolutionTexts.update(
                {
                    processedEvolutionTexts[i][0]: _process_record(
                        processedEvolutionTexts[i][1],
                        evolutionTexts[i]["principal_diagnostic"],
                    )
                }
            )

        # Calculate final metrics
        metrics = _calculate_metrics(evaluatedEvolutionTexts)

        return {
            "model": modelInfo,
            "performance": {
                "accuracy": metrics["accuracy"],
                "incorrectOutputs": metrics["incorrectOutputs"],
                "errors": metrics["errors"],
                "processingTime": {
                    "duration": round(
                        endTime["endDuration"] - startTime["startDuration"], 2
                    ),
                    "startDateTime": time.strftime(dateFormat, startTime["startDate"]),
                    "endDateTime": time.strftime(dateFormat, endTime["endDate"]),
                },
                "numBatches": argsBatches,
                "totalETProcessed": argsNumEvolutionTexts,
            },
            "evaluatedEvolutionTexts": evaluatedEvolutionTexts,
        }

    if len(model_s) == 1:
        evaluationResults = evaluate(model_s[0])
        print_evaluated_results(evaluationResults)
        write_results(testingResultsDir /
                     f"results_{model_s[0]['modelName'].replace(':', '_')}.json", evaluationResults)
    else:
        allEvaluationsResults = []
        for model in model_s:
            if (check_model):
                evaluationResults = evaluate(model)
                update_results(evaluationResults, allEvaluationsResults)

import time
from pathlib import Path

from scripts._validator import validateResult
from scripts.analyzer_HF import evolutionTextAnalysis
from scripts.auxiliary_functions import checkModel, updateResults


def evaluateLLM(
    modelInfo: dict,
    evolutionTexts: list,
    numBatches: int,
    systemPrompt: str,
    totalEvolutionTextsToProcess: int,
):
    dateFormat = "%H:%M:%S %d-%m-%Y"

    # Función para evaluar un registro
    def processRecord(processedEvolutionText: dict, correctDiagnostic: str):
        try:
            isValid = validateResult(
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

    modelInfo.values()

    def calculateMetrics(evaluatedEvolutionTexts):
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

    startTime = {"startDuration": time.time(), "startDate": time.localtime()}
    processedEvolutionTexts = evolutionTextAnalysis(
        modelInfo["modelName"],
        evolutionTexts,
        numBatches,
        systemPrompt,
        totalEvolutionTextsToProcess,
    )
    endTime = {"endDuration": time.time(), "endDate": time.localtime()}

    evaluatedEvolutionTexts = {}
    processedEvolutionTexts = list(processedEvolutionTexts.items())

    for i in range(totalEvolutionTextsToProcess):
        evaluatedEvolutionTexts.update(
            {
                processedEvolutionTexts[i][0]: processRecord(
                    processedEvolutionTexts[i][1],
                    evolutionTexts[i]["principal_diagnostic"],
                )
            }
        )

    # Calculate final metrics
    metrics = calculateMetrics(evaluatedEvolutionTexts)

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
            "numBatches": numBatches,
            "totalETProcessed": totalEvolutionTextsToProcess,
        },
        "evaluatedEvolutionTexts": evaluatedEvolutionTexts,
    }


def evaluateListedLLMs(
    models: list,
    evolutionTexts: list,
    numBatches: int,
    systemPrompt: str,
    totalEvolutionTextsToProcess: int,
    testingResultsDir: Path
):
    modelsResults = []
    for mIdx, model in enumerate(models):
        if checkModel(model):
            partialResults = evaluateLLM(
                model,
                evolutionTexts,
                numBatches,
                systemPrompt,
                totalEvolutionTextsToProcess,
            )
            updateResults(testingResultsDir, partialResults, modelsResults)

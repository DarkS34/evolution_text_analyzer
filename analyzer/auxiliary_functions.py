from pathlib import Path
from argparse import ArgumentParser
from pydantic import ByteSize

import json
import ollama
import requests
import os
import pandas as pd

# https://mediately.co/_next/data/ca334f6100043fcbd2d00ec1242b3b547e1f226a/es/icd.json?classificationCode=


def checkOllamaConnected(url="http://localhost:11434"):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return True
        else:
            print("There is a problem. Try to restart Ollama to see available models.")
            return False
    except requests.ConnectionError as e:
        print(
            f"Error:\n{e}.\nOllama is not running. Start ollama to see available models.\n"
        )
        return False


def getArgs():
    parser = ArgumentParser(description="Script for processing with labeled modes.")
    parser.add_argument(
        "-mode",
        type=int,
        default=1,
        required=False,
        choices=[1, 2],
        help="Operation mode (1 or 2)",
    )
    parser.add_argument(
        "-batches",
        type=int,
        default=2,
        required=False,
        help="Number of batches for parallel evolution texts processing (5-20, default: 5)",
    )
    parser.add_argument(
        "-installed",
        action="store_true",
        help="Use only installed models (default: False)",
    )
    parser.add_argument(
        "-reason",
        action="store_true",
        help="Models reason the result for each processed evolution text (default: False)",
    )

    return parser.parse_args()


def getEvolutionTexts(path: Path):
    evolutionTextsList = []
    fileExtension = path.suffix
    try:
        with open(path, mode="r", encoding="utf-8") as file:
            if fileExtension == ".csv":
                evolutionTextsList = pd.read_csv(
                    file,
                    sep="|",
                    usecols=["ID", "principal_diagnostic", "evolution_text"],
                ).to_dict(orient='records')
            elif fileExtension == ".json":
                evolutionTextsList = json.load(file)
            else:
                raise ValueError(
                    "Evolution Texts - Extension not supported. Extension must be: '.json', '.csv'"
                )
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{path}' not found")
    return evolutionTextsList


def getModels(modelsListPath: Path, installedFlag: bool):
    def _process_model_info(rawModel, installedModels):
        installedModelInfo = next(
            (model for model in installedModels if model["model"] == rawModel),
            None,
        )

        size = None
        if installedModelInfo and "size" in installedModelInfo:
            size = f"{str(round(ByteSize(installedModelInfo['size']).to('GB'),1,))} GB"

        parameter_size = None
        if installedModelInfo and "details" in installedModelInfo:
            parameter_size = installedModelInfo["details"].get("parameter_size")

        quantization_level = None
        if installedModelInfo and "details" in installedModelInfo:
            quantization_level = installedModelInfo["details"].get("quantization_level")

        return {
            "modelName": rawModel,
            "installed": installedModelInfo is not None,
            "size": size,
            "parameterSize": parameter_size,
            "quantizationLevel": quantization_level,
        }

    if not modelsListPath.is_file():
        raise FileNotFoundError(f"File does not exist in: {modelsListPath}")

    if modelsListPath.suffix != ".json":
        raise ValueError(
            "List of models - Extension not supported. Extension must be: '.json'"
        )

    try:
        with open(modelsListPath, mode="r", encoding="utf-8") as modelsListFile:
            rawModelsList = json.load(modelsListFile)
            installedModels = ollama.list()["models"]
            modelsList = [
                _process_model_info(model, installedModels) for model in rawModelsList
            ]
            if not modelsList:
                raise ValueError("No models to use.")
            if installedFlag:
                modelsList = [model for model in modelsList if model["installed"]]

            return modelsList

    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Error while loading models: {e}")


def downloadModel(model: dict):
    success = False
    try:
        print(f"\rDownloading: '{model['modelName']}'", end="")
        ollama.pull(model["modelName"])
        success = True
    except (ollama.ResponseError, requests.RequestException) as e:
        print(f"Error: {e}")
        success = False
    finally:
        installedModelInfo = next(
            (
                installedModel
                for installedModel in ollama.list()["models"]
                if installedModel["model"] == model["modelName"]
            ),
            None,
        )
        if installedModelInfo:
            if not model.get("size"):
                model.update(
                    {
                        "size": str(
                            round(ByteSize(installedModelInfo["size"]).to("GB"), 1)
                        )
                        + " GB"
                    }
                )
            if not model.get("parameterSize"):
                model.update(
                    {"parameterSize": installedModelInfo["details"]["parameter_size"]}
                )
            if not model.get("quantizationLevel"):
                model.update(
                    {
                        "quantizationLevel": installedModelInfo["details"][
                            "quantization_level"
                        ]
                    }
                )
            model.pop("installed")
            success = True

    return success


def checkModel(model: dict):
    if not model["installed"]:
        return downloadModel(model)
    model.pop("installed")
    return True


def chooseModel(modelsListPath: Path, installedFlag: bool):
    modelsList = getModels(modelsListPath, installedFlag)
    modelNameWidth = max(len(model["modelName"]) for model in modelsList) + 4
    installedWidth = len("Not installed") + 4
    detailsWidth = 15

    print(
        f"{'ID'.rjust(4)} {'NAME'.ljust(modelNameWidth)}{'AVAILABLE'.ljust(installedWidth)}{"SIZE".ljust(detailsWidth-5)}{'PARAMETERS'.ljust(detailsWidth)}{'QUANTIZATION'.ljust(detailsWidth)}"
    )
    for i, model in enumerate(modelsList, start=1):
        installedMsg = (
            f"{'Installed'.ljust(installedWidth)}{model['size'].ljust(detailsWidth-5)}{model['parameterSize'].ljust(detailsWidth)}{model['quantizationLevel'].ljust(detailsWidth)}"
            if model["installed"]
            else "Not installed"
        )
        print(
            f"{(str(i)+'.').rjust(4)} {model['modelName'].ljust(modelNameWidth)}{installedMsg}"
        )

    while True:
        j = input(f"\nSelect model (1 - {len(modelsList)}): ")
        if j.isnumeric():
            j = int(j) - 1
            if j in range(0, len(modelsList)):
                break

    choosenModel = modelsList[j]
    if checkModel(choosenModel):
        return choosenModel
    else:
        return None


def printProcessedResults(results: dict):
    for id, processedETResult in results["evolutionTextsResults"].items():
        print(
            f"""
        {id} - {processedETResult['valid']} {'-'*20}
        Model result: {processedETResult['processedOutput'].get('principal_diagnostic')} ({processedETResult['processedOutput'].get('icd_code')})
        Correct result: {processedETResult['correctOutput']['principal_diagnostic']}
        """,
            (
                f"Reasoning: {processedETResult['processedOutput'].get('reasoning')}\n"
                if processedETResult["processedOutput"].get("reasoning")
                else ""
            ),
            (
                f"Error: {processedETResult['processedOutput'].get('error')}\n"
                if processedETResult["processedOutput"].get("error")
                else ""
            ),
        )

    print(
        f"""
        Accuracy: {results['performance']['accuracy']}% ({int((results['performance']['accuracy']/100) * results['performance']['totalRecordsProcessed'])}/{results['performance']['totalRecordsProcessed']})
        Incorrect outputs: {results['performance']['incorrectOutputs']}% ({int((results['performance']['incorrectOutputs']/100) * results['performance']['totalRecordsProcessed'])}/{results['performance']['totalRecordsProcessed']})
        Errors: {results['performance']['errors']}% ({int((results['performance']['errors']/100) * results['performance']['totalRecordsProcessed'])}/{results['performance']['totalRecordsProcessed']})
        Batches: {results['performance']['numBatches']}
        Total records processed: {results['performance']['totalRecordsProcessed']}
         
        Duration: {results['performance']['processingTime']['duration']} s.
        Start time: {results['performance']['processingTime']['startDate']}
        End time: {results['performance']['processingTime']['endDate']}
    """
    )


def updateResults(resultsDirPath: Path, partialResult: dict, modelsResults: list):
    modelsResults.append(partialResult)
    modelsResults.sort(key=lambda x: x["performance"]["accuracy"], reverse=True)
    writeProcessedResult(resultsDirPath / "results_allListedModels.json", modelsResults)


def writeProcessedResult(resultsPath: str, results: dict):
    with open(resultsPath, "w") as f:
        json.dump(results, f, indent=3)


def printExecutionProgression(
    modelName: str,
    processedTexts: int,
    totalTexts: int,
    processedModels: int,
    totalModels: int,
):
    print(f"\r{' '*os.get_terminal_size().columns}", end="", flush=True)
    if totalModels == 1:
        print(
            f"\r{modelName} - Evolution texts processed {processedTexts}/{totalTexts}",
            end="",
            flush=True,
        )
    else:
        print(
            f"\rModels processed {processedModels}/{totalModels} | Currently {modelName} - Evolution texts processed {processedTexts}/{totalTexts}",
            end="",
            flush=True,
        )

from pathlib import Path
from argparse import ArgumentParser
from pydantic import ByteSize

import csv
import json
import ollama
import requests
import os


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
        default=5,
        required=False,
        help="Number of batches for parallel evolution texts processing (5-20, default: 5)",
    )

    return parser.parse_args()


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


def modelTemplate(
    modelName: str,
    installed: bool,
    size: str | None,
    parameter_size: str | None,
    quantization_level: str | None,
):
    return {
        "modelName": modelName,
        "installed": installed,
        "size": size,
        "parameterSize": parameter_size,
        "quantizationLevel": quantization_level,
    }


def getModels(modelsListPath: Path):
    modelsList = []
    if modelsListPath.is_file():
        try:
            with open(modelsListPath, mode="r", encoding="utf-8") as modelsListFile:
                if modelsListPath.suffix == ".json":
                    rawModelsList: list = json.load(modelsListFile)

                    for rawModel in rawModelsList:
                        installedModelInfo = next(
                            (
                                installedModel
                                for installedModel in ollama.list()["models"]
                                if installedModel["model"] == rawModel["modelName"]
                            ),
                            None,
                        )
                        modelsList.append(
                            modelTemplate(
                                rawModel["modelName"],
                                installedModelInfo != None,
                                rawModel.get(
                                    "size",
                                    (
                                        f"{str(round(ByteSize(installedModelInfo['size']).to('GB'),1,))} GB"
                                        if installedModelInfo
                                        and "size" in installedModelInfo
                                        else None
                                    ),
                                ),
                                rawModel.get(
                                    "parameter_size",
                                    (
                                        installedModelInfo["details"].get(
                                            "parameter_size"
                                        )
                                        if installedModelInfo
                                        and "details" in installedModelInfo
                                        else None
                                    ),
                                ),
                                rawModel.get(
                                    "quantization_level",
                                    (
                                        installedModelInfo["details"].get(
                                            "quantization_level"
                                        )
                                        if installedModelInfo
                                        and "details" in installedModelInfo
                                        else None
                                    ),
                                ),
                            )
                        )
                else:
                    print(
                        "List of models - Extension not supported. Extension must be: '.json'"
                    )
        except Exception as e:
            raise Exception(f"Error while loading models: {e}")
    else:
        raise Exception(f"File does not exist in: {modelsListPath}")

    if len(modelsList) > 0:
        return modelsList
    else:
        raise Exception("No models to use.")


def getEvolutionTexts(path: Path):
    evolutionTextsList = []
    fileExtension = path.suffix
    try:
        with open(path, mode="r", encoding="utf-8") as file:
            if fileExtension == ".csv":
                lector = csv.DictReader(file, delimiter="|")
                for line in lector:
                    evolutionTextsList.append(dict(line))
            elif fileExtension == ".json":
                evolutionTextsList = json.load(file)
            else:
                raise Exception(
                    "Evolution Texts - Extension not supported. Extension must be: '.json', '.csv'"
                )
    except FileNotFoundError:
        raise Exception(f"File '{path}' not found")
    except Exception as e:
        raise Exception(f"Error: {e}")
    return evolutionTextsList


def downloadModel(model: dict):
    try:
        print(f"\rDownloading: '{model['modelName']}'", end="")
        ollama.pull(model["modelName"])
    except Exception as e:
        print(f"Error: {e}")
        return False
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
            if not model.get("parametersSize"):
                model.update(
                    {"parametersSize": installedModelInfo["details"]["parameter_size"]}
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
            return True
        return False


def checkModel(model):
    if not model["installed"]:
        return downloadModel(model)
    model.pop("installed")
    return True


def chooseModel(modelsListPath: Path):
    modelsList = getModels(modelsListPath)
    modelNameWidth = max(len(model["modelName"]) for model in modelsList) + 4

    for i, model in enumerate(modelsList, start=1):
        print(
            f"{(str(i)+'.').rjust(3)} {model['modelName'].ljust(modelNameWidth)}{'Installed' if model['installed'] else 'Not installed'}"
        )

    while True:
        j = int(input(f"\nSelect model (1 - {len(modelsList)}): ")) - 1
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
            f"\n{id} - {processedETResult['valid']} {'-'*20}\nModel result: {processedETResult['processedOutput'].get('principal_diagnostic')} ({processedETResult['processedOutput'].get('icd_code')})\nCorrect result: {processedETResult['correctOutput']['principal_diagnostic']}"
        )
    print(f"\n\tAccuracy: {results['performance']['accuracy']:.2f}%")
    print(f"\tIncorrect outputs: {results['performance']['incorrectOutputs']:.2f}%")
    print(f"\tErrors: {results['performance']['errors']:.2f}%")
    print(f"\tProcessing time: {results['performance']['processingTime']} s.")


def updateResults(resultsDirPath: Path, partialResult: dict, modelsResults: list):
    modelsResults.append(partialResult)
    modelsResults.sort(key=lambda x: x["performance"]["accuracy"], reverse=True)
    writeProcessedResult(resultsDirPath / "detailedResults.json", modelsResults)


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

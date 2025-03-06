"""This is the example module.

This module does stuff.
"""
from pathlib import Path
from argparse import ArgumentParser
from pydantic import ByteSize

import os
import ollama
import pandas as pd
import requests
import json


def check_ollama_connected(url="http://localhost:11434"):
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


def get_args(numEvolutionTexts: int):
    parser = ArgumentParser(
        description="Script for processing with labeled modes.")
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
        "--num-texts",
        type=int,
        default=numEvolutionTexts,
        required=False,
        help="Number of evolution texts to process (default: 2)",
    )
    parser.add_argument(
        "-test",
        action="store_true",
        help="Test mode (default: False)",
    )
    parser.add_argument(
        "-installed",
        action="store_true",
        help="Use only installed models (default: False)",
    )
    parser.add_argument(
        "--test-prompts",
        action="store_true",
        help="Use all system prompts for testing (default: False)",
    )

    return parser.parse_args()


def get_evolution_texts(path: Path):
    evolutionTextsList = []
    fileExtension = path.suffix
    try:
        with open(path, mode="r", encoding="utf-8") as file:
            if fileExtension == ".csv":
                evolutionTextsList = pd.read_csv(
                    file,
                    sep="|",
                    quotechar="'",
                ).to_dict(orient="records")
            elif fileExtension == ".json":
                evolutionTextsList = pd.read_json(file)
            else:
                raise ValueError(
                    "Evolution Texts - Extension not supported. Extension must be: '.json', '.csv'"
                )
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{path}' not found")

    for et in evolutionTextsList:
        et["evolution_text"] = et["evolution_text"].replace("\n", " ")

    return evolutionTextsList


def get_optimal_analyzer_configuration(path: Path) -> tuple[tuple[str, str], list[str], list[str], str]:
    if not path.exists():
        raise FileNotFoundError(f"File '{path}' not found")
    try:
        config = pd.read_json(path, typ="series")
    except ValueError:
        raise ValueError(f"Invalid JSON format in file '{path}'")

    return ((config["opt_model_name"], config["opt_system_prompt"]), config["models"], config["system_prompts"], config["output_formatting"])


# def get_ICD_dataset(path: Path) -> dict:
#     icdList = {}
#     fileExtension = path.suffix
#     try:
#         with open(path, mode="r", encoding="utf-8") as file:
#             if fileExtension == ".csv":
#                 df = pd.read_csv(file, quotechar='"', encoding="utf-8")
#                 icdList = dict(zip(df["DIAGNOSTIC"], df["CODE"]))
#             elif fileExtension == ".json":
#                 icdList = pd.read_json(file)
#             else:
#                 raise ValueError(
#                     "ICD Dataset - Extension not supported. Extension must be: '.json', '.csv'"
#                 )
#     except FileNotFoundError:
#         raise FileNotFoundError(f"File '{path}' not found")

#     return icdList


def _process_model_info(raw_model: str, installed_models: list[dict]) -> dict:
    """Process raw model data into a structured dictionary."""
    installed_info = next(
        (m for m in installed_models if m["model"] == raw_model),
        None
    )

    size = None
    parameter_size = None
    quantization_level = None

    if installed_info:
        size = (f"{round(ByteSize(installed_info.get('size', 0)).to('GB'), 1)} GB"
                if "size" in installed_info else None)
        details = installed_info.get("details", {})
        parameter_size = details.get("parameter_size")
        quantization_level = details.get("quantization_level")

    return {
        "modelName": raw_model,
        "installed": bool(installed_info),
        "size": size,
        "parameterSize": parameter_size,
        "quantizationLevel": quantization_level
    }


def get_listed_models(rawModels, installed_only: bool = False) -> list[dict]:
    """Retrieve and process model list from JSON file."""
    try:
        if not rawModels:
            raise ValueError("No models found in the list")

        installed_models = ollama.list()["models"]
        models = [_process_model_info(model, installed_models)
                  for model in rawModels]

        return [m for m in models if m["installed"]] if installed_only else models

    except pd.errors.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in models list: {e}")


def download_model(model: dict) -> bool:
    """Download a model and update its metadata."""
    try:
        print(f"\rDownloading: '{model['modelName']}'", end="")
        ollama.pull(model["modelName"])

        # Update model info after successful download
        installed_info = next(
            (m for m in ollama.list()["models"]
             if m["model"] == model["modelName"]),
            None
        )

        if installed_info:
            model.update({
                "size": f"{round(ByteSize(installed_info['size']).to('GB'), 1)} GB",
                "parameterSize": installed_info["details"]["parameter_size"],
                "quantizationLevel": installed_info["details"]["quantization_level"]
            })
            model.pop("installed", None)
            return True

        return False

    except (ollama.ResponseError, requests.RequestException) as e:
        print(f"\nDownload failed: {e}")
        return False


def check_model(model: dict) -> bool:
    """Check if model is installed and download if necessary."""
    if not model["installed"]:
        return download_model(model)
    model.pop("installed", None)
    return True


def display_model_table(models: list[dict]) -> None:
    """Display models in a formatted table."""
    col_widths = {
        "name": max(len(m["modelName"]) for m in models) + 4,
        "available": len("Not installed") + 4,
        "details": 15
    }

    header = (
        f"{'ID'.rjust(4)}  "
        f"{'NAME'.ljust(col_widths['name'])}"
        f"{'AVAILABLE'.ljust(col_widths['available'])}"
        f"{'SIZE'.ljust(col_widths['details']-5)}"
        f"{'PARAMETERS'.ljust(col_widths['details'])}"
        f"{'QUANTIZATION'.ljust(col_widths['details'])}"
    )
    print(header)

    for i, model in enumerate(models, 1):
        details = (
            f"{'Installed'.ljust(col_widths['available'])}"
            f"{(model['size'] or '').ljust(col_widths['details']-5)}"
            f"{(model['parameterSize'] or '').ljust(col_widths['details'])}"
            f"{(model['quantizationLevel'] or '').ljust(col_widths['details'])}"
        ) if model["installed"] else "Not installed"

        print(
            f"{str(i).rjust(4)}. {model['modelName'].ljust(col_widths['name'])}{details}")


def choose_model(models, installed_only: bool = False) -> dict | None:
    """Interactive model selection interface."""
    try:
        models = get_listed_models(models, installed_only)
        if not models:
            print("No models available to choose from.")
            return None

        display_model_table(models)

        while True:
            choice = input(f"\nSelect model (1 - {len(models)}): ").strip()
            if choice.isnumeric() and 1 <= int(choice) <= len(models):
                selected_model = models[int(choice) - 1]
                return [selected_model] if check_model(selected_model) else None
            print("Invalid selection. Please try again.")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return None


def print_evaluated_results(results: dict):
    for id, eet in results["evaluatedEvolutionTexts"].items():
        print(
            f"""
        {id} - {eet["valid"]} {"-" * 20}
        Model result: {eet["processedOutput"].get("principal_diagnostic")} ({eet["processedOutput"].get("icd_code")})
        Correct result: {eet["correctOutput"]["principal_diagnostic"]}
        """,
            (
                f"Error: {eet['processedOutput'].get('processing_error')}\n"
                if eet["processedOutput"].get("processing_error")
                else ""
            ),
            (
                f"Error: {eet['processedOutput'].get('validation_error')}\n"
                if eet["processedOutput"].get("validation_error")
                else ""
            ),
        )

    print(
        f"""
        Accuracy: {results["performance"]["accuracy"]}% ({int((results["performance"]["accuracy"] / 100) * results["performance"]["totalETProcessed"])}/{results["performance"]["totalETProcessed"]})
        Incorrect outputs: {results["performance"]["incorrectOutputs"]}% ({int((results["performance"]["incorrectOutputs"] / 100) * results["performance"]["totalETProcessed"])}/{results["performance"]["totalETProcessed"]})
        Errors: {results["performance"]["errors"]}% ({int((results["performance"]["errors"] / 100) * results["performance"]["totalETProcessed"])}/{results["performance"]["totalETProcessed"]})
        Batches: {results["performance"]["numBatches"]}
        Total records processed: {results["performance"]["totalETProcessed"]}
         
        Duration: {results["performance"]["processingTime"]["duration"]} s.
        Start time: {results["performance"]["processingTime"]["startDateTime"]}
        End time: {results["performance"]["processingTime"]["endDateTime"]}
    """
    )


def update_results(resultsPath: Path, partialResult: dict, modelsResults: list):
    modelsResults.append(partialResult)
    modelsResults.sort(
        key=lambda x: x["performance"]["accuracy"], reverse=True)
    write_results(resultsPath, modelsResults)


def write_results(resultsPath: str, results: dict):
    with open(resultsPath, mode="w", encoding="utf-8") as file:
        json.dump(results, file, indent=3, ensure_ascii=False)


def print_execution_progression(
    modelName: str,
    processedTexts: int,
    totalTexts: int,
    processedModels: int = 1,
    totalModels: int = 1,
):
    print(f"\r{' ' * os.get_terminal_size().columns}", end="", flush=True)
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

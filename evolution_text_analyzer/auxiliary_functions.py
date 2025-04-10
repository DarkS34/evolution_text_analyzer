"""This is the example module.

This module does stuff.
"""
import json
import os
from argparse import ArgumentParser
from pathlib import Path

import ollama
import pandas as pd
import requests
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel, ByteSize

# ------------------------
# Data Models
# ------------------------


class ModelInfo(BaseModel):
    modelName: str
    installed: bool = False
    size: str | None = None
    parameterSize: str | None = None
    quantizationLevel: str | None = None

# ------------------------
# Connection & Argument Handling
# ------------------------


def check_ollama_connected(url="http://localhost:11434") -> bool:
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return True
        print("There is a problem. Try to restart Ollama to see available models.")
        return False
    except requests.ConnectionError as e:
        print(f"Error:\n{e}.\n\nOllama is not running.\n")
        return False


def get_args(numEvolutionTexts: int):
    parser = ArgumentParser(
        description="Script for processing with labeled modes.", allow_abbrev=False)
    parser.add_argument("-m", "--mode", type=int,
                        choices=[1, 2], default=1, help="Operation mode (1 or 2)")
    parser.add_argument("-b", "--batches", type=int,
                        default=1, dest="numBatches")
    parser.add_argument("-n", "--num-texts", type=int,
                        default=numEvolutionTexts, dest="numEvolutionTexts")
    parser.add_argument("-t", "--test", action="store_true", dest="test")
    parser.add_argument("-i", "--installed",
                        action="store_true", dest="onlyInstalledModels")
    parser.add_argument("-tp", "--test-prompts",
                        action="store_true", dest="testPrompts")
    parser.add_argument("-v", "--verbose",
                        action="store_true", dest="verboseMode")

    args = parser.parse_args()

    return args


# ------------------------
# File and Config Handling
# ------------------------

def _color_text(text, color="green"):
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "bold": "\033[1m",
        # "blue": "\033[94m",
        # "magenta": "\033[95m",
        # "cyan": "\033[96m",
    }
    return f"{colors.get(color, '')}{text}\033[0m"


def get_evolution_texts(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File '{path}' not found")

    ext = path.suffix
    with open(path, mode="r", encoding="utf-8") as file:
        if ext == ".csv":
            texts = pd.read_csv(file, sep="|", quotechar="'").to_dict(
                orient="records")
        elif ext == ".json":
            texts = pd.read_json(file)
        else:
            raise ValueError("Extension not supported. Must be .json or .csv")

    for et in texts:
        et["evolution_text"] = et["evolution_text"].replace("\n", " ")

    return texts


def get_analyzer_configuration(path: Path) -> tuple[tuple[str, str], list[str], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"File '{path}' not found")
    try:
        return pd.read_json(path, typ="series").to_dict()
    except ValueError:
        raise ValueError(f"Invalid JSON format in file '{path}'")


def check_chroma_db(indexPath: str = "icd_vector_db", csvPath: str = "icd_dataset.csv", modelName: str = "nomic-embed-text:latest") -> bool:
    indexDir = Path(indexPath)

    if indexDir.exists() and any(indexDir.glob("*")):
        return True
    
    print(
        f"{_color_text('[INFO]')} Indice Chroma no encontrado. Creando nuevo desde {csvPath}...\r", end="")
    df = pd.read_csv(csvPath)
    df["text"] = df["code"] + ": " + df["description"]
    texts = df["text"].tolist()
    metadatas = df[["code", "description"]].to_dict(orient="records")

    try:
        embeddings = OllamaEmbeddings(model=modelName)
        Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=str(indexDir)
        )
        
        return True
    except Exception as e:
        print(f"\n{_color_text('[ERROR]', 'red')} Failed to create Chroma DB: {e}")
        return False


def load_chroma_db(indexPath: str = "icd_vector_db", modelName: str = "nomic-embed-text:latest") -> Chroma:
    indexDir = Path(indexPath)

    if not indexDir.exists() or not any(indexDir.glob("*")):
        raise FileNotFoundError(
            f"No se encontró el índice Chroma en {indexDir}. ¿Olvidaste ejecutar `create_chroma_db_if_needed()`?")

    embeddings = OllamaEmbeddings(model=modelName)
    return Chroma(
        persist_directory=str(indexDir),
        embedding_function=embeddings
    )

# ------------------------
# Model Handling
# ------------------------


def _process_model_info(rawModel: str, installedModels: list[dict]) -> ModelInfo:
    installed = next(
        (m for m in installedModels if m["model"] == rawModel), None)
    size, parameterSize, quantLevel = None, None, None

    if installed:
        size = f"{round(ByteSize(installed.get('size', 0)).to('GB'), 1)} GB" if "size" in installed else None
        details = installed.get("details", {})
        parameterSize = details.get("parameter_size")
        quantLevel = details.get("quantization_level")

    return ModelInfo(
        modelName=rawModel,
        installed=bool(installed),
        size=size,
        parameterSize=parameterSize,
        quantizationLevel=quantLevel
    )


def get_listed_models(rawModels: list[str], installedOnly: bool = False) -> list[dict]:
    if not rawModels:
        raise ValueError("No models found in the list")

    installed = ollama.list()["models"]
    models = [_process_model_info(m, installed).__dict__ for m in rawModels]
    return [m for m in models if m["installed"]] if installedOnly else models


def download_model(model: dict) -> bool:
    try:
        print(f"\rDownloading: '{model['modelName']}'", end="")
        ollama.pull(model["modelName"])
        updated = next((m for m in ollama.list()[
                       "models"] if m["model"] == model["modelName"]), None)
        if updated:
            model.update({
                "size": f"{round(ByteSize(updated['size']).to('GB'), 1)} GB",
                "parameterSize": updated["details"]["parameter_size"],
                "quantizationLevel": updated["details"]["quantization_level"]
            })
            model.pop("installed", None)
            return True
        return False
    except (ollama.ResponseError, requests.RequestException) as e:
        print(f"\nDownload failed: {e}")
        return False


def check_model(model: dict) -> bool:
    if not model["installed"]:
        return download_model(model)
    model.pop("installed", None)
    return True


def display_model_table(models: list[dict]) -> None:
    colWidths = {
        "name": max(len(m["modelName"]) for m in models) + 4,
        "available": len("Not installed") + 4,
        "details": 15
    }

    header = (
        f"{'ID'.rjust(4)}  "
        f"{'NAME'.ljust(colWidths['name'])}"
        f"{'AVAILABLE'.ljust(colWidths['available'])}"
        f"{'SIZE'.ljust(colWidths['details']-5)}"
        f"{'PARAMETERS'.ljust(colWidths['details'])}"
        f"{'QUANTIZATION'.ljust(colWidths['details'])}"
    )
    print(header)

    for i, model in enumerate(models, 1):
        details = (
            "Installed".ljust(colWidths["available"]) +
            f"{model['size'] or ''}".ljust(colWidths["details"] - 5) +
            f"{model['parameterSize'] or ''}".ljust(colWidths["details"]) +
            f"{model['quantizationLevel'] or ''}".ljust(colWidths["details"])
        ) if model["installed"] else "Not installed"

        print(
            f"{str(i).rjust(4)}. {model['modelName'].ljust(colWidths['name'])}{details}")


def choose_model(models: list[str], installedOnly: bool = False) -> list[dict] | None:
    try:
        listed = get_listed_models(models, installedOnly)
        if not listed:
            print("No models available to choose from.")
            return None

        display_model_table(listed)
        while True:
            choice = input(f"\nSelect model (1 - {len(listed)}): ").strip()
            if choice.isnumeric() and 1 <= int(choice) <= len(listed):
                selected = listed[int(choice) - 1]
                return [selected] if check_model(selected) else None
            print(f"{_color_text('[ERROR]', 'red')}Invalid selection. Please try again.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return None


def print_evaluated_results(model: dict, results: BaseModel, verbose: bool) -> None:
    if verbose:
        for id, eet in results.evaluatedTexts.items():
            print(
                f"""
            {id} - {eet.valid} {"-" * 20}
            Model result: {eet.processedOutput.principalDiagnostic} ({eet.processedOutput.icdCode})
            Correct result: {eet.correctOutput["principal_diagnostic"]}
            """,
            )
            if eet.processedOutput.processingError:
                print(f"Error: {eet.processedOutput.processingError}")
            if eet.processedOutput.validationError:
                print(f"Error: {eet.processedOutput.validationError}")

    # Limpieza de la línea actual en consola
    print(f"\r{' ' * os.get_terminal_size().columns}", end="", flush=True)

    perf = results.performance
    accuracy = perf.accuracy
    incorrect = perf.incorrectOutputs
    errors = perf.errors
    total = perf.totalTexts

    print(f"""\r
    Model: {_color_text(model["modelName"], 'bold')}
    Performance:
    ------------------
        {_color_text('Accuracy')}: {accuracy}% ({int((accuracy / 100) * total)}/{total})
        Incorrect outputs: {incorrect}% ({int((incorrect / 100) * total)}/{total})
        {_color_text('Errors', 'red')}: {errors}% ({int((errors / 100) * total)}/{total})
        
        Total records processed: {total}
        Duration: {perf.duration} s.
    -------------------
    """, end="", flush=True)


def update_results(resultsPath: Path, partialResult: dict, modelsResults: list) -> None:
    modelsResults.append(partialResult)
    modelsResults.sort(
        key=lambda x: x.performance.accuracy, reverse=True)
    write_results(resultsPath, modelsResults)


def write_results(resultsPath: str, results: dict | BaseModel | list[BaseModel]) -> None:
    with open(resultsPath, mode="w", encoding="utf-8") as file:
        if isinstance(results, list):
            results = [result.model_dump(exclude_none=True)
                       for result in results]
        elif isinstance(results, BaseModel):
            results = results.model_dump(exclude_none=True)

        json.dump(results, file, indent=3, ensure_ascii=False)


def print_execution_progression(
    modelName: str,
    processedTexts: int,
    totalTexts: int,
    processedModels: int = 1,
    totalModels: int = 1,
) -> None:
    print(f"\r{' ' * os.get_terminal_size().columns}", end="", flush=True)
    if totalModels == 1:
        print(
            f"\r{_color_text('[PROCESSING]')} {modelName} - Evolution texts processed {processedTexts}/{totalTexts}",
            end="",
            flush=True,
        )
    else:
        print(
            f"\r{_color_text('[PROCESSING]')} Models processed {processedModels}/{totalModels} | Currently {modelName} - Evolution texts processed {processedTexts}/{totalTexts}",
            end="",
            flush=True,
        )

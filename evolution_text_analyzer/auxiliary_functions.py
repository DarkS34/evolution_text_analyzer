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
    model_name: str
    installed: bool = False
    size: str | None = None
    parameter_size: str | None = None
    quantization_level: str | None = None


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

# ------------------------
# Connection & Argument Handling
# ------------------------


def check_ollama_connection(url: str = "http://localhost:11434") -> None:
    try:
        if not requests.get(url).status_code == 200:
            print(
                f"{_color_text('[ERROR]', 'red')} An error ocurred. Imposible to connect to Ollama.\n")
            exit(1)
    except requests.ConnectionError:
        print(
            f"{_color_text('[ERROR]', 'red')} Ollama is not running.")
        exit(1)


def get_args(num_evolution_texts: int):
    parser = ArgumentParser(
        description="Script for processing with labeled modes.", allow_abbrev=False)
    parser.add_argument("-m", "--mode", type=int,
                        choices=[1, 2], default=1, help="Operation mode (1 or 2)")
    parser.add_argument("-b", "--batches", type=int,
                        default=1, dest="num_batches")
    parser.add_argument("-n", "--num-texts", type=int,
                        default=num_evolution_texts, dest="num_evolution_texts")
    parser.add_argument("-t", "--test", action="store_true", dest="test")
    parser.add_argument("-i", "--installed",
                        action="store_true", dest="only_installed_models")
    parser.add_argument("-tp", "--test-prompts",
                        action="store_true", dest="test_prompts")
    parser.add_argument("-v", "--verbose",
                        action="store_true", dest="verbose_mode")
    parser.add_argument("-N", "--no-normalization", action="store_false",
                        dest="normalize_results", help="Don't normalize results via RAG")
    parser.add_argument("-E", "--no-expansion", action="store_false",
                        dest="expansion_mode", help="Don't expand evolution texts")

    return parser.parse_args()


# ------------------------
# File and Config Handling
# ------------------------


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

    for evolution_text in texts:
        evolution_text["evolution_text"] = evolution_text["evolution_text"].replace(
            "\n", " ")

    return texts


def get_analyzer_configuration(path: Path) -> tuple[tuple[str, str], list[str], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"File '{path}' not found")
    try:
        return pd.read_json(path, typ="series").to_dict()
    except ValueError:
        raise ValueError(f"Invalid JSON format in file '{path}'")


def _create_chroma_db(index_path: str = "icd_vector_db", model_name: str = "nomic-embed-text:latest") -> bool:
    index_dir = Path(index_path)
    csv_path: str = "icd_dataset.csv"

    print(
        f"{_color_text('[INFO]')} Indice Chroma no encontrado. Creando nuevo desde {csv_path}...\r", end="")
    df = pd.read_csv(csv_path)
    df["text"] = df["code"] + ": " + df["description"]
    texts = df["text"].tolist()
    metadatas = df[["code", "description"]].to_dict(orient="records")

    try:
        embeddings = OllamaEmbeddings(model=model_name)
        chroma_db = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            persist_directory=str(index_dir)
        )

        return chroma_db
    except Exception as e:
        print(
            f"\n{_color_text('[ERROR]', 'red')} Failed to create Chroma DB: {e}")
        return False


def get_chroma_db(index_path: str = "icd_vector_db", model_name: str = "nomic-embed-text:latest") -> Chroma:
    index_dir = Path(index_path)

    if not index_dir.exists():
        _create_chroma_db(index_path, model_name)

    embeddings = OllamaEmbeddings(model=model_name)
    return Chroma(
        persist_directory=str(index_dir),
        embedding_function=embeddings
    )

# ------------------------
# Model Handling
# ------------------------


def _process_model_info(raw_model: str, installed_models: list[dict]) -> ModelInfo:
    installed = next(
        (model for model in installed_models if model["model"] == raw_model), None)
    size, parameter_size, quant_level = None, None, None

    if installed:
        size = f"{round(ByteSize(installed.get('size', 0)).to('GB'), 1)} GB" if "size" in installed else None
        details = installed.get("details", {})
        parameter_size = details.get("parameter_size")
        quant_level = details.get("quantization_level")

    return ModelInfo(
        model_name=raw_model,
        installed=bool(installed),
        size=size,
        parameter_size=parameter_size,
        quantization_level=quant_level
    )


def get_listed_models(raw_models: list[str], installed_only: bool = False) -> list[dict]:
    if not raw_models:
        raise ValueError("No models found in the list")

    installed = ollama.list()["models"]
    models = [_process_model_info(
        model, installed).__dict__ for model in raw_models]
    return [model for model in models if model["installed"]] if installed_only else models


def download_model(model: dict) -> bool:
    try:
        print(f"\rDownloading: '{model['model_name']}'", end="")
        ollama.pull(model["model_name"])
        updated = next((model for model in ollama.list()[
                       "models"] if model["model"] == model["model_name"]), None)
        if updated:
            model.update({
                "size": f"{round(ByteSize(updated['size']).to('GB'), 1)} GB",
                "parameter_size": updated["details"]["parameter_size"],
                "quantization_level": updated["details"]["quantization_level"]
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
    col_widths = {
        "name": max(len(model["model_name"]) for model in models) + 4,
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
            "Installed".ljust(col_widths["available"]) +
            f"{model['size'] or ''}".ljust(col_widths["details"] - 5) +
            f"{model['parameter_size'] or ''}".ljust(col_widths["details"]) +
            f"{model['quantization_level'] or ''}".ljust(col_widths["details"])
        ) if model["installed"] else "Not installed"

        print(
            f"{str(i).rjust(4)}. {model['model_name'].ljust(col_widths['name'])}{details}")


def choose_model(models: list[str], installed_only: bool = False) -> list[dict] | None:
    try:
        listed = get_listed_models(models, installed_only)
        if not listed:
            print("No models available to choose from.")
            return None

        display_model_table(listed)
        while True:
            choice = input(f"\nSelect model (1 - {len(listed)}): ").strip()
            if choice.isnumeric() and 1 <= int(choice) <= len(listed):
                selected = listed[int(choice) - 1]
                return [selected] if check_model(selected) else None
            print(
                f"{_color_text('[ERROR]', 'red')}Invalid selection. Please try again.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return None


def print_evaluated_results(model: dict, results: BaseModel, verbose: bool) -> None:
    # Limpieza de la línea actual en consola
    print(f"\r{' ' * os.get_terminal_size().columns}", end="", flush=True)
    if verbose:
        for id, evaluated_text in results.evaluated_texts.items():
            print(
                f"\n{id} - {_color_text(evaluated_text.valid, 'green' if evaluated_text.valid else 'red')}\n"
                f"\tModel dignostic: {evaluated_text.processed_output.principal_diagnostic} (Code - {evaluated_text.processed_output.icd_code})\n"
                f"\tCorrect diagnostic: {evaluated_text.correct_diagnostic}"
            )
            if evaluated_text.processed_output.processing_error:
                print(
                    f"Processing Error: {evaluated_text.processed_output.processing_error}")
            if evaluated_text.processed_output.validation_error:
                print(
                    f"Validation Error: {evaluated_text.processed_output.validation_error}")
        print()
    performance = results.performance
    accuracy = performance.accuracy
    incorrect = performance.incorrect_outputs
    errors = performance.errors
    total = performance.total_texts

    print(
        f"\rModel: {model['model_name']}\n"
        f"Performance:\n"
        f"\t{_color_text('Accuracy')}: {accuracy}% ({int((accuracy / 100) * total)}/{total})\n"
        f"\t{_color_text('Incorrect outputs', 'red')}: {incorrect}% ({int((incorrect / 100) * total)}/{total})\n"
        f"\tErrors: {errors}% ({int((errors / 100) * total)}/{total})\n\n"
        f"\tTotal records processed: {total}\n"
        f"\tDuration: {performance.duration} s.", end="", flush=True)


def update_results(results_path: Path, partial_result: dict, models_results: list) -> None:
    models_results.append(partial_result)
    models_results.sort(
        key=lambda x: x.performance.accuracy, reverse=True)
    write_results(results_path, models_results)


def write_results(results_path: str, results: dict | BaseModel | list[BaseModel]) -> None:
    with open(results_path, mode="w", encoding="utf-8") as file:
        if isinstance(results, list):
            results = [result.model_dump(exclude_none=True)
                       for result in results]
        elif isinstance(results, BaseModel):
            results = results.model_dump(exclude_none=True)

        json.dump(results, file, indent=3, ensure_ascii=False)


def print_execution_progression(
    model_name: str,
    processed_texts: int,
    total_texts: int,
    processed_models: int = 1,
    total_models: int = 1,
) -> None:
    print(f"\r{' ' * os.get_terminal_size().columns}", end="", flush=True)
    if total_models == 1:
        print(
            f"\r{_color_text('[PROCESSING]')} {model_name} - Evolution texts processed {processed_texts}/{total_texts}",
            end="",
            flush=True,
        )
    else:
        print(
            f"\r{_color_text('[PROCESSING]')} Models processed {processed_models}/{total_models} | Currently {model_name} - Evolution texts processed {processed_texts}/{total_texts}",
            end="",
            flush=True,
        )

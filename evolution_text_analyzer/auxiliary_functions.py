"""
Module containing auxiliary functions for the medical diagnostic analysis system.
This module provides helper functions for various aspects of the system including
model management, data loading, configuration, and visualization.
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

from .data_models import ModelInfo

EMBEDDINGS_MODEL = "nomic-embed-text:latest"


def _color_text(text, color="green"):
    """
    Format text with ANSI color codes for terminal output.
    
    Args:
        text: The text to be colored
        color: Color name to use (red, green, bold, cyan)
    
    Returns:
        String formatted with ANSI color codes
    """
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "cyan": "\033[96m",
    }
    return f"{colors.get(color, '')}[{text}]\033[0m"


def check_ollama_connection(url: str = "http://localhost:11434") -> None:
    """
    Verify that Ollama server is running and accessible.
    
    Args:
        url: URL of the Ollama server to check
    
    Returns:
        None, exits with error code 1 if connection fails
    """
    try:
        if not requests.get(url).status_code == 200:
            print(
                f"{_color_text('ERROR', 'red')} An error ocurred. Imposible to connect to Ollama.\n")
            exit(1)
    except requests.ConnectionError:
        print(
            f"{_color_text('ERROR', 'red')} Ollama is not running.")
        exit(1)


def get_args(num_evolution_texts: int):
    """
    Parse command line arguments for the application.
    
    Args:
        num_evolution_texts: Total number of evolution texts available
    
    Returns:
        Parsed command line arguments
    """
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
    parser.add_argument("-v", "--verbose",
                        action="store_true", dest="verbose_mode")
    parser.add_argument("-E", "--expand", action="store_true",
                        dest="expansion_mode", help="Expand evolution texts")
    parser.add_argument("-N", "--normalize", action="store_true",
                        dest="normalization_mode", help="Normalize results via RAG")

    return parser.parse_args()


def get_evolution_texts(path: Path):
    """
    Load evolution texts from CSV or JSON file.
    
    Args:
        path: Path to the file containing evolution texts
    
    Returns:
        List of dictionaries containing evolution texts
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is not supported
    """
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


def get_analyzer_configuration(path: Path):
    """
    Load analyzer configuration from JSON file.
    
    Args:
        path: Path to the configuration file
    
    Returns:
        Dictionary containing analyzer configuration
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the JSON format is invalid
    """
    if not path.exists():
        raise FileNotFoundError(f"File '{path}' not found")
    try:
        return pd.read_json(path, typ="series").to_dict()
    except ValueError:
        raise ValueError(f"Invalid JSON format in file '{path}'")


def _create_chroma_db(index_path: str = "icd_vector_db", model_name: str = "nomic-embed-text:latest") -> bool:
    """
    Create a new Chroma vector database for ICD data.
    
    Args:
        index_path: Directory path where the database will be stored
        model_name: Name of the embedding model to use
    
    Returns:
        Chroma database object if successful, False otherwise
    """
    index_dir = Path(index_path)
    csv_path: str = "icd_dataset.csv"

    print(
        f"\r{_color_text('INFO')} Chroma database not found. Creating new one from {csv_path}...", end="")

    df = pd.read_csv(csv_path, quotechar="\"")
    df["text"] = df["principal_diagnostic"] + ":" + df["icd_code"]
    texts = df["text"].tolist()
    metadatas = df[["principal_diagnostic", "icd_code"]
                   ].to_dict(orient="records")

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
            f"\n{_color_text('ERROR', 'red')} Failed to create Chroma DB: {e}")
        return False


def get_chroma_db(index_path: str = "icd_vector_db") -> Chroma:
    """
    Get Chroma vector database for ICD data, creating it if it doesn't exist.
    
    Args:
        index_path: Directory path where the database is stored
    
    Returns:
        Chroma database object
    """
    index_dir = Path(index_path)

    if not get_listed_models([EMBEDDINGS_MODEL], True)[0]["installed"]:
        print(
            f"{_color_text('DOWNLOADING', 'cyan')} '{EMBEDDINGS_MODEL}' ", end="")
        ollama.pull(EMBEDDINGS_MODEL)

    if not index_dir.exists():
        _create_chroma_db(index_path, EMBEDDINGS_MODEL)

    embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL)
    return Chroma(
        persist_directory=str(index_dir),
        embedding_function=embeddings
    )


def _process_model_info(raw_model: str, installed_models: list[dict]) -> ModelInfo:
    """
    Process raw model information into a structured ModelInfo object.
    
    Args:
        raw_model: Name of the model
        installed_models: List of installed models from Ollama
    
    Returns:
        ModelInfo object containing model details
    """
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
    """
    Get information about specified models, optionally filtering for installed ones.
    
    Args:
        raw_models: List of model names to get information for
        installed_only: If True, return only installed models
    
    Returns:
        List of dictionaries containing model information
    
    Raises:
        ValueError: If the raw_models list is empty
    """
    if not raw_models:
        raise ValueError("No models found in the list")

    installed = ollama.list()["models"]
    models = [_process_model_info(
        model, installed).__dict__ for model in raw_models]
    return [model for model in models if model["installed"]] if installed_only else models


def download_model(model: dict) -> bool:
    """
    Download a model using Ollama.
    
    Args:
        model: Dictionary containing model information
    
    Returns:
        True if download was successful, False otherwise
    """
    try:
        print(
            f"\r{_color_text('DOWNLOADING', 'cyan')} '{model['model_name']}'", end="")
        ollama.pull(model["model_name"])
        updated = next((ollama_model for ollama_model in ollama.list()[
                       "models"] if ollama_model["model"] == model["model_name"]), None)
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
        print(f"\n{_color_text('ERROR')}Download failed: {e}")
        return False


def check_model(model: dict) -> bool:
    """
    Check if a model is installed, downloading it if necessary.
    
    Args:
        model: Dictionary containing model information
    
    Returns:
        True if the model is available (installed or successfully downloaded), False otherwise
    """
    if not model["installed"]:
        return download_model(model)
    model.pop("installed", None)
    return True


def display_model_table(models: list[dict]) -> None:
    """
    Display a formatted table of models in the terminal.
    
    Args:
        models: List of dictionaries containing model information
    """
    col_widths = {
        "name": max(len(model["model_name"]) for model in models) + 4,
        "available": len("Not installed") + 4,
        "details": 15
    }

    header = (
        f"\r{'ID'.rjust(4)}  "
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
    """
    Display a list of models and prompt the user to choose one.
    
    Args:
        models: List of model names
        installed_only: If True, only show installed models
    
    Returns:
        List containing a single dictionary with the chosen model information, or None if selection failed
    
    Raises:
        SystemExit: If no models are available to choose from
    """
    listed = get_listed_models(models, installed_only)
    if not listed:
        print(
            f"{_color_text('ERROR', 'red')} No models available to choose from.")
        exit(1)

    display_model_table(listed)
    while True:
        choice = input(f"\nSelect model (1 - {len(listed)}): ").strip()

        if choice.isnumeric() and 1 <= int(choice) <= len(listed):
            selected = listed[int(choice) - 1]
            return [selected] if check_model(selected) else None

        print(
            f"{_color_text('[ERROR]', 'red')} Invalid selection. Please try again.")


def print_evaluated_results(model: dict, results: BaseModel, verbose: bool) -> None:
    """
    Print evaluation results for a model.
    
    Args:
        model: Dictionary containing model information
        results: Evaluation results as a BaseModel
        verbose: If True, print detailed results for each text
    """
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
                    f"\tProcessing Error: {evaluated_text.processed_output.processing_error}")
            if evaluated_text.processed_output.validation_error:
                print(
                    f"\tValidation Error: {evaluated_text.processed_output.validation_error}")
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
        f"\tDuration: {performance.duration} s.", end="\n\n", flush=True)


def write_results(results_path: str, results: dict) -> None:
    """
    Write analysis results to a JSON file.
    
    Args:
        results_path: Path to write the results to
        results: Dictionary containing analysis results
    """
    with open(results_path, mode="w", encoding="utf-8") as file:
        json.dump(results, file, indent=3, ensure_ascii=False)


def print_execution_progression(
    model_name: str,
    processed_texts: int,
    total_texts: int,
) -> None:
    """
    Print the progress of text processing in the terminal.
    
    Args:
        model_name: Name of the model being used
        processed_texts: Number of texts processed so far
        total_texts: Total number of texts to process
    """
    
    print(f"\r{' ' * os.get_terminal_size().columns}", end="", flush=True)
    
    print(
        f"\r{_color_text('TESTING')} {model_name} - Evolution texts processed {processed_texts}/{total_texts}",
        end="",
        flush=True,
    )
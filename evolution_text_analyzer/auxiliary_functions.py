"""
Auxiliary functions module for the medical diagnostic analysis system.

This module provides utility functions for various system components including:
- Configuration loading and validation
- Command-line argument parsing
- File I/O operations
- Model management and selection
- Progress reporting and visualization
- Vector database operations
"""

import json
import os
from argparse import ArgumentParser
from pathlib import Path
import subprocess

import ollama
import pandas as pd
import requests
from pydantic import BaseModel, ByteSize

from .data_models import ModelInfo

def color_text(text, color="green"):
    """
    Format text with ANSI color codes for terminal output.

    Args:
        text: The text to be colored
        color: Color name (red, green, cyan)

    Returns:
        String formatted with ANSI color codes
    """
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "cyan": "\033[96m",
    }
    return f"{colors.get(color, '')}[{text}]\033[0m"


def get_exclusion_terms() -> list[str]:
    return [
        # Spanish terms
        "con", "y", "de", "del", "la", "el", "en", "por", "sin", "a", "para",
        "debido", "asociado", "secundario", "primario", "crónico", "agudo",
        "al", "ambos", "ante", "cada", "como", "desde", "ella", "hasta",
        "las", "lo", "los", "que", "se", "según", "sí", "sobre", "su",
        "un", "una", "unas", "uno", "unos",

        # English terms
        "with", "and", "of", "the", "in", "by", "without", "to", "for",
        "due", "associated", "secondary", "primary", "chronic", "acute",
        "at", "both", "before", "each", "as", "from", "she", "until",
        "them", "it", "they", "that", "is", "according", "yes", "on", "his", "her", "its", "their",
        "a", "an", "some", "one", "ones"
    ]


def check_ollama_connection(url: str = "http://localhost:11434") -> None:
    """
    Verify that Ollama server is running and accessible.

    Attempts to connect to the Ollama API endpoint and exits the application
    with an error message if the connection fails.

    Args:
        url: URL of the Ollama server to check

    Raises:
        SystemExit: If connection to Ollama server fails
    """
    try:
        if not requests.get(url).status_code == 200:
            print(
                f"{color_text('ERROR', 'red')} An error occurred. Unable to connect to Ollama.\n"
            )
            exit(1)
    except requests.ConnectionError:
        print(f"{color_text('ERROR', 'red')} Ollama is not running.")
        exit(1)


def get_context_window_length(model_name: str) -> int:
    """
    Get raw model information directly using CLI command.

    Args:
        model_name: Name of the Ollama model

    Returns:
        Raw output from the 'ollama show' command
    """
    try:
        result = subprocess.run(
            ["ollama", "show", model_name],
            capture_output=True,
            text=True,
            check=True
        )
        for line in result.stdout.split('\n'):
            line = line.strip()
            if "context length" in line.lower():
                return int(line.split()[-1])
        return -1
    except subprocess.CalledProcessError as e:
        print(f"Error running ollama show: {e}")
        return -1
    except FileNotFoundError:
        print("Error: ollama command not found. Is it installed and in PATH?")
        return -1


def get_analyzer_configuration(path: Path):
    """
    Load and validate analyzer configuration from JSON file.

    Reads the configuration file and verifies that all required fields
    and prompts are present with appropriate values.

    Args:
        path: Path to the configuration file

    Returns:
        Dictionary containing validated analyzer configuration

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If JSON format is invalid or required fields are missing
    """
    if not path.exists():
        raise FileNotFoundError(f"File '{path}' not found")

    try:
        config = pd.read_json(path, typ="series").to_dict()
    except ValueError:
        raise ValueError(f"Invalid JSON format in file '{path}'")

    # Verify required fields
    required_fields = ["optimal_model", "models", "prompts"]
    missing_fields = [
        field for field in required_fields if field not in config]
    if missing_fields:
        raise ValueError(
            f"Missing required fields in configuration: {', '.join(missing_fields)}"
        )

    if not config["models"] or len(config["models"]) == 0:
        raise ValueError("The 'models' array cannot be empty")

    if not isinstance(config["prompts"], dict):
        raise ValueError("The 'prompts' field must be a dictionary")

    # Verify required prompts in the new structure
    required_prompts = [
        "gen_diagnostic_prompt",
        "gen_icd_code_prompt"
    ]
    missing_prompts = [
        prompt for prompt in required_prompts if prompt not in config["prompts"]
    ]
    if missing_prompts:
        raise ValueError(
            f"Missing required prompts: {', '.join(missing_prompts)}")

    # Validate optimal_model
    if not isinstance(config["optimal_model"], int) or not config[
        "optimal_model"
    ] == int(config["optimal_model"]):
        raise ValueError("The 'optimal_model' must be an integer")

    optimal_model_index = int(config["optimal_model"])
    if optimal_model_index < 0 or optimal_model_index >= len(config["models"]):
        raise ValueError(
            f"The 'optimal_model' index ({optimal_model_index}) is out of range for the models array (0-{len(config['models']) - 1})"
        )

    return config


def get_args():
    """
    Parse command line arguments for the application.

    Defines and processes all command line options for controlling the system's
    behavior, including operation mode, batch size, and processing flags.

    Returns:
        Parsed command line arguments object
    """
    parser = ArgumentParser(
        description="Medical evolution text analyzer with multiple operation modes.",
        allow_abbrev=False
    )
    parser.add_argument(
        "-f", "--filename",
        type=str,
        default="evolution_texts.csv",
        dest="et_filename",
        help="Filename for the evolution texts file"
    )
    parser.add_argument(
        "-m", "--mode",
        type=int,
        choices=[1, 2],
        default=1,
        dest="eval_mode",
        help="Operation mode: 1 for all models, 2 for model selection"
    )
    parser.add_argument(
        "-b", "--batches",
        type=int,
        default=1,
        dest="num_batches",
        help="Number of batches for parallel processing"
    )
    parser.add_argument(
        "-n", "--num-texts",
        type=int,
        default=None,
        dest="num_texts",
        help="Number of texts to process"
    )
    parser.add_argument(
        "-t", "--test",
        action="store_true",
        dest="test_mode",
        help="Run in test mode to evaluate model performance"
    )
    parser.add_argument(
        "-i", "--installed",
        action="store_true",
        dest="only_installed_models_mode",
        help="Only use models that are already installed"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        dest="verbose_mode",
        help="Print detailed output during processing"
    )
    parser.add_argument(
        "-N", "--normalize",
        action="store_true",
        dest="normalization_mode",
        help="Normalize results using SNOMED dataset"
    )

    return parser.parse_args()


def get_evolution_texts(path: Path):
    """
    Load evolution texts from CSV or JSON file.

    Reads medical evolution texts from a file and prepares them for processing,
    performing basic validation on the required fields.

    Args:
        path: Path to the file containing evolution texts

    Returns:
        List of dictionaries containing validated evolution texts

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If file format is unsupported or required fields are missing
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
            print(
                f"{color_text('ERROR', 'red')} Extension not supported. Must be .json or .csv"
            )
            exit(1)

    required_fields = ["id", "evolution_text"]
    for i, evolution_text in enumerate(texts):
        missing_fields = [
            field for field in required_fields if field not in evolution_text
        ]
        if missing_fields:
            raise ValueError(
                f"Missing required fields {missing_fields} in record {i}")

        evolution_text["evolution_text"] = evolution_text["evolution_text"].replace(
            "\n", " "
        )

    return texts


def get_installed_models(with_info: bool = False):
    """
    Get list of models installed in Ollama.

    Retrieves either model names or full model information from
    the local Ollama installation.

    Args:
        with_info: If True, return full model information, otherwise just names

    Returns:
        List of model names or model information dictionaries

    Raises:
        SystemExit: If Ollama API request fails
    """
    try:
        if with_info:
            installed_models = [
                installed_model_info for installed_model_info in ollama.list()["models"]
            ]
        else:
            installed_models = [
                installed_model_info["model"]
                for installed_model_info in ollama.list()["models"]
            ]

        return installed_models
    except Exception as e:
        print(color_text("ERROR", "red"), e)
        exit(1)


def model_installed(model_name: str) -> bool:
    """
    Check if a model is installed and download it if not.

    Verifies if the specified model is available locally in Ollama,
    and attempts to download it if not present.

    Args:
        model_name: Name of the model to check

    Returns:
        True if model is installed or successfully downloaded, False otherwise
    """
    if model_name in get_installed_models():
        return True
    else:
        return _download_model(model_name)


def _get_model_info(model_name: str) -> ModelInfo:
    """
    Get detailed information about a specific model.

    Retrieves and formats information about an Ollama model including
    its size, parameters, and quantization level.

    Args:
        model_name: Name of the model to get information for

    Returns:
        ModelInfo object with model details
    """
    installed_model_info = next(
        (
            installed_model_name
            for installed_model_name in get_installed_models(True)
            if installed_model_name["model"] == model_name
        ),
        None,
    )

    size, parameter_size, quant_level = None, None, None

    if installed_model_info:
        size = (
            f"{round(ByteSize(installed_model_info.get('size', 0)).to('GB'), 1)} GB"
            if "size" in installed_model_info
            else None
        )
        details = installed_model_info.get("details", {})
        parameter_size = details.get("parameter_size")
        quant_level = details.get("quantization_level")

    return ModelInfo(
        model_name=model_name,
        installed=bool(installed_model_info),
        size=size,
        parameter_size=parameter_size,
        quantization_level=quant_level,
    )


def get_listed_models_info(
    listed_model_names: list[str], installed_only: bool = False
) -> list[ModelInfo]:
    """
    Get information about multiple models from a list.

    Retrieves detailed information about multiple models, optionally
    filtering to only include models that are already installed.

    Args:
        listed_model_names: List of model names to get information for
        installed_only: If True, only return information for installed models

    Returns:
        List of ModelInfo objects with model details

    Raises:
        SystemExit: If no models are found in the list
    """
    if not listed_model_names:
        print(f"{color_text('ERROR', 'red')} No models found in config list")
        exit(1)

    models = [
        _get_model_info(listed_model_name) for listed_model_name in listed_model_names
    ]

    return (
        [model for model in models if model.installed] if installed_only else models
    )


def _download_model(model_name: str) -> bool:
    """
    Download a model from Ollama's model repository.

    Pulls the specified model from Ollama with a progress indicator,
    handling errors if the download fails.

    Args:
        model_name: Name of the model to download

    Returns:
        True if download successful, False otherwise
    """
    try:
        download_progress = ollama.pull(model_name, stream=True)
        total = 0
        completed = 0
        for partial_progress in download_progress:
            if 'total' in partial_progress:
                total = partial_progress.get('total', 0)
            if 'completed' in partial_progress:
                completed = partial_progress.get('completed', 0)

            if total > 0 and completed <= total:
                progress = (completed / total) * 100

                print(
                    f"\r{color_text('DOWNLOADING', 'cyan')} {model_name} {progress:.1f}%",
                    flush=True,
                    end=""
                )

        return True
    except (ollama.ResponseError, requests.RequestException) as e:
        print(f"\n{color_text('ERROR', 'red')} Download failed: {e}")
        return False


def _display_models_table(models: list[ModelInfo]) -> None:
    """
    Display a formatted table of model information.

    Creates a nicely formatted table showing model details including
    availability status, size, parameters, and quantization level.

    Args:
        models: List of ModelInfo objects to display
    """
    col_widths = {
        "name": max(len(model.model_name) for model in models) + 4,
        "available": len("Not installed") + 4,
        "details": 15,
    }

    header = (
        f"\r{'ID'.rjust(4)}  "
        f"{'NAME'.ljust(col_widths['name'])}"
        f"{'AVAILABLE'.ljust(col_widths['available'])}"
        f"{'SIZE'.ljust(col_widths['details'] - 5)}"
        f"{'PARAMETERS'.ljust(col_widths['details'])}"
        f"{'QUANTIZATION'.ljust(col_widths['details'])}"
    )
    print(header)

    for i, model in enumerate(models, 1):
        details = (
            (
                "Installed".ljust(col_widths["available"])
                + f"{model.size or ''}".ljust(col_widths["details"] - 5)
                + f"{model.parameter_size or ''}".ljust(col_widths["details"])
                + f"{model.quantization_level or ''}".ljust(col_widths["details"])
            )
            if model.installed
            else "Not installed"
        )

        print(
            f"{str(i).rjust(4)}. {model.model_name.ljust(col_widths['name'])}{details}"
        )


def choose_model(model_names: list[str], installed_only: bool = False) -> list[ModelInfo] | None:
    """
    Display available models and let the user choose one.

    Presents a list of models with details and prompts the user to select one,
    handling the download process if necessary.

    Args:
        model_names: List of model names to choose from
        installed_only: If True, only show models that are already installed

    Returns:
        List containing the selected ModelInfo object, or None if selection fails

    Raises:
        SystemExit: If no models are available to choose from
    """
    listed = get_listed_models_info(model_names, installed_only)
    if not listed:
        print(f"{color_text('ERROR', 'red')} No models available to choose from.")
        exit(1)

    _display_models_table(listed)
    while True:
        choice = input(f"\nSelect model (1 - {len(listed)}): ").strip()

        if choice.isnumeric() and 1 <= int(choice) <= len(listed):
            selected_model = listed[int(choice) - 1]
            if model_installed(selected_model.model_name):
                return selected_model

        print(
            f"{color_text('ERROR', 'red')} Invalid selection. Please try again.")


def print_evaluated_results(model: dict, results: BaseModel, verbose: bool) -> None:
    """
    Print evaluation results for a model.

    Displays model performance metrics and optionally detailed results
    for individual texts if verbose mode is enabled.

    Args:
        model: Dictionary containing model information
        results: Evaluation results as a BaseModel
        verbose: If True, print detailed results for each text
    """
    print(f"\r{' ' * os.get_terminal_size().columns}", end="", flush=True)
    if verbose:
        for id, evaluated_text in results.evaluated_texts.items():
            print(
                f"\n{id} - {color_text(evaluated_text.valid, 'green' if evaluated_text.valid else 'red')}\n"
                f"\tModel diagnostic: {evaluated_text.processed_output.principal_diagnostic} (Code - {evaluated_text.processed_output.icd_code})\n"
                f"\tCorrect diagnostic: {evaluated_text.correct_diagnostic}"
            )
            if evaluated_text.processed_output.processing_error:
                print(
                    f"\tProcessing Error: {evaluated_text.processed_output.processing_error}"
                )
            if evaluated_text.processed_output.validation_error:
                print(
                    f"\tValidation Error: {evaluated_text.processed_output.validation_error}"
                )
    print()

    performance = results.performance
    accuracy = performance.accuracy
    incorrect = performance.incorrect_outputs
    errors = performance.errors
    total = performance.total_texts

    print(
        f"\rModel: {model.model_name}\n"
        f"Performance:\n"
        f"\t{color_text('Accuracy')}: {accuracy}% ({int((accuracy / 100) * total)}/{total})\n"
        f"\t{color_text('Incorrect outputs', 'red')}: {incorrect}% ({int((incorrect / 100) * total)}/{total})\n"
        f"\tErrors: {errors}% ({int((errors / 100) * total)}/{total})\n\n"
        f"\tTotal records processed: {total}\n"
        f"\tDuration: {performance.duration} s.",
        end="\n",
        flush=True,
    )


def write_results(results_path: str, results: dict) -> None:
    """
    Write analysis results to a JSON file.

    Saves processed diagnostic results to a JSON file at the specified path,
    with appropriate formatting for readability.

    Args:
        results_path: Path to write the results to
        results: Dictionary containing analysis results
    """
    with open(results_path, mode="w", encoding="utf-8") as file:
        json.dump(results, file, indent=3, ensure_ascii=False)

    print(
        f"{color_text('COMPLETED')} Results available at:\n{results_path}", end="\n\n"
    )


def print_execution_progression(
    model_name: str,
    processed_texts: int,
    total_texts: int,
) -> None:
    """
    Print the progress of text processing in the terminal.

    Displays a progress indicator showing how many texts have been processed
    out of the total, providing real-time feedback during execution.

    Args:
        model_name: Name of the model being used
        processed_texts: Number of texts processed so far
        total_texts: Total number of texts to process
    """
    print(f"\r{' ' * os.get_terminal_size().columns}", end="", flush=True)

    print(
        f"\r{color_text('PROCESSING')} {model_name} - Evolution texts processed {processed_texts}/{total_texts}",
        end="",
        flush=True,
    )
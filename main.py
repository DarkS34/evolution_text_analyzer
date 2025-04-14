"""
Main entry point for the medical diagnostic analysis system.
"""
from pathlib import Path

from evolution_text_analyzer.analyzer import evolution_text_analysis
from evolution_text_analyzer.auxiliary_functions import (
    check_ollama_connection,
    choose_model,
    get_analyzer_configuration,
    get_args,
    get_evolution_texts,
    get_listed_models,
    get_chroma_db,
    write_results,
)
from evolution_text_analyzer.tester import evaluate_analysis


def run_test_analysis_mode(models: list[str], prompts: list[dict], evolution_texts: list, chroma_db, args):
    """
    Run the system in test analysis mode, evaluating model performance.

    Args:
        models: List of model names to evaluate
        prompts: List of prompts to use
        opt_prompt: Index of the optimal prompt
        evolution_texts: List of medical texts to analyze
        chroma_db: Chroma database for normalization
        args: Command line arguments
    """

    selected_models = (
        get_listed_models(models, args.only_installed_models)
        if args.mode == 1
        else choose_model(models, args.only_installed_models)
    )
    
    testing_results_dir = config_file.parent / "testing_results"
    testing_results_dir.mkdir(parents=True, exist_ok=True)

    evaluate_analysis(
        selected_models,
        prompts,
        evolution_texts,
        testing_results_dir,
        chroma_db,
        args.expansion_mode,
        args.num_batches,
        args.num_evolution_texts,
        args.verbose_mode,
    )


def run_analysis_mode(model: str, prompts: dict, evolution_texts: list, chroma_db, args):
    """
    Run the system in production analysis mode with a single model.

    Args:
        model: Name of the model to use
        prompts: Dictionary of prompts to use
        evolution_texts: List of medical texts to analyze
        chroma_db: Chroma database for normalization
        args: Command line arguments
    """
    results_dir = config_file.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results = evolution_text_analysis(
        model,
        prompts,
        evolution_texts,
        chroma_db,
        args.expansion_mode,
        args.num_batches,
        args.num_evolution_texts,
    )

    write_results(results_dir / "processed_evolution_texts.json", results)


if __name__ == "__main__":

    check_ollama_connection()

    base_path = Path(__file__).parent
    evolution_texts_file = base_path / "evolution_texts_resolved.csv"
    config_file = base_path / "config.json"

    config = get_analyzer_configuration(config_file)

    opt_model, models, prompts = config["optimal_model"], config["models"], config["prompts"]

    evolution_texts = get_evolution_texts(evolution_texts_file)
    args = get_args(len(evolution_texts))

    chroma_db = get_chroma_db() if args.normalization_mode else None

    if args.test or args.test_prompts:
        run_test_analysis_mode(
            models, prompts, evolution_texts, chroma_db, args)
    else:
        run_analysis_mode(
            models[opt_model], prompts, evolution_texts, chroma_db, args)

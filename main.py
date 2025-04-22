"""
Main entry point for the medical diagnostic analysis system.
This module orchestrates the system's workflow, handling configuration loading,
model selection, and execution of either test or analysis modes.
"""
from pathlib import Path

from evolution_text_analyzer.analyzer import evolution_text_analysis
from evolution_text_analyzer.auxiliary_functions import (
    check_ollama_connection,
    get_analyzer_configuration,
    get_args,
    get_context_window_length,
    get_evolution_texts,
    model_installed,
    write_results,
)
from evolution_text_analyzer.tester import evaluate_analysis


def run_test_analysis_mode(models: list[str], prompts: dict, args) -> None:
    evolution_texts = get_evolution_texts(
        base_path / "testing" / args.et_filename)

    testing_results_dir = config_file.parent / "testing" / "results"
    testing_results_dir.mkdir(parents=True, exist_ok=True)

    args.num_texts = args.num_texts if args.num_texts is not None else len(
        evolution_texts)

    evaluate_analysis(
        models,
        prompts,
        evolution_texts,
        testing_results_dir,
        args
    )


def run_analysis_mode(model: str, prompts: dict, args) -> None:
    evolution_texts = get_evolution_texts(base_path / args.et_filename)
    args.num_texts = args.num_texts if args.num_texts is not None else len(
        evolution_texts)

    results_dir = config_file.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    ctx_len = get_context_window_length(model)

    results = evolution_text_analysis(
        model,
        prompts,
        ctx_len,
        evolution_texts,
        args.num_batches,
        args.num_texts,
        args.normalization_mode
    )

    write_results(results_dir / "processed_evolution_texts.json", results)


if __name__ == "__main__":
    # Verify Ollama connection before proceeding
    check_ollama_connection()

    # Load configuration and setup paths
    base_path = Path(__file__).parent
    config_file = base_path / "config.json"

    config = get_analyzer_configuration(config_file)
    opt_model, models = config["optimal_model"], config["models"]

    # Process command line arguments
    args = get_args()

    # Run in appropriate mode based on command line arguments
    if args.test_mode:
        run_test_analysis_mode(
            models, config["prompts"], args)
    else:
        model_name = models[opt_model]
        if model_installed(model_name):
            run_analysis_mode(model_name,
                              config["prompts"], args)

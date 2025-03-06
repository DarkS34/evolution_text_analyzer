from pathlib import Path
from evolution_text_analyzer.auxiliary_functions import (
    check_ollama_connected,
    get_args,
    get_evolution_texts,
    get_listed_models,
    choose_model,
    get_optimal_analyzer_configuration,
    write_results
)

from evolution_text_analyzer.analyzer import evolution_text_analysis
from evolution_text_analyzer.llm_tester import evaluate_analysis

if __name__ == "__main__" and check_ollama_connected():
    EVOLUTION_TEXTS_FILENAME = Path(
        __file__).parent / "evolution_texts_resolved.csv"
    CONFIG_FILENAME = Path(__file__).parent / "config.json"
    # MODELS_LIST_FILENAME = Path(__file__).parent / "models.json"
    testingResultsDir = Path(__file__).parent / "testing_results"
    testingResultsDir.mkdir(parents=True, exist_ok=True)
    resultsDir = Path(__file__).parent / "results"
    resultsDir.mkdir(parents=True, exist_ok=True)

    evolutionTexts = get_evolution_texts(EVOLUTION_TEXTS_FILENAME)

    args = get_args(len(evolutionTexts))
    if args.test or args.test_prompts:
        _, models, systemPrompts, outputFormatting = get_optimal_analyzer_configuration(CONFIG_FILENAME)
        evaluate_analysis(
            get_listed_models(models, args.installed) if args.mode == 1 else choose_model(
                models, args.installed),
            evolutionTexts,
            systemPrompts if args.test_prompts else [systemPrompts[0]],
            outputFormatting,
            args.batches,
            args.num_texts,
            testingResultsDir,
        )
    else:
        opt, models, systemPrompts, outputFormatting = get_optimal_analyzer_configuration(
            CONFIG_FILENAME)
        results = evolution_text_analysis(
            models[opt[0]],
            evolutionTexts,
            systemPrompts[opt[1]]+outputFormatting,
            args.batches,
            args.num_texts,
        )
        write_results(resultsDir / "processed_evolution_texts.json", results)

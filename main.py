from pathlib import Path
from evolution_text_analyzer.auxiliary_functions import (
    check_ollama_connected,
    get_args,
    get_evolution_texts,
    get_listed_models,
    choose_model,
    get_analyzer_configuration,
    write_results
)

from evolution_text_analyzer.analyzer import evolution_text_analysis
from evolution_text_analyzer.tester import evaluate_analysis

if __name__ == "__main__" and check_ollama_connected():
    EVOLUTION_TEXTS_FILENAME = Path(
        __file__).parent / "evolution_texts_resolved.csv"
    CONFIG_FILENAME = Path(__file__).parent / "config.json"
    testingResultsDir = Path(__file__).parent / "testing_results"
    testingResultsDir.mkdir(parents=True, exist_ok=True)
    resultsDir = Path(__file__).parent / "results"
    resultsDir.mkdir(parents=True, exist_ok=True)

    evolutionTexts = get_evolution_texts(EVOLUTION_TEXTS_FILENAME)

    args = get_args(len(evolutionTexts))
    if args.test or args.testPrompts:
        _, models, systemPrompts, outputFormatting = get_analyzer_configuration(
            CONFIG_FILENAME)
        evaluate_analysis(
            get_listed_models(models, args.onlyInstalledModels) if args.mode == 1 else choose_model(
                models, args.onlyInstalledModels),
            evolutionTexts,
            systemPrompts if args.testPrompts else [systemPrompts[0]],
            outputFormatting,
            args.numBatches,
            args.numEvolutionTexts,
            testingResultsDir,
            args.verbose,
        )
    else:
        opt, models, systemPrompts, outputFormatting = get_analyzer_configuration(
            CONFIG_FILENAME)
        results = evolution_text_analysis(
            models[opt[0]],
            evolutionTexts,
            systemPrompts[opt[1]]+outputFormatting,
            args.numBatches,
            args.numEvolutionTexts,
        )
        write_results(resultsDir / "processed_evolution_texts.json", results)

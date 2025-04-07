from pathlib import Path
from evolution_text_analyzer.auxiliary_functions import (
    check_ollama_connected,
    get_args,
    get_evolution_texts,
    get_listed_models,
    choose_model,
    get_analyzer_configuration,
    write_results,
)
from evolution_text_analyzer.analyzer import evolution_text_analysis
from evolution_text_analyzer.tester import evaluate_analysis


def run_test_analysis_mode(configFile: Path, evolutionTexts: list, args):
    testingResultsDir = configFile.parent / "testing_results"
    testingResultsDir.mkdir(parents=True, exist_ok=True)

    _, models, systemPrompts, outputFormatting = get_analyzer_configuration(configFile)

    selectedModels = (
        get_listed_models(models, args.onlyInstalledModels)
        if args.mode == 1
        else choose_model(models, args.onlyInstalledModels)
    )
    selectedPrompts = systemPrompts if args.testPrompts else [systemPrompts[0]]

    evaluate_analysis(
        selectedModels,
        evolutionTexts,
        selectedPrompts,
        outputFormatting,
        args.numBatches,
        args.numEvolutionTexts,
        testingResultsDir,
        args.verboseMode,
    )


def run_analysis_mode(configFile: Path, evolutionTexts: list, args):
    resultsDir = configFile.parent / "results"
    resultsDir.mkdir(parents=True, exist_ok=True)

    opt, models, systemPrompts, outputFormatting = get_analyzer_configuration(configFile)

    results = evolution_text_analysis(
        models[opt[0]],
        evolutionTexts,
        systemPrompts[opt[1]] + outputFormatting,
        args.numBatches,
        args.numEvolutionTexts,
    )

    write_results(resultsDir / "processed_evolution_texts.json", results)


if __name__ == "__main__":
    if not check_ollama_connected():
        print("❌ No se detecta conexión con Ollama.")
        exit(1)

    basePath = Path(__file__).parent
    evolutionTextsFile = basePath / "evolution_texts_resolved.csv"
    configFile = basePath / "config.json"

    evolutionTexts = get_evolution_texts(evolutionTextsFile)
    args = get_args(len(evolutionTexts))

    if args.test or args.testPrompts:
        run_test_analysis_mode(configFile, evolutionTexts, args)
    else:
        run_analysis_mode(configFile, evolutionTexts, args)
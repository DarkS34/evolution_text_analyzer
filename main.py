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


def run_test_analysis_mode(models: list[str], prompts: list[dict], optPrompt: int, evolutionTexts: list, args):
    testingResultsDir = configFile.parent / "testing_results"
    testingResultsDir.mkdir(parents=True, exist_ok=True)

    selectedModels = (
        get_listed_models(models, args.onlyInstalledModels)
        if args.mode == 1
        else choose_model(models, args.onlyInstalledModels)
    )

    evaluate_analysis(
        selectedModels,
        (args.testPrompts, prompts),
        optPrompt,
        evolutionTexts,
        args.numBatches,
        args.numEvolutionTexts,
        testingResultsDir,
        args.verboseMode,
    )


def run_analysis_mode(model: str, prompt: dict, evolutionTexts: list, args):
    resultsDir = configFile.parent / "results"
    resultsDir.mkdir(parents=True, exist_ok=True)

    results = evolution_text_analysis(
        model,
        prompt,
        evolutionTexts,
        args.numBatches,
        args.numEvolutionTexts,
    )

    write_results(resultsDir / "processed_evolution_texts.json", results)


if __name__ == "__main__":
    if not check_ollama_connected():
        print("No connection to Ollama detected.")
        exit(1)

    basePath = Path(__file__).parent
    evolutionTextsFile = basePath / "evolution_texts_resolved.csv"
    configFile = basePath / "config.json"

    config = get_analyzer_configuration(
        configFile)

    opt, models, prompts = config["optimal"], config["models"], config["prompts"]

    evolutionTexts = get_evolution_texts(evolutionTextsFile)
    args = get_args(len(evolutionTexts))
    
    if args.test or args.testPrompts:
        run_test_analysis_mode(models, prompts, opt["prompt"], evolutionTexts, args)
    else:
        run_analysis_mode(
            models[opt[0]], prompts[opt[1]], evolutionTexts, args)

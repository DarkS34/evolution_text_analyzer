from pathlib import Path
from scripts.auxiliary_functions import (
    checkOllamaConnected,
    getArgs,
    getEvolutionTexts,
    chooseModel,
    getListedModels,
    getOptAnalyzerConfig,
    writeResults,
    printEvaluatedResults,
)

from scripts.analyzer import evolutionTextAnalysis
from scripts.llm_tester import evaluateLLM, evaluateListedLLMs

if __name__ == "__main__" and checkOllamaConnected():
    EVOLUTION_TEXTS_FILENAME = Path(__file__).parent / "evolution_texts_resolved.csv"
    CONFIG_FILENAME = Path(__file__).parent / "config.json"
    MODELS_LIST_FILENAME = Path(__file__).parent / "models.json"
    testingResultsDir = Path(__file__).parent / "testing_results"
    testingResultsDir.mkdir(parents=True, exist_ok=True)
    resultsDir = Path(__file__).parent / "results"
    resultsDir.mkdir(parents=True, exist_ok=True)

    evolutionTexts = getEvolutionTexts(EVOLUTION_TEXTS_FILENAME)

    args = getArgs(len(evolutionTexts))
    if not args.test:
        match args.mode:
            case 1:
                _, systemPrompt = getOptAnalyzerConfig(CONFIG_FILENAME)
                evaluateListedLLMs(
                    getListedModels(MODELS_LIST_FILENAME, args.installed),
                    evolutionTexts,
                    args.batches,
                    systemPrompt,
                    args.num_texts,
                    testingResultsDir,
                )
            case 2:
                model = chooseModel(MODELS_LIST_FILENAME, args.installed)
                _, systemPrompt = getOptAnalyzerConfig(CONFIG_FILENAME)
                results = evaluateLLM(
                    model,
                    evolutionTexts,
                    args.batches,
                    systemPrompt,
                    args.num_texts,
                )

                printEvaluatedResults(results)
                writeResults(
                    testingResultsDir
                    / f"results_{model['modelName'].replace(':', '_')}.json",
                    results,
                )
            case _:
                print("Modo no disponible")
    else:
        modelName, systemPrompt = getOptAnalyzerConfig(CONFIG_FILENAME)
        # if checkModel(modelName)
        results = evolutionTextAnalysis(
            modelName,
            evolutionTexts,
            args.batches,
            systemPrompt,
            args.num_texts,
        )
        writeResults(resultsDir / "processed_evolution_texts.json", results)

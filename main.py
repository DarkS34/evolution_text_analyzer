from pathlib import Path
from analyzer.auxiliary_functions import (
    checkOllamaConnected,
    getArgs,
    getEvolutionTexts,
    getModels,
    checkModel,
    chooseModel,
    updateResults,
    printProcessedResults,
)
from analyzer.parallel_ollama_et_analyzer import evolutionTextAnalysis

if __name__ == "__main__" and checkOllamaConnected():
    args = getArgs()
    EVOLUTION_TEXTS_FILENAME = Path(__file__).parent / "historiales_resueltos.csv"
    MODELS_LIST_FILENAME = Path(__file__).parent / "models.json"
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    medicalData = getEvolutionTexts(EVOLUTION_TEXTS_FILENAME)

    match args.mode:
        case 1:
            models = getModels(MODELS_LIST_FILENAME, args.installed)
            modelsResults = []
            for mIdx, model in enumerate(models):
                if checkModel(model):
                    partialResults = evolutionTextAnalysis(
                        model, medicalData, args.batches, mIdx, len(models)
                    )
                    updateResults(results_dir, partialResults, modelsResults)
        case 2:
            model = chooseModel(MODELS_LIST_FILENAME, args.installed)
            if model:
                results = evolutionTextAnalysis(model, medicalData, args.batches)
                updateResults(results_dir, results, [])
                printProcessedResults(results)
        case _:
            print("Modo no disponible")

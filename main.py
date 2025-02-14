from pathlib import Path
from analyzer.auxiliary_functions import (
    checkOllamaConnected,
    getArgs,
    getEvolutionTexts,
    getIcdDataset,
    getModels,
    checkModel,
    chooseModel,
    updateResults,
    printProcessedResults,
    writeProcessedResult,
)

# from analyzer.parallel_ollama_et_analyzer import evolutionTextAnalysis
from analyzer.parallel_ollama_et_analyzer_cie import evolutionTextAnalysis

if __name__ == "__main__" and checkOllamaConnected():
    args = getArgs()
    EVOLUTION_TEXTS_FILENAME = Path(__file__).parent / "historiales_resueltos.csv"
    ICD_DATASET_FILENAME = Path(__file__).parent / "dataset.csv"
    MODELS_LIST_FILENAME = Path(__file__).parent / "models.json"
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    medicalData = getEvolutionTexts(EVOLUTION_TEXTS_FILENAME)
    diagToIcdMap = getIcdDataset(ICD_DATASET_FILENAME)

    match args.mode:
        case 1:
            models = getModels(MODELS_LIST_FILENAME, args.installed)
            modelsResults = []
            for mIdx, model in enumerate(models):
                if checkModel(model):
                    partialResults = evolutionTextAnalysis(
                        model, medicalData, diagToIcdMap, args.batches, args.reason, mIdx, len(models)
                    )
                    updateResults(results_dir, partialResults, modelsResults)
        case 2:
            model = chooseModel(MODELS_LIST_FILENAME, args.installed)
            if model:
                # results = evolutionTextAnalysis(model, medicalData, args.batches)
                results = evolutionTextAnalysis(
                    model, medicalData, diagToIcdMap, args.batches, args.reason
                )
                printProcessedResults(results)
                writeProcessedResult(
                    results_dir / f"results_{model["modelName"].split(":")[0]}.json",
                    results,
                )
        case _:
            print("Modo no disponible")

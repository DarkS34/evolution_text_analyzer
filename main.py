from analyzer import parallel_ollama_et_analyzer, auxiliary_functions as aux_functions
from pathlib import Path

args = aux_functions.getArgs()

if __name__ == "__main__" and aux_functions.checkOllamaConnected():

    EVOLUTION_TEXTS_FILENAME = Path(__file__).parent / "historiales_resueltos.csv"
    MODELS_LIST_FILENAME = Path(__file__).parent / "models_list" / "models_list.json"
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    medicalData = aux_functions.convertToDict(EVOLUTION_TEXTS_FILENAME)

    match args.mode:
        case 1:
            model = aux_functions.chooseModel(MODELS_LIST_FILENAME)
            modelsResults = []
            if model:
                results = parallel_ollama_et_analyzer.evolutionTextAnalysis(
                    model, medicalData
                )
                aux_functions.updateResults(results_dir, results, modelsResults)
                aux_functions.printProcessedResults(results)
            print()
        case 2:
            models = aux_functions.getAllModels(MODELS_LIST_FILENAME)
            modelsResults = []
            for mIdx, model in enumerate(models):
                if aux_functions.checkModel(model):
                    partialResults = parallel_ollama_et_analyzer.evolutionTextAnalysis(
                        model, medicalData, mIdx, len(models)
                    )
                    aux_functions.updateResults(
                        results_dir, partialResults, modelsResults
                    )
            print()
        case _:
            print("Modo no disponible")

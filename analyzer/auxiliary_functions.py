from pydantic import ByteSize
from pathlib import Path
import csv, json, ollama, curses, requests, os, argparse, copy


def getArgs():
    parser = argparse.ArgumentParser(description="Script con etiquetas")
    parser.add_argument(
        "-mode",
        type=int,
        default=2,
        required=False,
        choices=[1, 2],
        help="Modo de operación (1, 2, etc.)",
    )
    parser.add_argument(
        "-installed",
        action="store_true",
        required=False,
        help="Solo se usan modelos instalados",
    )
    parser.add_argument(
        "-silent",
        action="store_true",
        required=False,
        help="Solo se usan modelos instalados",
    )

    global installedOnlyOption
    global silentOption
    installedOnlyOption = parser.parse_args().installed
    silentOption = parser.parse_args().silent
    return parser.parse_args()


def checkOllamaConnected(url="http://localhost:11434"):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return True
        else:
            print("There is a problem. Try to restart Ollama to see available models.")
            return False
    except requests.ConnectionError as e:
        print(
            f"Error:\n{e}.\nOllama is not running. Start ollama to see available models.\n"
        )
        return False


def getAllModels(modelsListpath: Path):
    modelsList = []

    def modelTemplate(
        modelName: str,
        size: ByteSize | str,
        parameters_size: str,
        quantization_level: str,
        installed: bool = True,
    ):
        return {
            "modelName": modelName,
            "size": (
                size
                if isinstance(size, str)
                else str(round(ByteSize(size).to("GB"), 1)) + " GB"
            ),
            "parametersSize": parameters_size,
            "quantizationLevel": quantization_level,
            "installed": installed,
        }

    def addInstalledOllamaModels(ml: list):
        installedOllamaModels = []
        for model in ollama.list()["models"]:
            installedOllamaModels.append(
                modelTemplate(
                    model["model"],
                    model["size"],
                    model["details"]["parameter_size"],
                    model["details"]["quantization_level"],
                )
            )
        installedOllamaModels.sort(key=lambda x: x["modelName"])
        ml += installedOllamaModels

    def addListModels(ml: list):
        listModels = []
        if not installedOnlyOption and os.path.isfile(modelsListpath):
            try:
                with open(modelsListpath, mode="r", encoding="utf-8") as modelsListFile:
                    fileExtension = modelsListpath.suffix
                    if fileExtension == ".json":
                        rawModelsList = json.load(modelsListFile)
                        for model in rawModelsList:
                            found = any(
                                installedModel["modelName"] == model["modelName"]
                                for installedModel in modelsList
                            )
                            if not found:
                                listModels.append(
                                    modelTemplate(
                                        model["modelName"],
                                        model["size"],
                                        model["parameters_size"],
                                        model["quantization_level"],
                                        False,
                                    )
                                )
                    else:
                        print(
                            "List of models - Extension not supported. Extension must be: '.json'"
                        )
            except Exception as e:
                raise (f"Error while loading list of models: {e}")
            listModels.sort(key=lambda x: x["modelName"])
        ml += listModels

    addInstalledOllamaModels(modelsList)
    addListModels(modelsList)
    if len(modelsList) > 0:
        return modelsList
    else:
        raise Exception("No models to use.")


def convertToDict(path: Path):
    evolutionTextsList = []
    fileExtension = path.suffix
    try:
        with open(path, mode="r", encoding="utf-8") as file:
            if fileExtension == ".csv":
                lector = csv.DictReader(file, delimiter="|")
                for fila in lector:
                    evolutionTextsList.append(dict(fila))
            elif fileExtension == ".json":
                evolutionTextsList = json.load(file)
            else:
                raise Exception(
                    "Evolution Texts - Extension not supported. Extension must be: '.json', '.csv'"
                )
    except FileNotFoundError:
        raise Exception(f"File '{path}' not found")
    except Exception as e:
        raise Exception(f"Error: {e}")

    return evolutionTextsList


def downloadModel(modelName):
    try:
        if not silentOption:
            print(f"Downloading model - '{modelName}' ...", end="\r")
        ollama.pull(modelName)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        return any(
            modelName == ollamaModel["model"] for ollamaModel in ollama.list()["models"]
        )


def checkModel(model):
    if not model["installed"]:
        return downloadModel(model["modelName"])
    else:
        return True


def chooseModel(modelsListpath: Path):
    modelsList = getAllModels(modelsListpath)

    model_width = max(len(modelInfo["modelName"]) for modelInfo in modelsList) + 2
    size_width = max(len(modelInfo["size"]) for modelInfo in modelsList) + 3
    paramStr = "Parameters Size"
    quantStr = "Quantization Level"
    instStr = "Installed"

    def modelsMenu(stdscr):
        current_option = 0

        curses.curs_set(0)
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)

        while True:
            stdscr.clear()
            stdscr.addstr(
                0,
                0,
                f"\t{'Model'.ljust(model_width)}|{'Size'.center(size_width)}|{paramStr.center(len(paramStr)+2)}|{quantStr.center(len(quantStr)+2)}|{instStr.center(len(instStr)+2)}",
                curses.A_BOLD,
            ),

            stdscr.addstr(
                1,
                0,
                f"\t{'‾'*(model_width+size_width+len(paramStr)+2+len(quantStr)+2+len(instStr)+2+4)}",
                curses.A_BOLD,
            )

            for idx, modelInfo in enumerate(modelsList):
                model = modelInfo["modelName"].ljust(model_width)
                size = modelInfo["size"].center(size_width)
                param = modelInfo["parametersSize"].center(len(paramStr) + 2)
                quant = modelInfo["quantizationLevel"].center(len(quantStr) + 2)
                installed = ("Yes" if modelInfo["installed"] else "No").center(
                    len(instStr) + 2
                )
                line = f"{idx+1}.\t{model}|{size}|{param}|{quant}|{installed}"
                if idx == current_option:
                    stdscr.addstr(
                        idx + 2,
                        0,
                        f">> {line}",
                        curses.A_BOLD | curses.color_pair(1),
                    )
                else:
                    stdscr.addstr(idx + 2, 0, f"   {line}")

            stdscr.refresh()
            key = stdscr.getch()

            if key == curses.KEY_UP and current_option > 0:
                current_option -= 1
            elif key == curses.KEY_DOWN and current_option < len(modelsList) - 1:
                current_option += 1
            elif key == ord("\n"):
                return modelsList[current_option]

    choosenModel = curses.wrapper(modelsMenu)
    if checkModel(choosenModel):
        choosenModel.pop("installed")
        return choosenModel
    else:
        return None


def printProcessedResults(results: dict):
    if not silentOption:
        for id, processedETResult in results["evolutionTextsResults"].items():
            print(
                f"\n{id} - {processedETResult['valid']} {'-'*20}\nResultado modelo: {processedETResult['processedOutput'].get('principal_diagnostic')} ({processedETResult['processedOutput'].get('icd_code')})\nResultado correcto: {processedETResult['correctOutput']['principal_diagnostic']}"
            )
        print(f"\n\tAccuracy: {results['performance']['accuracy']:.2f}%")
        print(f"\tIncorrect outputs: {results['performance']['incorrectOutputs']:.2f}%")
        print(f"\tErrors: {results['performance']['errors']:.2f}%")
        print(f"\tProcessing time: {results['performance']['processingTime']} s.")


def updateResults(resultsDirPath: Path, partialResult: dict, modelsResults: list):
    def simplifyProcess(result):
        result["performance"].pop("errors")
        result["performance"].pop("incorrectOutputs")
        result["performance"].pop("processingTime")
        result["performance"].pop("numBatches")
        for processedData in result["evolutionTextsResults"].values():
            processedData.pop("processedOutput")
            processedData.pop("correctOutput")

    modelsResults.append(partialResult)
    modelsResults.sort(key=lambda x: x["performance"]["accuracy"], reverse=True)

    simplifiedResults = copy.deepcopy(modelsResults)

    for res in simplifiedResults:
        simplifyProcess(res)

    detailedResults = modelsResults

    writeProcessedResult(resultsDirPath / "simplifiedResults.json", simplifiedResults)
    writeProcessedResult(resultsDirPath / "detailedResults.json", detailedResults)


def writeProcessedResult(resultsPath: str, results: dict):
    with open(resultsPath, "w") as f:
        json.dump(results, f, indent=3)


def printExecutionProgression(
    processedTexts: int,
    totalTexts: int,
    processedModels: int,
    totalModels: int,
    barLength: int = 40,
):
    if not silentOption:
        if totalModels == 1:
            processedPart = "*" * (barLength * processedTexts // totalTexts)
            emptyPart = " " * (barLength - (barLength * processedTexts // totalTexts))
            processedPercentage = f"{processedTexts / totalTexts * 100:.2f}%"
            print(
                f"Historiales procesados |{processedPart}{emptyPart}| {processedPercentage}",
                end="\r",
                flush=True,
            )
        else:
            modelsBarLenght = barLength + 20
            processedModelsPart = "*" * (
                modelsBarLenght * processedModels // totalModels
            )
            modelsEmptyPart = " " * (
                modelsBarLenght - (modelsBarLenght * processedModels // totalModels)
            )
            processedModelsPercentage = f"{processedModels / totalModels * 100:.2f}%"
            processedTextsPart = "*" * (barLength * processedTexts // totalTexts)
            textsEmptyPart = " " * (
                barLength - (barLength * processedTexts // totalTexts)
            )
            processedTextsPercentage = f"{processedTexts / totalTexts * 100:.2f}%"
            print(
                f"Modelos procesados |{processedModelsPart}{modelsEmptyPart}| {processedModelsPercentage} -- Historiales procesados |{processedTextsPart}{textsEmptyPart}| {processedTextsPercentage}{' '*5}",
                end="\r",
                flush=True,
            )

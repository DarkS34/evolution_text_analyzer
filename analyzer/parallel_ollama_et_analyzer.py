from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.runnables import RunnableLambda, RunnableParallel
from pydantic import BaseModel, Field
from analyzer.auxiliary_functions import printExecutionProgression
from analyzer.validator import validateResult
import time

PARALLEL_BATCH_SIZE = 5


def evolutionTextAnalysis(
    modelInfo: dict,
    medicalData: list,
    processedModels: int = 1,
    totalModels: int = 1,
):
    # Model
    model = OllamaLLM(
        model=modelInfo["modelName"],
        temperature=0,
        top_p=0.9,
        verbose=False,
        format="json",
        seed=2,
    )

    # Parser - Json Object
    class DeseaseAnalysis(BaseModel):
        icd_code: str = Field(
            title="Código CIE enfermedad",
            description="Código CIE de la enfermedad principal, basado en el historial del paciente",
        )
        principal_diagnostic: str = Field(
            title="Nombre enfermedad",
            description="Nombre de la enfermedad principal, basado en el historial del paciente",
        )

    parser = JsonOutputParser(pydantic_object=DeseaseAnalysis)

    # Prompt
    prompt = ChatPromptTemplate(
        messages=[
            (
                "system",
                "Eres un sistema médico especializado en el análisis de historiales médicos sobre enfermedades reumatológicas. Tu tarea es identificar y extraer el nombre y el código CIE correspondiente a la enfermedad principal mencionada en el siguiente historial.\n\n{instructions_format}",
            ),
            (
                "human",
                'El historial del paciente:\n\n""" historial\n{evolution_text}\n"""',
            ),
        ],
        input_variables=["evolution_text"],
        partial_variables={"instructions_format": parser.get_format_instructions()},
    )

    # Chain configuration
    chain = prompt | model | parser

    # Función para procesar un registro
    def processRecord(record: dict):
        try:
            processedChain = chain.invoke({"evolution_text": record["evolution_text"]})
            # Validar resultados
            return {
                "valid": validateResult(
                    processedChain["principal_diagnostic"],
                    record["principal_diagnostic"],
                ),
                "processedOutput": processedChain,
                "correctOutput": {
                    "principal_diagnostic": record["principal_diagnostic"],
                },
            }
        except Exception as e:
            return {
                "valid": False,
                "processedOutput": {
                    "icd_code": None,
                    "principal_diagnostic": None,
                    "error": str(e),
                },
                "correctOutput": {
                    "principal_diagnostic": record["principal_diagnostic"],
                },
            }

    # Parallel process execution (in batches)
    def executeInBatches(evolutionTexts: list):
        processedEvolutionTexts = {}
        startTime = time.time()

        for start in range(
            0,
            len(evolutionTexts),
            min(PARALLEL_BATCH_SIZE, len(evolutionTexts)),
        ):
            printExecutionProgression(
                modelInfo["modelName"],
                len(processedEvolutionTexts),
                len(evolutionTexts),
                processedModels,
                totalModels,
            )
            batch = evolutionTexts[start : start + PARALLEL_BATCH_SIZE]

            parallelRunner = RunnableParallel(
                {
                    str(batch[i]["ID"]): RunnableLambda(
                        lambda records, j=i: processRecord(records[j])
                    )
                    for i in range(len(batch))
                }
            )

            batchResults = parallelRunner.invoke(batch)
            processedEvolutionTexts.update(batchResults)
            printExecutionProgression(
                modelInfo["modelName"],
                len(processedEvolutionTexts),
                len(evolutionTexts),
                processedModels,
                totalModels,
            )
        endTime = time.time()

        accuracy = round(
            sum(1 for result in processedEvolutionTexts.values() if result["valid"])
            / len(processedEvolutionTexts)
            * 100,
            2,
        )
        errorOutputs = round(
            sum(
                1
                for result in processedEvolutionTexts.values()
                if result["valid"] == False and result["processedOutput"].get("error")
            )
            / len(processedEvolutionTexts)
            * 100,
            2,
        )

        return {
            "model": modelInfo,
            "performance": {
                "accuracy": accuracy,
                "incorrectOutputs": 100.00 - accuracy - errorOutputs,
                "errors": errorOutputs,
                "processingTime": round(endTime - startTime, 4),
                "numBatches": PARALLEL_BATCH_SIZE,
            },
            "evolutionTextsResults": processedEvolutionTexts,
        }

    return executeInBatches(medicalData)

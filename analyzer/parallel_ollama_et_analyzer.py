from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.runnables import RunnableLambda, RunnableParallel
from pydantic import BaseModel, Field
from analyzer.auxiliary_functions import printExecutionProgression
from analyzer.validator import validateResult
import time


def evolutionTextAnalysis(
    modelInfo: dict,
    medicalData: list,
    numBatches: int,
    reasoningMode: bool,
    processedModels: int = 1,
    totalModels: int = 1,
):
    isReasoningModel = modelInfo["modelName"].find("deepseek") != -1
    # Model
    model = OllamaLLM(
        model=modelInfo["modelName"],
        temperature=0,
        top_p=0.9,
        verbose=False,
        format="" if isReasoningModel else "json",
        seed=123,
    )
    numBatches = min(numBatches, len(medicalData))
    dateFormat = "%H:%M:%S %d-%m-%Y"

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
        if reasoningMode:
            reasoning: str = Field(
                title="Razonamiento",
                description="Breve razonamiento del sistema para identificar la enfermedad principal",
            )

    parser = JsonOutputParser(pydantic_object=DeseaseAnalysis)

    reasoning_prompt = (
        f"Además, debes proporcionar un breve razonamiento de 50 palabras sobre cómo llegaste a esa conclusión."
        if reasoningMode
        else ""
    )

    # Prompt
    prompt = ChatPromptTemplate(
        messages=[
            (
                "system",
                """Eres un sistema médico especializado en el análisis de historiales médicos sobre enfermedades reumatológicas. Tu tarea es identificar y extraer el nombre y el código CIE correspondiente a la enfermedad principal mencionada en el siguiente historial. {reasoning_prompt}
                Formato requerido para la respuesta:
                {instructions_format}""",
            ),
            (
                "human",
                'El historial del paciente:\n\n"""\n{evolution_text}\n"""',
            ),
        ],
        input_variables=["evolution_text"],
        partial_variables={
            "instructions_format": parser.get_format_instructions(),
            "reasoning_prompt": reasoning_prompt,
        },
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
                    **({"reasoning": None} if not reasoningMode else {}),
                    "error": str(e),
                },
                "correctOutput": {
                    "principal_diagnostic": record["principal_diagnostic"],
                },
            }

    # Parallel process execution (in batches)
    def executeInBatches(evolutionTexts: list):
        processedEvolutionTexts = {}
        startTime = {"startDuration": time.time(), "startDate": time.localtime()}

        # Helper function to calculate metrics
        def calculateMetrics(results):
            total = len(results)
            validCount = sum(1 for result in results.values() if result["valid"])
            errorCount = sum(
                1
                for result in results.values()
                if not result["valid"] and result["processedOutput"].get("error")
            )
            return {
                "accuracy": round(validCount / total * 100, 2),
                "incorrectOutputs": round(
                    100.00 - ((validCount + errorCount) / total * 100), 2
                ),
                "errors": round(errorCount / total * 100, 2),
            }

        # Process batches
        for start in range(0, len(evolutionTexts), numBatches):
            # Print progress once before processing
            printExecutionProgression(
                modelInfo["modelName"],
                len(processedEvolutionTexts),
                len(evolutionTexts),
                processedModels,
                totalModels,
            )

            batch = evolutionTexts[start : start + numBatches]

            # Simplified parallel runner creation
            parallelRunner = RunnableParallel(
                {
                    str(item["ID"]): RunnableLambda(lambda x, i=i: processRecord(x[i]))
                    for i, item in enumerate(batch)
                }
            )

            processedEvolutionTexts.update(parallelRunner.invoke(batch))

        endTime = {"endDuration": time.time(), "endDate": time.localtime()}
        # Calculate final metrics
        metrics = calculateMetrics(processedEvolutionTexts)

        return {
            "model": modelInfo,
            "performance": {
                "accuracy": metrics["accuracy"],
                "incorrectOutputs": metrics["incorrectOutputs"],
                "errors": metrics["errors"],
                "processingTime": {
                    "duration": round(
                        endTime["endDuration"] - startTime["startDuration"], 2
                    ),
                    "startDate": time.strftime(dateFormat, startTime["startDate"]),
                    "endDate": time.strftime(dateFormat, endTime["endDate"]),
                },
                "numBatches": numBatches,
                "totalRecordsProcessed": len(evolutionTexts),
            },
            "evolutionTextsResults": processedEvolutionTexts,
        }

    return executeInBatches(medicalData)

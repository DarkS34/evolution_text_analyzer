from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser
from langchain_ollama.llms import OllamaLLM
from langchain_core.runnables import RunnableLambda, RunnableParallel

from pydantic import BaseModel, Field

from analyzer.auxiliary_functions import printExecutionProgression
from analyzer._validator import validateResult
import time

# # You can add custom validation logic easily with Pydantic.
# @validator('likelihood_of_success')
# def check_score(cls, field):
#     if field >10:
#         raise ValueError("Badly formed Score")
#     return field
# https://mediately.co/_next/data/ca334f6100043fcbd2d00ec1242b3b547e1f226a/es/icd.json?classificationCode=


class DeseaseAnalysisBase(BaseModel):
    icd_code: str = Field(
        title="Código CIE enfermedad",
        description="Código CIE de la enfermedad principal, basado en el historial del paciente",
    )
    principal_diagnostic: str = Field(
        title="Nombre enfermedad",
        description="Nombre de la enfermedad principal, basado en el historial del paciente",
    )


class DeseaseAnalysisWithReasoning(DeseaseAnalysisBase):
    reasoning: str = Field(
        title="Razonamiento",
        description="Breve razonamiento del sistema para identificar la enfermedad principal",
    )


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
        num_ctx=8192,
        top_p=0.9,
        verbose=False,
        format="" if isReasoningModel else "json",
        seed=123,
    )
    
    numBatches = min(numBatches, len(medicalData))
    dateFormat = "%H:%M:%S %d-%m-%Y"

    # Parser
    parser = PydanticOutputParser(
        pydantic_object=(
            DeseaseAnalysisWithReasoning if reasoningMode else DeseaseAnalysisBase
        )
    )

    reasoning_prompt = (
        f"Además, debes proporcionar un breve razonamiento de 50 palabras sobre cómo llegaste a esa conclusión."
        if reasoningMode
        else ""
    )

    system_template = """Eres un sistema médico especializado en el análisis de historiales médicos sobre enfermedades reumatológicas.
    Tu tarea es identificar y extraer el nombre y el código CIE correspondiente a la enfermedad principal mencionada en el siguiente historial. {reasoning_prompt}
    
    Formato requerido para la respuesta:
    {instructions_format}"""

    # Prompt
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(
                'El historial del paciente:\n\n"""\n{evolution_text}\n"""'
            ),
        ],
    )

    prompt = prompt.partial(
        instructions_format=parser.get_format_instructions(),
        reasoning_prompt=reasoning_prompt,
    )
    # Chain configuration
    chain = prompt | model | parser

    # Función para procesar un registro
    def processRecord(record: dict):
        try:
            processedChain = chain.invoke({"evolution_text": record["evolution_text"]})
            processedChain = processedChain.model_dump()

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
            errorOutput = {
                "icd_code": None,
                "principal_diagnostic": None,
            }
            if reasoningMode:
                errorOutput["reasoning"] = None
            errorOutput["error"] = str(e)

            return {
                "valid": False,
                "processedOutput": errorOutput,
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

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel
from langchain_core.output_parsers import StrOutputParser

from ._custom_parser import CustomParser
from .auxiliary_functions import print_execution_progression


def evolution_text_analysis(
    modelName: str,
    prompts: dict[str],
    evolutionTexts: list[dict],
    numBatches: int,
    totalEvolutionTextsToProcess: int,
):
    expansionModel = OllamaLLM(
        model=modelName,
        temperature=0,
        verbose=False,
        seed=1,
    )

    diagnosticModel = OllamaLLM(
        model=modelName,
        temperature=0,
        verbose=False,
        format="json",
        seed=1,
    )

    totalEvolutionTextsToProcess = min(
        totalEvolutionTextsToProcess, len(evolutionTexts)
    )
    numBatches = min(numBatches, len(evolutionTexts))

    # Prompt

    # Prompt 1: Expansión de texto clínico
    expandPrompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "Eres un experto médico con amplia experiencia redactando y corrigiendo historiales clínicos en lenguaje técnico claro y preciso."
            ),
            HumanMessagePromptTemplate.from_template(
                """Tu tarea es tomar el siguiente historial evolutivo abreviado y:\n
                - Expandir abreviaciones médicas y tecnicismos.\n
                - Corregir errores ortográficos y gramaticales.\n
                - Mantener la **estructura y formato original** del texto.\n
                - No eliminar ni añadir información nueva.\n\n
                Texto original:\n\"\"\"{input_text}\"\"\"\n\n
                Devuelve únicamente el texto corregido y expandido, sin explicaciones."""
            ),
        ]
    )

    # Prompt 2: Diagnóstico
    diagnosisPrompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                prompts["system_prompt"]),
            HumanMessagePromptTemplate.from_template(prompts["human_prompt"]),
        ],
    )

    # Parser
    parser = CustomParser(modelName)

    # Expansion chain
    expandChain = expandPrompt | expansionModel | StrOutputParser()

    # Diagnosis chain
    diagnosisChain = diagnosisPrompt | diagnosticModel | parser

    analysisChain = (
        expandChain
        | RunnableLambda(lambda x: {"evolution_text": x})
        | diagnosisChain
    )

    # Función para procesar un registro
    def process_evolution_text(evolutionText: str):
        try:
            return analysisChain.invoke({"input_text": evolutionText})
        except Exception as e:
            return {
                "principal_diagnostic": None,
                "icd_code": None,
                "processing_error": str(e),
            }

    processedEvolutionTexts = {}

    # Process batches
    for start in range(0, totalEvolutionTextsToProcess, numBatches):
        # Print progress once before processing
        print_execution_progression(
            modelName,
            len(processedEvolutionTexts),
            totalEvolutionTextsToProcess,
        )

        batch = evolutionTexts[start: start + numBatches]

        # Simplified parallel runner creation
        parallelRunner = RunnableParallel(
            {
                str(item["id"]): RunnableLambda(
                    lambda x, i=i: process_evolution_text(
                        x[i]["evolution_text"])
                )
                for i, item in enumerate(batch)
            }
        )

        processedEvolutionTexts.update(parallelRunner.invoke(batch))

    return processedEvolutionTexts

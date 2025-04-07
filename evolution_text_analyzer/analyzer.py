from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel
from ._custom_parser import CustomParser

from .auxiliary_functions import print_execution_progression



def evolution_text_analysis(
    modelName: str,
    evolutionTexts: list,
    systemPrompt: str,
    numBatches: int,
    totalEvolutionTextsToProcess: int,
):
    model = OllamaLLM(
        model=modelName,
        temperature=0,
        num_ctx=8192,
        top_p=0.9,
        verbose=False,
        format="json",
        seed=123,
    )

    totalEvolutionTextsToProcess = min(
        totalEvolutionTextsToProcess, len(evolutionTexts)
    )
    numBatches = min(numBatches, len(evolutionTexts))

    # Prompt
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(systemPrompt),
            HumanMessagePromptTemplate.from_template(
                'El historial del paciente:\n\n"""\n{evolution_text}\n"""'
            ),
        ],
    )

    #Parser
    parser = CustomParser()

    # Chain configuration
    chain = prompt | model | parser

    # Función para procesar un registro
    def process_evolution_text(evolutionText: dict):
        try:
            processedChain: BaseModel = chain.invoke({"evolution_text": evolutionText})
            return processedChain.model_dump()
        except Exception as e:
            return {
                "principla_diagnostic": None,
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

        batch = evolutionTexts[start : start + numBatches]

        # Simplified parallel runner creation
        parallelRunner = RunnableParallel(
            {
                str(item["id"]): RunnableLambda(
                    lambda x, i=i: process_evolution_text(x[i]["evolution_text"])
                )
                for i, item in enumerate(batch)
            }
        )

        processedEvolutionTexts.update(parallelRunner.invoke(batch))

    return processedEvolutionTexts

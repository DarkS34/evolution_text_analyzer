from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable, RunnableParallel, RunnableLambda
from pydantic import BaseModel, Field
from transformers import pipeline
import torch

from scripts.auxiliary_functions import printExecutionProgression


class EvolutionTextDiagnostic(BaseModel):
    principal_diagnostic: str = Field(
        title="Nombre enfermedad",
        description="Nombre de la enfermedad principal, basado en el historial del paciente",
        min_length=3,
        max_length=100,
    )
    icd_code: str = Field(
        title="Código CIE-10 enfermedad",
        description="Código CIE-10 de la enfermedad principal, basado en el historial del paciente. Debe seguir el formato estándar.",
        examples=["M06.4", "M06.", "M05.0"],
        pattern=r"^[A-Z]\d{2}(\.\d{1,3})?$",
    )


class HuggingFaceLLM(Runnable):
    def __init__(self, model_name: str):
        self.pipe = pipeline(
            task="text-generation",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,  # Usa GPU si está disponible
            model_kwargs={"torch_dtype": torch.float16},
            trust_remote_code=True,
        )

    def invoke(self, input, config=None):
        try:
            # Convertir el input (ChatPromptTemplate renderizado) a string
            if hasattr(input, "messages"):
                prompt = "\n".join([msg.content for msg in input.messages])
            else:
                prompt = str(input)

            # Generar respuesta del modelo
            response = self.pipe(
                prompt,
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                return_full_text=False,
            )
            return response[0]["generated_text"]
        except Exception as e:
            raise Exception(f"Error en la generación del texto: {str(e)}")


def evolutionTextAnalysis(
    modelName: str,
    evolutionTexts: list,
    numBatches: int,
    systemPrompt: str,
    totalEvolutionTextsToProcess: int,
):
    # Model
    model = HuggingFaceLLM(modelName)

    totalEvolutionTextsToProcess = min(
        totalEvolutionTextsToProcess, len(evolutionTexts)
    )
    numBatches = min(numBatches, len(evolutionTexts))

    # Parser
    parser = PydanticOutputParser(pydantic_object=EvolutionTextDiagnostic)

    # Prompt
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(systemPrompt),
            HumanMessagePromptTemplate.from_template(
                'El historial del paciente:\n\n"""\n{evolution_text}\n"""'
            ),
        ],
    )

    # Chain configuration
    chain = prompt | model | parser

    # Función para procesar un registro
    def processEvolutionText(evolutionText: dict):
        try:
            processedChain: BaseModel = chain.invoke({"evolution_text": evolutionText})
            return processedChain.model_dump()
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
        printExecutionProgression(
            modelName,
            len(processedEvolutionTexts),
            totalEvolutionTextsToProcess,
        )

        batch = evolutionTexts[start : start + numBatches]

        # Simplified parallel runner creation
        parallelRunner = RunnableParallel(
            {
                str(item["id"]): RunnableLambda(
                    lambda x, i=i: processEvolutionText(x[i]["evolution_text"])
                )
                for i, item in enumerate(batch)
            }
        )

        processedEvolutionTexts.update(parallelRunner.invoke(batch))

    return processedEvolutionTexts

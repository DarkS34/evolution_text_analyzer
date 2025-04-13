from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

from ._custom_parser import CustomParser
from .auxiliary_functions import print_execution_progression


def evolution_text_analysis(
    model_name: str,
    prompts: dict[str],
    evolution_texts: list[dict],
    chroma_db,
    expansion_mode: bool,
    num_batches: int,
    total_evolution_texts_to_process: int,
):


    model = OllamaLLM(
        model=model_name,
        temperature=0
    )

    total_evolution_texts_to_process = min(
        total_evolution_texts_to_process, len(evolution_texts)
    )
    num_batches = min(num_batches, len(evolution_texts))

    # Prompt
    expand_prompt = PromptTemplate.from_template(prompts["expand_diagnostic_prompt"])

    diagnosis_prompt = PromptTemplate.from_template(prompts["diagnostic_prompt"])

    # Parser
    parser = CustomParser(chroma_db, model, prompts["rag_prompt"])

    # Expansion chain
    expand_chain = expand_prompt | model | StrOutputParser()

    # Diagnosis chain
    diagnosis_chain = diagnosis_prompt | model | parser

    full_chain = (
        expand_chain
        | RunnableLambda(lambda x: {"evolution_text": x})
        | diagnosis_chain
    )
    
    # Función para procesar un registro
    def process_evolution_text(evolution_text: str):
        try:
            analysis_chain = full_chain if expansion_mode else diagnosis_chain
            return analysis_chain.invoke({"evolution_text": evolution_text})
        except Exception as e:
            return {
                "principal_diagnostic": None,
                "icd_code": None,
                "processing_error": str(e),
            }

    processed_evolution_texts = {}

    # Process batches
    for start in range(0, total_evolution_texts_to_process, num_batches):
        # Print progress once before processing
        print_execution_progression(
            model_name,
            len(processed_evolution_texts),
            total_evolution_texts_to_process,
        )

        batch = evolution_texts[start: start + num_batches]

        # Simplified parallel runner creation
        parallel_runner = RunnableParallel(
            {
                str(item["id"]): RunnableLambda(
                    lambda x, i=i: process_evolution_text(
                        x[i]["evolution_text"]
                    )
                )
                for i, item in enumerate(batch)
            }
        )

        processed_evolution_texts.update(parallel_runner.invoke(batch))

    return processed_evolution_texts
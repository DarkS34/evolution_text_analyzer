from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_ollama.llms import OllamaLLM

from ._custom_parser import CustomParser
from .auxiliary_functions import print_execution_progression


MAX_EVOLUTION_TEXT_TOKENS = 2560
DEFAULT_CONTEXT_WINDOW = 3072


class EvolutionTextSummarizer:
    """Handles text summarization for long medical evolution texts."""

    def __init__(self, model: OllamaLLM,
                 max_tokens: int = MAX_EVOLUTION_TEXT_TOKENS,
                 overlap_ratio: float = 0.1):

        self.model = model
        self.max_tokens = max_tokens
        self.overlap_ratio = overlap_ratio
        self.summary_prompt = PromptTemplate.from_template("""
        Realiza un resumen conciso del siguiente texto clínico, manteniendo la continuidad temporal y la información crítica para el diagnóstico.
        
        Texto a resumir:
        {chunk}
        
        Resumen:
        """)

        self.text_splitter = CharacterTextSplitter(
            chunk_size=int(MAX_EVOLUTION_TEXT_TOKENS * 0.6),
            chunk_overlap=int(MAX_EVOLUTION_TEXT_TOKENS * overlap_ratio),
            separator=".",
            length_function=model.get_num_tokens,
        )

        self.summary_chain = self.summary_prompt | self.model | StrOutputParser()

    def needs_summarization(self, text: str) -> bool:
        """Check if text needs summarization based on token count."""
        tokens_per_word = 4
        word_chunk = int(1024 / tokens_per_word)

        words = text.split()
        total_words = len(words)
        num_blocks = (total_words + (word_chunk-1)) // word_chunk
        total_token_count = 0

        for i in range(num_blocks):
            start_idx = i * word_chunk
            end_idx = min((i + 1) * word_chunk, total_words)

            block_text = ' '.join(words[start_idx:end_idx])
            
            partial_token_count = 0
            try:
                partial_token_count = self.model.get_num_tokens(block_text)
                # Asegurarse de que sea un entero
                if isinstance(partial_token_count, list):
                    # Si es una lista, tomar el primer elemento o la suma
                    partial_token_count = partial_token_count[0] if partial_token_count else 0
                total_token_count += int(partial_token_count)
            except Exception as e:
                print(f"Error al calcular tokens: {e}")
                partial_token_count = int((end_idx - start_idx) * tokens_per_word)
                total_token_count += partial_token_count
        
        return total_token_count > MAX_EVOLUTION_TEXT_TOKENS

    def summarize_text(self, text: str) -> str:
        """Summarize long text by chunking and recursive summarization."""
        
        chunks = self.text_splitter.split_text(text)
        summaries = []
        for i, chunk in enumerate(chunks, 1):
            try:
                summary = self.summary_chain.invoke({"chunk": chunk})
                summaries.append(summary)
            except Exception as e:
                raise ValueError(e)

        combined_summary = " ".join(summaries)

        if self.needs_summarization(combined_summary):
            return self.summarize_text(combined_summary)

        return combined_summary


def evolution_text_analysis(
    model_name: str,
    prompts: dict[str],
    norm_mode: bool,
    model_context_window: int,
    evolution_texts: list[dict],
    num_batches: int,
    total_evolution_texts_to_process: int,
):
    model = OllamaLLM(
        model=model_name,
        temperature=0,
        num_ctx=DEFAULT_CONTEXT_WINDOW if model_context_window > DEFAULT_CONTEXT_WINDOW else model_context_window
    )
    summarizer = EvolutionTextSummarizer(model)

    diagnosis_prompt = PromptTemplate.from_template(
        prompts["gen_diagnostic_prompt"])

    parser = CustomParser(model, norm_mode, prompts["gen_icd_code_prompt"])

    # Diagnosis chain
    diagnosis_chain = diagnosis_prompt | model | parser

    # Función para procesar un registro

    def process_evolution_text(evolution_text: str):
        try:
            processed_text = evolution_text
            if summarizer.needs_summarization(evolution_text):
                processed_text = summarizer.summarize_text(evolution_text)
                print(processed_text)
            return diagnosis_chain.invoke({"evolution_text": processed_text})
        except Exception as e:
            return {
                "principal_diagnostic": None,
                "icd_code": None,
                "processing_error": str(e),
            }

    processed_evolution_texts = dict()

    num_batches = min(num_batches, len(evolution_texts))

    total_evolution_texts_to_process = min(
        total_evolution_texts_to_process, len(evolution_texts))

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

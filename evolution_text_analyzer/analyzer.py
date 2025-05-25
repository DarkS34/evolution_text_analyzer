from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_ollama.llms import OllamaLLM

from ._custom_output_parser import CustomOutputParser
from .auxiliary_functions import CustomStringOutputParser, print_execution_progression
from .data_models import SummarizerConfig


class EvolutionTextSummarizer:
    def __init__(
        self,
        model: any,
        summary_prompt: str,
        context_window: int,
    ):
        self.model = model
        self.context_window = context_window
        self.config = SummarizerConfig()

        # Determine the token counting function
        self.token_counter = getattr(model, 'get_num_tokens', None)
        if not self.token_counter:
            # Fallback to simple estimation if no token counter is available
            self.token_counter = lambda text: int(
                len(text.split()) * self.config.tokens_per_word)

        # Configure the text splitter
        self.text_splitter = CharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separator=self.config.separator,
            length_function=self.token_counter,
        )

        # Configure the summarization chain
        self.summary_chain = PromptTemplate.from_template(
            summary_prompt) | self.model | CustomStringOutputParser()

    def needs_summarization(self, text: str) -> bool:
        try:
            token_count = self.token_counter(text)
            return token_count > (self.context_window - self.config.safety_margin)
        except Exception as _:
            words = len(text.split())
            estimated_tokens = int(words * self.config.tokens_per_word)
            return estimated_tokens > (self.context_window - self.config.safety_margin)

    def summarize_text(self, text: str, max_recursion: int = 3, current_depth: int = 0) -> str:
        if current_depth >= max_recursion:
            return text

        if not self.needs_summarization(text):
            return text

        try:
            chunks = self.text_splitter.split_text(text)

            summaries: list[str] = []
            for i, chunk in enumerate(chunks, 1):
                try:
                    summary = self.summary_chain.invoke({"chunk": chunk})
                    summaries.append(summary)
                except Exception as _:
                    # Include a simple fallback if summarization failed
                    summaries.append(chunk)

            combined_summary = " ".join(summaries)

            # Check if more summarization is needed
            if self.needs_summarization(combined_summary):
                return self.summarize_text(
                    combined_summary,
                    max_recursion=max_recursion,
                    current_depth=current_depth + 1
                )

            return combined_summary

        except Exception as e:
            raise ValueError(f"Error summarizing text: {str(e)}")


def evolution_text_analysis(
    model_name: str,
    prompts: dict[str],
    context_window_tokens: int,
    evolution_texts: list[dict],
    num_batches: int,
    total_evolution_texts_to_process: int,
    norm_mode: bool,
    test_mode: bool = False,
):
    model = OllamaLLM(
        model=model_name,
        temperature=0,
        num_ctx=context_window_tokens
    )

    summarizer = EvolutionTextSummarizer(
        model, prompts["gen_summary_prompt"], context_window_tokens)

    diagnosis_prompt = PromptTemplate.from_template(
        prompts["gen_diagnostic_prompt"])

    parser = CustomOutputParser(model, norm_mode, prompts["gen_icd_code_prompt"])

    # Diagnosis chain
    diagnosis_chain = diagnosis_prompt | model | parser

    # Función para procesar un registro

    def process_evolution_text(evolution_text: str):
        try:
            processed_text = evolution_text
            result = dict()

            if summarizer.needs_summarization(evolution_text):
                processed_text = summarizer.summarize_text(evolution_text)
                if test_mode:
                    result.update({"summarized": True})
            else:
                if test_mode:
                    result.update({"summarized": False})

            result.update(diagnosis_chain.invoke(
                {"evolution_text": processed_text}))

            return result
        except Exception as e:
            return {
                "principal_diagnostic": None,
                "icd_code": None,
                "summarized": False,
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
            test_mode
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

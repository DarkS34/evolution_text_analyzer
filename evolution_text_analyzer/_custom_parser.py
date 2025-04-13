import re

import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.outputs import Generation
from langchain_ollama.llms import OllamaLLM

THRESHOLD: float = 0.90
FW_RATION: int = 80


class DiagnosticNormalizerRAG:
    def __init__(self, vector_store: Chroma, llm: OllamaLLM, prompt: str, csv_path: str):
        self.prompt = prompt
        self.df = pd.read_csv(csv_path)
        self.df['text'] = self.df['principal_diagnostic'] + \
            ':' + self.df['icd_code']
        self.vector_store = vector_store
        self.threshold = THRESHOLD
        self.llm = llm

    def normalize_diagnostic(self, principal_diagnostic: str) -> dict[str, str]:
        normalized_principal_diagnostic = principal_diagnostic.lower().strip()

        similar_diagnostics = self.vector_store.similarity_search_with_score(
            normalized_principal_diagnostic, k=5)

        if (similar_diagnostics[-1][1] >= self.threshold):
            return {
                "icd_code": similar_diagnostics[-1][0].metadata.get('icd_code', ''),
                "principal_diagnostic": similar_diagnostics[-1][0].metadata.get('principal_diagnostic', '')
            }

        similar_diagnostics_text = "\n".join(
            f"- {diag[0].metadata.get('principal_diagnostic', '')}" for diag in similar_diagnostics)

        prompt = PromptTemplate.from_template(self.prompt)

        similarity_chain = prompt | self.llm | StrOutputParser()

        try:
            result = similarity_chain.invoke({
                "principal_diagnostic": normalized_principal_diagnostic,
                "similar_diagnostics": similar_diagnostics_text,
            })

            for diag in similar_diagnostics:
                if result.lower().strip() == diag[0].metadata["principal_diagnostic"].lower():
                    return {
                        "icd_code": diag[0].metadata["icd_code"],
                        "principal_diagnostic": diag[0].metadata["principal_diagnostic"]
                    }

        except Exception as e:
            raise Exception(
                f"Error al invocar el modelo para normalizar el diagnóstico: {e}")

        import rapidfuzz.fuzz as fuzz
        best_match = None
        best_score = 0

        for item in similar_diagnostics:
            score = fuzz.ratio(result.lower(), item[0].metadata.get(
                'principal_diagnostic', '').lower())
            if score > FW_RATION and score > best_score:
                best_score = score
                best_match = item

        if best_match:
            return {
                "icd_code": best_match[0].metadata["icd_code"],
                "principal_diagnostic": best_match[0].metadata["principal_diagnostic"]
            }

        return {
            "icd_code": "N/A",
            "principal_diagnostic": "N/A"
        }


class CustomParser(BaseOutputParser):
    def __init__(self, chroma_db: Chroma | None, llm: OllamaLLM, prompt: str, csv_path: str = "icd_dataset.csv", **kwargs):
        super().__init__()
        self._llm = llm
        if chroma_db is None:
            self._rag_system = None
        else:
            self._rag_system = DiagnosticNormalizerRAG(
                chroma_db, llm, prompt, csv_path)

    def generate_icd_code(self, llm: OllamaLLM, principal_diagnostic: str) -> dict:
        prompt = PromptTemplate.from_template(
            """Eres un experto en medicina y diagnóstico.
            Tu tarea es encontrar el código CIE-10 para una enfermedad específica.
            ¿Cuál es el código CIE-10 para la enfermedad '{principal_diagnostic}'? 
            Sólamente devuelve el código CIE-10 sin ningún otro texto. 
            Por ejemplo: M06.4, M06.33, M05.0"""
        )

        icd_chain = prompt | llm | StrOutputParser()

        try:
            icd_code = icd_chain.invoke(
                {"principal_diagnostic": principal_diagnostic})
            icd_code_re = re.search(r'([A-Z]\d+\.\d+)', icd_code)
            icd_code_re = re.sub(r"[\n\r\s\.]", "", icd_code_re.group(1))
            
            return {
                "icd_code": icd_code,
                "principal_diagnostic": principal_diagnostic
            }
        except Exception as e:
            raise Exception(
                f"Error al invocar el modelo para normalizar el diagnóstico: {e}")

    def parse(self, result: list[Generation], *, partial: bool = False) -> dict:
        try:
            if isinstance(result, list) and all(isinstance(x, Generation) for x in result):
                principal_diagnostic = result[0].text.strip()
            elif isinstance(result, str):
                principal_diagnostic = result.strip()
            else:
                principal_diagnostic = str(result).strip()

            if self._rag_system is None:
                generated = self.generate_icd_code(
                    self._llm, principal_diagnostic)
                parsed = generated
            else:
                normalized = self._rag_system.normalize_diagnostic(
                    principal_diagnostic=principal_diagnostic)
                parsed = normalized

            if (len(parsed["icd_code"]) > 3):
                parsed["icd_code"] = parsed["icd_code"][:3] + \
                    "." + parsed["icd_code"][3:]

            return parsed
        except Exception as e:
            raise Exception(
                f"Error al intentar parsear el resultado: {e}")

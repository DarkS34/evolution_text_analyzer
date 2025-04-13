import re
from typing import Dict, Optional

import pandas as pd
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_core.outputs import Generation
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel, Field, PrivateAttr


class EvolutionTextDiagnosticSchema(BaseModel):
    principal_diagnostic: str = Field(
        title="Nombre enfermedad pricipal",
        description="Nombre de la enfermedad principal, basado en el historial del paciente",
    )
    # icd_code: str = Field(
    #     title="Código CIE-10 enfermedad",
    #     description="Código CIE-10 de la enfermedad principal, basado en el historial del paciente. Debe seguir el formato estándar.",
    #     examples=["M06.4", "M06.33", "M05.0"]
    # )


class DiagnosticNormalizerRAG:
    def __init__(self, csv_path: str, llm: OllamaLLM, vector_store: Chroma, prompt: str):
        self.prompt = prompt
        self.df = pd.read_csv(csv_path)
        self.df_search = self.df.copy()
        self.df_search['text'] = self.df_search['principal_diagnostic'] + \
            ':' + self.df_search['icd_code']
        self.vector_store = vector_store
        self.llm = llm

    def normalize_diagnostic(self, principal_diagnostic: str) -> Dict[str, str]:
        normalized_principal_diagnostic = principal_diagnostic.lower().strip()

        diagnostic_matches = self.df[
            self.df['principal_diagnostic'].str.contains(
                normalized_principal_diagnostic, case=False, na=False)
        ]

        if not diagnostic_matches.empty:
            match = diagnostic_matches.iloc[0]
            return {
                "icd_code": match["icd_code"],
                "principal_diagnostic": match["principal_diagnostic"]
            }

        similar_diagnostics = self.vector_store.similarity_search(
            normalized_principal_diagnostic, k=5)

        similar_diagnostics_with_icd = [
            {
                "icd_code": diag.metadata.get('icd_code', ''),
                "principal_diagnostic": diag.metadata.get('principal_diagnostic', '')
            }
            for diag in similar_diagnostics
        ]

        similar_diagnostics_text = "\n".join(
            f"- {desc.metadata.get('principal_diagnostic', '')}" for desc in similar_diagnostics)

        prompt = PromptTemplate.from_template(self.prompt)

        similarity_chain = prompt | self.llm

        try:
            result = similarity_chain.invoke({
                "principal_diagnostic": normalized_principal_diagnostic,
                "similar_diagnostics": similar_diagnostics_text,
            })

            for item in similar_diagnostics_with_icd:
                if result == item["principal_diagnostic"]:
                    return {
                        "icd_code": item["icd_code"],
                        "principal_diagnostic": item["principal_diagnostic"]
                    }

        except Exception as e:
            raise Exception(
                f"Error al invocar el modelo para normalizar el diagnóstico: {e}")

        return {
            "icd_code": similar_diagnostics_with_icd[0]["icd_code"],
            "principal_diagnostic": similar_diagnostics_with_icd[0]["principal_diagnostic"]
        }


class CustomParser(PydanticOutputParser):
    _rag_system: DiagnosticNormalizerRAG | None = PrivateAttr()
    _llm: OllamaLLM | None = PrivateAttr()

    def __init__(self, chroma_db: Chroma | None, llm: OllamaLLM, prompt: str, csv_path: str = "icd_dataset.csv", **kwargs):
        kwargs.setdefault("pydantic_object", EvolutionTextDiagnosticSchema)
        super().__init__(**kwargs)
        if chroma_db is None:
            self._rag_system = None
            self._llm = llm
        else:
            self._rag_system = DiagnosticNormalizerRAG(
                csv_path, llm, chroma_db, prompt)

    def include_icd_code(self, llm: OllamaLLM, principal_diagnostic: str) -> dict:
        prompt = PromptTemplate.from_template(
            "¿Cuál es el código CIE-10 para la enfermedad '{principal_diagnostic}'? "
            "Sólamente devuelve el código CIE-10 sin ningún otro texto. "
            "Por ejemplo: 'M06.4', 'M06.33', 'M05.0'"
        )

        icd_chain = prompt | llm

        try:
            icd_code = icd_chain.invoke(
                {"principal_diagnostic": principal_diagnostic})
            icd_code = re.sub(r"[\n\r\s\.]", "", icd_code.strip())
            return {
                "icd_code": icd_code,
                "principal_diagnostic": principal_diagnostic
            }
        except Exception as e:
            raise Exception(
                f"Error al invocar el modelo para normalizar el diagnóstico: {e}")

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Optional[EvolutionTextDiagnosticSchema]:
        try:
            parsed: BaseModel = super().parse_result(result)
            parsed = parsed.model_dump()

            if self._rag_system is None:
                generated = self.include_icd_code(
                    self._llm, parsed["principal_diagnostic"])
                parsed = generated
            else:
                normalized = self._rag_system.normalize_diagnostic(
                    principal_diagnostic=parsed["principal_diagnostic"])
                parsed = normalized

            if (len(parsed["icd_code"]) > 3):
                parsed["icd_code"] = parsed["icd_code"][:3] + \
                    "." + parsed["icd_code"][3:]

            return parsed
        except Exception:
            if partial:
                return None
            raise

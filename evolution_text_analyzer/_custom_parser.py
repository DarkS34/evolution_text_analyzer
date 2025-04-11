from typing import Dict, Optional

from langchain_chroma import Chroma
import pandas as pd
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.outputs import Generation
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel, Field, PrivateAttr


class _EvolutionTextDiagnosticSchema(BaseModel):
    principal_diagnostic: str = Field(
        title="Nombre enfermedad pricipal",
        description="Nombre de la enfermedad principal, basado en el historial del paciente",
        min_length=3,
        max_length=100,
    )
    icd_code: str = Field(
        title="Código CIE-10 enfermedad",
        description="Código CIE-10 de la enfermedad principal, basado en el historial del paciente. Debe seguir el formato estándar.",
        examples=["M06.4", "M06.33", "M05.0"],
        pattern=r"^[A-Z0-9]{1,3}(\.\d{1,5})?$",
    )


class diagnosticNormalizerRAG:
    def __init__(self, csv_path: str, llm: OllamaLLM, vector_store: Chroma):
        self.df = pd.read_csv(csv_path)
        self.df_search = self.df.copy()
        self.df_search['text'] = self.df_search['code'] + \
            ': ' + self.df_search['description']
        self.vectorstore = vector_store
        self.llm = llm

    def normalize_diagnostic(self, principal_diagnostic: str, icd_code: str) -> Dict[str, str]:
        normalized_code = icd_code.strip().replace(".", "").replace("-", "").upper()
        normalized_diagnostic = principal_diagnostic.strip()

        description_matches = self.df[
            self.df['description'].str.contains(
                normalized_diagnostic, case=False, na=False)
        ]

        code_matches = self.df[
            self.df['code'].str.contains(normalized_code, case=False, na=False)
        ]

        common_matches = description_matches.merge(code_matches, how='inner')

        exact_match = common_matches[
            (common_matches['description'].str.lower() == normalized_diagnostic.lower()) &
            (common_matches['code'].str.upper() == normalized_code.upper())
        ]

        if not exact_match.empty:
            match = exact_match.iloc[0]
            return {
                "icd_code": match["code"],
                "principal_diagnostic": match["description"]
            }

        query = f"{normalized_code}: {normalized_diagnostic}"
        similar_docs = self.vectorstore.similarity_search(query, k=5)

        if not similar_docs:
            return {
                "icd_code": normalized_code,
                "principal_diagnostic": normalized_diagnostic
            }

        similar_codes = "\n".join(
            f"- {doc.page_content}" for doc in similar_docs)

        # 4. Crear prompt
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    "Eres un experto médico con amplia experiencia en la identificación de códigos CIE-10 y normalización de diagnósticos."
                ),
                HumanMessagePromptTemplate.from_template(
                    """
                    Necesito normalizar el siguiente diagnóstico médico con su código CIE-10 correspondiente:

                    El diagnóstico original es:
                    {icd_code}: {principal_diagnostic}

                    Basado en la siguiente lista de códigos CIE-10 similares:
                    {similar_codes}

                    Tu objetivo es seleccionar el código CIE-10 correcto y su descripción correspondiente normalizados que mejor coincidan con el diagnóstico.

                    La respuesta debe estar formateada como un objeto JSON que cumpla con este esquema:
                    ```
                    {{
                        "principal_diagnostic": "string (3-100 caracteres)",
                        "icd_code": "string con formato CIE-10, ej: M06.4"
                    }}
                    ```
                    No agregues explicaciones adicionales, solo devuelve el JSON.
                    """
                ),
            ]
        )

        parser = CustomParser()
        similarity_chain = prompt | self.llm | parser

        try:
            result = similarity_chain.invoke({
                "similar_codes": similar_codes,
                "principal_diagnostic": normalized_diagnostic,
                "icd_code": normalized_code
            })

            if hasattr(result, "model_dump"):
                return result.model_dump()

            return {
                "icd_code": normalized_code,
                "principal_diagnostic": normalized_diagnostic
            }

        except Exception:
            return {
                "icd_code": normalized_code,
                "principal_diagnostic": normalized_diagnostic
            }


class CustomParser(PydanticOutputParser):
    _llm: Optional[OllamaLLM] = PrivateAttr()
    _csv_path: str = PrivateAttr()
    _vector_store: Optional[Chroma] = PrivateAttr()

    def __init__(self, chromaDB=None, llm: OllamaLLM = None, csv_path: str = "icd_dataset.csv", **kwargs):
        kwargs.setdefault("pydantic_object", _EvolutionTextDiagnosticSchema)
        super().__init__(**kwargs)
        self._vector_store = chromaDB
        self._llm = llm
        self._csv_path = csv_path

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Optional[_EvolutionTextDiagnosticSchema]:
        try:
            parsed: BaseModel = super().parse_result(result)
            parsed = parsed.model_dump()
            
            # Solo para la primera iteración
            if self._vector_store is not None:
                rag_system = diagnosticNormalizerRAG(
                    csv_path=self._csv_path,
                    llm=self._llm,
                    vector_store=self._vector_store
                )

                normalized = rag_system.normalize_diagnostic(
                    principal_diagnostic=parsed["principal_diagnostic"], icd_code=parsed["icd_code"])

                parsed = normalized

            return parsed
        except Exception:
            if partial:
                return None
            raise

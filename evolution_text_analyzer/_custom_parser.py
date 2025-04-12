from typing import Dict, Optional

from langchain_chroma import Chroma
import pandas as pd
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.outputs import Generation
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel, Field, PrivateAttr

class EvolutionTextDiagnosticSchema(BaseModel):
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

class DiagnosticNormalizerRAG:
    def __init__(self, csv_path: str, llm: OllamaLLM, vector_store: Chroma):
        self.df = pd.read_csv(csv_path)
        self.df_search = self.df.copy()
        self.df_search['text'] = self.df_search['code'] + ': ' + self.df_search['description']
        self.vector_store = vector_store
        self.llm = llm

    def normalize_diagnostic(self, normalized_diagnostic: str, normalized_code: str) -> Dict[str, str]:
        description_matches = self.df[
            self.df['description'].str.contains(normalized_diagnostic, case=False, na=False)
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
        similar_docs = self.vector_store.similarity_search(query, k=5)

        if not similar_docs:
            return {
                "icd_code": normalized_code,
                "principal_diagnostic": normalized_diagnostic
            }

        similar_codes = "\n".join(
            f"- {doc.page_content}" for doc in similar_docs)

        prompt = PromptTemplate.from_template(
            """
            Eres un experto médico con amplia experiencia en la clasificación y normalización de diagnósticos utilizando códigos CIE-10.

            Tu tarea es analizar un diagnóstico médico original y seleccionar el código CIE-10 más adecuado a partir de una lista de códigos similares proporcionada.

            ### Objetivo:
            Seleccionar el código CIE-10 correcto que mejor represente el diagnóstico original y devolver su descripción normalizada.

            ### Instrucciones:
            1. Lee el diagnóstico médico original.
            2. Examina la lista de códigos CIE-10 similares proporcionada.
            3. Elige el código CIE-10 que más se ajuste semánticamente al diagnóstico original.
            4. Normaliza la descripción del diagnóstico para que sea clara, concisa y esté alineada con la terminología CIE-10.
            5. Devuelve únicamente un objeto JSON con el siguiente formato:

            ```json
            {{
            "principal_diagnostic": "string (entre 3 y 100 caracteres)",
            "icd_code": "string con formato CIE-10, por ejemplo: M06.4"
            }}
            ```

            ### Diagnóstico original:
            {icd_code}: {principal_diagnostic}

            ### Lista de códigos CIE-10 similares:
            {similar_codes}

            No expliques tu razonamiento ni añadas ningún texto adicional. Devuelve solo el JSON.
            """
        )

        parser = CustomParser()
        similarity_chain = prompt | self.llm | parser

        try:
            result = similarity_chain.invoke({
                "similar_codes": similar_codes,
                "principal_diagnostic": normalized_diagnostic,
                "icd_code": normalized_code
            })

            return result if isinstance(result, dict) else {
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

    def __init__(self, chroma_db=None, llm: OllamaLLM = None, csv_path: str = "icd_dataset.csv", **kwargs):
        kwargs.setdefault("pydantic_object", EvolutionTextDiagnosticSchema)
        super().__init__(**kwargs)
        self._vector_store = chroma_db
        self._llm = llm
        self._csv_path = csv_path

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Optional[EvolutionTextDiagnosticSchema]:
        try:
            parsed: BaseModel = super().parse_result(result)
            parsed = parsed.model_dump()
            
            normalized_code = parsed["icd_code"].strip().replace(".", "").replace("-", "").upper()
            normalized_diagnostic = parsed["principal_diagnostic"].strip()

            if self._vector_store is not None:
                rag_system = DiagnosticNormalizerRAG(
                    csv_path=self._csv_path,
                    llm=self._llm,
                    vector_store=self._vector_store
                )

                normalized = rag_system.normalize_diagnostic(
                    normalized_diagnostic=normalized_diagnostic, normalized_code=normalized_code)

                parsed = normalized

            if (len(normalized_code) > 3):
                normalized_code = normalized_code[:3] + "." + normalized_code[3:]
            
            return parsed
        except Exception:
            if partial:
                return None
            raise

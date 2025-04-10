from typing import Dict, Optional

import pandas as pd
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_ollama import OllamaEmbeddings
from langchain_core.outputs import Generation
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel, Field, PrivateAttr

from evolution_text_analyzer.auxiliary_functions import load_chroma_db


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
    def __init__(self, csv_path: str, model_name: str):
        """
        Inicializa el sistema RAG para códigos CIE-10

        Args:
            csv_path: Ruta al archivo CSV con los códigos CIE-10
            model_name: Nombre del modelo en Ollama
        """
        self.df = pd.read_csv(csv_path)
        self.df_search = self.df.copy()

        self.df_search['text'] = self.df_search['code'] + \
            ': ' + self.df_search['description']

        # Comprobar que existe el modelo nomic-embed-text:latest

        self.embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        self.vectorstore = load_chroma_db()
        self.llm = OllamaLLM(model=model_name)

    def normalize_diagnostic(self, principal_diagnostic: str, icd_code: str) -> Dict[str, str]:
        # 1. Preprocesamiento de entrada
        normalized_code = icd_code.strip().replace(".", "").replace("-", "").upper()
        normalized_diagnostic = principal_diagnostic.strip()

        # 2. Intentar coincidencia exacta en el DataFrame
        exact_match = self.df[
            self.df['code'].str.contains(normalized_code, case=False, na=False) |
            self.df['description'].str.contains(
                normalized_diagnostic, case=False, na=False)
        ]

        if not exact_match.empty:
            match = exact_match.iloc[0]
            return {
                "icd_code": normalized_code,
                "principal_diagnostic": match["description"]
            }

        # 3. Búsqueda vectorial con RAG
        query = f"{normalized_code}: {normalized_diagnostic}"
        similar_docs = self.vectorstore.similarity_search(query, k=5)
        
        if not similar_docs:
            return {
                "icd_code": normalized_code,
                "principal_diagnostic": normalized_diagnostic
            }

        similar_codes = "\n".join(
            f"- {doc.page_content}" for doc in similar_docs)
        
        print(similar_codes)

        # 4. Crear prompt
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    "Eres un experto médico con amplia experiencia en la identificación de códigos CIE-10 y normalización de diagnósticos."
                ),
                HumanMessagePromptTemplate.from_template(
                    """
                    Necesito normalizar el siguiente diagnóstico médico con su código CIE-10 correspondiente:

                    Código CIE-10 original: {icd_code}
                    Nombre enfermedad principal original: {principal_diagnostic}

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
    _model_name: str = PrivateAttr()
    _csv_path: str = PrivateAttr()

    def __init__(self, model_name: str = None, csv_path: str = "icd_dataset.csv", **kwargs):
        kwargs.setdefault("pydantic_object", _EvolutionTextDiagnosticSchema)
        super().__init__(**kwargs)
        self._model_name = model_name
        self._csv_path = csv_path

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Optional[_EvolutionTextDiagnosticSchema]:
        try:
            # Obtenemos el objeto Pydantic usando la implementación base
            parsed: BaseModel = super().parse_result(result)
            print(parsed)
            # Solo para la primera iteración
            if self._model_name is not None:
                rag_system = diagnosticNormalizerRAG(
                    csv_path=self._csv_path,
                    model_name=self._model_name
                )

                normalized = rag_system.normalize_diagnostic(
                    principal_diagnostic=parsed.principal_diagnostic, icd_code=parsed.icd_code)

                parsed = normalized

            return parsed
        except Exception:
            if partial:
                return None
            raise

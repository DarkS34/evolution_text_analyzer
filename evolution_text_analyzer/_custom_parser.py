import re

import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.outputs import Generation
from langchain_ollama import OllamaLLM

from .auxiliary_functions import get_exclusion_terms


class CustomParser(BaseOutputParser):
    def __init__(self, llm: OllamaLLM, normalization_mode: bool, prompt: str, **kwargs):
        super().__init__()
        self._llm = llm
        self._normalization_mode = normalization_mode
        self._prompt = prompt
        self._snomed_df = pd.read_csv("./snomed_description_icd_normalized.csv", sep="\t")

    def _check_match(self, diagnostic: str) -> dict[str, str] | None:
        # List of terms to exclude in Spanish and English
        
        
        # Function to remove exclusion terms
        def remove_exclusion_terms(text: str) -> str:
            text_lower = text.lower().strip()
            words = text_lower.split()
            filtered_words = [word for word in words if word not in get_exclusion_terms()]
            return " ".join(filtered_words)
        
        diagnostic_lower = diagnostic.lower().strip()
        diagnostic_filtered = remove_exclusion_terms(diagnostic_lower)

        # Check columns in specified order: description_es, description_en, description_es_normalized
        for column in ['description_es_normalized', 'description_es', 'description_en']:
            if column in self._snomed_df.columns:
                # Process dataframe column to remove exclusion terms
                processed_column = self._snomed_df[column].fillna("").str.lower().str.strip().apply(remove_exclusion_terms)
                
                # Exact match with filtered terms
                exact_match = self._snomed_df[processed_column == diagnostic_filtered]
                if not exact_match.empty:
                    match_row = exact_match.iloc[0]
                    return {
                        "icd_code": match_row.get('icd_code', ''),
                        "principal_diagnostic": match_row.get('description_es_normalized', diagnostic)
                    }

                # Partial match with filtered terms
                partial_match_mask = processed_column.str.contains(diagnostic_filtered, regex=False)
                partial_matches = self._snomed_df[partial_match_mask]
                if not partial_matches.empty:
                    match_row = partial_matches.iloc[0]
                    return {
                        "icd_code": match_row.get('icd_code', ''),
                        "principal_diagnostic": match_row.get('description_es_normalized', diagnostic)
                    }

        return {
            "icd_code": "N/A",
            "principal_diagnostic": diagnostic
        }

    def _generate_icd_code(self, diagnostic: str) -> dict:
        prompt = PromptTemplate.from_template(self._prompt)

        icd_chain = prompt | self._llm | StrOutputParser()

        try:
            icd_code = icd_chain.invoke(
                {"principal_diagnostic": diagnostic})

            # Look for ICD pattern
            icd_match = re.search(r'([A-Z]\d+\.?\d*)', icd_code)

            # Check if a match was found
            if icd_match:
                cleaned_icd = re.sub(r"[\n\r\s]", "", icd_match.group(1))
                return {
                    "icd_code": cleaned_icd,
                    "principal_diagnostic": diagnostic
                }
            else:
                return {
                    "icd_code": "N/A",
                    "principal_diagnostic": diagnostic
                }
        except Exception as e:
            raise Exception(
                f"Error al invocar el modelo para normalizar el diagnóstico: {e}")

    def parse(self, result: list[Generation], *, partial: bool = False) -> dict:
        try:
            # Extraer texto del resultado según su tipo
            if isinstance(result, list) and result and all(isinstance(x, Generation) for x in result):
                # Si es una lista de Generation
                generated_diagnostic = result[0].text.strip()
            elif isinstance(result, Generation):
                # Si es un solo Generation
                generated_diagnostic = result.text.strip()
            elif isinstance(result, str):
                # Si ya es un string
                generated_diagnostic = result.strip()
            else:
                # Cualquier otro tipo
                generated_diagnostic = str(result).strip()

            # Aplicar el método adecuado según el modo
            if self._normalization_mode:
                return self._check_match(generated_diagnostic)
            else:
                return self._generate_icd_code(generated_diagnostic)
        except Exception as e:
            raise Exception(
                f"An error ocurred while trying to parse the result: {e}")

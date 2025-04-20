"""
Custom parser module for medical diagnostic analysis.
This module provides classes for parsing and normalizing medical diagnoses
using language models, with optional RAG-based normalization capabilities.
"""
import re

import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.outputs import Generation
from langchain_ollama.llms import OllamaLLM

# Threshold for similarity in RAG-based normalization
THRESHOLD: float = 0.9
# Minimum fuzzy ratio for accepting a match
FW_RATIO: int = 75


class DiagnosticNormalizerRAG:
    """
    RAG-based normalizer for medical diagnoses.
    
    This class uses Retrieval Augmented Generation (RAG) to normalize
    medical diagnoses by finding similar diagnoses in a reference dataset
    and using a language model to select the best match.
    """
    def __init__(self, vector_store: Chroma, llm: OllamaLLM, prompt: str, csv_path: str):
        """
        Initialize the RAG-based normalizer.
        
        Args:
            vector_store: Chroma vector database containing reference diagnoses
            llm: Language model to use for normalization
            prompt: Prompt template for the language model
            csv_path: Path to the CSV file containing reference diagnoses and ICD codes
        """
        self.prompt = prompt
        self.df = pd.read_csv(csv_path, sep="\t")
        self.df['text'] = self.df['principal_diagnostic'] + \
            ':' + self.df['icd_code']
        self.vector_store = vector_store
        self.threshold = THRESHOLD
        self.llm = llm

    def normalize_diagnostic(self, principal_diagnostic: str) -> dict[str, str]:
        """
        Normalize a medical diagnosis using RAG.
        
        This method follows a multi-step approach:
        1. Find similar diagnoses in the vector database
        2. If a highly similar diagnosis is found, return it directly
        3. Otherwise, use a language model to select the best match
        4. If that fails, use fuzzy matching as a fallback
        
        Args:
            principal_diagnostic: The diagnosis to normalize
            
        Returns:
            Dictionary with normalized ICD code and principal diagnostic
        """
        normalized_principal_diagnostic = principal_diagnostic.lower().strip()

        # Find similar diagnoses in the vector store
        similar_diagnostics = self.vector_store.similarity_search_with_score(
            normalized_principal_diagnostic, k=5)

        # If the similarity is above threshold, use the match directly
        if (similar_diagnostics[-1][1] >= self.threshold):
            return {
                "icd_code": similar_diagnostics[-1][0].metadata.get('icd_code', ''),
                "principal_diagnostic": similar_diagnostics[-1][0].metadata.get('principal_diagnostic', '')
            }

        # Create a text representation of similar diagnoses for the LLM
        similar_diagnostics_text = "\n".join(
            f"- {diag[0].metadata.get('principal_diagnostic', '')}" for diag in similar_diagnostics)

        # Create a prompt for the language model
        prompt = PromptTemplate.from_template(self.prompt)

        # Create a chain for finding the most similar diagnosis
        similarity_chain = prompt | self.llm | StrOutputParser()

        try:
            # Query the language model to find the best match
            result = similarity_chain.invoke({
                "principal_diagnostic": normalized_principal_diagnostic,
                "similar_diagnostics": similar_diagnostics_text,
            })

            # Look for an exact match with the LLM result
            for diag in similar_diagnostics:
                if result.lower().strip() == diag[0].metadata["principal_diagnostic"].lower():
                    return {
                        "icd_code": diag[0].metadata["icd_code"],
                        "principal_diagnostic": diag[0].metadata["principal_diagnostic"]
                    }

        except Exception as e:
            raise Exception(
                f"Error al invocar el modelo para normalizar el diagnóstico: {e}")

        # Fallback: Use fuzzy matching if LLM method didn't find a match
        import rapidfuzz.fuzz as fuzz
        best_match = None
        best_score = 0

        for item in similar_diagnostics:
            score = fuzz.ratio(result.lower(), item[0].metadata.get(
                'principal_diagnostic', '').lower())
            if score > FW_RATIO and score > best_score:
                best_score = score
                best_match = item

        if best_match:
            return {
                "icd_code": best_match[0].metadata["icd_code"],
                "principal_diagnostic": best_match[0].metadata["principal_diagnostic"]
            }

        # If no match found, return N/A
        return {
            "icd_code": "N/A",
            "principal_diagnostic": "N/A"
        }


class CustomParser(BaseOutputParser):
    """
    Custom parser for medical diagnoses.
    
    This class parses model outputs to extract normalized diagnoses and ICD codes,
    optionally using RAG-based normalization if a Chroma database is provided.
    """
    def __init__(self, chroma_db: Chroma | None, llm: OllamaLLM, prompt: str, csv_path: str = "icd_dataset.csv", **kwargs):
        """
        Initialize the custom parser.
        
        Args:
            chroma_db: Chroma database for RAG-based normalization (None to disable)
            llm: Language model to use for parsing
            prompt: Prompt template for the language model
            csv_path: Path to the CSV file containing reference diagnoses and ICD codes
            **kwargs: Additional arguments for the base parser
        """
        super().__init__()
        self._llm = llm
        if chroma_db is None:
            self._rag_system = None
            self._prompt = prompt
        else:
            self._rag_system = DiagnosticNormalizerRAG(
                chroma_db, llm, prompt, csv_path)

    def generate_icd_code(self, principal_diagnostic: str) -> dict:
        """
        Generate an ICD code for a diagnosis using a language model.
        
        This method uses a language model to generate an ICD code for a diagnosis
        when RAG-based normalization is not available.
        
        Args:
            principal_diagnostic: The diagnosis to generate an ICD code for
            
        Returns:
            Dictionary with the ICD code and principal diagnostic
            
        Raises:
            Exception: If an error occurs while invoking the model
        """
        prompt = PromptTemplate.from_template(self._prompt)

        icd_chain = prompt | self._llm | StrOutputParser()

        try:
            icd_code = icd_chain.invoke(
                {"principal_diagnostic": principal_diagnostic})

            # Look for ICD pattern
            icd_match = re.search(r'([A-Z]\d+\.?\d*)', icd_code)

            # Check if a match was found
            if icd_match:
                cleaned_icd = re.sub(r"[\n\r\s]", "", icd_match.group(1))
                return {
                    "icd_code": cleaned_icd,
                    "principal_diagnostic": principal_diagnostic
                }
            else:
                return {
                    "icd_code": "N/A",
                    "principal_diagnostic": principal_diagnostic
                }
        except Exception as e:
            raise Exception(
                f"Error al invocar el modelo para normalizar el diagnóstico: {e}")

    def parse(self, result: list[Generation], *, partial: bool = False) -> dict:
        """
        Parse language model output to extract a diagnosis and ICD code.
        
        This method implements the BaseOutputParser interface to parse 
        the output of a language model. It handles different input formats
        and uses either RAG-based normalization or direct ICD code generation.
        
        Args:
            result: Output from the language model
            partial: Whether this is a partial result (unused, for interface compatibility)
            
        Returns:
            Dictionary with the parsed ICD code and principal diagnostic
            
        Raises:
            Exception: If an error occurs during parsing
        """
        try:
            # Handle different input formats
            if isinstance(result, list) and all(isinstance(x, Generation) for x in result):
                principal_diagnostic = result[0].text.strip()
            elif isinstance(result, str):
                principal_diagnostic = result.strip()
            else:
                principal_diagnostic = str(result).strip()

            # Use RAG system if selected, otherwise generate ICD code directly
            if self._rag_system is None:
                generated = self.generate_icd_code(principal_diagnostic)
                parsed = generated
            else:
                normalized = self._rag_system.normalize_diagnostic(
                    principal_diagnostic=principal_diagnostic)
                parsed = normalized

            return parsed
        except Exception as e:
            raise Exception(f"An error ocurred while trying to parse the result: {e}")
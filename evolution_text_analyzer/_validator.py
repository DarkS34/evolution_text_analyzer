"""
Diagnosis validation module for the medical diagnostic analysis system.
This module provides functions to validate and compare extracted diagnoses with reference
diagnoses, applying various similarity and normalization techniques.
"""
import unicodedata
import re
from rapidfuzz import fuzz

from .auxiliary_functions import get_exclusion_terms


# Dictionary of diagnostic abbreviations and their expanded forms
EXTENDED_DIAGNOSTICS: dict[str, str] = {
    "PMR": "Polimialgia Reumática",
    "HAVI": "Hiperuricemia Asintomática",
    "ACG": "Arteritis de Células Gigantes",
    "OP": "Osteoporosis",
    "VASCULITIS": "Vasculitis",
    "Raynaud": "Fenómeno de Raynaud",
    "Dupuytren": "Enfermedad de Dupuytren",
    "ESCLEROSIS SISTÉMICA": "Esclerosis Sistémica",
    "LES": "Lupus Eritematoso Sistémico",
    "AIJ": "Artritis Idiopática Juvenil",
    "EDPP": "Espondilopatía Degenerativa Primaria",
    "ENF. INDIFERENCIADA DEL TEJIDO CONECTIVO": "Enfermedad Indiferenciada del Tejido Conectivo",
    "STC": "Síndrome del Túnel Carpiano",
    "Morton": "Neuroma de Morton",
    "Baker": "Quiste de Baker",
    "Miopatia inflamatoria idiop.": "Miopatía Inflamatoria Idiopática",
    "Paget": "Enfermedad de Paget",
    "SAPHO": "Síndrome SAPHO",
    "Behcet": "Enfermedad de Behçet",
    "S. Autoinflamatorio": "Síndrome Autoinflamatorio",
    "SAF": "Síndrome Antifosfolípido",
    "EMTC": "Enfermedad Mixta del Tejido Conectivo",
    "EPID": "Enfermedad Pulmonar Intersticial Difusa",
    "EII": "Enfermedad Inflamatoria Intestinal",
    "Microcristalina": "Artritis Microcristalina",
    "Sind. de SJÖGREN": "Síndrome de Sjögren",
    "Autoinflamatorio": "Síndrome Autoinflamatorio",
    "FOP": "Fibrodisplasia Osificante Progresiva",
    "Beçhet": "Enfermedad de Behçet",
    "Artritis gotosa aguda": "Gota",
    "Artritis goutosa": "Gota",
    "Gouto crónico": "Gota"
}

# Threshold for similarity scores to consider diagnoses as matching
SIMILARITY_THRESHOLD: float = 70.0


def normalize_name(name: str) -> str:
    """
    Normalize a diagnosis name to facilitate comparison.

    This function performs several normalization steps:
    1. Expands known abbreviations using the extendedDiagMap
    2. Removes accents and diacritical marks
    3. Converts text to lowercase
    4. Removes punctuation and special characters
    5. Removes extra whitespace

    Args:
        name: The diagnosis name to normalize

    Returns:
        Normalized diagnosis name (lowercase, without accents, expanded if abbreviated)
    """
    if name is None:
        return ""

    # 1. Expand known abbreviations
    expanded_name = EXTENDED_DIAGNOSTICS.get(name.strip(), name)

    # 2. Normalize: remove accents, convert to lowercase
    normalized = "".join(
        c for c in unicodedata.normalize("NFKD", expanded_name)
        if not unicodedata.combining(c)
    ).lower()

    # 3. Remove punctuation and special characters
    normalized = re.sub(r'[^\w\s]', ' ', normalized)

    # 4. Remove multiple spaces and spaces at beginning/end
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    return normalized


def tokenize_diagnosis(diagnosis: str) -> list[str]:
    """
    Split a diagnosis into significant tokens, removing common words.

    This function breaks a diagnosis into individual words and removes common
    words that don't contribute significantly to the diagnosis meaning.

    Args:
        diagnosis: Normalized diagnosis text

    Returns:
        List of significant tokens from the diagnosis
    """
    tokens = diagnosis.split()

    # Remove exclusion terms
    return [token for token in tokens if token not in get_exclusion_terms()]


def get_key_terms(diagnosis: str) -> list[str]:
    """
    Extract key terms from a diagnosis.

    This function normalizes a diagnosis and extracts significant tokens
    that represent the key medical concepts.

    Args:
        diagnosis: Original diagnosis text

    Returns:
        List of key terms from the diagnosis
    """
    normalized = normalize_name(diagnosis)

    return tokenize_diagnosis(normalized)


def calculate_similarity_scores(processed: str, correct: str) -> tuple[float, float, float]:
    """
    Calculate various similarity scores between two diagnosis strings.

    This function uses different fuzzy matching algorithms to assess how
    similar two diagnosis strings are to each other.

    Args:
        processed: The processed (extracted) diagnosis text
        correct: The correct reference diagnosis text

    Returns:
        Tuple of (ratio, partial_ratio, token_sort_ratio) similarity scores
    """
    ratio = fuzz.ratio(processed, correct)
    partial_ratio = fuzz.partial_ratio(processed, correct)
    token_sort_ratio = fuzz.token_sort_ratio(processed, correct)

    return (ratio, partial_ratio, token_sort_ratio)


def validate_result(processed_diag: str, correct_diag: str) -> bool:
    """
    Validate if a processed diagnosis matches the correct diagnosis.

    This function applies multiple validation strategies to determine if
    an extracted diagnosis is considered equivalent to the reference diagnosis:
    1. Direct comparison after normalization
    2. Checking if all key terms in the correct diagnosis are present in the processed one
    3. Using fuzzy string matching with various similarity metrics
    4. Special handling for short and long diagnoses

    Args:
        processed_diag: The processed (extracted) diagnosis to validate
        correct_diag: The correct reference diagnosis to compare against

    Returns:
        Boolean indicating whether the processed diagnosis is valid
    """
    # Handle null cases
    if processed_diag is None or correct_diag is None:
        return False
    if isinstance(processed_diag, str) and processed_diag.strip() == "":
        return False
    if isinstance(correct_diag, str) and correct_diag.strip() == "":
        return False

    # Normalize both diagnoses
    processed_norm = normalize_name(processed_diag)
    correct_norm = normalize_name(correct_diag)

    # 1. Direct match after normalization
    if processed_norm == correct_norm:
        return True

    # 2. Check if all key terms in correct diagnosis are in processed diagnosis
    processed_terms = set(get_key_terms(processed_diag))
    correct_terms = set(get_key_terms(correct_diag))
    if len(correct_terms) > 0 and correct_terms.issubset(processed_terms):
        return True

    # 3. Using fuzzy string matching with various similarity metrics
    ratio, partial_ratio, token_sort_ratio = calculate_similarity_scores(
        processed_norm, correct_norm)
    if max(ratio, token_sort_ratio) >= SIMILARITY_THRESHOLD:
        return True
    if len(correct_norm) < 15 and partial_ratio >= 90:
        return True

    # 4. Special case for substring relationships
    if len(correct_norm) > 10 and len(processed_norm) > 10:
        if (correct_norm in processed_norm) or (processed_norm in correct_norm):
            return True

    return False

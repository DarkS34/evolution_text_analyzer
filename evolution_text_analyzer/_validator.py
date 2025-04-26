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
    "Beçhet": "Enfermedad de Behçet"
}

# Threshold for similarity scores to consider diagnoses as matching
SIMILARITY_THRESHOLD: float = 70.0


def _normalize_name(name: str) -> str:
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


def _tokenize_diagnosis(diagnosis: str) -> list[str]:
    tokens = diagnosis.split()

    # Remove exclusion terms
    return [token for token in tokens if token not in get_exclusion_terms()]


def _get_key_terms(diagnosis: str) -> list[str]:
    normalized = _normalize_name(diagnosis)

    return _tokenize_diagnosis(normalized)


def _calculate_similarity_scores(processed: str, correct: str) -> tuple[float, float, float]:
    ratio = fuzz.ratio(processed, correct)
    partial_ratio = fuzz.partial_ratio(processed, correct)
    token_sort_ratio = fuzz.token_sort_ratio(processed, correct)

    return (ratio, partial_ratio, token_sort_ratio)


def validate_result(processed_diag: str, correct_diag: str) -> bool:
    # Handle null cases
    if processed_diag is None or correct_diag is None:
        return False
    if isinstance(processed_diag, str) and processed_diag.strip() == "":
        return False
    if isinstance(correct_diag, str) and correct_diag.strip() == "":
        return False

    # Normalize both diagnoses
    processed_norm = _normalize_name(processed_diag)
    correct_norm = _normalize_name(correct_diag)

    # 1. Direct match after normalization
    if processed_norm == correct_norm:
        return True

    # 2. Check if all key terms in correct diagnosis are in processed diagnosis
    processed_terms = set(_get_key_terms(processed_diag))
    correct_terms = set(_get_key_terms(correct_diag))
    if len(correct_terms) > 0 and correct_terms.issubset(processed_terms):
        return True

    # 3. Using fuzzy string matching with various similarity metrics
    ratio, partial_ratio, token_sort_ratio = _calculate_similarity_scores(
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

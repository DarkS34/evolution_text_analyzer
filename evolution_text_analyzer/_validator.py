import unicodedata
import re
from rapidfuzz import fuzz


extendedDiagMap: dict[str, str] = {
    "Baker": "Quiste de Baker",
    "Behcet": "Enfermedad de Behçet",
    "ACG": "Arteritis de Células Gigantes",
    "PMR": "Polimialgia Reumática",
    "HAVI": "Hiperuricemia Asintomática",
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
    "Miopatia inflamatoria idiop.": "Miopatía Inflamatoria Idiopática",
    "Paget": "Enfermedad de Paget",
    "SAPHO": "Síndrome SAPHO",
    "S. Autoinflamatorio": "Síndrome Autoinflamatorio",
    "SAF": "Síndrome Antifosfolípido",
    "EMTC": "Enfermedad Mixta del Tejido Conectivo",
    "EPID": "Enfermedad Pulmonar Intersticial Difusa",
    "EII": "Enfermedad Inflamatoria Intestinal",
    "Microcristalina": "Artritis Microcristalina",
    "Sind. de SJÖGREN": "Síndrome de Sjögren",
    "Autoinflamatorio": "Síndrome Autoinflamatorio",
    "FOP": "Fibrodisplasia Osificante Progresiva",
}


EXCLUSION_TERMS: list[str] = [
    "con", "y", "de", "del", "la", "el", "en", "por", "sin", "a", "para",
    "debido", "asociado", "secundario", "primario", "crónico", "agudo"
]


SIMILARITY_THRESHOLD: float = 80.0  


def normalize_name(name: str) -> str:
    """
    Normaliza un nombre de diagnóstico para facilitar la comparación.
    
    Args:
        name: Nombre del diagnóstico a normalizar
        
    Returns:
        Nombre normalizado en minúsculas, sin acentos y expandido si es una abreviatura
    """
    if name is None:
        return ""
        
    # Expandir abreviaturas conocidas
    expanded_name = extendedDiagMap.get(name.strip(), name)
    
    # Normalizar: eliminar acentos, convertir a minúsculas
    normalized = "".join(
        c for c in unicodedata.normalize("NFKD", expanded_name) 
        if not unicodedata.combining(c)
    ).lower()
    
    # Eliminar puntuación y caracteres especiales
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
    
    # Eliminar espacios múltiples y espacios al inicio/fin
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized


def tokenize_diagnosis(diagnosis: str) -> list[str]:
    """
    Divide un diagnóstico en tokens significativos, eliminando palabras comunes.
    
    Args:
        diagnosis: Diagnóstico normalizado
        
    Returns:
        Lista de tokens significativos
    """
    tokens = diagnosis.split()
    
    # Eliminar términos de exclusión
    return [token for token in tokens if token not in EXCLUSION_TERMS]


def get_key_terms(diagnosis: str) -> list[str]:

    normalized = normalize_name(diagnosis)
    
    return tokenize_diagnosis(normalized)


def calculate_similarity_scores(processed: str, correct: str) -> tuple[float, float, float]:
    ratio = fuzz.ratio(processed, correct)
    partial_ratio = fuzz.partial_ratio(processed, correct)
    token_sort_ratio = fuzz.token_sort_ratio(processed, correct)
    
    return (ratio, partial_ratio, token_sort_ratio)


def validate_result(processed_diag: str, correct_diag: str) -> bool:
    
    if processed_diag is None or correct_diag is None:
        return False
    
    if isinstance(processed_diag, str) and processed_diag.strip() == "":
        return False
        
    if isinstance(correct_diag, str) and correct_diag.strip() == "":
        return False
    
    
    processed_norm = normalize_name(processed_diag)
    correct_norm = normalize_name(correct_diag)
    
    if processed_norm == correct_norm:
        return True
    
    processed_terms = set(get_key_terms(processed_diag))
    correct_terms = set(get_key_terms(correct_diag))

    if len(correct_terms) > 0 and correct_terms.issubset(processed_terms):
        return True
    
    ratio, partial_ratio, token_sort_ratio = calculate_similarity_scores(processed_norm, correct_norm)
    
    if max(ratio, token_sort_ratio) >= SIMILARITY_THRESHOLD:
        return True
        
    if len(correct_norm) < 15 and partial_ratio >= 90:
        return True
    
    if len(correct_norm) > 10 and len(processed_norm) > 10:
        if (correct_norm in processed_norm) or (processed_norm in correct_norm):
            return True
    
    return False
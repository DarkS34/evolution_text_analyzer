import unicodedata
import re
from rapidfuzz import fuzz

from .auxiliary_functions import get_exclusion_terms


# Dictionary of diagnostic abbreviations and their expanded forms
EXTENDED_DIAGNOSTICS: dict[str, str] = {
    "LES": "Lupus eritematoso sistémico",
    "PMR": "Polimialgia reumática",
    "AIJ": "Artritis idiopática juvenil",
    "ACG": "Arteritis de células gigantes",
    "SAF": "Síndrome antifosfolípido",
    "EMTC": "Enfermedad mixta del tejido conectivo",
    "EPID": "Enfermedad pulmonar intersticial difusa",
    "EII": "Enfermedad inflamatoria intestinal",
    "Artritis reumatoide": "Artritis reumatoide con factor reumatoide, no especificada",
    "Artrosis": "Artrosis no especificada, localización no especificada",
    "Artritis psoriásica": "Artropatía psoriásica, no especificada",
    "Espondilitis anquilosante": "Espondilitis anquilosante de localizaciones múltiples de la columna vertebral",
    "Gota": "Gota, no especificada",
    "Polimialgia reumática": "Polimialgia reumática",
    "Lumbalgia": "Dolor en la parte inferior de la espalda",
    "Osteoporosis": "Osteoporosis con fractura patológica, no especificada",
    "Tendinopatía": "Tendinitis, localización no especificada",
    "Lupus eritematoso sistémico": "Lupus eritematoso sistémico, no especificado",
    "Artralgias": "Dolor articular, no especificado",
    "Espondiloartrosis": "Espondilosis, no especificada",
    "Enfermedad degenerativa de las pequeñas articulaciones": "Artrosis de otras articulaciones, no especificada",
    "Artritis idiopática juvenil": "Artritis juvenil, no especificada",
    "Síndrome de Sjögren": "Síndrome de Sjögren, no especificado",
    "Ciática": "Ciática",
    "Fibromialgia": "Fibromialgia",
    "Artritis": "Artritis, no especificada",
    "Esclerosis sistémica": "Esclerosis sistémica progresiva",
    "Gonalgia": "Dolor en la rodilla",
    "Fenómeno de Raynaud": "Fenómeno de Raynaud",
    "Trocanteritis": "Bursitis del trocánter mayor",
    "Arteritis de células gigantes": "Arteritis de células gigantes",
    "Vasculitis": "Vasculitis, no especificada",
    "Dorsalgia": "Dolor en la parte superior de la espalda",
    "Miopatía inflamatoria idiopática": "Miopatía inflamatoria, no especificada",
    "Mialgias": "Dolor muscular",
    "Cervicobraquialgia": "Síndrome cervicobraquial",
    "Fascitis": "Fascitis, no especificada",
    "Aplastamiento vertebral": "Fractura por compresión de la vértebra, no especificada",
    "Dedo en resorte": "Tenosinovitis estenosante",
    "Enfermedad de Paget ósea": "Osteítis deformante",
    "Estenosis de canal raquídeo": "Estenosis espinal, no especificada",
    "Gonartrosis": "Artrosis de rodilla, no especificada",
    "Contractura muscular": "Contractura muscular",
    "Epicondilitis": "Epicondilitis lateral",
    "Discopatía": "Trastorno degenerativo del disco intervertebral, no especificado",
    "Enfermedad de Behçet": "Enfermedad de Behçet",
    "Artralgias mecánicas": "Dolor articular, no especificado",
    "Síndrome del túnel carpiano": "Síndrome del túnel carpiano",
    "Enfermedad indiferenciada del tejido conectivo": "Enfermedad del tejido conectivo, no especificada",
    "Bursitis": "Bursitis, no especificada",
    "Coxalgia": "Dolor en la cadera",
    "Déficit de vitamina D": "Deficiencia de vitamina D",
    "Esclerodermia": "Esclerosis sistémica progresiva",
    "Hernia discal": "Hernia de disco intervertebral, no especificada",
    "Sarcoidosis": "Sarcoidosis, no especificada",
    "Artritis indiferenciada": "Artritis, no especificada",
    "Artrosis axial": "Artrosis de la columna vertebral, no especificada",
    "Síndrome autoinflamatorio": "Trastorno autoinflamatorio, no especificado",
    "Omalgia": "Dolor en el hombro",
    "Enfermedad mixta del tejido conectivo": "Enfermedad mixta del tejido conectivo",
    "Tenosinovitis": "Tenosinovitis, no especificada",
    "Escoliosis": "Escoliosis, no especificada",
    "Síndrome antifosfolípido": "Síndrome antifosfolípido primario",
    "Osteopenia": "Osteopenia",
    "Condropatía rotuliana": "Condromalacia rotuliana",
    "Hiperplasia angiofibromatosa vulvar idiopática": "Hiperplasia angiofibromatosa vulvar idiopática",
    "Meniscopatía": "Lesión del menisco, no especificada",
    "Dermatomiositis": "Dermatomiositis, no especificada",
    "Artritis microcristalina": "Artritis por cristales, no especificada",
    "Síndrome SAPHO": "Síndrome SAPHO",
    "Quiste de Baker": "Quiste sinovial de la fosa poplítea",
    "Hiperparatiroidismo": "Hiperparatiroidismo, no especificado",
    "Mal apoyo plantar": "Trastorno del pie, no especificado",
    "Lumbartrosis": "Artrosis de la región lumbar",
    "Uveítis": "Uveítis, no especificada",
    "Artritis séptica": "Artritis piógena, no especificada",
    "Fibrodisplasia osificante progresiva": "Fibrodisplasia osificante progresiva",
    "Coxopatía": "Trastorno de la cadera, no especificado",
    "Enfermedad de Dupuytren": "Contractura de Dupuytren",
    "Crioglobulinemia": "Crioglobulinemia esencial",
    "Neuroma de Morton": "Neuroma de Morton",
    "Miopatía inflamatoria": "Miopatía inflamatoria, no especificada",
    "Enfermedad autoinflamatoria": "Trastorno autoinflamatorio, no especificado",
    "Enfermedad pulmonar intersticial difusa": "Enfermedad pulmonar intersticial, no especificada",
    "Miopatía": "Miopatía, no especificada",
    "Condromalacia rotuliana": "Condromalacia rotuliana",
    "Enfermedad inflamatoria intestinal": "Enfermedad inflamatoria intestinal, no especificada"
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

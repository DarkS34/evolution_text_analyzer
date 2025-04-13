"""This is the example module.

This module does stuff.
"""

import unicodedata

extendedDiagMap = {
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


def normalize_name(name: str) -> str:
    newName = extendedDiagMap.get(name, name)
    return "".join(c for c in unicodedata.normalize("NFKD", newName) if not unicodedata.combining(c)).lower()


def validate_result(modelName: str, processedDiag: str, correctDiag: str):
    processedDiagNorm = normalize_name(processedDiag)
    correctDiagNorm = normalize_name(correctDiag)

    staticalCond = (processedDiagNorm == correctDiagNorm) or processedDiagNorm.find(
        correctDiagNorm) != -1 or correctDiagNorm.find(processedDiagNorm) != -1

    return staticalCond

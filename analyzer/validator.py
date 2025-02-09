import unicodedata

extendedDiagMap = {
    "PMR": "Polimialgia Reumática",
    "HAVI": "Hiperuricemia Asintomática",
    "ACG": "Arteritis de Células Gigantes",
    "OP": "Osteoporosis",
    "VASCULITIS": "Vasculitis",
    "Raynaud": "Fenómeno de Raynaud",
    "Dupuytren": "Enfermedad de Dupuytren",
    "ESCLEROSIS SISTÉMICA": "Esclerosis Sistémica",
    "LES": "Lupus Eritematoso Sistémico",
    "Artritis reumatoide": "Artritis Reumatoide",
    "AIJ": "Artritis Idiopática Juvenil",
    "EDPP": "Espondilopatía Degenerativa Primaria",
    "ENF. INDIFERENCIADA DEL TEJIDO CONECTIVO": "Enfermedad Indiferenciada del Tejido Conectivo",
    "Deficit Vitamina D": "Déficit de Vitamina D",
    "STC": "Síndrome del Túnel Carpiano",
    "Esclerodermia": "Esclerodermia",
    "Artritis": "Artritis",
    "Escoliosis": "Escoliosis",
    "Discopatia": "Discopatía",
    "Morton": "Neuroma de Morton",
    "Estenosis de canal": "Estenosis de canal",
    "Artrosis axial": "Artrosis axial",
    "Baker": "Quiste de Baker",
    "Miopatia inflamatoria idiop.": "Miopatía Inflamatoria Idiopática",
    "Paget": "Enfermedad de Paget",
    "SAPHO": "Síndrome SAPHO",
    "Behcet": "Enfermedad de Behçet",
    "S. Autoinflamatorio": "Síndrome Autoinflamatorio",
    "SAF": "Síndrome Antifosfolípido",
    "EMTC": "Enfermedad Mixta del Tejido Conectivo",
    "Artritis septica": "Artritis séptica",
    "EPID": "Enfermedad Pulmonar Intersticial Difusa",
    "EII": "Enfermedad Inflamatoria Intestinal",
    "Microcristalina": "Artritis Microcristalina",
    "Sind. de SJÖGREN": "Síndrome de Sjögren",
    "Condromalacea": "Condromalacia",
    "Autoinflamatorio": "Síndrome Autoinflamatorio",
    "FOP": "Fibrodisplasia Osificante Progresiva",
    "Miopatia inflamatoria": "Miopatía Inflamatoria",
    "Beçhet": "Enfermedad de Behçet",
    "Artritis gotosa aguda": "Gota",
}


def _normalizeText(texto):
    return "".join(
        c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c)
    ).lower()


def validateResult(processedDiag: str, correctDiag: str):
    diagE1 = _normalizeText(extendedDiagMap.get(processedDiag, processedDiag))
    diagE2 = _normalizeText(extendedDiagMap.get(correctDiag, correctDiag))

    return (diagE1 == diagE2) or diagE1.find(diagE2) != -1 or diagE2.find(diagE1) != -1

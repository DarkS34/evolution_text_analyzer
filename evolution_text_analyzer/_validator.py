"""This is the example module.

This module does stuff.
"""

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel, Field


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
}


def normalize_name(name: str) -> str:
    newName = extendedDiagMap.get(name, name)
    return "".join(
        c for c in unicodedata.normalize("NFKD", newName) if not unicodedata.combining(c)
    ).lower()


def model_validation(modelName: str, diag1: str, diag2: str):
    class DiagnosticComparator(BaseModel):
        comparation: bool = Field(
            title="Comparación de Diagnósticos",
            description="Indica si las dos enfermedades mencionadas son la misma (True) o diferentes (False).",
            examples=[True, False]
        )
    model = OllamaLLM(
        model=modelName,
        temperature=0,
        top_p=0.9,
        verbose=False,
        format="json",
        seed=123,
    )
    parser = PydanticOutputParser(pydantic_object=DiagnosticComparator)

    prompt = ChatPromptTemplate(
        messages=[
        SystemMessagePromptTemplate.from_template(
            """Eres un sistema médico experto en diagnosticar y comparar enfermedades. 
            Tu tarea es analizar si dos enfermedades son la misma basándote en criterios médicos rigurosos, como: síntomas, causas, patología, tratamientos y clasificación médica oficial (ej. CIE-10, DSM-5). "
            Si los nombres son diferentes pero la enfermedad es la misma según estos criterios, indica 'True'. 
            Si hay diferencias significativas en cualquiera de estos aspectos, indica 'False'. 
            No asumas que dos nombres similares significan la misma enfermedad sin evidencia clara."""
        ),
        HumanMessagePromptTemplate.from_template(
            '¿La enfermedad "{diag1}" es exactamente la misma que "{diag2}" según criterios médicos oficiales?'
            'Responde solo con "True" o "False".'
        ),
        ],
        input_variables=["diag1", "diag2"],
        partial_variables={
            "instructionsFormat": parser.get_format_instructions()}
    )

    validationChain = prompt | model | parser

    try:
        result: DiagnosticComparator = validationChain.invoke(
            {"diag1": diag1, "diag2": diag2})
    except Exception:
        return False

    return result.comparation


def validate_result(modelName: str, processedDiag: str, correctDiag: str):
    processedDiagNorm = normalize_name(processedDiag)
    correctDiagNorm = normalize_name(correctDiag)

    if (processedDiagNorm == correctDiagNorm) or processedDiagNorm.find(correctDiagNorm) != -1 or correctDiagNorm.find(processedDiagNorm) != -1:
        return True
    else:
        print("\r Validating reuslts")
        return model_validation(modelName, processedDiagNorm, correctDiagNorm)

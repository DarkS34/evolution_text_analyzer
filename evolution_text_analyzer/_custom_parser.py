from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class EvolutionTextDiagnosticSchema(BaseModel):
    principal_diagnostic: str = Field(
        title="Nombre enfermedad",
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

class CustomParser(PydanticOutputParser):
    def __init__(self, **kwargs):
        # Si ya te pasan el pydantic_object, usalo; si no, poné el tuyo
        kwargs.setdefault("pydantic_object", EvolutionTextDiagnosticSchema)
        super().__init__(**kwargs)

    def parse(self, text: str) -> EvolutionTextDiagnosticSchema:
        parsed = super().parse(text)
        print("🎯 Resultado parseado:", parsed)
        return parsed
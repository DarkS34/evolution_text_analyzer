from transformers import pipeline
from pydantic import BaseModel, Field, ValidationError
import json

# Definimos el esquema de salida


class EvolutionTextDiagnostic(BaseModel):
    principal_diagnostic: str = Field(
        title="Nombre enfermedad",
        description="Nombre de la enfermedad principal, basado en el historial del paciente",
        min_length=3,
        max_length=100,
    )


# Cargamos el modelo y tokenizer
try:
    pipe = pipeline("text2text-generation", model="HiTZ/Medical-mT5-large", device=0)
except Exception as e:
    print("Error al cargar el modelo:", e)

# Prompt para el modelo


def build_prompt(historial_medico: str) -> str:
    return (
        "Instrucción: A partir del siguiente historial médico, identifica el diagnóstico principal del paciente. "
        "Devuelve únicamente un objeto JSON con la clave 'principal_diagnostic'. "
        "El valor debe ser el nombre completo de la enfermedad más importante, escrito en español. "
        "Ejemplo de formato esperado:\n"
        '{"principal_diagnostic": "Neumonía adquirida en la comunidad"}\n\n'
        f"Historial médico:\n{historial_medico}"
    )

# Función principal


def analizar_historial(historial_medico: str):
    if pipe is None:
            print("Pipeline no disponible.")
            return None

    prompt = build_prompt(historial_medico)
    response = pipe(prompt, max_new_tokens=256)[0]["generated_text"]

    print(f"Respuesta generada:\n{response}\n")

    # Intentamos convertir la respuesta a JSON y validarla con pydantic
    try:
        data = json.loads(response)
        resultado = EvolutionTextDiagnostic(**data)
        return resultado.dict()
    except (json.JSONDecodeError, ValidationError) as e:
        print("Error al procesar la respuesta:")
        print(e)
        return None

# Ejemplo de uso
if __name__ == "__main__":
    historial = """
    La paciente acude por seguimiento mensual. Refiere mejoría parcial en cuanto a la fatiga y el rash malar, aunque persiste proteinuria (1.2 g/24 h). Al examen físico (EF), no se evidencian lesiones cutáneas activas ni edema periférico. La presión arterial (TA) es de 140/90 mmHg, por lo que se ajustará el manejo antihipertensivo. En laboratorio, se observa proteinuria en orina de 24 horas persistente, pero en descenso. La creatinina se ha reducido a 1.1 mg/dL (previo 1.4 mg/dL) y el nitrógeno ureico en sangre (BUN) está dentro de los límites normales. Se ajustará la dosis de micofenolato mofetil (MMF) a 2 g/día y se mantiene la prednisona en 10 mg/día, con descenso progresivo según evolución. Los anticuerpos anti-ADN de doble cadena se mantienen moderadamente elevados (50 UI/mL) y los niveles de complemento (C3 y C4) permanecen bajos pero estables.La paciente tolera el tratamiento sin eventos adversos reportados. Se le indica continuar con fotoprotección estricta para evitar exacerbaciones cutáneas. Además, se inicia hidroxicloroquina 200 mg/día para el manejo a largo plazo. Dado el antecedente de hipertensión arterial (HTA), se solicita un ecocardiograma para evaluar posibles afectaciones cardiovasculares relacionadas con el lupus. También se programa una evaluación de fondo de ojo debido a la introducción de hidroxicloroquina.En la siguiente consulta, la paciente refiere disminución en la fatiga y en el rash malar. La proteinuria ha bajado a 0.9 g/24 h, y la creatinina se mantiene estable en 1.0 mg/dL. El ecocardiograma revela hipertrofia ventricular izquierda leve, probablemente secundaria a la HTA. Se ajusta el antihipertensivo a losartán 50 mg al día, dado su efecto renoprotector. Se insiste en la adherencia al tratamiento y se sugiere iniciar actividad física moderada bajo supervisión para mejorar su estado general.En el tercer mes de seguimiento, la paciente reporta remisión completa del rash y una mejoría significativa en su energía diaria. La proteinuria desciende a 0.5 g/24 h, indicando un control adecuado de la nefritis lúpica. Los anticuerpos anti-ADN doble cadena han disminuido a 25 UI/mL y los niveles de complemento están en recuperación (C3 82 mg/dL, C4 18 mg/dL). La paciente no reporta efectos secundarios al tratamiento con hidroxicloroquina. Se programa seguimiento cada dos meses, con controles regulares de laboratorio y monitoreo oftalmológico anual.
    """
    resultado = analizar_historial(historial)
    print("Resultado validado:")
    print(resultado)
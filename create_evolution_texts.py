import asyncio
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_ollama.llms import OllamaLLM



diagnostics = [
    "Polimialgia Reumática",
    "Hiperuricemia Asintomática",
    "Arteritis de Células Gigantes",
    "Osteoporosis",
    "Vasculitis",
    "Fenómeno de Raynaud",
    "Enfermedad de Dupuytren",
    "Esclerosis Sistémica",
    "Lupus Eritematoso Sistémico",
    "Artritis Idiopática Juvenil",
    "Espondilopatía Degenerativa Primaria",
    "Enfermedad Indiferenciada del Tejido Conectivo",
    "Síndrome del Túnel Carpiano",
    "Neuroma de Morton",
    "Quiste de Baker",
    "Miopatía Inflamatoria Idiopática",
    "Enfermedad de Paget",
    "Síndrome SAPHO",
    "Enfermedad de Behçet",
    "Síndrome Autoinflamatorio",
    "Síndrome Antifosfolípido",
    "Enfermedad Mixta del Tejido Conectivo",
    "Enfermedad Pulmonar Intersticial Difusa",
    "Enfermedad Inflamatoria Intestinal",
    "Artritis Microcristalina",
    "Síndrome de Sjögren",
    "Síndrome Autoinflamatorio",
    "Fibrodisplasia Osificante Progresiva",
    "Enfermedad de Behçet",
    "Gota",
]

async def stream_response(diagnostic):
    model = OllamaLLM(
        model="llama3.1:8b-instruct-q8_0",
        temperature=0,
        num_ctx=8192,
        verbose=False,
    )
    
    prompt_template = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template("Eres un médico especializado en reumatología."),
            HumanMessagePromptTemplate.from_template(
                """Genera un texto evolutivo detallado de un paciente con una enfermedad reumatológica. El texto debe cumplir con las siguientes reglas estrictas:
                1. Formato:
                    - Debe seguir la estructura: diagnóstico principal, historial médico del paciente.
                    - El nombre de la enfermedad no debe aparecer en el historial, solo en diagnostico principal.
                    - Debe escribirse en un estilo médico realista, como si fuera parte de una historia clínica.
                    - Incluir tecnicismos y abreviaciones médicas para mayor autenticidad.
                    - Debe escribirse de manera redactada con grandes parrafos y con unos pocos saltos de linea.
                2. Contenido:a
                    - Describir la evolución de la enfermedad {diagnostic} en un paciente de manera detallada y progresiva.
                    - Reflejar cambios en la sintomatología a lo largo del tiempo y modificaciones en el tratamiento.
                3. Ortografía y estilo:
                    - Introducir un 10% de errores ortográficos leves de manera natural.
                    - Mantener un tono profesional y técnico, simulando una historia médica real.
                4. Extensión: generar un texto de aproximadamente 1000 palabras.
                5. Condiciones adicionales: El texto debe ser realista, basado en el curso clínico habitual de la enfermedad seleccionada.
                """
            ),
        ]
    )

    # Aplicamos el formato con el diagnóstico específico
    formatted_prompt = prompt_template.format(diagnostic=diagnostic)

    try:
        async for chunk in model.astream(formatted_prompt):
            yield chunk  # Enviar el texto por partes en lugar de imprimir
    except Exception as e:
        yield f"Error: {str(e)}"

# Si quieres probarlo directamente en consola
async def main():
    async for text in stream_response(diagnostics[0]):
        print(text, end="", flush=True)

asyncio.run(main())

# Medical Evolution Text Analyzer

![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen)

Un sistema avanzado para el análisis de textos médicos de evolución que extrae diagnósticos principales y códigos CIE mediante modelos de lenguaje.

## 📋 Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Características](#características)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
  - [Argumentos de línea de comandos](#argumentos-de-línea-de-comandos)
  - [Ejemplos de ejecución](#ejemplos-de-ejecución)
- [Arquitectura](#arquitectura)
- [Flujo de Procesamiento](#flujo-de-procesamiento)
- [Estructura de Directorios](#estructura-de-directorios)
- [Formato de Datos](#formato-de-datos)
- [Resultados](#resultados)
- [Licencia](#licencia)

## 🔍 Descripción General

El **Medical Evolution Text Analyzer** es un sistema basado en Python diseñado para procesar textos médicos de evolución, extraer diagnósticos principales y códigos CIE (Clasificación Internacional de Enfermedades), y validar estos diagnósticos contra datos de referencia. Utiliza modelos de lenguaje a través del framework Ollama para realizar análisis semántico avanzado de textos médicos, con especial enfoque en enfermedades reumatológicas.

## ✨ Características

- **Extracción automática de diagnósticos** a partir de notas clínicas
- **Normalización de diagnósticos** mediante RAG (Retrieval Augmented Generation)
- **Asignación de códigos CIE** a los diagnósticos extraídos
- **Procesamiento en paralelo** para optimizar el tiempo de ejecución
- **Evaluación de precisión** de diferentes modelos de lenguaje
- **Expansión de texto** opcional para mejorar la extracción de información
- **Interfaz de línea de comandos** flexible y potente

## 📋 Requisitos

- Python 3.9+
- [Ollama](https://ollama.ai/) instalado y en ejecución
- Gestor de paquetes UV (opcional, pero recomendado)

## 💻 Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/username/medical-evolution-text-analyzer.git
   cd medical-evolution-text-analyzer
   ```

2. Instala las dependencias utilizando UV:
   ```bash
   pip install uv
   uv sync
   ```
   
3. Sincroniza las dependencias:
   ```bash
   uv sync
   ```

3. Asegúrate de que Ollama esté ejecutándose:
   ```bash
   ollama start
   ```

## 🚀 Uso

El script principal se ejecuta a través de la línea de comandos con varios argumentos para personalizar el análisis.

### Argumentos de línea de comandos

```
python main.py [opciones]
```

| Argumento | Descripción |
|-----------|-------------|
| `-m`, `--mode` | Modo de operación: `1` para todos los modelos, `2` para selección de modelo (predeterminado: `1`) |
| `-b`, `--batches` | Número de lotes para procesamiento paralelo (predeterminado: `1`) |
| `-n`, `--num-texts` | Número de textos a procesar (predeterminado: todos) |
| `-t`, `--test` | Modo de prueba |
| `-tp`, `--test-prompts` | Probar diferentes prompts |
| `-i`, `--installed` | Usar solo modelos instalados |
| `-v`, `--verbose` | Modo detallado |
| `-E`, `--expand` | Expandir textos de evolución |
| `-N`, `--normalize` | Normalizar resultados mediante RAG |

### Ejemplos de ejecución

```bash
# Seleccionar un modelo específico, modo prueba, solo usar modelos instalados, modo verboso, 
python main.py -tiv -m2

# Ejecutar en modo prueba con expansión de texto y normalización, con solo modelos instalados
python main.py -tiEN -m2
```

## 🏗️ Arquitectura

El sistema se estructura en varios módulos principales:

1. **analyzer.py**: Coordina el proceso de análisis de textos médicos
2. **_custom_parser.py**: Parsea y normaliza los diagnósticos extraídos
3. **auxiliary_functions.py**: Proporciona funciones auxiliares para el manejo de datos
4. **_validator.py**: Valida los resultados del diagnóstico
5. **tester.py**: Evalúa la precisión de los modelos

## 📊 Flujo de Procesamiento

1. **Inicialización**:
   - Verificación de conexión con Ollama
   - Carga de configuración y textos de evolución
   - Procesamiento de argumentos de la línea de comandos

2. **Análisis de Textos**:
   - El texto se procesa en lotes paralelos
   - Opcionalmente se expande mediante un modelo de lenguaje
   - Se extraen diagnósticos principales y códigos CIE

3. **Normalización** (opcional):
   - Los diagnósticos se normalizan mediante RAG (Retrieval Augmented Generation)
   - Se utiliza una base de datos vectorial Chroma para encontrar diagnósticos similares

4. **Validación**:
   - Los diagnósticos extraídos se comparan con los valores de referencia
   - Se calculan métricas de precisión, errores y salidas incorrectas

5. **Resultados**:
   - Los resultados se almacenan en archivos JSON
   - Se proporcionan métricas detalladas de rendimiento

## 📁 Estructura de Directorios

```
.
├── evolution_text_analyzer/
│   ├── __init__.py
│   ├── analyzer.py                # Lógica de análisis principal
│   ├── _custom_parser.py          # Parseador de diagnósticos
│   ├── _validator.py              # Validación de resultados
│   ├── auxiliary_functions.py     # Funciones auxiliares
│   └── tester.py                  # Evaluación de modelos
├── main.py                        # Punto de entrada principal
├── config.json                    # Configuración del sistema
├── icd_dataset.csv                # Conjunto de datos de códigos CIE
├── evolution_texts_resolved.csv   # Textos médicos de evolución
└── README.md                      # Documentación
```

## 📋 Formato de Datos

### Archivos de entrada

Los textos de evolución médica deben estar en formato CSV o JSON con los siguientes campos:

- `id`: Identificador único del registro
- `evolution_text`: Texto médico de evolución
- `principal_diagnostic`: Diagnóstico principal correcto (para evaluación)

### Configuración

El archivo `config.json` debe contener:

- `models`: Lista de modelos a evaluar
- `prompts`: Lista de prompts para usar con los modelos
- `optimal`: Configuración óptima (índices de modelo y prompt)

## 📈 Resultados

Los resultados se almacenan en directorios según el modo de ejecución:

- **Modo normal**: Archivos JSON en el directorio `results/`
- **Modo prueba**: Archivos JSON en el directorio `testing_results/`

El formato de resultados incluye:

- Información del modelo (nombre, tamaño, parámetros)
- Métricas de rendimiento (precisión, errores, tiempo de procesamiento)
- Detalles de cada diagnóstico procesado
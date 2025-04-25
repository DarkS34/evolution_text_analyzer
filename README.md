# Medical Evolution Text Analyzer

![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)

An advanced system for analyzing rheumatological medical evolution texts that extracts principal diagnoses and ICD codes using local language models with SNOMED-CT based normalization capabilities.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Command Line Arguments](#command-line-arguments)
  - [Execution Examples](#execution-examples)
- [Architecture](#architecture)
- [Processing Flow](#processing-flow)
- [Directory Structure](#directory-structure)
- [Data Format](#data-format)
- [Results](#results)

## Overview

The **Medical Evolution Text Analyzer** is a Python-based application designed to process medical evolution texts, extract principal rheumatological diagnoses and ICD codes (International Classification of Diseases), and validate these diagnoses against SNOMED-CT reference data. It uses language models through the Ollama framework to perform advanced semantic analysis of medical texts, with a special focus on rheumatological diseases.

## Features

- **Automatic diagnosis extraction** from clinical notes using local LLMs via Ollama
- **Diagnosis normalization** using SNOMED-CT dataset
- **ICD code assignment** to extracted diagnoses
- **Parallel processing** to optimize execution time
- **Comprehensive model evaluation** with detailed performance metrics
- **Text summarization** for long clinical notes to fit context windows
- **Visual performance comparisons** between different models
- **Support for Spanish medical texts** with specialized prompts
- **Flexible and powerful command line interface**
- **Multi-model evaluation** for comparing different LLMs

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- Large Language Models (Llama, Gemma, Mixtral, etc.) available through Ollama
- SNOMED-CT dataset for normalization

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/evolution-text-analysis.git
   cd evolution-text-analysis
   ```

2. Install dependencies:
   ```bash
   pip install .
   ```
   
3. Make sure Ollama is installed:
   ```bash
   https://ollama.com/download
   ```

4. Prepare the SNOMED-CT dataset (run once):
   ```bash
   python create_snomed_normalized_icd_dataset.py
   ```

## Usage

The main script is executed through the command line with various arguments to customize the analysis.

### Command Line Arguments

```
python main.py [options]
```

| Argument | Description |
|-----------|-------------|
| `-f`, `--filename` | Filename for the evolution texts file (default: `evolution_texts.csv`) |
| `-m`, `--mode` | Operation mode: `1` for all models, `2` for model selection (default: `1`) |
| `-b`, `--batches` | Number of batches for parallel processing (default: `1`) |
| `-n`, `--num-texts` | Number of texts to process (default: all) |
| `-W`, `--context-window` | Size of context window in tokens (default: `3072`) |
| `-t`, `--test` | Run in test mode to evaluate model performance |
| `-i`, `--installed` | Only use models that are already installed |
| `-v`, `--verbose` | Print detailed output during processing |
| `-N`, `--normalize` | Normalize results using SNOMED dataset |

### Execution Examples

```bash
# Run with the optimal model (from config) in standard mode
python main.py

# Run with all installed models in test mode with verbose output
python main.py -tiv

# Select a specific model in test mode and use normalization
python main.py -tN -m2

# Process 50 texts with 4 parallel batches
python main.py -n50 -b4
```

## Architecture

The system is structured into several main modules:

1. **analyzer.py**: Coordinates the medical text analysis process
2. **_custom_parser.py**: Parses and normalizes extracted diagnoses using SNOMED-CT
3. **_validator.py**: Validates diagnosis results using multiple strategies
4. **auxiliary_functions.py**: Provides utility functions for data handling
5. **tester.py**: Evaluates model accuracy and performance
6. **data_models.py**: Contains Pydantic models for data structures
7. **results_manager.py**: Manages storage and visualization of results

## Processing Flow

1. **Initialization**:
   - Verification of connection with Ollama
   - Loading configuration and evolution texts
   - Processing command line arguments

2. **Text Analysis**:
   - Text is processed in parallel batches
   - Long texts are summarized to fit LLM context windows
   - Principal diagnoses and ICD codes are extracted

3. **Normalization** (optional):
   - Diagnoses are normalized using SNOMED-CT dataset
   - Exact and fuzzy matching is applied to standardize terminology
   - Exclusion terms are filtered to improve matching quality

4. **Validation**:
   - Extracted diagnoses are compared with reference values
   - Multiple validation strategies are applied (direct comparison, key terms, fuzzy matching)
   - Metrics for accuracy, errors, and incorrect outputs are calculated

5. **Results**:
   - Results are stored in JSON files
   - Detailed performance metrics are provided
   - Visualizations are generated for model comparison

## Directory Structure

```
.
├── evolution_text_analyzer/
│   ├── __init__.py                # Package initialization
│   ├── analyzer.py                # Main analysis logic
│   ├── _custom_parser.py          # Diagnosis parsing and normalization
│   ├── _validator.py              # Results validation
│   ├── auxiliary_functions.py     # Utility functions
│   ├── data_models.py             # Pydantic data models
│   ├── results_manager.py         # Results management and visualization
│   └── tester.py                  # Model evaluation
├── main.py                        # Main entry point
├── config.json                    # System configuration with models and prompts
├── create_snomed_normalized_icd_dataset.py  # SNOMED-CT dataset creator
├── snomed_description_icd_normalized.csv    # Generated normalization dataset
├── testing/                       # Testing resources
│   └── evolution_texts.csv        # Test medical evolution texts
├── results/                       # Analysis results
├── testing_results/               # Model evaluation results
└── pyproject.toml                 # Project dependencies
```

## Data Format

### Input Files

Medical evolution texts must be in CSV or JSON format with the following fields:

- `id`: Unique record identifier
- `evolution_text`: Medical evolution text (in Spanish)
- `principal_diagnostic`: Correct principal diagnosis (for evaluation)

### Configuration

The `config.json` file contains:

- `models`: List of language models for diagnosis extraction
- `optimal_model`: Index of the recommended model in the models list
- `prompts`: Structured prompts for text summarization, diagnosis extraction, and ICD coding

## Results

Results are organized in directories based on the execution mode:

- **Normal mode**: JSON files in the `results/` directory
- **Test mode**: JSON files and visualizations in the `testing_results/` directory

The results format includes:

- Model information (name, size, parameters, quantization level)
- Performance metrics (accuracy, errors, processing time)
- Details of each processed diagnosis
- Performance comparison charts and visualizations
# Medical Evolution Text Analyzer

![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen)

An advanced system for analyzing medical evolution texts that extracts principal diagnoses and ICD codes using language models with RAG (Retrieval Augmented Generation) capabilities.

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

The **Medical Diagnostic Analysis System** is a Python-based application designed to process medical evolution texts, extract principal diagnoses and ICD codes (International Classification of Diseases), and validate these diagnoses against reference data. It uses language models through the Ollama framework to perform advanced semantic analysis of medical texts, with a special focus on rheumatological diseases.

## Features

- **Automatic diagnosis extraction** from clinical notes
- **Diagnosis normalization** using RAG (Retrieval Augmented Generation)
- **ICD code assignment** to extracted diagnoses
- **Parallel processing** to optimize execution time
- **Comprehensive model evaluation** with detailed performance metrics
- **Text expansion** option to improve information extraction
- **Visual performance comparisons** between different models
- **Flexible and powerful command line interface**

## Requirements

- Python 3.10
- [Ollama](https://ollama.ai/) installed and running
- UV package manager

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/medical-diagnostic-analysis.git
   cd medical-diagnostic-analysis
   ```

2. Install uv package manager:
   ```bash
   https://docs.astral.sh/uv/getting-started/installation/
   ```
   
3. Make sure Ollama is install:
   ```bash
   https://ollama.com/download
   ```

4. Install all necesary dependencies:
   ```bash
   uv sync
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
| `-t`, `--test` | Run in test mode to evaluate model performance |
| `-i`, `--installed` | Only use models that are already installed |
| `-v`, `--verbose` | Print detailed output during processing |
| `-E`, `--expand` | Expand evolution texts before processing |
| `-N`, `--normalize` | Normalize results using RAG |

### Execution Examples

```bash
# Run with the optimal model (from config) in standard mode
python main.py

# Run with all installed models in test mode with verbose output
python main.py -tiv

# Select a specific model, use text expansion and normalization
python main.py -EN -m2

# Process 50 texts with 4 parallel batches
python main.py -n8 -b4
```

## Architecture

The system is structured into several main modules:

1. **analyzer.py**: Coordinates the medical text analysis process
2. **_custom_parser.py**: Parses and normalizes extracted diagnoses
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
   - Optionally expanded using a language model
   - Principal diagnoses and ICD codes are extracted

3. **Normalization** (optional):
   - Diagnoses are normalized using RAG (Retrieval Augmented Generation)
   - A Chroma vector database is used to find similar diagnoses
   - Fuzzy matching is used as a fallback

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
├── config.json                    # System configuration
├── icd_dataset.csv                # ICD code dataset
├── testing/                       # Testing resources
│   └── evolution_texts.csv        # Test medical evolution texts
├── results/                       # Analysis results
├── testing_results/               # Model evaluation results
└── pyproject.toml/                # Project requirements
```

## Data Format

### Input Files

Medical evolution texts must be in CSV or JSON format with the following fields:

- `id`: Unique record identifier
- `evolution_text`: Medical evolution text
- `principal_diagnostic`: Correct principal diagnosis (for evaluation)

### Configuration

The `config.json` file contains:

- `models`: List of language models for diagnosis extraction
- `optimal_model`: Index of the recommended model in the models list
- `prompts`: Structured prompts for text expansion, diagnosis extraction, and ICD coding

## Results

Results are organized in directories based on the execution mode:

- **Normal mode**: JSON files in the `results/` directory
- **Test mode**: JSON files in the `testing_results/` directory with visualizations

The results format includes:

- Model information (name, size, parameters)
- Performance metrics (accuracy, errors, processing time)
- Details of each processed diagnosis
- Performance comparison charts and visualizations

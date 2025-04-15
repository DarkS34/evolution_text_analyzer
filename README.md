# Medical Evolution Text Analyzer

An advanced system for analyzing medical evolution texts that extracts principal diagnoses and ICD codes using language models.

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

The **Medical Evolution Text Analyzer** is a Python-based system designed to process medical evolution texts, extract principal diagnoses and ICD codes (International Classification of Diseases), and validate these diagnoses against reference data. It uses language models through the Ollama framework to perform advanced semantic analysis of medical texts, with a special focus on rheumatological diseases.

## Features

- **Automatic diagnosis extraction** from clinical notes
- **Diagnosis normalization** using RAG (Retrieval Augmented Generation)
- **ICD code assignment** to extracted diagnoses
- **Parallel processing** to optimize execution time
- **Accuracy evaluation** of different language models
- **Text expansion** option to improve information extraction
- **Flexible and powerful command line interface**

## Requirements

- Python 3.10
- [Ollama](https://ollama.ai/) installed and running
- UV package manager

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/medical-evolution-text-analyzer.git
   cd medical-evolution-text-analyzer
   ```

2. Install dependencies using UV:
   ```bash
   pip install uv
   ```
   
3. Sync dependencies:
   ```bash
   uv sync
   ```

3. Make sure Ollama is running:
   ```bash
   ollama start
   ```

## 🚀 Usage

The main script is executed through the command line with various arguments to customize the analysis.

### Command Line Arguments

```
python main.py [options]
```

| Argument | Description |
|-----------|-------------|
| `-m`, `--mode` | Operation mode: `1` for all models, `2` for model selection (default: `1`) |
| `-b`, `--batches` | Number of batches for parallel processing (default: `1`) |
| `-n`, `--num-texts` | Number of texts to process (default: all) |
| `-t`, `--test` | Test mode |
| `-i`, `--installed` | Use only installed models |
| `-v`, `--verbose` | Verbose mode |
| `-E`, `--expand` | Expand evolution texts |
| `-N`, `--normalize` | Normalize results using RAG |

### Execution Examples

```bash
# Select a specific model, test mode, only use installed models, verbose mode
python main.py -tiv -m2

# Run in test mode with text expansion and normalization, with only installed models
python main.py -tiEN -m2
```

## Architecture

The system is structured into several main modules:

1. **analyzer.py**: Coordinates the medical text analysis process
2. **_custom_parser.py**: Parses and normalizes extracted diagnoses
3. **auxiliary_functions.py**: Provides auxiliary functions for data handling
4. **_validator.py**: Validates diagnosis results
5. **tester.py**: Evaluates model accuracy
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

4. **Validation**:
   - Extracted diagnoses are compared with reference values
   - Metrics for accuracy, errors, and incorrect outputs are calculated

5. **Results**:
   - Results are stored in JSON files
   - Detailed performance metrics are provided
   - Visualizations are generated for model comparison

## Directory Structure

```
.
├── evolution_text_analyzer/
│   ├── __init__.py
│   ├── analyzer.py                # Main analysis logic
│   ├── _custom_parser.py          # Diagnosis parser
│   ├── _validator.py              # Results validation
│   ├── auxiliary_functions.py     # Auxiliary functions
│   ├── data_models.py             # Pydantic data models
│   ├── results_manager.py         # Results management and visualization
│   └── tester.py                  # Model evaluation
├── main.py                        # Main entry point
├── config.json                    # System configuration
├── icd_dataset.csv                # ICD code dataset
├── evolution_texts_resolved.csv   # Medical evolution texts
└── README.md                      # Documentation
```

## Data Format

### Input Files

Medical evolution texts must be in CSV or JSON format with the following fields:

- `id`: Unique record identifier
- `evolution_text`: Medical evolution text
- `principal_diagnostic`: Correct principal diagnosis (for evaluation)

### Configuration

The `config.json` file must contain:

- `models`: List of models to evaluate
- `prompts`: List of prompts to use with models
- `optimal_model`: Index of the optimal model

## Results

Results are stored in directories based on the execution mode:

- **Normal mode**: JSON files in the `results/` directory
- **Test mode**: JSON files in the `testing_results/` directory with visualizations

The results format includes:

- Model information (name, size, parameters)
- Performance metrics (accuracy, errors, processing time)
- Details of each processed diagnosis
- Performance comparison charts and visualizations

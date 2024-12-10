# Medical Evolution Text Analyzer

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Directory Structure](#directory-structure)
---

## Overview
The **Medical Evolution Text Analyzer** is a Python-based system designed to process medical evolution texts, validate diagnoses, and analyze performance across different language models. It provides detailed results for reumatological disease diagnostics based on ICD codes and principal diseases.

---

## Features
- **Parallel Text Processing**: Efficiently processes medical evolution texts in batches.
- **Model Compatibility**: Works with multiple language models using the Ollama LLM framework.
- **Validation Mechanism**: Validates results against known diagnoses for accuracy.
- **Comprehensive Output**: Generates performance metrics including accuracy, errors, and processing time.

---

## Installation
1. Clone the repository:

2. Ensure you have uv package manager and Python 3.9+ installed.
    ```bash
    pip install uv
    ```

3. Install required Python packages with **uv**:
    ```bash
    uv sync
    ```

4. Ensure the **Ollama** framework is running locally:
    ```bash
    ollama start
    ```

---

## Usage
Run the main script with the appropriate arguments:

### Modes
- **Mode 1**: Evaluate all models listed in `models.json`.
- **Mode 2**: Choose a specific model for evaluation.

### Command-Line Arguments
- `-mode`: Specify operation mode (`1` or `2`, by default: `1`).
- `-batches`: Number of batches for parallel processing (by default: 5).

### Example
```bash
python main.py -mode 1 -batches 10
```

---

## Directory structure

```bash
.
├── analyzer/
│   ├── auxiliary_functions.py   # Helper functions
│   ├── parallel_ollama_et_analyzer.py   # Core processing logic
│   ├── validator.py   # Validation of diagnostic results
├── main.py   # Entry point of the application
├── models.json   # List of models for evaluation
├── [evolution_texts].csv   # Input medical evolution texts
├── results/   # Directory for output results
└── README.md   # Project documentation
```

## Additional information

- The **evolution_texts** file can be in either `.json` or `.csv` format. It must contain three main fields: `ID`, `principal_diagnostic`, and `evolution_text`.
- The script creates a **results** directory. When a model finishes processing the data, it generates a `detailedResults` file inside this directory. Additionally, the output includes the ICD code for each evolution text.
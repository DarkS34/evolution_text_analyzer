# Medical Evolution Text Analyzer

## Table of Contents
1. [Overview](#overview)
2. [Process Explanation](#process-explanation)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Directory Structure](#directory-structure)
---

## Overview
The **Medical Evolution Text Analyzer** is a Python-based system designed to process medical evolution texts, validate diagnoses, and analyze performance across different language models. It provides detailed results for reumatological disease diagnostics based on ICD codes and principal diseases.

---

## Process Explanation

The script follows a structured process to analyze medical evolution texts using various llms. Below is a detailed step-by-step explanation:

### 1. Initialization
- When the script starts, it connects to the **Ollama framework** to ensure the required environment is active.
- It parses the input arguments to determine the operational mode (`-mode 1` or `-mode 2`, by default 1) and the number of batches for processing (`-batches`, by default 5).

### 2. Mode Selection
- **Mode 1**: 
  - The program automatically iterates through all models listed in the `models.json` file.
  - Each model is processed sequentially using a `for` loop.
- **Mode 2**: 
  - The program presents a list of models to the user.
  - The user selects a specific model for processing.

### 3. Batch Processing
- The script processes input data in batches to optimize execution time.
- It uses the **RunnableParallel** functionality from LangChain to process multiple records in parallel.
- This parallelization significantly reduces processing time compared to sequential execution.

### 4. Evolution Texts Analysis
For each batch of input data, the program:
1. Passes the medical evolution texts to the selected model.
2. Extracts diagnostic information, including:
   - `principal_diagnostic`: The principal diagnosis.
   - `icd_code`: The corresponding International Classification of Diseases code.
3. Validates the model's output against the known correct diagnosis using a validation function.

### 5. Result Tracking
After processing each model (in both modes 1 and 2), the results are stored in a `detailedResults.json` file located in the `results` directory. Results file includes the following information per processed model:
- Model information (name, size, parameter_size, and quantization_level).
- Performance metrics include:
  - Accuracy.
  - Percentage of incorrect outputs.
  - Percentage of errors.
  - Processing time for each model.
- All processed data. 

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
python main.py -batches 10
```
Or:
```bash
uv run main.py -batches 10
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
- There is no need to include all fields in the **models.json file**. 
    - Only **required** field is `modelName`.
    - Other fields (`size`, `parameter_size`, and `quantization_level`) are **optional**.
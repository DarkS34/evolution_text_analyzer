# Evolution Texts Analyzer

## Description

**Evolution Texts Analyzer** is a tool designed to process texts using various installed models. It offers different operational modes, allowing you to manually select models or automate the data processing.

## Setup
To install all the necessary dependencies, run the following command in the root directory of the repository:

```bash
pip install -r requirements.txt
```

**Evolution texts** file (`.csv` or `.json`) must be in the root directory of the repository.

## Configuration Tags
Tags are used to customize the behavior of the script. Below are the main tags you can use:

| Tag | Description |
| ---- | ---------- |
| `-mode` | Defines the operation mode of the script. See the next section for more details.|
| `-installed` | Limits the script to process only installed models. |
| `-silent` |Suppresses terminal output (except errors). Useful for background processing. |
>### `-mode` Tag Explanation
>| Mode | Description |
>| ---- | ---------- |
>| `-mode 1` | Allows the user to manually select a model to process the data. |
>| `-mode 2` | Automatically iterates through all available models and processes the data, saving the results. |

## Usage
To run the script with your desired tags, execute the following command:
```bash
python script.py -mode 2 -silent
```
This command will run the script in automatic mode (`-mode 2`), suppressing terminal output (`-silent`).
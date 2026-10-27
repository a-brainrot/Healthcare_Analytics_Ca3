========== README.md ==========

# Healthcare Analytics: Identifying Diabetes Risk Factors

A Python project using the Pima Indians Diabetes Dataset to explore diabetes risk factors and train a logistic regression model.

## Features

- Dataset inspection and summary statistics
- Median imputation of invalid zero measurements
- Exploratory visualisations and correlation analysis
- Welch’s two-sample t-test for glucose levels
- Logistic regression and performance evaluation

## Required Files

- `run_analysis.py`
- `requirements.txt`
- `README.md`

## How to Run

Install Python 3, download and extract this repository, then open a terminal in the project folder.

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the environment

Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the project

```bash
python run_analysis.py
```

On macOS/Linux, use `python3` instead of `python` when creating the environment if needed.

## Results

The script prints analysis results and evaluation metrics in the terminal. Charts are automatically saved in an `outputs/` folder.

## Dataset

The dataset contains 768 observations, eight predictor variables and a binary diabetes outcome.

An internet connection is required because the script downloads the dataset automatically.

Source: https://github.com/npradaschnor/Pima-Indians-Diabetes-Dataset

## Limitations

Missing-value imputation is performed before the train/test split, which may make evaluation optimistic. This project is intended for educational analysis.

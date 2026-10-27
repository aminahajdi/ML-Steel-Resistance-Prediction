# Machine Learning for Steel Structural Resistance Prediction

Research code developed for *A Systematic Framework for Machine Learning-Based Structural Resistance Prediction* (manuscript in preparation).

This repository contains Python implementations for four benchmark studies investigating machine learning (ML) for steel structural resistance prediction. The analyses cover database assessment, preprocessing, model development, validation, engineering interpretation, and applicability assessment.

## Benchmark studies

| Study | Benchmark database | Main analyses |
|---|---|---|
| **1 — IDEA StatiCa CWC** | Numerical simulations of column web in transverse compression | Data generation, conventional DNN/RF/XGBoost development, database diagnostics, dataset-size sensitivity |
| **2 — Analytical CWC** | EN 1993-1-8 resistance calculations | Data generation, preprocessing, Optuna-based model and hyperparameter optimisation, validation sensitivity, SHAP |
| **3 — Experimental CWC** | Experimental column-web transverse-compression resistance | AutoML comparison, repeated-split analysis, feature engineering, SHAP, yielding- and buckling-specific analyses |
| **4 — Welded joints** | Nonlinear finite-element simulations of welded beam-to-column joints | Fixed DNN, AutoML comparison, preprocessing, dataset-size and validation sensitivity, SHAP, feature ablation |

The studies investigate three levels of ML development: conventional model development, automated model-family and hyperparameter optimisation using Optuna, and full AutoML using AutoGluon, H2O AutoML, and MLJAR.

## Repository structure

```text
ML-Steel-Resistance-Prediction/
├── 01_IDEA_CWC/
│   ├── 01_Data_Generation/
│   ├── 02_Conventional_Model_Development/
│   ├── 03_Dataset_Size_Sensitivity/
│   └── 04_Database_Analysis/
├── 02_Analytical_CWC/
│   ├── 01_Data_Generation/
│   ├── 02_Database_Analysis/
│   ├── 03_Preprocessing/
│   ├── 04_Optuna/
│   ├── 05_Dataset_Size/
│   └── 06_Validation_sensitivity/
├── 03_Experimental_CWC/
│   ├── 01_Database_Analysis/
│   ├── 02_AutoML/
│   ├── 03_Seed_Sensitivity/
│   ├── 04_Feature_Engineering/
│   └── 05_Yielding_vs_Buckling/
├── 04_Welded_Joints/
│   ├── 01_Database_Analysis/
│   ├── 02_Conventional_DNN/
│   ├── 03_AutoML/
│   ├── 04_Preprocessing/
│   ├── 05_Dataset_Size/
│   ├── 06_Validation_Sensitivity/
│   ├── 07_Ablation_Study/
│   └── 08_Explainability/
├── data/
│   └── README.md
├── requirements.txt
└── README.md
```

Individual directories contain the corresponding scripts, methodological details, and available reference results.

## Installation and execution

Clone the repository and install the required Python dependencies:

```bash
git clone https://github.com/aminahajdi/ML-Steel-Resistance-Prediction.git
cd ML-Steel-Resistance-Prediction
pip install -r requirements.txt
```

The repository uses PyTorch, scikit-learn, Optuna, XGBoost, SHAP, and AutoML frameworks, among other libraries. Some frameworks may require separate environments depending on version compatibility.

Place the required datasets in the `data/` directory:

| Study | Dataset location |
|---|---|
| 1 — IDEA StatiCa CWC | `data/study_01_data.xlsx` |
| 2 — Analytical CWC | `data/study_02_data.xlsx` |
| 3 — Experimental CWC | `data/study_03_data.xlsx` |
| 4 — Welded joints | `data/study_04_data.xlsx` |

Data-generation scripts and supporting inputs are provided where applicable. The experimental CWC dataset is not publicly distributed. See `data/README.md` for further information on dataset requirements and availability.

Run the analysis scripts from the repository root, following the instructions in the relevant study-level README.

## Methodology and evaluation

The analyses follow the five-stage ML development framework investigated in the manuscript: database assessment and preprocessing, model development and optimisation, validation, engineering interpretation, and applicability assessment.

The principal benchmark evaluation protocol uses an 80/20 training/test split, with five-fold cross-validation within the training data for model selection. Performance is assessed primarily through prediction-to-reference resistance ratios (bias and coefficient of variation, CoV) and the coefficient of determination (R²).

Additional sensitivity analyses investigate database size, preprocessing, validation settings, model stability, and feature relevance. Study-specific configurations and evaluation procedures are documented in the corresponding directories.

## Data availability and reproducibility

The repository provides analysis scripts and selected reference results generated during the research.

Reproducing individual analyses requires the corresponding databases and compatible software dependencies. Numerical results from stochastic training and time-limited optimisation may vary with random seeds, software versions, and computational resources.

Experimental data and other datasets subject to access restrictions are not redistributed in this repository.

## Citation

This repository was developed in connection with the manuscript:

Hajdarević, A., Ljubinković, F., Vild, M., and Simões da Silva, L. *A Systematic Framework for Machine Learning-Based Structural Resistance Prediction* (manuscript in preparation).

Publication details will be added when available.

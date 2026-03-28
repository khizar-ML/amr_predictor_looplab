# 🧬 Antimicrobial Resistance (AMR) Phenotype Predictor
**By The Elites**  
**Project:** CUI · LoopVerse 2.0 ML Module

## Overview
This repository contains a machine learning pipeline and a Streamlit web application designed to predict Antimicrobial Resistance (AMR) phenotypes based on genome data, antibiotic types, and clinical measurements. The data originates from the BVBRC genome AMR dataset.


## Core Components
* **`exploration copy 3.ipynb`**: The comprehensive, documented Jupyter Notebook that builds the entire ML prediction pipeline. It covers:
  * **Data Loading & Cleaning:** Null value handling to prevent data leakage in training/validation/test splits.
  * **Exploratory Data Analysis (EDA) & Statistical Significance:** Validates relationships between categorical (`Evidence`, `has_measurement`) and continuous (`Measurement Value`) features against target phenotypes. Includes Chi-Square and T-Tests.
  * **Feature Engineering:** Target Encoding high-cardinality features (`Genome Name`), One-Hot Encoding (`Antibiotic`), and Ordinal Mapping (`Measurement Sign`, `Evidence`).
  * **Model Tuning and Training:** Training an XGBoost Classifier via `Optuna`, heavily penalizing the minority class (`Resistant`) to handle extreme class imbalances.
  * **Evaluation:** Analyzing model diagnostics through Feature Importance, threshold optimization, F1-Score, and AUC-ROC.
* **`app/app.py`**: A deployment-ready Streamlit web application interface that implements the trained XGBoost model and preprocessing pipeline for live user inference.

## Web Application Features
* **CSV Data Upload**: Seamlessly accepts BVBRC AMR formatted datasets (requires properties like `Genome Name`, `Antibiotic`, `Measurement Value`, `Evidence`, `Measurement Sign`).
* **Robust Preprocessing Pipeline**: Automatically applies the pre-fitted Target, One-Hot, and Label encoders built in the exploration notebook onto newly ingested data. Force-maps columns structurally for the XGBoost model.
* **Dynamic Inference Mode**: Modifies classification metric evaluations automatically depending on whether the target variable (`Resistant Phenotype`) is present or absent in the uploaded dataset. Falls back to generating probability inferences regardless.
* **Detailed Insights & Classification Reports**: Outputs diagnostic metrics such as Weighted F1-score, AUC-ROC, a plotted Confusion Matrix, and horizontally barred Top 10 Feature Importances when verifiable targets are provided.
* **Downloadable Predictions List**: Offers a one-click CSV download for all generated model predictions (Susceptible/Resistant) along with output confidence probabilities (`y_probs`).

## 👨‍💻 Credits
* Khizar Abbas Khan
* Zeeshan Ahmad
* Muhammad Humais

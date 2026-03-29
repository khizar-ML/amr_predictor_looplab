# 🧬 Antimicrobial Resistance (AMR) Phenotype Predictor
**By The Elites**  
**Project:** CUI · LoopVerse 2.0 ML Module

🔗 **Live Application:** [AMR Phenotype Predictor (Streamlit)](https://amrpredictorlooplab-j7dzjq5aryzicsxzo2jsva.streamlit.app/)

**Presentation:** [Video Presentation Link (Google Drive)](https://drive.google.com/file/d/1Ufxsv2RFu272FilhMZL1h31B7GVLNS5C/view?usp=drive_link)


---

## Table of Contents
1. [Project Description](#1-project-description)
2. [Approach](#2-approach)
3. [Folder Structure](#3-folder-structure)
4. [Results](#4-results)
5. [Final Remarks & Credits](#5-final-remarks--credits)

---

## 1. Project Description
This repository contains a comprehensive machine learning pipeline and a deployment-ready Streamlit web application designed to predict Antimicrobial Resistance (AMR) phenotypes. By ingesting genome data, antibiotic types, and clinical laboratory measurements originating from the **BVBRC genome AMR dataset**, this module identifies strains as either **Resistant** or **Susceptible**.

The aim of this project is to provide a robust predictive interface that natively handles severe class imbalances and high-cardinality genomic data. This results in reliable phenotypic predictions, serving as an effective diagnostic aid to map out susceptibility vectors against antimicrobial treatments.

## 2. Approach
The methodology is meticulously laid out in our primary exploration notebook (`notebooks/exploration copy 3.ipynb`) and operates in the following stages:

* **Data Loading & Rigorous Cleaning:** Null value handling and data stratification to prevent target leakage across the training, validation, and testing sample splits. Ambiguous intermediate phenotypes are structurally dropped to optimize binary prediction confidence.
* **Exploratory Data Analysis (EDA) & Statistical Tests:** Validates relationships between categorical variables (like `Evidence` and measurement presence) and continuous variables (`Measurement Value`) against target phenotypes using Chi-Square and T-Tests.
* **Feature Engineering:**
  * *Target Encoding* applied to high-cardinality features like `Genome Name` to extract structural patterns without dimensionality explosion.
  * *One-Hot Encoding* mapped strictly upon leading `Antibiotic` features.
  * *Ordinal Mapping* implemented for `Measurement Sign` and `Evidence` to secure categorical preservation.
* **Model Tuning and Training:** We initialized an **XGBoost Classifier** guided via **Optuna** to heavily optimize hyperparameter distributions. Crucially, the approach structurally applies a `scale_pos_weight` that heavily penalizes misclassification against the minority class (`Resistant`) to handle extreme imbalances securely.
* **Inference Pipeline:** A Streamlit app (`app/app.py`) reconstructs this entire data processing payload dynamically onto any user-uploaded CSV dataset.

## 3. Folder Structure
* **`app/`**: Contains the Streamlit web application (`app.py`), required web dependencies (`requirements.txt`), and the serialized `.pkl` models/encoders for deployment.
* **`bonus_files/`**: Contains additional supplementary files or experimental scripts beyond the primary prediction scope.
* **`catboost_info/`**: Automatically generated log directories from CatBoost model training runs.
* **`notebooks/`**: Houses the core machine learning pipelines, including EDA, data preprocessing, and model training in Jupyter Notebooks.
* **`reports/`**: Contains diagnostic model evaluations and metric reports (e.g. `model_evaluation_report.md`).
* **`saved_files/`**: Output directory used for storing globally serialized model states (`.pkl` encoders), datasets, or outputs.
* **`visualizations/`**: Stores exported plots, graphs, and visual findings generated during exploratory data analysis and module evaluation.

## 4. Results
The deployed XGBoost model was evaluated specifically against a rigorous validation sample with heavy class imbalance (57,912 Susceptible vs. 25,184 Resistant samples).

* **Accuracy:** **91%** 
* **Weighted F1-Score:** **91%** — Strong validation that the model effectively recognizes both classes without heavily falling into majority class bias.
* **AUC-ROC:** **97%** — Proving excellent separability thresholds between Susceptibility and Resistance parameters.

**Web Dashboard Resiliency:** The live application natively accommodates evaluation omissions. If users upload data with missing Target labels, it gracefully drops back safely to full prediction logic (providing confidence probabilities) without raising structural execution errors.

## 5. Final Remarks & Credits
This predictive module successfully bridges the gap between massive genomic surveillance records and instant predictive evaluations. It highlights a secure implementation that not only maintains high Susceptibility recall but minimizes False Negatives in recognizing Antimicrobial Resistance—serving as a solid foundation capable of supporting healthcare monitoring and antibiotic stewardship heuristics.

**Developed By:**
* Khizar Abbas Khan
* Zeeshan Ahmad
* Muhammad Humais

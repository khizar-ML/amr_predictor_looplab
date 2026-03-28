# Intuition & Approach: AMR Phenotype Prediction

This document outlines the core intuition and reasoning behind the critical design decisions made during the construction of the Antimicrobial Resistance (AMR) machine learning pipeline.

## 1. Feature Selection
**Decision:** Selected a highly specific subset of columns (Genome ID, Genome Name, Antibiotic, Measurement Value, Evidence, Measurement Sign) rather than utilizing the entire BVBRC dataset.

**Intuition:** The raw dataset contained numerous sparsely populated or redundant metadata columns (e.g., specific vendor details, PubMed IDs, testing years). By strictly isolating explicit biological, measurement-based, and methodological traits, we drastically reduced noise and avoided overfitting to irrelevant laboratory administrative strings.

## 2. Handling the Target Variable
**Decision:** Dropped rows where the target variable (`Resistant Phenotype`) is absent instead of imputing it.

**Intuition:** The target variable is the absolute ground truth. Imputing a medical outcome like Antimicrobial Susceptibility or Resistance introduces severe bias and synthetic confidence into the model. If we do not explicitly know the phenotype, we cannot securely train our models or evaluate its accuracy against it.

## 3. Retaining Nulls in Feature Columns
**Decision:** We did NOT impute or drop null values in independent feature columns (e.g., Missing Measurement Values). 

**Intuition:** Nulls in clinical/genomic data often carry structural meaning (e.g., a missing measurement often implies a specific threshold or testing standard wasn't required or met for that specific lab run). Because our selected algorithm (XGBoost) natively learns optimal branching directions for missing values (`NaN`), leaving them untouched securely preserves this implicit "missingness" signal without distorting the data with synthetic mean/median approximations.

## 4. Encoding Strategies
**Decision:** Using an ensemble of distinct encoder pipelines depending strictly on column cardinality and type.

**Intuition:** 
* **Target Encoding (`Genome Name`):** High-cardinality categorical features would drastically overfit or bloat memory if One-Hot Encoded (creating thousands of sub-columns). Target encoding smoothly replaces these strings with the expected probability of the target.
* **One-Hot Encoding (`Antibiotic`):** Safely isolates the top, most frequent antibiotics into binary flags without assuming any mathematical or ordinal relationship between completely different drugs.
* **Ordinal Mapping (`Measurement Sign`, `Evidence`):** Maps logical ordinal relationships (e.g., `<`, `<=`, `=`, `>`, `>=`) directly to scaled numerical weights appropriately so their relational severity is retained.

## 5. Handling Class Imbalance
**Decision:** Applied `scale_pos_weight` directly in the objective function.

**Intuition:** The AMR data features a massive imbalance (Susceptible samples drastically outweigh Resistant ones). Rather than relying on synthetic upsampling (like SMOTE)—which can controversially generate unrealistic genomic intersections—we instructed the algorithm to heavily penalize misclassifications of the minority "Resistant" class during gradient descent. This natively balances output accuracy across both labels.

## 6. Model Selection & Fine-Tuning
**Decision:** Selecting XGBoost guided by Optuna over alternatives like Random Forest, CatBoost, and LightGBM.

**Intuition:** During exploration, we trained and evaluated several robust tree-based ensemble architectures including Random Forests, CatBoost, and LightGBM. **XGBoost definitively stood out**, offering the absolute best tradeoff between peak AUC-ROC performance, inference speed, and seamless handling of sparse, un-imputed datasets. 

We then utilized **Optuna** to rigorously fine-tune the XGBoost hyperparameter space, efficiently discovering the optimal tree depth, learning rates, and subsampling ratios for maximum predictive stability.

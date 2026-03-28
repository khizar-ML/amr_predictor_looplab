# Antimicrobial Resistance (AMR) Prediction Model: Evaluation Report

## 1. Overview
This report evaluates the performance of the deployed XGBoost Classifier for predicting Antimicrobial Resistance phenotypes (Susceptible vs. Resistant). The evaluation is generated across the validated samples within the BVBRC genome AMR dataset excluding ambiguous intermediate phenotypes.

## 2. Key Metrics Summary

* **Accuracy:** 0.91 (91%)
  * *Explanation:* Accuracy measures the overall percentage of correct predictions out of the total predictions made. While highly intuitive, it can sometimes be misleading in imbalanced datasets like ours, but a 91% accuracy shows strong baseline predictability.
* **Weighted F1-Score:** 0.91 (91%)
  * *Explanation:* The F1-Score calculates the harmonic mean of Precision and Recall. The weighted adjustment accounts for the severe class imbalance in our AMR dataset (57,912 Susceptible vs 25,184 Resistant instances) by calculating the metrics for each label and finding their average weighted by support. A 91% weighted F1-score confirms the model effectively recognizes both classes and doesn't superficially guess the majority class.
* **AUC-ROC:** 0.95 (95%)
  * *Explanation:* Area Under the Receiver Operating Characteristic Curve (AUC-ROC). It indicates the model's ability to distinguish between Susceptible (Class 0) and Resistant (Class 1) classes across different probability thresholds. An AUC of 0.95 indicates an excellent degree of separability.

## 3. Confusion Matrix Breakdown

The confusion matrix maps the exact counts of our predicted categorizations against the actual valid labels (Support: 83,096).

### Interpretation:
* **True Negatives (Susceptible correctly predicted):** Instances where the sample was correctly identified as Susceptible. The model successfully identified 91% of the 57,912 Susceptible samples.
* **False Positives (Susceptible incorrectly predicted):** Instances where the model mistakenly predicted a Susceptible sample as Resistant (Type I Error).
* **False Negatives (Resistant incorrectly predicted):** Instances where the model failed to identify a Resistant sample, classifying it as Susceptible instead (Type II Error). In AMR, minimizing False Negatives is crucial.
* **True Positives (Resistant correctly predicted):** Instances where the model correctly identified a Resistant sample. The model successfully identified 91% of the 25,184 Resistant samples.

## 4. Conclusion
The XGBoost model robustly captures Antimicrobial Resistance traits with an impressive recall of 91% for the minority class, efficiently handling the large class imbalance present in the BVBRC genome AMR dataset and minimizing dangerous false negatives.

# Antimicrobial Resistance (AMR) Prediction Model: Evaluation Report

## 1. Overview
This report evaluates the performance of the deployed XGBoost Classifier for predicting Antimicrobial Resistance phenotypes (Susceptible vs. Resistant). The evaluation is generated across the validated samples within the BVBRC genome AMR dataset excluding ambiguous intermediate phenotypes.

## 2. Key Metrics Summary

* **Accuracy:** 0.9074
  * *Explanation:* Accuracy measures the overall percentage of correct predictions out of the total predictions made. While highly intuitive, it can sometimes be misleading in imbalanced datasets like ours.
* **Weighted F1-Score:** 0.9089
  * *Explanation:* The F1-Score calculates the harmonic mean of Precision and Recall. The weighted adjustment accounts for the severe class imbalance in our AMR dataset by calculating the metrics for each label (Resistant, Susceptible), and finding their average weighted by support (the number of true instances for each label).
* **AUC-ROC:** 0.9770
  * *Explanation:* Area Under the Receiver Operating Characteristic Curve (AUC-ROC). It indicates the model's ability to distinguish between Susceptible (Class 0) and Resistant (Class 1) classes across different probability thresholds. An AUC closer to 1.0 indicates an excellent degree of separability.

## 3. Confusion Matrix

The confusion matrix breaks down the exact counts of True Positives, True Negatives, False Positives, and False Negatives:

```text
Actual \ Predicted     Susceptible (0)     Resistant (1)
---------------------------------------------------------
Susceptible (0)        [52496   ]          [5415    ]
Resistant (1)          [2276    ]          [22908   ]
```

### Interpretation of Confusion Matrix:
* **True Negatives (52496):** Instances where the sample was correctly identified as Susceptible.
* **False Positives (5415):** Instances where the model mistakenly predicted a Susceptible sample as Resistant (Type I Error).
* **False Negatives (2276):** Instances where the model failed to identify a Resistant sample, classifying it as Susceptible instead (Type II Error). In AMR, minimizing False Negatives is often crucial.
* **True Positives (22908):** Instances where the model correctly identified a Resistant sample.

## 4. Conclusion
With a customized probability threshold of **0.65**, the model robustly captures Resistance traits while maintaining strong Susceptibility recall. 

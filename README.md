##Telco Customer Churn Prediction

#Track 1 – Tabular Machine Learning

#Final Results

Best Model: XGBoost

ROC-AUC: 0.8598527863233746
Accuracy: 0.79

Classification Report:

              precision    recall  f1-score   support

           0       0.89      0.81      0.85      1035
           1       0.58      0.72      0.64       374

    accuracy                           0.79      1409
   macro avg       0.73      0.77      0.75      1409
weighted avg       0.81      0.79      0.79      1409


Final AUC (Validation): 0.8598

Primary evaluation metric: ROC-AUC

Objective

Predict whether a telecom customer will churn (Yes/No).

Binary classification problem:

0 = No Churn

1 = Churn

Dataset

Telco Customer Churn dataset.
Target variable: Churn.

Features include customer demographics, service usage, and billing information.

Train / Validation Split

80-20 split

Stratified sampling to preserve class distribution

random_state = 42

Implemented using train_test_split

Data Preprocessing

Removed missing values

Converted TotalCharges to numeric

One-hot encoded categorical variables

Converted target variable to binary (0/1)

Models Used

Baseline Model:

Logistic Regression

Improved Model:

XGBoost Classifier

XGBoost selected as final model based on ROC-AUC performance.

Evaluation Criteria

Since churn is imbalanced, multiple metrics were used:

Accuracy

Precision

Recall

F1-score

ROC-AUC (Primary metric)

ROC-AUC used for model comparison because it measures class separability independent of threshold.

Error Analysis

Recall for churn class (1): 0.72
Model identifies 72% of actual churners.

Precision for churn class (1): 0.58
Some false positives present.

Strong performance on non-churn class (0).

Tradeoff chosen to prioritize detecting churners over minimizing false positives.
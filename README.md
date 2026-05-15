# Customer Churn Prediction

A complete end-to-end machine learning pipeline for predicting customer churn using 8 classification models, built and evaluated on Kaggle.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Models Used](#models-used)
- [Results](#results)
- [Keywords](#key-words)
- [Inference from Results](#inference-from-results)
- [Dependencies Used](#dependencies-used)

---

## Overview

Customer churn — when a customer stops doing business with a company — is one of the most costly problems in subscription-based industries. This project builds a supervised binary classification system to predict whether a customer will churn (`1`) or stay (`0`), using demographic and behavioural features.

The pipeline covers data cleaning, encoding, preprocessing, model training, evaluation via multiple metrics, and feature importance analysis — all structured to avoid common pitfalls like data leakage and double encoding.

---

## Dataset

**Source:** [Customer Churn Dataset — Kaggle](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset)

### About the Dataset
The testing file for the churn dataset consists of 64,374 customer records and serves as a separate dataset for evaluating the performance and generalization capabilities of trained churn prediction models. Each record in the testing file corresponds to a customer and contains the same set of features as the training file, such as age, gender, tenure, usage frequency, support calls, payment delay, subscription type, contract length, total spend, and last interaction. However, the churn labels are not included in the testing file as they are used for assessing the accuracy and effectiveness of the churn prediction models. The testing file allows businesses to assess the predictive power of their trained models on unseen data and gain insights into how well the models generalize to new customers. By analyzing the model's performance on the testing file, businesses can gauge the effectiveness of their churn prediction strategies and make informed decisions to optimize customer retention efforts. 



### Dataset notable points
| Property | Value |
|---|---|
| File | `customer_churn_dataset-training-master.csv` |
| Target column | `Churn` (Yes / No → 1 / 0) |
| Split | 80% train / 20% test (stratified) |

### Dataset Parameters with details
<img width="1631" height="815" alt="image" src="https://github.com/user-attachments/assets/4f70b995-2942-4a26-86c1-958e4f9ca0a8" />
<img width="1632" height="817" alt="image" src="https://github.com/user-attachments/assets/cb47586f-c2a5-49c2-8599-65029d968a96" />
<img width="1632" height="371" alt="image" src="https://github.com/user-attachments/assets/7658fcca-caf6-49dd-b20f-0343e96b24d1" />




---

## Project Workflow

<img width="876" height="1024" alt="image" src="https://github.com/user-attachments/assets/b392f928-811d-4f14-9311-fd3d39ec33dc" />



## Models Used

### 1. Logistic Regression
A linear model that estimates the probability of churn using a logistic (sigmoid) function. Fast, interpretable, and a strong baseline for binary classification. Works well when the decision boundary is approximately linear.

### 2. Naive Bayes (Gaussian)
A probabilistic classifier based on Bayes' theorem, assuming features are conditionally independent given the class. Extremely fast to train, performs well with limited data, and handles high-dimensional spaces gracefully — at the cost of the (often unrealistic) independence assumption.

### 3. Decision Tree
A non-linear model that recursively splits the feature space into rectangular regions based on information gain or Gini impurity. Highly interpretable (can be visualised as a tree), but prone to overfitting without pruning.

### 4. Random Forest
An ensemble of decision trees trained on random subsets of the data and features (bagging). Predictions are aggregated by majority vote. Reduces the variance of a single tree, is robust to outliers, and naturally provides feature importances. Used here with 300 estimators.

### 5. Gradient Boosting
An ensemble method that builds trees sequentially, where each tree corrects the residual errors of the previous one. More accurate than bagging-based methods on many tabular datasets, but slower to train.

### 6. XGBoost
An optimised and regularised implementation of gradient boosting with built-in handling for missing values, parallel tree construction, and L1/L2 regularisation. One of the most widely used algorithms in structured data competitions.

### 7. LightGBM
A gradient boosting framework that uses histogram-based splits and leaf-wise (rather than level-wise) tree growth. Significantly faster than XGBoost on large datasets while maintaining comparable accuracy.

### 8. CatBoost
A gradient boosting library with native support for categorical features (no manual encoding required). Uses ordered boosting to reduce overfitting on small datasets and is particularly effective when categorical cardinality is high.

> **Note:** SVM and K-Nearest Neighbours were excluded. Both have O(n²) or worse complexity at prediction/training time and become impractically slow on datasets with tens of thousands of rows.

---

## Results

All models evaluated on the held-out 20% test set. Results sorted by ROC-AUC.
### 1. Algorithm Comparison and Analysis
<img width="494" height="316" alt="image" src="https://github.com/user-attachments/assets/dd5e5dbe-a5a6-497a-8686-0910db0897ad" />

### 2. ROC Curve
<img width="1022" height="779" alt="image" src="https://github.com/user-attachments/assets/e31d845b-6152-4f93-8047-5a71a0eaab29" />

### 3. Confusion Matrices

### i. Logistic Regression
<img width="428" height="397" alt="image" src="https://github.com/user-attachments/assets/bc84eadb-25fd-4da5-9320-054765e0057d" />

### ii. Naive Bayes
<img width="448" height="444" alt="image" src="https://github.com/user-attachments/assets/af701c46-3409-4b86-885f-6932b8ad98b3" />

### iii. Decision Tree
<img width="446" height="439" alt="image" src="https://github.com/user-attachments/assets/b24d5866-e2a0-4b17-bfa6-d93b47abf99a" />

### iv. Random Forest
<img width="498" height="448" alt="image" src="https://github.com/user-attachments/assets/4859b256-ce1b-4ccc-93fe-8b27fd57b4ce" />

### v. CatBoost
<img width="455" height="438" alt="image" src="https://github.com/user-attachments/assets/5d6d2a82-1e13-4b5b-ab0e-6b6bcc87932b" />

### vi.LightGBM
<img width="451" height="453" alt="image" src="https://github.com/user-attachments/assets/b2139312-70b3-4eca-b287-db079be72920" />

### vii. XGBoost
<img width="458" height="449" alt="image" src="https://github.com/user-attachments/assets/d8084c09-bbd4-42cc-85b7-81e5a5c1d74f" />

### viii. Gradient Boosting
<img width="478" height="441" alt="image" src="https://github.com/user-attachments/assets/3a00e023-2706-44c8-b7f8-9c429d77bb12" />

### 4. Feature Importance
<img width="956" height="682" alt="image" src="https://github.com/user-attachments/assets/badb8e2f-bef7-4a0b-a52f-e60cc8b6cdcf" />

---
---
## Key Words

### 1. ROC
ROC (Receiver Operating Characteristic) is a graph that shows how well a binary classifier performs across all classification thresholds.

|Axis| Representation|
|---|---|
|X-axis| False Positive Rate (FPR) — how often the model wrongly predicts positive|
|Y-axis| True Positive Rate (TPR / Recall) — how often the model correctly predicts positive|

A perfect classifier hugs the top-left corner. A random (useless) classifier follows the diagonal line.

### 2. AUC
AUC (Area Under Curve) is a single number that summarizes the ROC curve — literally the area under it.

|AUC Value| Meaning|
|---|---|
|1| Perfect Classifier|
|0.9+ but < 1| Excellent|
|0.7-0.9| Good|
|0.5| Random Guess|
|<0.5|Worst|

AUC is threshold-independent, making it great for comparing models regardless of the cutoff chosen.

### 3. Pipelining
A Pipeline chains preprocessing and modeling steps into a single, reproducible workflow and is used for **Preventing Data Leakage**, **Reproducibility and Clean Code**, **Easy cross-validation and hyperparameter tuning**.

### 4. Confusion Matrix
A Confusion Matrix is a summary table that breaks down the predictions of a classifier against the actual ground-truth labels, revealing not just how many mistakes were made, but what kind.
| Square | Name | Model Predicted | Reality |
|--------|------|-----------------|---------|
|  TP | True Positive | Positive | Positive |
|  TN | True Negative | Negative | Negative |
|  FP | False Positive | Positive | Negative |
|  FN | False Negative | Negative | Positive |

### 4a) Metrics Derived Fron Confustion Matrix

### i. Accuracy
Accuracy is "How often is the model right overall?"


$$\text{Accuracy} = \frac{TP+TN}{Total}$$

### ii. Precision
Precision is "When it predicts positive, how often is it correct?"

$$\text{Precision} = \frac{TP}{TP + FP}$$

### iii. Recall
Recall is "Of all actual positives, how many did it catch?"

$$\text{Recall} = \frac{TP}{TP + FN}$$

### iv. F1 Score
F1 score determines the "Balance between Precision and Recall"

$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### v. Specificity
Specificity is based on "Of all actual negatives, how many did it correctly reject?"

$$\text{Specificity} = \frac{TN}{TN + FP}$$
---

## Inference from Results

**1. Boosting models dominate**
CatBoost, LightGBM, and XGBoost consistently outperform all other models in both accuracy and ROC-AUC. Their ability to model complex non-linear feature interactions makes them well-suited to tabular churn data.

**2. Random Forest is the best interpretable option**
While slightly behind the boosting trio, Random Forest provides meaningful feature importances and is far more robust to overfitting than a single Decision Tree — making it the preferred choice when explainability matters.

**3. Logistic Regression underperforms — the relationship is non-linear**
LR's lower accuracy indicates that the churn decision boundary cannot be well captured by a linear combination of features. This suggests interactions between variables (e.g. high usage + short tenure) that linear models cannot express.

**4. Naive Bayes has the lowest AUC**
The conditional independence assumption is violated in this dataset — features like tenure, usage, and contract type are correlated. This explains NB's relatively poor probability calibration, reflected in its lower ROC-AUC.

**5. Decision Tree overfits without pruning**
An unpruned tree achieves reasonable accuracy but lower AUC compared to ensemble methods, confirming high variance that gets corrected when trees are aggregated.

**6. Feature importance (Random Forest)**
The top features driving churn predictions are typically tenure, contract type, and monthly charges — customers with shorter tenure, month-to-month contracts, and higher charges are significantly more likely to churn. These features should be prioritised in any business retention strategy.

**7. Class imbalance is well-handled by stratification**
Using `stratify=y` in the train/test split ensures both folds mirror the original class distribution, preventing the model from being evaluated on an unrepresentative test set.

---


## Dependencies Used

| Package | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Preprocessing, pipelines, metrics |
| `matplotlib` | Plotting |
| `seaborn` | Feature importance bar chart |
| `xgboost` | XGBoost classifier |
| `lightgbm` | LightGBM classifier |
| `catboost` | CatBoost classifier |

# Supervised Learning. Classification

## 🚀 Goal

This project aims to explore and implement classification models for the **Don'tGetKicked competition** using various machine learning algorithms. The goal is to create models that predict whether a car is a "lemon" (bad buy) or not based on several features, using proper validation techniques and performance metrics.

---

## 📊 Task Overview

1. **Dataset**: Download and process data from the Kaggle **Don'tGetKicked competition**.
2. **Train/Validation/Test Split**: Use the `PurchDate` field to split the dataset into three parts (train, validation, test).
3. **Feature Engineering**: Apply techniques to create additional features and perform feature selection.
4. **Modeling**: Implement and evaluate `Logistic Regression`, `KNN`, and `GaussianNB` models. Compare their performance using **Gini score**, **AUC**, and other metrics.
5. **Hyperparameter Tuning**: Optimize hyperparameters for `ElasticNet` and other models using Grid Search, Random Search, and **Optuna**.
6. **Final Evaluation**: Assess the models based on Gini score, PR-AUC, and other evaluation metrics.

---

## 🧑‍💻 Structure

The project is organized as follows:

### 📂 `data/`
- **raw/** — Original dataset files (e.g., `dont_get_kicked.csv`).
- **processed/** — Preprocessed data ready for modeling.

### 📂 `notebooks/`
- **00_problem_and_metrics.ipynb** — Problem description, metrics overview (Gini, AUC, etc.).
- **01_data_split_and_preprocessing.ipynb** — Train/validation/test split and preprocessing.
- **02_baseline_models_sklearn.ipynb** — Implement basic classification models (Logistic Regression, KNN, GaussianNB).
- **03_metrics_gini_auc_pr.ipynb** — Implement and evaluate Gini score, ROC AUC, PR AUC.
- **04_custom_models.ipynb** — Implement custom classifiers for Logistic Regression, KNN, and Naive Bayes.
- **05_feature_engineering.ipynb** — Create new features and assess their impact.
- **06_feature_selection.ipynb** — Feature selection with Lasso, correlation, and other methods.
- **07_hyperparameter_tuning.ipynb** — Hyperparameter tuning with Grid Search, Random Search, and Optuna.
- **08_final_evaluation.ipynb** — Final evaluation and performance comparison.

### 📂 `src/`
- **data_split.py** — Functions for splitting data based on date and other strategies.
- **preprocessing.py** — Data preprocessing functions (e.g., encoding, scaling).
- **metrics.py** — Implementations of Gini score, AUC, Precision, Recall, F1, etc.
- **models/**
  - **logistic_regression.py** — Custom implementation of logistic regression.
  - **knn.py** — Custom KNN implementation.
  - **naive_bayes.py** — Custom Naive Bayes implementation.
- **feature_engineering.py** — Feature engineering functions (e.g., creating new features).

---

## 🚧 Instructions

1. **Clone this repository**:
   ```bash
   git clone https://github.com/yourusername/supervised-learning-classification.git
   cd supervised-learning-classification

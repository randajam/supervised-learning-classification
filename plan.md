## 🧑‍💻 План с галочками

### ✅ 01_data_split_and_preprocessing.ipynb
- [x] Download the dataset and load it into a pandas DataFrame.
- [x] Split data into train, validation, and test based on `PurchDate`.
- [x] Apply LabelEncoding or OneHotEncoding to categorical variables.
- [x] Normalize the features.

### ✅ 02_baseline_models_sklearn.ipynb
- [x] Train **Logistic Regression**, **GaussianNB**, and **KNN** on the training dataset.
- [x] Check the performance using **validation** dataset.
- [x] Calculate the Gini score for each model.

### ✅ 03_metrics_gini_auc_pr.ipynb
- [x] Implement and calculate **Gini score**, **ROC AUC**, **PR AUC**.
- [x] Compare performance of the models using these metrics.

### ✅ 04_custom_models.ipynb
- [x] Implement custom **Logistic Regression**, **KNN**, and **GaussianNB** models.
- [x] Train and evaluate the custom models.
- [x] Compare the results with the sklearn models.

### ✅ 05_feature_engineering.ipynb
- [x] Create new features (e.g., fractions, groupby features).
- [x] Train models with new features and evaluate Gini score.
- [x] Did feature engineering improve the model?

### ✅ 06_feature_selection.ipynb
- [x] Implement **L1 regularization**, **feature selection by correlation**.
- [x] Refit model with selected features and evaluate performance.
- [x] Compare different feature selection methods.

### ✅ 07_hyperparameter_tuning.ipynb
- [x] Implement **Grid Search** and **Random Search** for hyperparameter tuning.
- [x] Implement **Optuna** for hyperparameter optimization.
- [x] Compare results with Grid Search and Random Search.

### ✅ 08_final_evaluation.ipynb
- [x] Evaluate the final model on **training**, **validation**, and **test** datasets.
- [x] Check the performance difference between validation and test set (is the model overfitting?).
- [x] Implement **Recall**, **Precision**, **F1 score**, and **AUC PR** metrics.
- [x] Select the final model based on performance metrics.

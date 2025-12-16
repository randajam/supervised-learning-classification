
---

## 🧑‍💻 План с галочками

Вот тебе подробный план с задачами для **поэтапного выполнения**:

```text
### ✅ 01_data_split_and_preprocessing.ipynb
- [ ] Download the dataset and load it into a pandas DataFrame.
- [ ] Split data into train, validation, and test based on `PurchDate`.
- [ ] Apply LabelEncoding or OneHotEncoding to categorical variables.
- [ ] Normalize the features.

### ✅ 02_baseline_models_sklearn.ipynb
- [ ] Train **Logistic Regression**, **GaussianNB**, and **KNN** on the training dataset.
- [ ] Check the performance using **validation** dataset.
- [ ] Calculate the Gini score for each model.

### ✅ 03_metrics_gini_auc_pr.ipynb
- [ ] Implement and calculate **Gini score**, **ROC AUC**, **PR AUC**.
- [ ] Compare performance of the models using these metrics.

### ✅ 04_custom_models.ipynb
- [ ] Implement custom **Logistic Regression**, **KNN**, and **Naive Bayes** models.
- [ ] Train and evaluate the custom models.
- [ ] Compare the results with the sklearn models.

### ✅ 05_feature_engineering.ipynb
- [ ] Create new features (e.g., fractions, groupby features).
- [ ] Train models with new features and evaluate Gini score.
- [ ] Did feature engineering improve the model?

### ✅ 06_feature_selection.ipynb
- [ ] Implement **L1 regularization**, **feature selection by correlation**.
- [ ] Refit model with selected features and evaluate performance.
- [ ] Compare different feature selection methods.

### ✅ 07_hyperparameter_tuning.ipynb
- [ ] Implement **Grid Search** and **Random Search** for hyperparameter tuning.
- [ ] Implement **Optuna** for hyperparameter optimization.
- [ ] Compare results with Grid Search and Random Search.

### ✅ 08_final_evaluation.ipynb
- [ ] Evaluate the final model on **training**, **validation**, and **test** datasets.
- [ ] Check the performance difference between validation and test set (is the model overfitting?).
- [ ] Implement **Recall**, **Precision**, **F1 score**, and **AUC PR** metrics.
- [ ] Select the final model based on performance metrics.

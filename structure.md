supervised-learning-classification/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   └── dont_get_kicked.csv
│   └── processed/
│
├── notebooks/
│   ├── 00_problem_and_metrics.ipynb
│   ├── 01_data_split_and_preprocessing.ipynb
│   ├── 02_baseline_models_sklearn.ipynb
│   ├── 03_metrics_gini_auc_pr.ipynb
│   ├── 04_custom_models.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_feature_selection.ipynb
│   ├── 07_hyperparameter_tuning.ipynb
│   ├── 08_final_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_split.py
│   ├── preprocessing.py
│   ├── metrics.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── logistic_regression.py
│   │   ├── knn.py
│   │   └── naive_bayes.py
│   └── feature_engineering.py
│
└── experiments/
    └── results_summary.csv

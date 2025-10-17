## Directory Structure

```
iris-mlops-pipeline-w4/
├── .github/
│   └── workflows/
│       ├── ci-dev.yml
│       └── ci-main.yml
├── feature_repo/
│   ├── __init__.py
│   ├── feature_store.yaml
│   ├── iris_features.py
│   └── data_sources.py
├── data/
│   ├── .gitignore
│   └── iris.csv
├── models/
│   └── .gitignore
├── outputs/
│   └── .gitignore
├── tests/
│   ├── __init__.py
│   ├── test_data_validation.py
│   ├── test_model_evaluation.py
│   └── test_feast_integration.py  ← NEW
├── src/
│   ├── __init__.py
│   ├── train.py  ← UPDATED with Feast
│   ├── inference.py  ← UPDATED with Feast
│   ├── utils.py
│   └── feast_utils.py  ← NEW
├── .dvc/
├── .gitignore
├── requirements.txt  ← UPDATED
├── params.yaml
└── README.md
```

## Descriptions


import pandas as pd
import numpy as np
import yaml

def load_config(config_path='params.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def validate_data_schema(df, required_columns):
    """Validate that dataframe has required columns"""
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return True

def check_data_quality(df):
    """Check data quality metrics"""
    checks = {
        'null_values': df.isnull().sum().sum() == 0,
        'duplicate_rows': df.duplicated().sum() == 0,
        'valid_shape': df.shape[0] > 0 and df.shape[1] > 0
    }
    return checks

import pytest
import pandas as pd
import numpy as np
import yaml
import os

def load_params():
    """Load parameters"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

@pytest.fixture
def iris_data():
    """Fixture to load iris data"""
    params = load_params()
    data_path = params['data']['raw_data']
    df = pd.read_csv(data_path)
    return df

@pytest.fixture
def validation_params():
    """Fixture to load validation parameters"""
    params = load_params()
    return params['validation']

class TestDataValidation:
    """Test suite for data validation"""
    
    def test_data_file_exists(self):
        """Test if data file exists"""
        params = load_params()
        data_path = params['data']['raw_data']
        assert os.path.exists(data_path), f"Data file not found at {data_path}"
    
    def test_data_not_empty(self, iris_data):
        """Test if dataset is not empty"""
        assert len(iris_data) > 0, "Dataset is empty"
    
    def test_minimum_samples(self, iris_data, validation_params):
        """Test if dataset has minimum required samples"""
        min_samples = validation_params['min_samples']
        assert len(iris_data) >= min_samples, \
            f"Dataset has {len(iris_data)} samples, minimum required: {min_samples}"
    
    def test_required_columns(self, iris_data, validation_params):
        """Test if all required columns are present"""
        required_cols = validation_params['required_columns']
        missing_cols = set(required_cols) - set(iris_data.columns)
        assert len(missing_cols) == 0, f"Missing columns: {missing_cols}"
    
    def test_no_null_values(self, iris_data):
        """Test if dataset has no null values"""
        null_count = iris_data.isnull().sum().sum()
        assert null_count == 0, f"Dataset contains {null_count} null values"
    
    def test_no_duplicate_rows(self, iris_data):
        """Test if dataset has no duplicate rows"""
        dup_count = iris_data.duplicated().sum()
        assert dup_count == 0, f"Dataset contains {dup_count} duplicate rows"
    
    def test_numeric_columns_type(self, iris_data):
        """Test if numeric columns have correct data type"""
        numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(iris_data[col]), \
                f"Column {col} is not numeric"
    
    def test_species_values(self, iris_data):
        """Test if species column has valid values"""
        valid_species = ['setosa', 'versicolor', 'virginica']
        unique_species = iris_data['species'].unique()
        for species in unique_species:
            assert species in valid_species, \
                f"Invalid species value: {species}"
    
    def test_positive_values(self, iris_data):
        """Test if numeric columns have positive values"""
        numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for col in numeric_cols:
            assert (iris_data[col] > 0).all(), \
                f"Column {col} contains non-positive values"
    
    def test_data_ranges(self, iris_data):
        """Test if data values are within reasonable ranges"""
        # Sepal length typically 4-8 cm
        assert iris_data['sepal_length'].between(3, 10).all(), \
            "Sepal length out of expected range"
        
        # Sepal width typically 2-5 cm
        assert iris_data['sepal_width'].between(1, 6).all(), \
            "Sepal width out of expected range"
        
        # Petal length typically 1-7 cm
        assert iris_data['petal_length'].between(0, 8).all(), \
            "Petal length out of expected range"
        
        # Petal width typically 0.1-3 cm
        assert iris_data['petal_width'].between(0, 4).all(), \
            "Petal width out of expected range"
    
    def test_balanced_classes(self, iris_data):
        """Test if classes are reasonably balanced"""
        class_counts = iris_data['species'].value_counts()
        min_count = class_counts.min()
        max_count = class_counts.max()
        
        # Check if ratio is not more than 3:1
        ratio = max_count / min_count
        assert ratio <= 3.0, \
            f"Classes are imbalanced with ratio {ratio:.2f}:1"

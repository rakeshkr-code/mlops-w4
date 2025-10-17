import pytest
import pandas as pd
import os
import yaml
from datetime import datetime
import sys
# sys.path.insert(0, 'src')

from src.feast_utils import (
    initialize_feast_store,
    prepare_iris_data_for_bigquery
)

def load_params():
    """Load parameters"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

@pytest.fixture
def feast_store():
    """Fixture to initialize Feast store"""
    try:
        fs = initialize_feast_store()
        return fs
    except Exception as e:
        pytest.skip(f"Feast store not available: {e}")

@pytest.fixture
def iris_data():
    """Fixture to load iris data"""
    params = load_params()
    data_path = params['data']['raw_data']
    df = pd.read_csv(data_path)
    return df

class TestFeastIntegration:
    """Test suite for Feast integration"""
    
    def test_feast_repo_exists(self):
        """Test if Feast repository exists"""
        assert os.path.exists('feature_repo'), "Feast feature_repo directory not found"
        assert os.path.exists('feature_repo/feature_store.yaml'), \
            "feature_store.yaml not found"
    
    def test_feast_feature_definitions_exist(self):
        """Test if Feast feature definitions exist"""
        assert os.path.exists('feature_repo/iris_features.py'), \
            "iris_features.py not found"
    
    def test_feast_store_initialization(self, feast_store):
        """Test if Feast store can be initialized"""
        assert feast_store is not None, "Feast store initialization failed"
    
    def test_prepare_data_for_bigquery(self, iris_data):
        """Test data preparation for BigQuery"""
        feature_df = prepare_iris_data_for_bigquery(iris_data)
        
        # Check required columns exist
        assert 'sample_id' in feature_df.columns, "sample_id column missing"
        assert 'event_timestamp' in feature_df.columns, "event_timestamp column missing"
        assert 'sepal_length' in feature_df.columns, "sepal_length column missing"
        
        # Check data types
        assert feature_df['sample_id'].dtype == 'int64', "sample_id should be int64"
        
        # Check no null values
        assert feature_df.isnull().sum().sum() == 0, "Feature data contains null values"
    
    def test_feast_feature_list(self, feast_store):
        """Test if Feast has correct features registered"""
        feature_views = feast_store.list_feature_views()
        
        feature_view_names = [fv.name for fv in feature_views]
        assert 'iris_features' in feature_view_names, \
            "iris_features view not found in Feast"
    
    def test_feast_entity_definition(self, feast_store):
        """Test if Feast entity is properly defined"""
        entities = feast_store.list_entities()
        entity_names = [e.name for e in entities]
        
        assert 'sample_id' in entity_names, \
            "sample_id entity not found in Feast"
    
    def test_feast_offline_store_config(self, feast_store):
        """Test if offline store (BigQuery) is configured"""
        config = feast_store.config
        
        assert config.offline_store.type == 'bigquery', \
            "Offline store should be BigQuery"

    def test_feature_data_schema(self, iris_data):
        """Test if feature data has correct schema for Feast"""
        feature_df = prepare_iris_data_for_bigquery(iris_data)
        
        expected_features = ['sepal_length', 'sepal_width', 
                            'petal_length', 'petal_width']
        
        for feature in expected_features:
            assert feature in feature_df.columns, \
                f"Feature {feature} missing from prepared data"

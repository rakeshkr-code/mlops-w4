import pytest
import pandas as pd
import numpy as np
import joblib
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_params():
    """Load parameters"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

@pytest.fixture
def trained_model():
    """Fixture to load trained model"""
    params = load_params()
    model_path = params['model']['model_path']
    
    if not os.path.exists(model_path):
        pytest.skip(f"Model not found at {model_path}")
    
    model = joblib.load(model_path)
    return model

@pytest.fixture
def test_data():
    """Fixture to load and prepare test data"""
    params = load_params()
    data_path = params['data']['raw_data']
    df = pd.read_csv(data_path)
    
    # Map species to numeric
    species_map = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    df['species'] = df['species'].map(species_map)
    
    X = df.drop('species', axis=1)
    y = df['species']
    
    # Split data
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=params['train']['test_size'],
        random_state=params['train']['random_state']
    )
    
    return X_test, y_test

class TestModelEvaluation:
    """Test suite for model evaluation"""
    
    def test_model_file_exists(self):
        """Test if model file exists"""
        params = load_params()
        model_path = params['model']['model_path']
        assert os.path.exists(model_path), f"Model file not found at {model_path}"
    
    def test_model_can_predict(self, trained_model, test_data):
        """Test if model can make predictions"""
        X_test, _ = test_data
        predictions = trained_model.predict(X_test)
        assert len(predictions) == len(X_test), "Prediction length mismatch"
    
    def test_prediction_shape(self, trained_model, test_data):
        """Test if predictions have correct shape"""
        X_test, _ = test_data
        predictions = trained_model.predict(X_test)
        assert predictions.shape == (len(X_test),), "Incorrect prediction shape"
    
    def test_prediction_values(self, trained_model, test_data):
        """Test if predictions are valid class labels"""
        X_test, _ = test_data
        predictions = trained_model.predict(X_test)
        valid_classes = [0, 1, 2]
        assert all(pred in valid_classes for pred in predictions), \
            "Invalid class predictions"
    
    def test_minimum_accuracy(self, trained_model, test_data):
        """Test if model meets minimum accuracy requirement"""
        params = load_params()
        min_accuracy = params['validation']['min_accuracy']
        
        X_test, y_test = test_data
        predictions = trained_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        assert accuracy >= min_accuracy, \
            f"Model accuracy {accuracy:.4f} is below minimum {min_accuracy}"
    
    def test_precision_score(self, trained_model, test_data):
        """Test if model has reasonable precision"""
        X_test, y_test = test_data
        predictions = trained_model.predict(X_test)
        precision = precision_score(y_test, predictions, average='weighted')
        
        assert precision >= 0.7, \
            f"Model precision {precision:.4f} is too low"
    
    def test_recall_score(self, trained_model, test_data):
        """Test if model has reasonable recall"""
        X_test, y_test = test_data
        predictions = trained_model.predict(X_test)
        recall = recall_score(y_test, predictions, average='weighted')
        
        assert recall >= 0.7, \
            f"Model recall {recall:.4f} is too low"
    
    def test_f1_score(self, trained_model, test_data):
        """Test if model has reasonable F1 score"""
        X_test, y_test = test_data
        predictions = trained_model.predict(X_test)
        f1 = f1_score(y_test, predictions, average='weighted')
        
        assert f1 >= 0.7, \
            f"Model F1 score {f1:.4f} is too low"
    
    def test_no_constant_predictions(self, trained_model, test_data):
        """Test if model doesn't predict only one class"""
        X_test, _ = test_data
        predictions = trained_model.predict(X_test)
        unique_predictions = np.unique(predictions)
        
        assert len(unique_predictions) > 1, \
            "Model predicts only one class"
    
    def test_model_attributes(self, trained_model):
        """Test if model has required attributes"""
        assert hasattr(trained_model, 'predict'), \
            "Model missing predict method"
        assert hasattr(trained_model, 'predict_proba'), \
            "Model missing predict_proba method"
    
    def test_probability_predictions(self, trained_model, test_data):
        """Test if probability predictions sum to 1"""
        X_test, _ = test_data
        probabilities = trained_model.predict_proba(X_test)
        
        # Check if probabilities sum to 1 for each sample
        prob_sums = probabilities.sum(axis=1)
        assert np.allclose(prob_sums, 1.0), \
            "Probabilities don't sum to 1"
    
    def test_metrics_file_exists(self):
        """Test if metrics file exists"""
        assert os.path.exists('metrics.json'), \
            "Metrics file not found"

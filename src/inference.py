import pandas as pd
import numpy as np
import joblib
import yaml
import json
import os
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
from feast_utils import (
    initialize_feast_store,
    get_online_features
)

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def load_model(model_path):
    """Load trained model"""
    model = joblib.load(model_path)
    return model

def load_data(data_path):
    """Load test data"""
    df = pd.read_csv(data_path)
    return df

def fetch_online_features_from_feast(sample_ids):
    """
    Fetch features from Feast online store (Firestore) for inference
    """
    print("Fetching features from Feast online store (Firestore)...")
    
    # Initialize Feast
    fs = initialize_feast_store()
    
    # Get online features
    feature_vector = get_online_features(fs, sample_ids)
    
    # Convert to DataFrame
    feature_df = pd.DataFrame({
        'sepal_length': feature_vector['sepal_length'],
        'sepal_width': feature_vector['sepal_width'],
        'petal_length': feature_vector['petal_length'],
        'petal_width': feature_vector['petal_width'],
    })
    
    print(f"Retrieved {len(feature_df)} samples from online store")
    
    return feature_df

def preprocess_data(df):
    """Preprocess the data"""
    species_map = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    df['species'] = df['species'].map(species_map)
    
    X = df.drop('species', axis=1)
    y = df['species']
    
    return X, y

def run_inference(model, X):
    """Run inference on data"""
    predictions = model.predict(X)
    return predictions

def evaluate_predictions(y_true, y_pred):
    """Evaluate predictions"""
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Inference Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    return accuracy

def main():
    """Main inference pipeline with Feast integration"""
    print("=" * 20)
    print("Starting IRIS Inference Pipeline with Feast Feature Store")
    print("=" * 20)
    
    # Load parameters
    params = load_params()
    
    # Load model
    print("\n1. Loading model...")
    model = load_model(params['model']['model_path'])
    
    # Load test data (for labels)
    print("\n2. Loading data...")
    df = load_data(params['data']['raw_data'])
    
    # Get sample IDs
    sample_ids = list(range(len(df)))
    
    # Fetch features from Feast online store
    print("\n3. Fetching features from Feast online store...")
    X = fetch_online_features_from_feast(sample_ids)
    
    # Get true labels
    _, y = preprocess_data(df)
    
    # Run inference
    print("\n4. Running inference...")
    predictions = run_inference(model, X)
    
    # Evaluate
    print("\n5. Evaluating predictions...")
    accuracy = evaluate_predictions(y, predictions)
    
    print("\n" + "=" * 20)
    print("Inference completed!")
    print("=" * 20)

if __name__ == "__main__":
    main()

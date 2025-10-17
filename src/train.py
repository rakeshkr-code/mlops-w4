import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import yaml
import json
from datetime import datetime, timedelta
# import matplotlib.pyplot as plt
# import seaborn as sns
from feast_utils import (
    initialize_feast_store,
    prepare_iris_data_for_bigquery,
    upload_features_to_bigquery,
    materialize_features_to_online_store,
    get_training_features
)

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def load_data(data_path):
    """Load iris dataset"""
    df = pd.read_csv(data_path)
    return df

def setup_feast_features(df, project_id):
    """
    Setup Feast: Upload features to BigQuery and materialize to Firestore
    
    Returns:
        FeatureStore instance and feature DataFrame
    """
    print("\n=== Setting up Feast Feature Store ===")
    
    # Initialize Feast
    fs = initialize_feast_store()
    
    # Prepare data for BigQuery
    feature_df = prepare_iris_data_for_bigquery(df)
    
    # Upload to BigQuery (offline store)
    upload_features_to_bigquery(feature_df, project_id)
    
    # Materialize to Firestore (online store)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    materialize_features_to_online_store(fs, start_date, end_date)
    
    print("=== Feast setup complete ===\n")
    
    return fs, feature_df

def fetch_features_from_feast(fs, df):
    """
    Fetch features from Feast offline store for training
    """
    print("Fetching features from Feast offline store...")
    
    # Create entity DataFrame for Feast
    entity_df = pd.DataFrame({
        'sample_id': range(len(df)),
        'event_timestamp': [datetime.now()] * len(df)
    })
    
    # Get historical features from BigQuery
    feature_df = get_training_features(fs, entity_df)
    
    # Merge with labels
    feature_df['species'] = df['species'].values
    
    print(f"Retrieved {len(feature_df)} samples with {len(feature_df.columns)} features")
    
    return feature_df

def preprocess_data(df):
    """Preprocess the data"""
    # Map species to numeric values
    species_map = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    df['species'] = df['species'].map(species_map)
    
    # Extract features (excluding metadata columns)
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = df[feature_cols]
    y = df['species']
    
    return X, y

def train_model(X_train, y_train, params):
    """Train RandomForest model"""
    model = RandomForestClassifier(
        n_estimators=params['train']['n_estimators'],
        max_depth=params['train']['max_depth'],
        random_state=params['train']['random_state']
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy, y_pred

# def plot_confusion_matrix(y_test, y_pred, timestamp):
#     """Plot and save confusion matrix"""
#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.title('Confusion Matrix')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
    
#     output_dir = f"outputs/{timestamp}"
#     os.makedirs(output_dir, exist_ok=True)
#     plt.savefig(f"{output_dir}/confusion_matrix.png")
#     plt.close()

def save_metrics(accuracy, timestamp):
    """Save metrics to JSON"""
    output_dir = f"outputs/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = {
        'accuracy': float(accuracy),
        'timestamp': timestamp,
        'feature_store': 'feast',
        'offline_store': 'bigquery',
        'online_store': 'firestore'
    }
    
    with open(f"{output_dir}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Also save to root for CI/CD
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

def save_model(model, timestamp):
    """Save trained model"""
    output_dir = f"outputs/{timestamp}"
    models_dir = "models"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Save to timestamped output
    joblib.dump(model, f"{output_dir}/iris_model.pkl")
    
    # Save to models directory for latest version
    joblib.dump(model, f"{models_dir}/iris_model.pkl")
    
    print(f"Model saved to {output_dir}/iris_model.pkl")

def main():
    """Main training pipeline with Feast integration"""
    print("=" * 20)
    print("Starting IRIS Training Pipeline with Feast Feature Store")
    print("=" * 20)
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load parameters
    params = load_params()
    gcp_project_id = "rapid-sphinx-473108-u4"
    project_id = params.get('gcp', {}).get('project_id', gcp_project_id)
    
    # Load raw data
    print("\n1. Loading raw data...")
    df = load_data(params['data']['raw_data'])
    print(f"Loaded {len(df)} samples")
    
    # Setup Feast and upload features
    print("\n2. Setting up Feast Feature Store...")
    fs, feature_df_with_meta = setup_feast_features(df, project_id)
    
    # Fetch features from Feast for training
    print("\n3. Fetching features from Feast...")
    feature_df = fetch_features_from_feast(fs, df)
    
    # Preprocess data
    print("\n4. Preprocessing data...")
    X, y = preprocess_data(feature_df)
    
    # Split data
    print("\n5. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params['train']['test_size'],
        random_state=params['train']['random_state']
    )
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # Train model
    print("\n6. Training model...")
    model = train_model(X_train, y_train, params)
    
    # Evaluate model
    print("\n7. Evaluating model...")
    accuracy, y_pred = evaluate_model(model, X_test, y_test)
    
    # # Plot confusion matrix
    # print("\n8. Generating visualizations...")
    # plot_confusion_matrix(y_test, y_pred, timestamp)
    
    # Save metrics
    print("\n9. Saving metrics...")
    save_metrics(accuracy, timestamp)
    
    # Save model
    print("\n10. Saving model...")
    save_model(model, timestamp)
    
    print("\n" + "=" * 20)
    print(f"Training completed! Outputs saved to outputs/{timestamp}/")
    print("=" * 20)

if __name__ == "__main__":
    main()

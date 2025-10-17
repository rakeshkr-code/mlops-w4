import pandas as pd
from feast import FeatureStore
from datetime import datetime
from google.cloud import bigquery
import os

def initialize_feast_store():
    """Initialize Feast feature store"""
    return FeatureStore(repo_path="feature_repo/")

def prepare_iris_data_for_bigquery(df):
    """Prepare iris data for BigQuery ingestion"""
    # Add required columns for Feast
    df = df.copy()
    df['sample_id'] = range(len(df))
    df['event_timestamp'] = pd.Timestamp.utcnow()  # UTC timezone aware ||datetime.now()
    
    # Rename columns to match Feast feature names (remove species for now)
    feature_df = df[['sample_id', 'event_timestamp', 'sepal_length', 
                      'sepal_width', 'petal_length', 'petal_width']].copy()
    
    return feature_df

def upload_features_to_bigquery(df, project_id, dataset_id='feast_iris_dataset', 
                                 table_id='iris_features'):
    """Upload features to BigQuery"""
    client = bigquery.Client(project=project_id)
    
    # Create dataset if it doesn't exist
    dataset_ref = f"{project_id}.{dataset_id}"
    try:
        client.get_dataset(dataset_ref)
        print(f"Dataset {dataset_ref} already exists")
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        client.create_dataset(dataset)
        print(f"Created dataset {dataset_ref}")
    
    # Upload to BigQuery
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
    )
    
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()  # Wait for job to complete
    
    print(f"Loaded {len(df)} rows to {table_ref}")
    return table_ref

def materialize_features_to_online_store(fs, start_date, end_date):
    """Materialize features from offline (BigQuery) to online store (Firestore)"""
    print("Materializing features to online store...")
    fs.materialize(start_date=start_date, end_date=end_date)
    print("Materialization complete!")

def get_training_features(fs, entity_df):
    """
    Fetch features from Feast offline store for training
    
    Args:
        fs: FeatureStore instance
        entity_df: DataFrame with sample_id and event_timestamp
    
    Returns:
        DataFrame with features
    """
    training_df = fs.get_historical_features(
        entity_df=entity_df,
        features=[
            "iris_features:sepal_length",
            "iris_features:sepal_width",
            "iris_features:petal_length",
            "iris_features:petal_width",
        ],
    ).to_df()
    
    return training_df

def get_online_features(fs, sample_ids):
    """
    Fetch features from Feast online store for inference
    
    Args:
        fs: FeatureStore instance
        sample_ids: List of sample IDs
    
    Returns:
        Dictionary with features
    """
    entity_rows = [{"sample_id": sid} for sid in sample_ids]
    
    feature_vector = fs.get_online_features(
        features=[
            "iris_features:sepal_length",
            "iris_features:sepal_width",
            "iris_features:petal_length",
            "iris_features:petal_width",
        ],
        entity_rows=entity_rows,
    ).to_dict()
    
    return feature_vector

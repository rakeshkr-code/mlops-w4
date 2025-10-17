import pandas as pd
from google.cloud import bigquery

# Load IRIS data
df = pd.read_csv('data/iris_data_for_feast.csv')
df["sample_id"] = range(len(df))              # Needed for Feast entity
df["event_timestamp"] = pd.Timestamp.utcnow()    # Needed for Feast timestamp

# BigQuery info
project_id = 'rapid-sphinx-473108-u4'
dataset_id = 'feast_iris_dataset'
table_id = 'iris_features'
table_ref = f"{project_id}.{dataset_id}.{table_id}"

# Upload dataframe as a new table
client = bigquery.Client(project=project_id)
client.load_table_from_dataframe(df, table_ref).result()
print(f"Table created: {table_ref}")

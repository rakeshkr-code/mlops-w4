# from feast import Entity, Feature, FeatureView, BigQuerySource, ValueType
from feast import Entity, Field, FeatureView, BigQuerySource, ValueType
# from feast.data_source.bigquery import BigQuerySource
from datetime import timedelta
from feast.types import Float32

# Define entity
iris_sample = Entity(
    name="sample_id",
    value_type=ValueType.INT64,
    description="Unique identifier for each iris sample"
)

# Define BigQuery data source for features
iris_source = BigQuerySource(
    table="rapid-sphinx-473108-u4.feast_iris_dataset.iris_features",
    timestamp_field="event_timestamp",
    # created_timestamp_column="created_timestamp"
)

# Define feature view
iris_feature_view = FeatureView(
    name="iris_features",
    entities=[iris_sample],
    ttl=timedelta(days=365),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
    ],
    online=True,
    source=iris_source,
    tags={"team": "ml_team"},
)

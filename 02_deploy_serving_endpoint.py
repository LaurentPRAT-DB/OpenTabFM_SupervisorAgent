# Databricks notebook source
# MAGIC %md
# MAGIC # Step 2: Deploy Model Serving Endpoint
# MAGIC
# MAGIC Creates a serverless Model Serving endpoint for the registered TabFM model.

# COMMAND ----------

# MAGIC %pip install mlflow databricks-sdk pandas
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

CATALOG = "serverless_stable_3n0ihb_catalog"
SCHEMA = "hf_tabularpredict"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.tabfm_forecast"
ENDPOINT_NAME = "tabfm-forecast-endpoint"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get latest model version

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()

model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest_version = max(model_versions, key=lambda v: int(v.version))
print(f"Latest version: {latest_version.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create or update serving endpoint

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)

w = WorkspaceClient()

served_entities = [
    ServedEntityInput(
        entity_name=MODEL_NAME,
        entity_version=latest_version.version,
        workload_size="Small",
        scale_to_zero_enabled=True,
    )
]

from databricks.sdk.errors import NotFound, ResourceAlreadyExists

try:
    endpoint = w.serving_endpoints.get(ENDPOINT_NAME)
    print(f"Endpoint '{ENDPOINT_NAME}' exists. Updating...")
    w.serving_endpoints.update_config_and_wait(
        name=ENDPOINT_NAME,
        served_entities=served_entities,
    )
    print("Endpoint updated successfully.")
except NotFound:
    print(f"Creating endpoint '{ENDPOINT_NAME}'...")
    w.serving_endpoints.create_and_wait(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(served_entities=served_entities),
    )
    print("Endpoint created successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the endpoint

# COMMAND ----------

import time

time.sleep(5)

response = w.serving_endpoints.query(
    name=ENDPOINT_NAME,
    dataframe_records=[
        {
            "MedInc": 8.3252,
            "HouseAge": 41.0,
            "AveRooms": 6.984,
            "AveOccup": 2.556,
        },
        {
            "MedInc": 3.5,
            "HouseAge": 20.0,
            "AveRooms": 5.0,
            "AveOccup": 3.0,
        },
    ],
)

print("Endpoint response:")
print(response.predictions)

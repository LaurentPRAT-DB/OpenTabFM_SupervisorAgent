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

import time
from datetime import timedelta

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound, ResourceConflict
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


def wait_for_endpoint_ready(name, timeout_minutes=30):
    """Poll until endpoint has no pending config update."""
    deadline = time.time() + timeout_minutes * 60
    while time.time() < deadline:
        ep = w.serving_endpoints.get(name)
        config_update = getattr(ep, "config_update", None)
        if config_update is None or config_update == "NOT_UPDATING":
            return ep
        print(f"  Endpoint updating... (config_update={config_update})")
        time.sleep(30)
    raise TimeoutError(f"Endpoint '{name}' did not finish updating within {timeout_minutes} min")


try:
    endpoint = w.serving_endpoints.get(ENDPOINT_NAME)
    current_entity = endpoint.config.served_entities[0] if endpoint.config.served_entities else None
    pending = getattr(endpoint, "pending_config", None)

    if current_entity and current_entity.entity_version == latest_version.version:
        print(f"Endpoint '{ENDPOINT_NAME}' already serving v{latest_version.version}. Skipping.")
    elif pending:
        print(f"Endpoint '{ENDPOINT_NAME}' has pending update. Waiting for it to complete...")
        wait_for_endpoint_ready(ENDPOINT_NAME)
        print("Pending update completed.")
    else:
        print(f"Endpoint '{ENDPOINT_NAME}' exists. Updating to v{latest_version.version}...")
        try:
            w.serving_endpoints.update_config_and_wait(
                name=ENDPOINT_NAME,
                served_entities=served_entities,
                timeout=timedelta(minutes=30),
            )
        except ResourceConflict:
            print("Update already in progress. Waiting for completion...")
            wait_for_endpoint_ready(ENDPOINT_NAME)
        print("Endpoint updated successfully.")
except NotFound:
    print(f"Creating endpoint '{ENDPOINT_NAME}'...")
    w.serving_endpoints.create_and_wait(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(served_entities=served_entities),
        timeout=timedelta(minutes=30),
    )
    print("Endpoint created successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify endpoint is ready

# COMMAND ----------

endpoint = w.serving_endpoints.get(ENDPOINT_NAME)
print(f"Endpoint state: {endpoint.state.ready.value}")
assert endpoint.state.ready.value == "READY", f"Endpoint not ready: {endpoint.state}"
print("Endpoint is READY — deployment successful.")

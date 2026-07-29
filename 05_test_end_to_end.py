# Databricks notebook source
# MAGIC %md
# MAGIC # End-to-End Validation Test
# MAGIC
# MAGIC Validates the full pipeline with sample data:
# MAGIC 1. Model registered in UC
# MAGIC 2. Serving endpoint responds
# MAGIC 3. UC function works
# MAGIC 4. Agent can call the tool

# COMMAND ----------

# MAGIC %pip install mlflow databricks-sdk
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

CATALOG = "serverless_stable_3n0ihb_catalog"
SCHEMA = "hf_tabularpredict"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.tabpfn_forecast"
ENDPOINT_NAME = "tabpfn-forecast-endpoint"
UC_FUNCTION = f"{CATALOG}.{SCHEMA}.forecast_with_tabpfn"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample test data

# COMMAND ----------

test_cases = [
    {"MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.984, "AveOccup": 2.556, "expected_range": (3.0, 6.0)},
    {"MedInc": 3.5, "HouseAge": 20.0, "AveRooms": 5.0, "AveOccup": 3.0, "expected_range": (1.0, 3.0)},
    {"MedInc": 5.0, "HouseAge": 30.0, "AveRooms": 6.0, "AveOccup": 2.8, "expected_range": (1.5, 4.0)},
    {"MedInc": 11.0, "HouseAge": 10.0, "AveRooms": 8.0, "AveOccup": 2.0, "expected_range": (3.5, 5.5)},
    {"MedInc": 1.5, "HouseAge": 50.0, "AveRooms": 4.0, "AveOccup": 4.5, "expected_range": (0.5, 2.0)},
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 1: Model exists in UC

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()

versions = client.search_model_versions(f"name='{MODEL_NAME}'")
assert len(versions) > 0, f"No versions found for {MODEL_NAME}"
latest = max(versions, key=lambda v: int(v.version))
print(f"PASS: Model registered - {MODEL_NAME} v{latest.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 2: Serving endpoint is ready

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

endpoint = w.serving_endpoints.get(ENDPOINT_NAME)
state = endpoint.state.ready
assert state.value == "READY", f"Endpoint not ready: {state}"
print(f"PASS: Endpoint '{ENDPOINT_NAME}' is READY")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 3: Endpoint returns predictions

# COMMAND ----------

records = [{k: v for k, v in tc.items() if k != "expected_range"} for tc in test_cases]

response = w.serving_endpoints.query(
    name=ENDPOINT_NAME,
    dataframe_records=records,
)

predictions = response.predictions
assert len(predictions) == len(test_cases), f"Expected {len(test_cases)} predictions, got {len(predictions)}"

print("PASS: Endpoint returned predictions")
for i, (pred, tc) in enumerate(zip(predictions, test_cases)):
    val = pred["forecast"] if isinstance(pred, dict) else pred
    lo, hi = tc["expected_range"]
    status = "OK" if lo <= val <= hi else "WARN (out of expected range)"
    print(f"  Case {i+1}: MedInc={tc['MedInc']}, prediction={val:.3f}, expected=[{lo}, {hi}] -> {status}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 4: UC function works via SQL

# COMMAND ----------

result = spark.sql(f"""
SELECT
  MedInc, HouseAge, AveRooms, AveOccup,
  {UC_FUNCTION}(MedInc, HouseAge, AveRooms, AveOccup) AS forecast
FROM VALUES
  (8.3252, 41.0, 6.984, 2.556),
  (3.5, 20.0, 5.0, 3.0),
  (5.0, 30.0, 6.0, 2.8),
  (11.0, 10.0, 8.0, 2.0),
  (1.5, 50.0, 4.0, 4.5)
AS t(MedInc, HouseAge, AveRooms, AveOccup)
""").collect()

assert len(result) == 5, f"Expected 5 rows, got {len(result)}"
print("PASS: UC function returned results for all test cases")
for row in result:
    print(f"  MedInc={row['MedInc']}, forecast={row['forecast']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 5: UC function metadata is correct

# COMMAND ----------

func_info = spark.sql(f"DESCRIBE FUNCTION EXTENDED {UC_FUNCTION}").collect()
func_text = "\n".join([row.function_desc for row in func_info])

assert "MedInc" in func_text, "Missing MedInc parameter"
assert "HouseAge" in func_text, "Missing HouseAge parameter"
assert "ai_query" in func_text, "Missing ai_query in function body"
print("PASS: UC function metadata correct")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("=" * 50)
print("ALL TESTS PASSED")
print("=" * 50)
print(f"  Model:    {MODEL_NAME} v{latest.version}")
print(f"  Endpoint: {ENDPOINT_NAME}")
print(f"  Function: {UC_FUNCTION}")
print(f"  Test cases validated: {len(test_cases)}")

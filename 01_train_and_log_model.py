# Databricks notebook source
# MAGIC %md
# MAGIC # Step 1: Load Google TabFM from HuggingFace & Register in UC
# MAGIC
# MAGIC This notebook demonstrates using **Google's TabFM** — a zero-shot foundation model
# MAGIC for tabular data — downloaded from HuggingFace and deployed via Databricks.
# MAGIC
# MAGIC - **Model**: [google/tabfm-1.0.0-pytorch](https://huggingface.co/google/tabfm-1.0.0-pytorch)
# MAGIC - **Type**: Zero-shot pretrained transformer for tabular regression/classification
# MAGIC - **Mechanism**: Model weights downloaded from HuggingFace Hub. `fit()` provides
# MAGIC   in-context examples — no gradient updates occur (weights are frozen).
# MAGIC
# MAGIC Architecture: 24-block causal ICL transformer with alternating row/column attention.
# MAGIC Trained on hundreds of millions of synthetic datasets using structural causal models.

# COMMAND ----------

# MAGIC %pip install tabfm[pytorch] mlflow scikit-learn pandas safetensors
# MAGIC %restart_python

# COMMAND ----------

import mlflow
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

CATALOG = "serverless_stable_3n0ihb_catalog"
SCHEMA = "hf_tabularpredict"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.tabfm_forecast"

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load TabFM from HuggingFace
# MAGIC
# MAGIC `tabfm_v1_0_0.load()` downloads the pretrained model weights from
# MAGIC HuggingFace Hub (`google/tabfm-1.0.0-pytorch`).
# MAGIC
# MAGIC No authentication required — model is publicly accessible.

# COMMAND ----------

from tabfm import TabFMRegressor
from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0

# Downloads pretrained weights from HuggingFace Hub (google/tabfm-1.0.0-pytorch)
model = tabfm_v1_0_0.load(model_type="regression")
regressor = TabFMRegressor(model=model, n_estimators=2)

print("TabFM loaded from HuggingFace (google/tabfm-1.0.0-pytorch)")
print("Architecture: 24-block causal ICL transformer")
print("Type: Zero-shot foundation model for tabular regression")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare sample tabular data
# MAGIC
# MAGIC Using California Housing as representative structured data.

# COMMAND ----------

housing = fetch_california_housing(as_frame=True)
df = housing.frame

feature_cols = ["MedInc", "HouseAge", "AveRooms", "AveOccup"]
target_col = "MedHouseVal"

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TabFM uses in-context learning — defaults to 100 context rows per estimator
MAX_CONTEXT_SAMPLES = 100
rng = np.random.RandomState(42)
idx = rng.choice(len(X_train), MAX_CONTEXT_SAMPLES, replace=False)
X_context = X_train.iloc[idx]
y_context = y_train.iloc[idx]

print(f"Context samples (in-context examples for transformer): {len(X_context)}")
print(f"Test samples: {len(X_test)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Provide context to the foundation model
# MAGIC
# MAGIC `fit()` does NOT train the model. It stores context examples that the
# MAGIC pretrained transformer conditions on during inference (like few-shot prompting).

# COMMAND ----------

regressor.fit(X_context, y_context)
print("Context provided to TabFM (zero-shot — no weight updates)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run inference

# COMMAND ----------

X_eval = X_test.head(100)
y_eval = y_test.head(100)
y_pred = regressor.predict(X_eval)
rmse = root_mean_squared_error(y_eval, y_pred)
r2 = r2_score(y_eval, y_pred)
print(f"Test RMSE: {rmse:.4f}")
print(f"Test R²: {r2:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register in Unity Catalog via MLflow sklearn flavor
# MAGIC
# MAGIC TabFM implements the sklearn estimator interface (fit/predict),
# MAGIC so we use `mlflow.sklearn.log_model` — MLflow handles serialization
# MAGIC natively. No custom wrapper class needed.

# COMMAND ----------

from mlflow.models.signature import infer_signature

mlflow.set_registry_uri("databricks-uc")

input_example = X_eval.head(3)
signature = infer_signature(input_example, y_pred[:3])

with mlflow.start_run(run_name="tabfm_v1_hf_foundation_model") as run:
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_param("model_source", "HuggingFace: google/tabfm-1.0.0-pytorch")
    mlflow.log_param("model_type", "zero_shot_foundation_model")
    mlflow.log_param("context_samples", MAX_CONTEXT_SAMPLES)
    mlflow.log_param("n_features", len(feature_cols))
    mlflow.log_param("architecture", "24-block causal ICL transformer")

    model_info = mlflow.sklearn.log_model(
        sk_model=regressor,
        artifact_path="model",
        serialization_format="cloudpickle",
        extra_pip_requirements=["tabfm[pytorch]", "safetensors"],
        signature=signature,
        input_example=input_example,
        registered_model_name=MODEL_NAME,
    )

print(f"Registered in UC: {MODEL_NAME}")
print(f"Model URI: {model_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify the registered model

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
result = loaded_model.predict(X_eval.head(5))
print("Predictions from registered model:")
print(result)

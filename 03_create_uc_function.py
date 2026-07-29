# Databricks notebook source
# MAGIC %md
# MAGIC # Step 3: Create Unity Catalog Function
# MAGIC
# MAGIC Wraps the Model Serving endpoint in a UC function that can be used by
# MAGIC SQL users and Supervisor Agents.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

CATALOG = "serverless_stable_3n0ihb_catalog"
SCHEMA = "hf_tabularpredict"
ENDPOINT_NAME = "tabfm-forecast-endpoint"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the UC function
# MAGIC
# MAGIC This function provides a clean, governed interface to the TabFM model.
# MAGIC Agents and SQL users call this instead of hitting the endpoint directly.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.forecast_with_tabfm(
  MedInc DOUBLE COMMENT 'Median income in block group (in tens of thousands)',
  HouseAge DOUBLE COMMENT 'Median house age in block group',
  AveRooms DOUBLE COMMENT 'Average number of rooms per household',
  AveOccup DOUBLE COMMENT 'Average number of household members'
)
RETURNS STRING
COMMENT 'Predicts median house value using TabFM-v2-reg model. Returns JSON with forecast value.'
RETURN ai_query(
  '{ENDPOINT_NAME}',
  named_struct(
    'MedInc', MedInc,
    'HouseAge', HouseAge,
    'AveRooms', AveRooms,
    'AveOccup', AveOccup
  )
)
""")

print(f"UC Function created: {CATALOG}.{SCHEMA}.forecast_with_tabfm")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the UC function

# COMMAND ----------

result = spark.sql(f"""
SELECT {CATALOG}.{SCHEMA}.forecast_with_tabfm(
  8.3252,
  41.0,
  6.984,
  2.556
) AS forecast_result
""")

result.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test with multiple rows

# COMMAND ----------

result = spark.sql(f"""
SELECT
  MedInc, HouseAge, AveRooms, AveOccup,
  {CATALOG}.{SCHEMA}.forecast_with_tabfm(MedInc, HouseAge, AveRooms, AveOccup) AS forecast
FROM VALUES
  (8.3252, 41.0, 6.984, 2.556),
  (3.5, 20.0, 5.0, 3.0),
  (5.0, 30.0, 6.0, 2.8)
AS t(MedInc, HouseAge, AveRooms, AveOccup)
""")

result.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Describe the function (verify governance metadata)

# COMMAND ----------

spark.sql(f"DESCRIBE FUNCTION EXTENDED {CATALOG}.{SCHEMA}.forecast_with_tabfm").display()

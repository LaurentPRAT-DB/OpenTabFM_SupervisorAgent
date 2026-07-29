# Databricks notebook source
# MAGIC %md
# MAGIC # Step 4: Supervisor Agent Integration
# MAGIC
# MAGIC Demonstrates attaching the UC function to a Supervisor Agent so it can
# MAGIC use Google TabFM as a forecasting tool.

# COMMAND ----------

# MAGIC %pip install databricks-agents mlflow databricks-sdk
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

CATALOG = "serverless_stable_3n0ihb_catalog"
SCHEMA = "hf_tabularpredict"
UC_FUNCTION = f"{CATALOG}.{SCHEMA}.forecast_with_tabfm"
AGENT_ENDPOINT_NAME = "tabfm-supervisor-agent"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Supervisor Agent with forecasting tool

# COMMAND ----------

import mlflow
from databricks.sdk import WorkspaceClient

mlflow.set_registry_uri("databricks-uc")
w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create agent using UC function as tool
# MAGIC
# MAGIC The agent can call `forecast_with_tabfm` when asked to predict house values.

# COMMAND ----------

from mlflow.models import ModelConfig

agent_config = {
    "llm_endpoint": "databricks-claude-sonnet-4",
    "instructions": """You are a forecasting assistant that predicts median house values
in California based on neighborhood features. When asked to predict or forecast house values,
use the forecast_with_tabfm tool.

The tool expects:
- MedInc: Median income (tens of thousands)
- HouseAge: Median house age (years)
- AveRooms: Average rooms per household
- AveOccup: Average household members

Explain your predictions in context.""",
    "tools": [
        {"uc_function": UC_FUNCTION}
    ],
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define agent code

# COMMAND ----------

# MAGIC %%writefile /tmp/tabfm_agent.py

import mlflow
from mlflow.models import set_model


@mlflow.trace
def create_agent():
    from langchain_community.chat_models import ChatDatabricks
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_community.tools.databricks import UCFunctionToolkit

    config = mlflow.models.ModelConfig(development_config=agent_config)

    llm = ChatDatabricks(endpoint=config.get("llm_endpoint"))

    toolkit = UCFunctionToolkit(
        function_names=[config.get("tools")[0]["uc_function"]]
    )
    tools = toolkit.get_tools()

    prompt = ChatPromptTemplate.from_messages([
        ("system", config.get("instructions")),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


agent_executor = create_agent()
set_model(agent_executor)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test agent locally

# COMMAND ----------

import mlflow

# Quick test without full deployment
agent_config_local = agent_config.copy()

with mlflow.start_run(run_name="tabfm_agent_test"):
    mlflow.log_dict(agent_config_local, "agent_config.json")
    print("Agent config logged. To deploy:")
    print(f"  1. Register agent model in UC: {CATALOG}.{SCHEMA}.tabfm_agent")
    print(f"  2. Deploy as agent endpoint: {AGENT_ENDPOINT_NAME}")
    print(f"  3. Agent will use UC function: {UC_FUNCTION}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary: End-to-End Architecture
# MAGIC
# MAGIC ```
# MAGIC User Question
# MAGIC     │
# MAGIC     ▼
# MAGIC Supervisor Agent (LLM)
# MAGIC     │
# MAGIC     ▼
# MAGIC UC Function: forecast_with_tabfm()
# MAGIC     │
# MAGIC     ▼
# MAGIC ai_query → Model Serving Endpoint
# MAGIC     │
# MAGIC     ▼
# MAGIC TabFM-v2-reg (MLflow pyfunc)
# MAGIC     │
# MAGIC     ▼
# MAGIC Forecast Response → Agent → User
# MAGIC ```

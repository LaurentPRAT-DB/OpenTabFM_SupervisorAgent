# Serving Open Weight Models with Databricks

End-to-end guide: download a HuggingFace open weight model, register it in Unity Catalog, deploy a serving endpoint, and expose it as a tool in a Supervisor Agent.

**Example model**: [Prior-Labs/TabPFN-v2-reg](https://huggingface.co/Prior-Labs/tabpfn-v2-regressor) — a pretrained transformer for tabular regression.

## Architecture

```
User Question
    │
    ▼
Supervisor Agent (LLM)
    │
    ▼
UC Function: forecast_with_tabpfn()
    │
    ▼
ai_query() → Model Serving Endpoint
    │
    ▼
TabPFN-v2-reg (MLflow pyfunc)
    │
    ▼
Prediction → Agent → User
```

---

## Step 1: Register the Model in Unity Catalog

**Notebook**: `01_train_and_log_model.py`

### 1.1 Load the open weight model from HuggingFace

```python
from tabpfn import TabPFNRegressor

# Downloads pretrained weights from HuggingFace Hub
regressor = TabPFNRegressor()
```

No training occurs here. TabPFN is a foundation model pretrained on millions of synthetic datasets — `fit()` simply stores in-context examples (like few-shot prompting an LLM).

### 1.2 Provide context examples

```python
# Store context for the transformer (no gradient updates)
regressor.fit(X_context, y_context)
```

### 1.3 Wrap in MLflow pyfunc and register

```python
import mlflow
from mlflow.pyfunc import PythonModel

class TabPFNForecastModel(PythonModel):
    def load_context(self, context):
        with open(context.artifacts["fitted_model"], "rb") as f:
            self.model = pickle.load(f)
        self.feature_cols = ["MedInc", "HouseAge", "AveRooms", "AveOccup"]

    def predict(self, context, model_input, params=None):
        df = pd.DataFrame(model_input)
        predictions = self.model.predict(df[self.feature_cols])
        return pd.DataFrame({"forecast": predictions})
```

Log and register:

```python
mlflow.set_registry_uri("databricks-uc")

mlflow.pyfunc.log_model(
    artifact_path="model",
    python_model=TabPFNForecastModel(),
    artifacts={"fitted_model": model_path},
    pip_requirements=["mlflow", "pandas", "scikit-learn", "tabpfn"],
    signature=signature,
    input_example=input_example,
    registered_model_name="catalog.schema.tabpfn_forecast",
)
```

The model is now versioned and governed in Unity Catalog.

---

## Step 2: Deploy a Model Serving Endpoint

**Notebook**: `02_deploy_serving_endpoint.py`

### 2.1 Create the endpoint

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)

w = WorkspaceClient()

w.serving_endpoints.create_and_wait(
    name="tabpfn-forecast-endpoint",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name="catalog.schema.tabpfn_forecast",
                entity_version="1",
                workload_size="Small",
                scale_to_zero_enabled=True,
            )
        ]
    ),
)
```

Key options:
- **`scale_to_zero_enabled`**: Saves cost when idle; endpoint cold-starts on first request.
- **`workload_size`**: `Small` / `Medium` / `Large` — scales GPU/memory allocation.

### 2.2 Test the endpoint

```python
response = w.serving_endpoints.query(
    name="tabpfn-forecast-endpoint",
    dataframe_records=[
        {"MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.984, "AveOccup": 2.556}
    ],
)
print(response.predictions)
```

---

## Step 3: Create a Unity Catalog Function

**Notebook**: `03_create_uc_function.py`

Wrap the endpoint in a SQL-callable UC function using `ai_query()`:

```sql
CREATE OR REPLACE FUNCTION catalog.schema.forecast_with_tabpfn(
  MedInc DOUBLE COMMENT 'Median income in block group (tens of thousands)',
  HouseAge DOUBLE COMMENT 'Median house age in block group',
  AveRooms DOUBLE COMMENT 'Average number of rooms per household',
  AveOccup DOUBLE COMMENT 'Average number of household members'
)
RETURNS STRING
COMMENT 'Predicts median house value using TabPFN-v2-reg model. Returns JSON with forecast.'
RETURN ai_query(
  'tabpfn-forecast-endpoint',
  named_struct('MedInc', MedInc, 'HouseAge', HouseAge, 'AveRooms', AveRooms, 'AveOccup', AveOccup)
)
```

This gives you:
- **Governance**: UC permissions control who can call the model
- **Discoverability**: Function shows up in catalog with parameter docs
- **SQL access**: Any SQL user can call the model directly
- **Agent integration**: UC functions are first-class tools for agents

Test it:

```sql
SELECT catalog.schema.forecast_with_tabpfn(8.3252, 41.0, 6.984, 2.556) AS forecast
```

---

## Step 4: Use as Tool in a Supervisor Agent

**Notebook**: `04_agent_integration.py`

### 4.1 Define agent configuration

```python
agent_config = {
    "llm_endpoint": "databricks-claude-sonnet-4",
    "instructions": """You are a forecasting assistant that predicts median house values
in California. Use the forecast_with_tabpfn tool when asked to predict house values.

The tool expects:
- MedInc: Median income (tens of thousands)
- HouseAge: Median house age (years)
- AveRooms: Average rooms per household
- AveOccup: Average household members""",
    "tools": [
        {"uc_function": "catalog.schema.forecast_with_tabpfn"}
    ],
}
```

### 4.2 Build the agent with LangChain

```python
from langchain_community.chat_models import ChatDatabricks
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.databricks import UCFunctionToolkit

llm = ChatDatabricks(endpoint="databricks-claude-sonnet-4")

toolkit = UCFunctionToolkit(function_names=["catalog.schema.forecast_with_tabpfn"])
tools = toolkit.get_tools()

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
```

### 4.3 How it works at runtime

1. User asks: *"What's a house worth in a neighborhood with median income $83k, 41-year-old homes, 7 rooms, 2.5 people per household?"*
2. Supervisor Agent parses the request and calls `forecast_with_tabpfn(8.3252, 41.0, 6.984, 2.556)`
3. UC function calls `ai_query()` → hits the serving endpoint
4. Endpoint runs TabPFN inference and returns prediction
5. Agent explains the result in natural language

---

## Prerequisites

- Databricks workspace with Unity Catalog enabled
- `CREATE FUNCTION` privilege on target schema
- Model Serving access (serverless compute)
- Python packages: `tabpfn`, `mlflow`, `databricks-sdk`, `langchain-community`

## Key Concepts

| Concept | Role |
|---------|------|
| **HuggingFace model** | Source of pretrained open weights |
| **MLflow pyfunc** | Packaging format for serving |
| **Unity Catalog model** | Versioned, governed model registry |
| **Model Serving endpoint** | Real-time inference API |
| **UC function + ai_query()** | SQL/agent-friendly wrapper |
| **Supervisor Agent** | LLM that uses the function as a tool |

## File Structure

```
01_train_and_log_model.py       # Load from HF, register in UC
02_deploy_serving_endpoint.py   # Create serving endpoint
03_create_uc_function.py        # Wrap endpoint as UC function
04_agent_integration.py         # Attach to Supervisor Agent
```

# Google TabFM on Databricks: Zero-Shot Tabular Prediction with a Supervisor Agent

End-to-end pipeline demonstrating how to take **Google's TabFM** — a zero-shot foundation model for tabular data — from HuggingFace, deploy it on Databricks with full Unity Catalog governance, and expose it as an intelligent tool through a Supervisor Agent.

## The Model: Google TabFM

**TabFM** (Tabular Foundation Model) is a research release from Google that brings the foundation model paradigm to structured/tabular data.

- **Paper**: [Introducing TabFM: A Zero-Shot Foundation Model for Tabular Data](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/)
- **HuggingFace**: [google/tabfm-1.0.0-pytorch](https://huggingface.co/google/tabfm-1.0.0-pytorch)
- **GitHub**: [google-research/tabfm](https://github.com/google-research/tabfm)
- **License**: Apache 2.0 (code) / tabfm-non-commercial-v1.0 (weights)

### Architecture

TabFM is a **24-block causal In-Context Learning (ICL) transformer** with:
- Alternating **row attention** and **column attention** layers
- Embedding dimension: 256, 8 attention heads
- SwiGLU activation functions
- Trained on **hundreds of millions of synthetic datasets** generated via Structural Causal Models (SCMs)

### How It Works (Zero-Shot)

Unlike traditional ML where you train a model on your data, TabFM uses **in-context learning**:

1. You provide labeled "context" rows (like few-shot examples for an LLM)
2. The pretrained transformer conditions on these examples
3. It generates predictions for new rows in a **single forward pass**
4. **No gradient updates occur** — the model weights remain frozen

This means:
- No hyperparameter tuning
- No cross-validation
- No training pipeline to maintain
- Instant deployment of new prediction tasks

### Performance

On the [TabArena benchmark](https://github.com/autogluon/tabrepo) (51 datasets), TabFM in zero-shot mode **outperforms heavily-tuned supervised baselines** including gradient-boosted decision trees (XGBoost, LightGBM, CatBoost) that require extensive tuning.

---

## Architecture: HuggingFace → Agent

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERACTION                         │
│  "What's a house worth with $83k income, 41yo homes?"       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              SUPERVISOR AGENT (Claude Sonnet 4)             │
│  Parses intent, selects tool, formats parameters            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│          UC FUNCTION: forecast_with_tabfm()                 │
│  Governed, discoverable, SQL-callable interface             │
│  Permissions: catalog.schema.forecast_with_tabfm            │
└─────────────────────┬───────────────────────────────────────┘
                      │ ai_query()
                      ▼
┌─────────────────────────────────────────────────────────────┐
│           MODEL SERVING ENDPOINT                            │
│  tabfm-forecast-endpoint                                    │
│  Serverless • Auto-scaling • Scale-to-zero                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│          MLFLOW SKLEARN (TabFMRegressor)                    │
│  Serialized via sklearn flavor (fit/predict interface)      │
│  Context examples baked in • Runs inference                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│          GOOGLE TabFM v1.0.0 (PyTorch)                      │
│  24-block causal ICL transformer                            │
│  Weights: google/tabfm-1.0.0-pytorch (HuggingFace Hub)      │
└─────────────────────────────────────────────────────────────┘
```

---

## Unity Catalog Integration (Deep Dive)

Unity Catalog (UC) is the governance layer that transforms a raw HuggingFace model into a production-grade, auditable, team-ready asset. Here's how each UC capability is leveraged:

### Model Registry in UC

TabFM implements the sklearn estimator interface (fit/predict), so we use the native sklearn MLflow flavor — no custom wrapper needed:

```python
mlflow.set_registry_uri("databricks-uc")

mlflow.sklearn.log_model(
    sk_model=regressor,
    artifact_path="model",
    pip_requirements=["tabfm[pytorch]", "safetensors", "scikit-learn", "pandas", "numpy"],
    signature=signature,
    input_example=input_example,
    registered_model_name="catalog.schema.tabfm_forecast",
)
```

**What this gives you:**
- **Versioning**: Every model iteration gets a monotonically increasing version number. Roll back instantly if a new context dataset degrades quality.
- **Lineage**: UC tracks which MLflow experiment, run, and artifacts produced each version. Full provenance from HuggingFace source to deployed endpoint.
- **Access Control**: `GRANT USE MODEL ON catalog.schema.tabfm_forecast TO data_scientists` — fine-grained permissions on who can read, deploy, or modify the model.
- **Cross-workspace sharing**: Register once, serve from any workspace in the same metastore.

### UC Function as API Contract

```sql
CREATE OR REPLACE FUNCTION catalog.schema.forecast_with_tabfm(
  MedInc DOUBLE COMMENT 'Median income in block group (tens of thousands)',
  HouseAge DOUBLE COMMENT 'Median house age in block group',
  AveRooms DOUBLE COMMENT 'Average number of rooms per household',
  AveOccup DOUBLE COMMENT 'Average number of household members'
)
RETURNS STRING
COMMENT 'Predicts median house value using Google TabFM zero-shot model.'
RETURN ai_query(
  'tabfm-forecast-endpoint',
  named_struct(
    'MedInc', MedInc,
    'HouseAge', HouseAge,
    'AveRooms', AveRooms,
    'AveOccup', AveOccup
  )
);
```

**What this gives you:**
- **Typed contract**: Parameters have names, types, and human-readable descriptions. Breaking changes are visible at the schema level.
- **Discoverability**: Shows up in the Catalog Explorer. Data analysts find it alongside tables and views — no separate API docs needed.
- **Permissions**: `GRANT EXECUTE ON FUNCTION ... TO analysts` — separate from model permissions. You can let people call the function without granting them access to the raw model or endpoint.
- **Audit trail**: Every invocation is logged in UC system tables. Know who called the model, when, and with what parameters.
- **SQL-native access**: Any tool that speaks SQL (BI dashboards, dbt, notebooks) can call the model without Python SDKs.
- **Agent-ready**: UC functions are first-class tools in Databricks agents — no adapter code needed.

### Governance Chain

```
HuggingFace (open source)
    → MLflow sklearn flavor (packaging + reproducibility)
        → UC Model Registry (versioning + ACL)
            → Model Serving (infra + scaling)
                → UC Function (typed API + audit)
                    → Supervisor Agent (natural language access)
```

At every layer, UC provides:
- **Who** can access it (permissions)
- **What** happened (audit logs)
- **Where** it came from (lineage)
- **When** it changed (versioning)

---

## Serving & Inference Performance

### Model Serving Architecture

Databricks Model Serving for TabFM operates as a **serverless, auto-scaling inference platform**:

| Aspect | Behavior |
|--------|----------|
| **Cold start** | ~30-60s (loads PyTorch model + weights from artifact store) |
| **Warm latency** | ~50-200ms per batch (depends on context size + batch size) |
| **Scale-to-zero** | Endpoint spins down after idle period — zero cost when unused |
| **Auto-scaling** | Horizontally scales replicas under load |
| **Workload sizes** | Small (2 CPU) / Medium (4 CPU) / Large (8 CPU) |

### Inference Speed Characteristics

TabFM's inference speed is governed by:

1. **Context size** (`n_context_rows`): More context = better predictions but slower inference. TabFM uses an attention mechanism over context rows — cost scales roughly O(n²) with context size.

2. **Batch size** (`n_test_rows`): Multiple predictions in one request amortize overhead. Single-row queries pay full per-request cost.

3. **Number of estimators** (`n_estimators`): TabFM can ensemble over random subsets of context rows for improved accuracy. Each estimator = one forward pass. Default: 8.

4. **Feature count** (`n_features`): Column attention scales with feature dimensionality. Up to 500 features supported.

**Typical latency profile (CPU, Small workload):**

| Context rows | Test rows | Estimators | Latency |
|-------------|-----------|------------|---------|
| 100 | 1 | 4 | ~100ms |
| 100 | 100 | 4 | ~200ms |
| 500 | 1 | 8 | ~400ms |
| 500 | 100 | 8 | ~800ms |

### Optimization Strategies

1. **Reduce context size**: TabFM is effective with as few as 50-100 context rows. More isn't always better — diminishing returns after ~500.

2. **Lower `n_estimators`**: Default 8 is conservative. For real-time serving, 2-4 often suffices with minimal accuracy loss.

3. **Batch predictions**: If you have multiple rows to predict, send them in one request rather than individual calls.

4. **Warm pool**: Disable `scale_to_zero` for latency-critical endpoints. Keeps at least one replica warm (~$0.07/hr for Small).

5. **Precomputed context**: The MLflow pyfunc wrapper serializes the fitted model (context included). No re-fitting at serving time — context is baked into the artifact.

### Comparison to Traditional Approaches

| Approach | Training Time | Inference Latency | Maintenance |
|----------|--------------|-------------------|-------------|
| XGBoost (tuned) | Hours (with HPO) | <10ms | Retrain on drift |
| TabFM (zero-shot) | 0 (no training) | 50-400ms | Swap context data |
| AutoML pipeline | Hours-days | <10ms | Full pipeline maintenance |

**TabFM wins when:**
- You need predictions on a **new task** within minutes, not hours
- Your data changes frequently and retraining pipelines are expensive
- You want a **single model** that generalizes across many tabular schemas
- Latency requirements are relaxed (>50ms acceptable)

**Traditional ML wins when:**
- Sub-10ms latency is critical (high-frequency trading, ad serving)
- You have a stable schema with months of labeled data
- Maximum accuracy on a single fixed task justifies tuning effort

### Cost Model

| Configuration | Monthly Cost (estimated) |
|--------------|------------------------|
| Scale-to-zero, ~100 req/day | ~$5-10 |
| Always-warm Small, ~10k req/day | ~$50 |
| Always-warm Medium, ~100k req/day | ~$150 |

TabFM's zero-shot nature means you skip the training compute entirely — no GPU hours for fine-tuning, no HPO sweeps.

---

## Notebooks

| File | Purpose |
|------|---------|
| `01_train_and_log_model.py` | Download TabFM from HuggingFace, provide context examples, evaluate, register in UC |
| `02_deploy_serving_endpoint.py` | Create Model Serving endpoint with auto-scaling |
| `03_create_uc_function.py` | Wrap endpoint as governed UC function via `ai_query()` |
| `04_agent_integration.py` | Attach UC function to Supervisor Agent |
| `05_test_end_to_end.py` | Validate full pipeline: model, endpoint, UC function, predictions |

---

## Installation & Setup

### Prerequisites

- Databricks workspace with Unity Catalog enabled
- [Databricks CLI](https://docs.databricks.com/dev-tools/cli/index.html) installed and configured
- `CREATE FUNCTION` privilege on target schema
- Model Serving access

### Step 1: Clone the repo

```bash
git clone https://github.com/LaurentPRAT-DB/OpenTabFM_SupervisorAgent.git
cd OpenTabFM_SupervisorAgent
```

### Step 2: Configure the Databricks CLI profile

Make sure you have a Databricks CLI profile pointing to your workspace:

```bash
databricks auth login --host https://<your-workspace>.cloud.databricks.com
```

Then update `databricks.yml` → set `workspace.profile` to your profile name.

### Step 3: Deploy with Databricks Asset Bundles (DABs)

```bash
databricks bundle validate
databricks bundle deploy
```

This uploads all notebooks and creates the job with the correct serverless environment (Python 3.12 + all dependencies).

### Step 4: Run the pipeline

```bash
databricks bundle run hf_tabularpredict_pipeline
```

The job runs 5 tasks sequentially:

1. **01_train_and_log_model** — Downloads TabFM from HuggingFace, provides context examples, evaluates, registers model in Unity Catalog
2. **02_deploy_serving_endpoint** — Creates a serverless Model Serving endpoint with auto-scaling
3. **03_create_uc_function** — Wraps the endpoint as a governed UC function via `ai_query()`
4. **04_agent_integration** — Attaches the UC function to a Supervisor Agent
5. **05_test_end_to_end** — Validates the full pipeline end-to-end

### Alternative: Run notebooks individually

You can also run each notebook one by one from the Databricks workspace UI. Upload the `.py` files to your workspace and run them as notebooks (they use Databricks notebook format with `# COMMAND ----------` separators).

### Environment Configuration

The `databricks.yml` bundle config specifies the serverless environment:

```yaml
environments:
  - environment_key: "default"
    spec:
      client: "4"  # Python 3.12 (required for TabFM)
      dependencies:
        - tabfm[pytorch]
        - mlflow
        - scikit-learn
        - pandas
        - safetensors
```

No local Python environment needed — all dependencies are resolved on Databricks serverless compute.

---

## Key Takeaways

1. **Open weight models from HuggingFace integrate natively** with Databricks through MLflow — no custom infrastructure needed.

2. **Unity Catalog provides full governance** at every layer: model versioning, endpoint access control, function permissions, and audit logging.

3. **Zero-shot models like TabFM eliminate the training pipeline** — swap context data to serve entirely new prediction tasks without redeploying.

4. **The UC Function → Agent pattern** makes any ML model accessible via natural language, governed by the same permission model as your data.

5. **Serverless serving + scale-to-zero** keeps costs near zero for experimental/low-traffic use cases while supporting production auto-scaling.

---

## License & Commercial Use

**Code**: Apache 2.0 — use freely, modify, distribute.

**Pretrained weights** (`google/tabfm-1.0.0-pytorch`): `tabfm-non-commercial-v1.0` — **commercial and production use is not permitted** with the default weights.

| Use Case | Allowed? |
|----------|----------|
| Internal research / evaluation | Yes |
| Academic / educational | Yes |
| Proof-of-concept / demo | Yes |
| Production predictions (any industry) | **No** |
| Internal trading models | **No** |
| Customer-facing products | **No** |

### Path to Commercial Use

1. **Train your own weights** — the architecture (Apache 2.0 code) is fully open. Use Google's synthetic data generation pipeline (Structural Causal Models) to produce your own checkpoint. Your weights = your license.
2. **Contact Google** — licensing terms may evolve. Non-commercial first, then relaxed licensing is a common release pattern.
3. **Use an alternative model** — e.g., TabPFN (PriorLabs offers commercial API tiers), or train a custom transformer on proprietary data using the open TabFM architecture.

> **Bottom line**: This repo demonstrates the Databricks integration pattern (HuggingFace → UC → Serving → Agent). For production/commercial deployment, swap in commercially-licensed weights or a differently-licensed model using the same pipeline.

---

## References

- [Google Research Blog: Introducing TabFM](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/)
- [TabFM on HuggingFace](https://huggingface.co/google/tabfm-1.0.0-pytorch)
- [TabFM GitHub](https://github.com/google-research/tabfm)
- [Databricks Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html)
- [Unity Catalog Functions](https://docs.databricks.com/en/sql/language-manual/sql-ref-functions-udf.html)
- [ai_query() Reference](https://docs.databricks.com/en/sql/language-manual/functions/ai_query.html)
- [Databricks Supervisor Agents](https://docs.databricks.com/en/generative-ai/agent-framework/index.html)

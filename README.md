# ML Playground (Dash + MLflow)

Small interactive playground to run classification experiments on the Titanic dataset and track runs with MLflow.

Features
- Configure features, scaling and model (LogisticRegression / RandomForest)
- Run experiments and log params/metrics/artifacts to MLflow
- Compare runs via table and radar chart

Quickstart (local)

1. Create a Python 3.11 venv and install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

2. Start MLflow server (recommended - SQLite backend)

```bash
# from project root
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

Set `MLFLOW_TRACKING_URI` to point to the server (e.g. `http://localhost:5000`) or omit to use a local `mlruns` folder.

3. Run the Dash app

```bash
python -m app.app
# open http://127.0.0.1:8050
```

Deploy to Google Cloud Run

1. Build a container image (example)

```bash
gcloud builds submit --tag gcr.io/PROJECT-ID/ml-playground
gcloud run deploy ml-playground --image gcr.io/PROJECT-ID/ml-playground --platform managed --region REGION --allow-unauthenticated
```

Notes about MLflow on GCP
- Cloud Run containers are stateless; for durable MLflow storage use Cloud SQL (backend-store) and a GCS bucket for artifacts.
- You can run `mlflow server --backend-store-uri mysql://... --default-artifact-root gs://...` on a VM or Cloud Run with appropriate permissions.

Files
- `app/` — Dash app and UI
- `ml/` — pipeline and training orchestration
- `experiments/` — MLflow wrapper utilities

License: MIT-style for the sample code (no license file included).
# ML Experiment Playground

An interactive machine learning experiment playground built with **Dash**, **Plotly**, **scikit-learn**, and **MLflow**.

Configure pipelines in the sidebar, run experiments, and compare results visually — all from your browser.

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Dash app

Always run from the **project root** (so MLflow can find its SQLite database at `./mlflow.db`):

```bash
python app/app.py
```

Open **http://localhost:8050** in your browser.

### 3. (Optional) Open the MLflow UI

In a separate terminal, from the project root:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open **http://localhost:5000** to browse all runs, compare metrics, and download logged model artefacts.

---

## How to use

| Step | Action |
|------|--------|
| 1 | **Select features** — pick any subset of the seven available columns |
| 2 | **Choose scaling** — None or StandardScaler |
| 3 | **Pick a model** — Logistic Regression or Random Forest |
| 4 | **Tune hyperparameters** — sliders update the model config |
| 5 | **Name the run** (optional) — helps you identify it in the table |
| 6 | Click **Run Experiment** |
| 7 | **Select rows** in the Experiment Runs table to compare them in the radar chart |

---

## Dataset

Titanic dataset loaded via `seaborn.load_dataset("titanic")`.
Seaborn downloads it automatically (~60 KB) on first use and caches it locally.

**Features (after preprocessing)**

| Column | Description |
|--------|-------------|
| `pclass` | Passenger class (1, 2, 3) |
| `age` | Age in years |
| `sibsp` | Number of siblings / spouses aboard |
| `parch` | Number of parents / children aboard |
| `fare` | Ticket fare |
| `sex` | Sex — female=0, male=1 |
| `embarked` | Port of embarkation — C=0, Q=1, S=2 |

**Target:** `survived` (0 = no, 1 = yes)

**Missing values:** numeric columns filled with column median; categorical columns filled with column mode.

---

## Project structure

```
ml-playground/
├── app/
│   ├── __init__.py
│   ├── app.py          # Dash layout and callbacks
│   └── components.py   # Sidebar UI component
├── ml/
│   ├── __init__.py
│   ├── pipeline.py     # sklearn Pipeline builder
│   └── train.py        # Dataset loading and model evaluation
├── experiments/
│   ├── __init__.py
│   └── tracker.py      # MLflow wrapper (start_run, log_*, list_runs)
├── requirements.txt
└── README.md
```

---

## MLflow artefacts

| Artefact | Location |
|----------|----------|
| Run database | `./mlflow.db` (SQLite, created automatically) |
| Model artefacts | `./mlruns/` (created automatically) |

Both are excluded from git via `.gitignore`.

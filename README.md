# EvidenceMap
Search, extract, compare and verify scientific evidence, with every result linked to its source passage.

Scope (MVP): machine learning + computer vision. See `configs/project.yaml`.

## Quick start
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,semantic,mlops]"
python -m pytest -q
python -m uvicorn backend.app.main:app --reload   # http://127.0.0.1:8000/health
```

## Current pipeline stage
The project uses a pretrained MiniLM transformer for semantic paper retrieval, with a lightweight lexical baseline for comparison. Its 15 OpenAlex queries cover the ML and computer-vision subtopics configured in `configs/project.yaml`.
Use the API endpoint:
```bash
curl "http://127.0.0.1:8000/search?q=deep%20learning&domain=machine%20learning&mode=semantic&limit=5"
```

Semantic retrieval uses the pretrained MiniLM transformer configured in `configs/project.yaml`. The first semantic request encodes the corpus and caches its embeddings locally; later runs reuse that cache.
```powershell
curl "http://127.0.0.1:8000/search?q=graph%20neural%20network%20recommendation&domain=machine%20learning&mode=semantic&limit=5"
```

Compare lexical retrieval with neural semantic retrieval and log both to MLflow:
```powershell
python -m pipelines.evaluation.track_search --top-k 5
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```
The current MLflow runs record runtime and retrieved results. Retrieval-quality metrics require human-judged relevant-paper labels; the OpenAlex search queries are not treated as ground truth.

## Data and search experiment tracking
Install the MLOps tools:
```powershell
python -m pip install -e ".[dev,mlops]"
```

Initialize DVC, set a machine-local remote, then track and upload the collected corpus:
```powershell
python -m dvc init
python -m dvc remote add --default --local local "$env:LOCALAPPDATA\EvidenceMap\dvc-storage"
python -m dvc add data/raw data/processed
python -m dvc push
```
The local remote is not shared with other machines. Commit the generated `.dvc` files and `.dvc/config` to Git; keep `.dvc/config.local` and DVC cache files local.

Run one MLflow experiment per configured query, then view results at `http://127.0.0.1:5000`:
```powershell
python -m pipelines.evaluation.track_search --top-k 5
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```
These runs record retrieval latency, result counts, and top results; relevance metrics need a human-labeled query set.

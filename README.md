# EvidenceMap
Search, extract, compare and verify scientific evidence, with every result linked to its source passage.

Scope (MVP): machine learning + computer vision. See `configs/project.yaml`.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
make install
cp .env.example .env
make test
make run   # http://127.0.0.1:8000/health
```

## Current pipeline stage
The project has an in-memory search index over the processed corpus. Its 15 OpenAlex queries cover the ML and computer-vision subtopics configured in `configs/project.yaml`.
Use the API endpoint:
```bash
curl "http://127.0.0.1:8000/search?q=deep%20learning&domain=machine%20learning&limit=5"
```

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

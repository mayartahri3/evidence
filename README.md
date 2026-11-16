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

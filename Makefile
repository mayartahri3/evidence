install:
	pip install -e ".[dev]"
test:
	pytest -q
run:
	uvicorn backend.app.main:app --reload
lint:
	ruff check .

install:
	pip install -e ".[dev,semantic,mlops]"
test:
	pytest -q
run:
	uvicorn backend.app.main:app --reload
lint:
	ruff check .

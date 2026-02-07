.PHONY: install run test lint format clean

install:
	pip install -r requirements.txt

run:
	streamlit run src/app.py

run-api:
	uvicorn src.api:app --reload --port 8000

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

clean:
	rm -rf __pycache__ .pytest_cache htmlcov .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

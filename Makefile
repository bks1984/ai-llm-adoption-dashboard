install:
	pip install -r requirements-dev.txt

run:
	streamlit run app.py

test:
	pytest

lint:
	ruff check .

sample:
	python scripts/generate_sample_data.py

data:
	python scripts/download_data.py

process:
	python scripts/build_processed.py

.PHONY: install run test lint sample data process

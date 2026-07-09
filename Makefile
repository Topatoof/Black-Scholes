.PHONY: install install-dev test run lint clean

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e .

install-dev: install
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v

run:
	$(VENV)/bin/streamlit run ui/app.py

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
